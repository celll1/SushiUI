"""Dataset index reconciliation independent of HTTP routing."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
import time
import warnings
from datetime import datetime
from typing import Optional

from PIL import Image
from sqlalchemy.orm import Session

from config.settings import settings
from database.models import Dataset, DatasetCaption, DatasetItem
from core.datasets.statistics import compute_tag_statistics
from utils import create_thumbnail, dataset_thumbnail_key
from utils.taglist_cache import taglist_cache


class DatasetScanNotFoundError(LookupError):
    pass


class DatasetScanPathError(ValueError):
    pass


def _send_progress(
    callback: Optional[Callable[[int, int, str], None]],
    step: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(step, total, message)


async def scan_dataset_index(
    dataset_id: int,
    db: Session,
    *,
    incremental: bool = False,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
):
    """Scan dataset directory and register images/captions.

    When *incremental* is True (training pre-flight rescan):
      - If nothing structurally changed (items_found==0, items_purged==0),
        the existing tag_statistics is kept as-is (Case 1).
      - Otherwise, tag_statistics is updated by adding/subtracting only
        the counts for new/purged items (Case 2).
    Both modes avoid the O(total_captions) full recomputation.

    Internal callers may stop the filesystem walk through ``should_cancel``.
    """
    warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise DatasetScanNotFoundError("Dataset not found")

    if not os.path.exists(dataset.path):
        raise DatasetScanPathError(f"Directory not found: {dataset.path}")

    # Auto-detect dataset structure before scanning
    from utils.dataset_structure_detector import detect_dataset_structure

    structure_detection_result = None
    if not dataset.reference_suffixes and not dataset.target_suffixes:
        print(f"[Dataset Scan] Auto-detecting dataset structure...")
        import asyncio
        _loop = asyncio.get_event_loop()
        structure_detection_result = await _loop.run_in_executor(
            None,
            lambda: detect_dataset_structure(
                dataset.path,
                recursive=dataset.recursive,
                max_depth=dataset.max_depth if hasattr(dataset, 'max_depth') and dataset.max_depth else None,
            )
        )

        if structure_detection_result["structure_type"] == "paired":
            dataset.reference_suffixes = structure_detection_result["reference_suffixes"]
            dataset.target_suffixes = structure_detection_result["target_suffixes"]
            dataset.caption_suffixes_for_reference = structure_detection_result.get("caption_suffixes_for_reference", [])
            db.commit()
            db.refresh(dataset)
            print(f"[Dataset Scan] Detected paired structure: "
                  f"ref={structure_detection_result['reference_suffixes']}, "
                  f"target={structure_detection_result['target_suffixes']}, "
                  f"caption={structure_detection_result.get('caption_suffixes_for_reference', [])}, "
                  f"confidence={structure_detection_result['confidence']:.3f}")
        else:
            print(f"[Dataset Scan] Normal dataset structure detected")
    else:
        print(f"[Dataset Scan] Using existing suffix configuration: "
              f"ref={dataset.reference_suffixes}, target={dataset.target_suffixes}")

    # Supported image + video + audio extensions
    from utils.dataset_scanner import (
        VIDEO_EXTS as video_exts,
        AUDIO_EXTS as audio_exts,
        probe_video_metadata,
        probe_audio_metadata,
        extract_poster_frame,
        relative_group_key,
    )
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    media_exts = image_exts | video_exts | audio_exts
    caption_exts = {".txt", ".json"}

    from utils.taglist_loader import load_all_tags
    from utils.caption_detector import classify_field, scan_json_fields, read_exif_captions
    print(f"[Dataset Scan] Loading taglist for format detection...")
    # Gelbooru supplement + alias table improve both format detection (match rate)
    # and category resolution; graceful when taglist_gel/ is absent.
    taglist = load_all_tags(settings.root_dir, include_gelbooru=True)
    taglist_cache.initialize(settings.root_dir, enable_gelbooru=True)
    print(f"[Dataset Scan] Loaded {len(taglist)} tags for format detection")

    read_exif_enabled = bool(getattr(dataset, "read_exif", False))
    exif_caption_fields = getattr(dataset, "exif_caption_fields", None) or None
    if read_exif_enabled:
        print(f"[Dataset Scan] read_exif enabled (fields={exif_caption_fields or 'default set'})")

    # Per-field-bucket scan counters (added/updated this run), so the progress &
    # result are intuitive instead of conflating images with caption rows. The
    # two training fields are reported on their own; everything else (image.*,
    # source.*, savedAt, exif.*, suffix fields) is aggregated as "other".
    #   tags = the danbooru-tags field;  caption = the natural-language field.
    _fstats = {"tags_add": 0, "tags_upd": 0, "cap_add": 0, "cap_upd": 0,
               "other_add": 0, "other_upd": 0}

    def _fbucket(caption_type: str) -> str:
        if caption_type == "tags":
            return "tags"
        if caption_type in ("natural_language", "caption"):
            return "cap"
        return "other"

    def _fstat_bump(caption_type: str, added: bool) -> None:
        _fstats[f"{_fbucket(caption_type)}_{'add' if added else 'upd'}"] += 1

    def _fstat_msg() -> str:
        """Compact per-field summary (+N new / ~N updated) for progress lines."""
        return (f"tags +{_fstats['tags_add']}/~{_fstats['tags_upd']} | "
                f"caption +{_fstats['cap_add']}/~{_fstats['cap_upd']} | "
                f"other +{_fstats['other_add']}/~{_fstats['other_upd']}")

    def _build_tag_data_json(content: str) -> str:
        """Build tag_data JSON string from comma-separated tag content."""
        import json as _json
        tags = [t.strip() for t in content.split(",") if t.strip()]
        if not tags:
            return "[]"
        cats = taglist_cache.get_categories_batch(tags)
        return _json.dumps(
            [{"tag": t, "category": cats.get(t, "Unknown")} for t in tags],
            ensure_ascii=False,
        )

    def _upsert_caption(item_id_local: int, result: dict) -> bool:
        """Insert or update a caption row keyed by (item_id, caption_type).

        Unifies JSON / EXIF field handling so a non-tags field (image.filename,
        source.*, exif.*, …) is UPDATED in place instead of being re-added on
        every rescan (which previously duplicated those rows). Returns True when a
        new row was added, False when an existing row was updated.
        """
        ctype = result["caption_type"]
        _is_tags = result["is_tags_format"]
        _tag_data = _build_tag_data_json(result["content"]) if _is_tags else None
        existing = db.query(DatasetCaption).filter(
            DatasetCaption.item_id == item_id_local,
            DatasetCaption.caption_type == ctype,
        ).first()
        if existing:
            existing.content = result["content"]
            existing.field_category = result["field_category"]
            existing.is_tags_format = _is_tags
            existing.tag_match_rate = result["tag_match_rate"]
            existing.source = "file"
            existing.source_field = result["source_field"]
            existing.tag_data = _tag_data
            existing.updated_at = datetime.utcnow()
            _fstat_bump(ctype, added=False)
            return False
        db.add(DatasetCaption(
            item_id=item_id_local,
            caption_type=ctype,
            content=result["content"],
            field_category=result["field_category"],
            is_tags_format=_is_tags,
            tag_match_rate=result["tag_match_rate"],
            tag_data=_tag_data,
            source="file",
            source_field=result["source_field"],
        ))
        _fstat_bump(ctype, added=True)
        return True

    # Pre-scan with 2-pass scanner: detect suffix captions + count images in one pass
    from utils.dataset_scanner import scan_directory_structure
    import asyncio
    print(f"[Dataset Scan] Pre-scanning directory structure...")
    loop = asyncio.get_event_loop()
    from core.training.rescan_control import RescanSkipped
    try:
        pre_scan_groups = await loop.run_in_executor(
            None,
            lambda: scan_directory_structure(
                dir_path=dataset.path,
                recursive=dataset.recursive,
                max_depth=dataset.max_depth if dataset.max_depth else None,
                reference_suffixes=dataset.reference_suffixes or [],
                target_suffixes=dataset.target_suffixes or [],
                should_cancel=should_cancel,
            )
        )
    except RescanSkipped:
        # Skipped during the pre-scan walk — nothing written yet.
        print(f"[Dataset Scan] Skipped during pre-scan walk for dataset {dataset_id}")
        return {
            "items_found": 0, "captions_found": 0, "captions_updated": 0,
            "items_purged": 0, "cancelled": True, "dataset": dataset.to_dict(),
        }

    suffix_captions_by_stem = {}
    detected_suffixes = set()
    total_images = 0
    for group_name, group_data in pre_scan_groups.items():
        # Count main/target images (not reference)
        total_images += sum(1 for img in group_data["images"] if img["role"] in ("main", "target"))
        # Collect suffix captions
        suffix_caps = [(c["suffix"], c["path"]) for c in group_data["captions"] if c["suffix"]]
        if suffix_caps:
            suffix_captions_by_stem[group_name] = suffix_caps
            for s, _ in suffix_caps:
                detected_suffixes.add(s)
    if detected_suffixes:
        print(f"[Dataset Scan] Detected caption suffixes: {sorted(detected_suffixes)}")
        existing_suffixes = dataset.caption_suffixes or []
        dataset.caption_suffixes = sorted(set(existing_suffixes) | detected_suffixes)

    print(f"[Dataset Scan] Found {total_images} images, {len(suffix_captions_by_stem)} groups with suffix captions")

    from sqlalchemy import exists

    has_file_caption = exists().where(
        DatasetCaption.item_id == DatasetItem.id,
        DatasetCaption.source == "file",
    )
    existing_items_rows = db.query(
        DatasetItem.id,
        DatasetItem.image_path,
        has_file_caption.label("has_file_caption"),
    ).filter(DatasetItem.dataset_id == dataset_id).all()
    existing_paths: dict[str, tuple[int, bool]] = {
        row.image_path: (row.id, bool(row.has_file_caption))
        for row in existing_items_rows
    }
    # Track which existing paths are still on disk (for purge at the end)
    seen_existing_paths: set[str] = set()
    # mtime threshold: captions updated after this are re-processed
    last_scanned_ts = dataset.last_scanned_at.timestamp() if dataset.last_scanned_at else 0.0
    # Track new item IDs for incremental tag_statistics update
    new_item_ids: list[int] = []
    print(f"[Dataset Scan] Loaded {len(existing_paths)} existing items from DB (path-based dedup)")

    # Scan directory
    items_found = 0
    captions_found = 0
    captions_updated = 0
    files_processed = 0
    revision_bumped = False

    def bump_scan_revision() -> None:
        nonlocal revision_bumped
        if revision_bumped:
            return
        from core.datasets.revisions import bump_dataset_revision

        bump_dataset_revision(dataset)
        revision_bumped = True

    # Progress tracking: Phase 1 (File scan): 0-90%, Phase 2 (Tag stats): 90-100%
    # We'll use a unified total_steps = total_images * 1.1 (rounded)
    # This way: file scan uses steps 0 to total_images (90.9%), tag stats uses remaining (9.1%)
    total_steps = int(total_images * 1.1) if total_images > 0 else 100

    print(f"[Dataset Scan] Sending initial progress to frontend...")
    _send_progress(progress, 0, total_steps, f"Starting scan: 0/{total_images} images to process")
    print(f"[Dataset Scan] Starting directory scan...")

    def scan_directory(dir_path, current_depth=0):
        nonlocal items_found, captions_found, captions_updated, files_processed

        # Cooperative cancellation (training pre-flight skip): checked per
        # directory and per image-group below. Raises RescanSkipped which the
        # registration await catches to commit partial progress.
        if should_cancel is not None and should_cancel():
            raise RescanSkipped()

        try:
            entries = os.listdir(dir_path)
        except PermissionError:
            print(f"[Dataset Scan] Permission denied: {dir_path}")
            return

        print(f"[Dataset Scan] Scanning directory: {dir_path} ({len(entries)} entries)")

        reference_suffixes = dataset.reference_suffixes or []
        target_suffixes = dataset.target_suffixes or []
        caption_suffixes_for_ref = dataset.caption_suffixes_for_reference or []

        use_reference_mode = bool(reference_suffixes and target_suffixes)
        if use_reference_mode:
            print(f"[Dataset Scan] Reference mode enabled: ref_suffixes={reference_suffixes}, target_suffixes={target_suffixes}")

        # Helper function to strip suffix and get base group name
        def get_group_name_and_type(filename):
            """
            For reference mode: extract group name and file type from filename.
            Example with suffixes ["_source"] and ["_target"]:
              - "20251026_01k8e370_01k8e370_source.webp" -> ("20251026_01k8e370_01k8e370", "reference")
              - "20251026_01k8e370_01k8e370_target.webp" -> ("20251026_01k8e370_01k8e370", "target")
              - "20251026_01k8e370_01k8e370_instruction.txt" -> ("20251026_01k8e370_01k8e370", "caption")
              - "normal_image.png" -> ("normal_image", "normal")
            """
            base_name, ext = os.path.splitext(filename)

            if use_reference_mode:
                for suffix in reference_suffixes:
                    if base_name.endswith(suffix):
                        group_name = base_name[:-len(suffix)]
                        return group_name, "reference"

                for suffix in target_suffixes:
                    if base_name.endswith(suffix):
                        group_name = base_name[:-len(suffix)]
                        return group_name, "target"

                for suffix in caption_suffixes_for_ref:
                    if base_name.endswith(suffix):
                        group_name = base_name[:-len(suffix)]
                        return group_name, "caption"

            # Normal mode: use base_name as group name
            return base_name, "normal"

        file_groups = {}
        entries_processed = 0
        for entry in entries:
            entries_processed += 1
            # Log progress every 10k entries
            if entries_processed % 10000 == 0:
                print(f"[Dataset Scan] Grouped {entries_processed}/{len(entries)} entries in {dir_path}")
            entry_path = os.path.join(dir_path, entry)

            if os.path.isfile(entry_path):
                base_name, ext = os.path.splitext(entry)
                ext_lower = ext.lower()
                group_name, file_type = get_group_name_and_type(entry)

                if group_name not in file_groups:
                    file_groups[group_name] = {
                        "images": [],       # Normal images (no suffix)
                        "captions": [],     # Normal captions (no suffix)
                        "reference": [],    # Reference images (_source suffix)
                        "target": [],       # Target images (_target suffix)
                        "ref_captions": [], # Reference mode captions (_instruction suffix)
                    }

                if ext_lower in media_exts:
                    if file_type == "reference":
                        file_groups[group_name]["reference"].append(entry_path)
                    elif file_type == "target":
                        file_groups[group_name]["target"].append(entry_path)
                    else:
                        file_groups[group_name]["images"].append(entry_path)
                elif ext_lower in caption_exts:
                    if file_type == "caption":
                        file_groups[group_name]["ref_captions"].append(entry_path)
                    else:
                        file_groups[group_name]["captions"].append(entry_path)

            elif os.path.isdir(entry_path) and dataset.recursive:
                max_depth = dataset.max_depth if dataset.max_depth else float('inf')
                if current_depth < max_depth:
                    scan_directory(entry_path, current_depth + 1)

        print(f"[Dataset Scan] Grouped {len(file_groups)} file groups in {dir_path}, starting processing...")

        groups_processed = 0
        for base_name, files in file_groups.items():
            groups_processed += 1
            # Log progress every 1k groups
            if groups_processed % 1000 == 0:
                print(f"[Dataset Scan] Processed {groups_processed}/{len(file_groups)} file groups in {dir_path}")
            # Cooperative cancellation: poll every 200 image-groups so a skip
            # aborts a large flat directory promptly (RescanSkipped propagates).
            if should_cancel is not None and groups_processed % 200 == 0 and should_cancel():
                raise RescanSkipped()

            # Determine which image to use as main and which as reference
            main_images = []
            reference_images = []
            caption_files = []

            if use_reference_mode:
                # Reference mode: use target as main, reference as related
                main_images = files["target"]
                reference_images = files["reference"]
                caption_files = files["ref_captions"] if files["ref_captions"] else files["captions"]
            else:
                # Normal mode: use images as main, no reference
                main_images = files["images"]
                caption_files = files["captions"]

            if not main_images:
                continue

            image_path = main_images[0]
            group_key = relative_group_key(dataset.path, dir_path, base_name)

            _t_item = time.time()
            try:
                # --- Path-based dedup (replaces SHA256 hash + per-item DB query) ---
                # Decide existing-vs-new BEFORE opening the image so unchanged
                # existing items skip the PIL open entirely — their dimensions are
                # already stored. Only NEW images need to be opened for width/height.
                existing_entry = existing_paths.get(image_path)
                caption_refresh_started = None
                if existing_entry is not None:
                    existing_item_id, had_file_caption = existing_entry
                    # Image already registered — mark as seen (for purge logic)
                    seen_existing_paths.add(image_path)
                    files_processed += 1

                    any_caption_updated = False
                    for cp in caption_files:
                        try:
                            if os.path.getmtime(cp) > last_scanned_ts:
                                any_caption_updated = True
                                break
                        except OSError:
                            pass
                    # Also check suffix captions
                    if not any_caption_updated and group_key in suffix_captions_by_stem:
                        for _, sp in suffix_captions_by_stem[group_key]:
                            try:
                                if os.path.getmtime(sp) > last_scanned_ts:
                                    any_caption_updated = True
                                    break
                            except OSError:
                                pass
                    # With read_exif on, a newer image file can itself carry updated
                    # embedded captions — treat a newer image mtime as an update.
                    if not any_caption_updated and read_exif_enabled:
                        try:
                            if os.path.getmtime(image_path) > last_scanned_ts:
                                any_caption_updated = True
                        except OSError:
                            pass

                    has_current_sidecar = bool(
                        caption_files or suffix_captions_by_stem.get(group_key)
                    )
                    if had_file_caption and not has_current_sidecar and not read_exif_enabled:
                        any_caption_updated = True

                    if not any_caption_updated:
                        # No changes — skip entirely (no Image.open: dimensions are
                        # already stored on the existing item).
                        if files_processed % 10 == 0 or total_images < 100:
                            _send_progress(progress, 
                                files_processed,
                                total_steps,
                                f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                            )
                        continue

                    # Captions updated — re-process for this existing item. No
                    # Image.open needed (dimensions already in DB).
                    item_id_for_captions = existing_item_id
                    caption_refresh_started = datetime.utcnow()
                    if files_processed % 10 == 0 or total_images < 100:
                        _send_progress(progress, 
                            files_processed,
                            total_steps,
                            f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                        )
                else:
                    _ext_lower = os.path.splitext(image_path)[1].lower()
                    is_video = _ext_lower in video_exts
                    is_audio = _ext_lower in audio_exts
                    video_meta = None
                    audio_meta = None

                    if is_video:
                        # New video — probe metadata via ffprobe WITHOUT decoding
                        # all frames. A probe failure skips the file (logged).
                        video_meta = probe_video_metadata(image_path)
                        if not video_meta:
                            print(f"[Dataset Scan] Skipping unreadable video {image_path}")
                            files_processed += 1
                            if files_processed % 10 == 0 or total_images < 100:
                                _send_progress(progress, 
                                    files_processed,
                                    total_steps,
                                    f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                                )
                            continue
                        width = video_meta["width"]
                        height = video_meta["height"]
                    elif is_audio:
                        # New audio clip — probe metadata via soundfile/ffprobe
                        # WITHOUT decoding the whole file. Audio clips have no
                        # spatial dimensions (width/height stored as 0).
                        audio_meta = probe_audio_metadata(image_path)
                        if not audio_meta:
                            print(f"[Dataset Scan] Skipping unreadable audio {image_path}")
                            files_processed += 1
                            if files_processed % 10 == 0 or total_images < 100:
                                _send_progress(progress, 
                                    files_processed,
                                    total_steps,
                                    f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                                )
                            continue
                        width = 0
                        height = 0
                    else:
                        # New image — open it ONCE for dimensions, then register.
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                with Image.open(image_path) as img:
                                    width, height = img.size
                        except Exception as img_error:
                            # Skip images that can't be opened (corrupt, unsupported, etc.)
                            print(f"[Dataset Scan] Skipping corrupt/unsupported image {image_path}: {img_error}")
                            files_processed += 1
                            if files_processed % 10 == 0 or total_images < 100:
                                _send_progress(progress, 
                                    files_processed,
                                    total_steps,
                                    f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                                )
                            continue

                    file_size = os.path.getsize(image_path)

                    related_images_data = {}
                    if use_reference_mode and reference_images:
                        related_images_data["reference"] = reference_images
                        print(f"[Dataset Scan] Group '{base_name}': {len(reference_images)} reference image(s)")

                    if is_video:
                        item_type = "video"
                    elif is_audio:
                        item_type = "audio"
                    elif use_reference_mode:
                        item_type = "reference"
                    else:
                        item_type = "single"

                    item = DatasetItem(
                        dataset_id=dataset_id,
                        # image_path stores the video/audio file path for
                        # video/audio items (it is just a path string).
                        # Per-clip metadata lives in exif_data (surfaced as
                        # video_meta / audio_meta in to_dict).
                        item_type=item_type,
                        base_name=base_name,
                        image_path=image_path,
                        width=width,
                        height=height,
                        file_size=file_size,
                        image_hash=None,  # SHA256 no longer computed at scan time
                        exif_data=video_meta if is_video else (audio_meta if is_audio else None),
                        related_images=related_images_data if related_images_data else None
                    )
                    db.add(item)
                    db.flush()  # Get item.id
                    item_id_for_captions = item.id
                    items_found += 1
                    files_processed += 1
                    new_item_ids.append(item.id)

                    # Poster thumbnail for videos: extract frame 0 via cv2 and run
                    # it through the shared PNG+WebP thumbnail generator keyed by
                    # base_name, so the dataset UI has a preview to show.
                    if is_video:
                        try:
                            import tempfile
                            poster_tmp = os.path.join(tempfile.gettempdir(), f"_dsposter_{base_name}.png")
                            if extract_poster_frame(image_path, poster_tmp):
                                # create_thumbnail keys the output by the source
                                # basename; rename target so it matches base_name.
                                poster_named = os.path.join(tempfile.gettempdir(), f"{base_name}.png")
                                try:
                                    if poster_tmp != poster_named:
                                        os.replace(poster_tmp, poster_named)
                                    create_thumbnail(
                                        poster_named,
                                        output_key=dataset_thumbnail_key(image_path),
                                    )
                                finally:
                                    for _p in (poster_tmp, poster_named):
                                        try:
                                            os.remove(_p)
                                        except OSError:
                                            pass
                        except Exception as _pe:
                            print(f"[Dataset Scan] poster thumbnail failed for {image_path}: {_pe}")

                    # Waveform thumbnail for audio clips: render a peak-envelope
                    # PNG via soundfile + the shared audio waveform writer, then
                    # run it through the same PNG+WebP thumbnail generator keyed
                    # by base_name, so the dataset UI has a preview to show
                    # (mirrors the video poster-frame path above).
                    if is_audio:
                        try:
                            import tempfile
                            import soundfile as sf
                            from utils.audio_utils import _write_waveform_png
                            wave_named = os.path.join(tempfile.gettempdir(), f"{base_name}.png")
                            try:
                                data, _sr = sf.read(image_path, dtype="float32", always_2d=True)
                                # soundfile returns [samples, channels]; waveform
                                # writer expects [channels, samples].
                                arr = data.T
                                _write_waveform_png(arr, wave_named)
                                create_thumbnail(
                                    wave_named,
                                    output_key=dataset_thumbnail_key(image_path),
                                )
                            finally:
                                try:
                                    os.remove(wave_named)
                                except OSError:
                                    pass
                        except Exception as _ae:
                            print(f"[Dataset Scan] waveform thumbnail failed for {image_path}: {_ae}")

                    if files_processed % 10 == 0 or total_images < 100:
                        _send_progress(progress, 
                            files_processed,
                            total_steps,
                            f"Scanning: {files_processed}/{total_images} images | {items_found} new img | {_fstat_msg()}"
                        )

                _t_caps = time.time()
                _jr = _sjf = _ups = 0.0  # json-read / scan_json_fields / upsert sub-times
                _cf = _btd = _sfx = _exif = 0.0  # classify / build_tag_data / suffix / exif
                _txr = _txq = 0.0  # .txt file read / .txt migration query
                for caption_path in caption_files:
                    try:
                        _, ext = os.path.splitext(caption_path)
                        ext_lower = ext.lower()

                        if ext_lower == '.txt':
                            # TXT file: Read content and detect format
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                _ts = time.time()
                                content = f.read().strip()
                                _txr += time.time() - _ts
                                if content:
                                    _ts = time.time()
                                    field_category, is_tags_format, match_rate = classify_field("tags", content, taglist)
                                    _cf += time.time() - _ts

                                    # Determine caption_type based on detected format
                                    detected_caption_type = "tags" if is_tags_format else "natural_language"

                                    # A .txt sidecar yields exactly ONE caption whose type ('tags' or
                                    # 'natural_language') depends on detection. Find it regardless of its
                                    # CURRENT stored type so a re-detection that flips the type — e.g. a
                                    # fixed detector now recognising a sidecar as tags, or a repaired
                                    # sidecar — MIGRATES the same row instead of leaving a stale
                                    # natural_language row and adding a duplicate tags row.
                                    _ts = time.time()
                                    existing_cap = db.query(DatasetCaption).filter(
                                        DatasetCaption.item_id == item_id_for_captions,
                                        DatasetCaption.source == "file",
                                        DatasetCaption.caption_type.in_(["tags", "natural_language"]),
                                    ).first()
                                    _txq += time.time() - _ts

                                    if existing_cap:
                                        existing_cap.caption_type = detected_caption_type
                                        existing_cap.content = content
                                        existing_cap.field_category = field_category
                                        existing_cap.is_tags_format = is_tags_format
                                        existing_cap.tag_match_rate = match_rate
                                        existing_cap.source = "file"
                                        existing_cap.source_field = detected_caption_type
                                        _ts = time.time()
                                        existing_cap.tag_data = _build_tag_data_json(content) if is_tags_format else None
                                        _btd += time.time() - _ts
                                        existing_cap.updated_at = datetime.utcnow()
                                        captions_updated += 1
                                        _fstat_bump(detected_caption_type, added=False)
                                    else:
                                        _ts = time.time()
                                        _td_new = _build_tag_data_json(content) if is_tags_format else None
                                        _btd += time.time() - _ts
                                        caption = DatasetCaption(
                                            item_id=item_id_for_captions,
                                            caption_type=detected_caption_type,
                                            content=content,
                                            field_category=field_category,
                                            is_tags_format=is_tags_format,
                                            tag_match_rate=match_rate,
                                            tag_data=_td_new,
                                            source="file",
                                            source_field=detected_caption_type
                                        )
                                        db.add(caption)
                                        captions_found += 1
                                        _fstat_bump(detected_caption_type, added=True)

                        elif ext_lower == '.json':
                            # JSON file: Recursively scan all fields
                            import json

                            _ts = time.time()
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            _jr += time.time() - _ts

                            # Scan all fields. Every field (the single tags field
                            # AND the non-tags fields) is upserted by caption_type,
                            # so a rescan UPDATES each row in place instead of
                            # re-adding non-tags fields (which previously duplicated
                            # them on every scan).
                            _ts = time.time()
                            caption_results = scan_json_fields(json_data, taglist)
                            _sjf += time.time() - _ts
                            _ts = time.time()
                            for result in caption_results:
                                if _upsert_caption(item_id_for_captions, result):
                                    captions_found += 1
                                else:
                                    captions_updated += 1
                            _ups += time.time() - _ts

                    except Exception as e:
                        print(f"[Dataset Scan] Failed to read caption {caption_path}: {e}")

                _ts = time.time()
                if group_key in suffix_captions_by_stem:
                    for suffix, suffix_path in suffix_captions_by_stem[group_key]:
                        try:
                            _, sext = os.path.splitext(suffix_path)
                            if sext.lower() == '.txt':
                                with open(suffix_path, 'r', encoding='utf-8') as f:
                                    content = f.read().strip()
                                if content:
                                    field_category, is_tags_format, match_rate = classify_field(
                                        suffix, content, taglist
                                    )
                                    existing_cap = db.query(DatasetCaption).filter(
                                        DatasetCaption.item_id == item_id_for_captions,
                                        DatasetCaption.caption_type == suffix
                                    ).first()
                                    if existing_cap:
                                        existing_cap.content = content
                                        existing_cap.field_category = field_category
                                        existing_cap.is_tags_format = is_tags_format
                                        existing_cap.tag_match_rate = match_rate
                                        existing_cap.source = "file"
                                        existing_cap.source_field = suffix
                                        if is_tags_format:
                                            existing_cap.tag_data = _build_tag_data_json(content)
                                        existing_cap.updated_at = datetime.utcnow()
                                        captions_updated += 1
                                        _fstat_bump(suffix, added=False)
                                    else:
                                        caption = DatasetCaption(
                                            item_id=item_id_for_captions,
                                            caption_type=suffix,
                                            content=content,
                                            field_category=field_category,
                                            is_tags_format=is_tags_format,
                                            tag_match_rate=match_rate,
                                            tag_data=_build_tag_data_json(content) if is_tags_format else None,
                                            source="file",
                                            source_field=suffix
                                        )
                                        db.add(caption)
                                        captions_found += 1
                                        _fstat_bump(suffix, added=True)
                        except Exception as e:
                            print(f"[Dataset Scan] Failed to read suffix caption {suffix_path}: {e}")
                _sfx += time.time() - _ts

                _ts = time.time()
                if read_exif_enabled:
                    try:
                        for result in read_exif_captions(image_path, taglist, exif_caption_fields):
                            if _upsert_caption(item_id_for_captions, result):
                                captions_found += 1
                            else:
                                captions_updated += 1
                    except Exception as e:
                        print(f"[Dataset Scan] Failed to read EXIF captions for {image_path}: {e}")
                _exif += time.time() - _ts

                if caption_refresh_started is not None:
                    db.flush()
                    stale_file_captions = db.query(DatasetCaption).filter(
                        DatasetCaption.item_id == item_id_for_captions,
                        DatasetCaption.source == "file",
                        (
                            DatasetCaption.updated_at.is_(None)
                            | (DatasetCaption.updated_at < caption_refresh_started)
                        ),
                    )
                    removed_captions = stale_file_captions.count()
                    if removed_captions:
                        stale_file_captions.delete(synchronize_session=False)
                        captions_updated += removed_captions

                # Per-item timing probe: surface which items (and which phase) stall
                # in the live backend, since every phase is fast in isolation.
                _caps_ms = (time.time() - _t_caps) * 1000
                _item_ms = (time.time() - _t_item) * 1000
                if _item_ms > 200:
                    _has_json = any(str(c).lower().endswith(".json") for c in caption_files)
                    print(f"[Dataset Scan][SLOW] {_item_ms:.0f}ms (caps {_caps_ms:.0f} = "
                          f"jsonRead {_jr*1000:.0f} + scanFields {_sjf*1000:.0f} + upsert {_ups*1000:.0f} + "
                          f"classify {_cf*1000:.0f} + buildTagData {_btd*1000:.0f} + txtRead {_txr*1000:.0f} + "
                          f"txtQuery {_txq*1000:.0f} + suffix {_sfx*1000:.0f} + "
                          f"exif {_exif*1000:.0f}) json={_has_json} ncaps={len(caption_files)} {os.path.basename(image_path)}")

            except Exception as e:
                print(f"[Dataset Scan] Failed to process image {image_path}: {e}")

            # Periodic commit: flush accumulated changes and let SQLAlchemy release
            # them. With autoflush off and a single end-of-scan commit, every
            # touched caption ORM object stays pinned (dirty objects are strong-refs
            # in the unit of work) for the whole scan — so the session grows to
            # tens of thousands of objects and per-item cost climbs (measured ~4x
            # slower, with the growth ~11x faster for JSON sidecars that yield
            # ~11 caption rows per image). Committing in batches keeps the working
            # set bounded and per-item cost flat. Partial progress is committed,
            # which also matches the cancel/skip-commits-partial behaviour.
            if files_processed > 0 and files_processed % 300 == 0:
                try:
                    if items_found or captions_found or captions_updated:
                        bump_scan_revision()
                    db.commit()
                except Exception as _ce:
                    print(f"[Dataset Scan] Periodic commit failed: {_ce}")

    # Run scan in thread pool to avoid blocking event loop (enables WebSocket progress updates)
    # SQLite is configured with check_same_thread=False, so cross-thread access is safe
    import asyncio
    loop = asyncio.get_event_loop()
    _scan_cancelled = False
    try:
        await loop.run_in_executor(None, lambda: scan_directory(dataset.path))
    except RescanSkipped:
        _scan_cancelled = True
        print(f"[Dataset Scan] Rescan skipped mid-walk for dataset {dataset_id}; "
              f"committing {items_found} new items, skipping purge")

    if _scan_cancelled:
        # Skip the purge: we did not finish seeing every on-disk file, so the
        # stale-path diff is incomplete and purging would wrongly delete items
        # we simply hadn't reached. Commit the new items/captions added so far
        # and leave last_scanned_at unchanged so the next pre-flight re-detects
        # drift (already-applied changes stay, per the skip contract).
        if items_found or captions_found or captions_updated:
            bump_scan_revision()
        db.commit()
        db.refresh(dataset)
        _send_progress(progress, 
            total_steps, total_steps,
            f"Rescan skipped: committed {items_found} new items (partial)"
        )
        return {
            "items_found": items_found,
            "captions_found": captions_found,
            "captions_updated": captions_updated,
            "items_purged": 0,
            "cancelled": True,
            "dataset": dataset.to_dict(),
        }

    stale_paths = set(existing_paths.keys()) - seen_existing_paths
    items_purged = 0
    # For incremental mode: read purged captions BEFORE deletion so we can
    # subtract their tag counts from the existing tag_statistics.
    purged_tag_counts: dict[str, int] = {}   # tag -> count to subtract
    if stale_paths:
        stale_item_ids = [existing_paths[p][0] for p in stale_paths]
        if incremental and dataset.tag_statistics:
            import json as _json_purge
            purged_caps = db.query(DatasetCaption).filter(
                DatasetCaption.item_id.in_(stale_item_ids),
                DatasetCaption.caption_type == "tags",
            ).all()
            for cap in purged_caps:
                tags: list[str] = []
                if cap.tag_data:
                    try:
                        tags = [t.get("tag", "").strip() for t in _json_purge.loads(cap.tag_data)]
                    except Exception:
                        pass
                if not tags and cap.content:
                    tags = [t.strip() for t in cap.content.split(",")]
                for tag in tags:
                    if tag:
                        purged_tag_counts[tag] = purged_tag_counts.get(tag, 0) + 1
        db.query(DatasetCaption).filter(
            DatasetCaption.item_id.in_(stale_item_ids)
        ).delete(synchronize_session=False)
        db.query(DatasetItem).filter(
            DatasetItem.id.in_(stale_item_ids)
        ).delete(synchronize_session=False)
        items_purged = len(stale_item_ids)
        print(f"[Dataset Scan] Purged {items_purged} items whose files no longer exist on disk")

    # File scan complete - progress is now at ~90%
    _send_progress(progress, 
        total_images,
        total_steps,
        f"File scan complete: {files_processed} processed, {items_purged} purged | Starting tag statistics..."
    )

    # Compute tag statistics -----------------------------------------------
    # incremental=True (training rescan):
    #   Case 1: no structural change → keep existing stats as-is
    #   Case 2: structural change → differential update (add new, subtract purged)
    # incremental=False (regular UI scan): always full recompute
    if incremental:
        existing_stats: dict = dataset.tag_statistics or {}
        if (
            items_found == 0
            and items_purged == 0
            and captions_found == 0
            and captions_updated == 0
        ):
            # Case 1: nothing changed structurally — reuse cached stats
            print(f"[Dataset Scan] No structural change — reusing cached tag statistics ({len(existing_stats)} tags)")
            tag_statistics = existing_stats
        elif captions_found or captions_updated:
            print("[Dataset Scan] Caption changes detected — rebuilding tag statistics")
            tag_statistics = compute_tag_statistics(
                dataset_id,
                db,
                root_dir=settings.root_dir,
                progress=lambda processed, unique_tags: _send_progress(
                    progress,
                    total_images,
                    total_steps,
                    f"Computing tag statistics: {processed} captions, {unique_tags} unique tags",
                ),
            )
        else:
            # Case 2: differential update
            print(f"[Dataset Scan] Incremental tag statistics update: -{items_purged} / +{items_found} items")
            import json as _json_incr
            stats: dict = {tag: dict(v) for tag, v in existing_stats.items()}

            # Subtract counts for purged items (collected before deletion)
            for tag, cnt in purged_tag_counts.items():
                if tag in stats:
                    stats[tag]["count"] -= cnt
                    if stats[tag]["count"] <= 0:
                        del stats[tag]

            if new_item_ids:
                new_caps = db.query(DatasetCaption).filter(
                    DatasetCaption.item_id.in_(new_item_ids),
                    DatasetCaption.caption_type == "tags",
                ).all()
                for cap in new_caps:
                    tag_cat_pairs: list[tuple[str, str]] = []
                    if cap.tag_data:
                        try:
                            tag_cat_pairs = [
                                (t.get("tag", "").strip(), t.get("category", "Unknown"))
                                for t in _json_incr.loads(cap.tag_data)
                            ]
                        except Exception:
                            pass
                    if not tag_cat_pairs and cap.content:
                        tag_cat_pairs = [(t.strip(), "Unknown") for t in cap.content.split(",")]
                    for tag, category in tag_cat_pairs:
                        if not tag:
                            continue
                        if tag in stats:
                            stats[tag]["count"] += 1
                            # Upgrade category if currently Unknown
                            if stats[tag]["category"] == "Unknown" and category != "Unknown":
                                stats[tag]["category"] = category
                        else:
                            # Resolve Unknown categories via taglist_cache
                            if category == "Unknown":
                                resolved = taglist_cache.get_categories_batch([tag])
                                category = resolved.get(tag, "Unknown")
                            stats[tag] = {"count": 1, "category": category}

            tag_statistics = stats
            print(f"[Dataset Scan] Incremental update complete: {len(tag_statistics)} unique tags")
    else:
        print(f"[Dataset Scan] Computing tag statistics...")
        tag_statistics = compute_tag_statistics(
            dataset_id,
            db,
            root_dir=settings.root_dir,
            progress=lambda processed, unique_tags: _send_progress(
                progress,
                total_images,
                total_steps,
                f"Computing tag statistics: {processed} captions, {unique_tags} unique tags",
            ),
        )

    _send_progress(progress, 
        total_steps,
        total_steps,
        f"Scan complete: {items_found} new images, {items_purged} purged | "
        f"{_fstat_msg()} | {len(tag_statistics)} unique tags"
    )

    # Normalize is_tags_format by majority vote per caption_type
    # Prevents a few misdetected tag-format files from causing issues
    # in predominantly natural-language datasets (and vice versa)
    caption_types_in_dataset = db.query(DatasetCaption.caption_type).filter(
        DatasetCaption.item_id.in_(
            db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset_id)
        )
    ).distinct().all()
    normalized_captions = 0
    for (ct,) in caption_types_in_dataset:
        captions_of_type = db.query(DatasetCaption).filter(
            DatasetCaption.item_id.in_(
                db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset_id)
            ),
            DatasetCaption.caption_type == ct
        ).all()
        if not captions_of_type:
            continue
        tags_count = sum(1 for c in captions_of_type if c.is_tags_format)
        nl_count = len(captions_of_type) - tags_count
        majority_is_tags = tags_count > nl_count
        minority_count = nl_count if majority_is_tags else tags_count
        if minority_count > 0:
            normalized_captions += minority_count
            majority_type_name = "tags" if majority_is_tags else "natural_language"
            print(f"[Dataset Scan] caption_type='{ct}': {tags_count} tags, {nl_count} NL -> "
                  f"normalizing {minority_count} to {majority_type_name}")
            for c in captions_of_type:
                if c.is_tags_format != majority_is_tags:
                    c.is_tags_format = majority_is_tags
                    # Also update caption_type for default txt fields
                    if ct in ("tags", "natural_language"):
                        c.caption_type = majority_type_name

    # Update dataset statistics (count all items in DB, not just newly added).
    # total_tags = images with danbooru tags; total_captions = images with a
    from core.datasets.statistics import caption_item_counts

    dataset.total_items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).count()
    dataset.total_tags, dataset.total_captions = caption_item_counts(db, dataset_id)
    dataset.tag_statistics = tag_statistics
    dataset.last_scanned_at = datetime.utcnow()

    if items_found or captions_found or captions_updated or items_purged or normalized_captions:
        bump_scan_revision()

    db.commit()
    db.refresh(dataset)

    # Per-field scan summary. The two training fields (tags / caption) report
    # updated this run, how many images currently HAVE that field, and the total
    # image count; "other" aggregates the metadata fields (image.*, source.*,
    # exif.*, …). total_tags/total_captions = images-with-that-field (see
    # caption_item_counts()).
    field_summary = {
        "total_images": dataset.total_items,
        "tags":    {"added": _fstats["tags_add"],  "updated": _fstats["tags_upd"],
                    "images_with": dataset.total_tags},
        "caption": {"added": _fstats["cap_add"],   "updated": _fstats["cap_upd"],
                    "images_with": dataset.total_captions},
        "other":   {"added": _fstats["other_add"], "updated": _fstats["other_upd"]},
    }

    response = {
        "items_found": items_found,
        "captions_found": captions_found,
        "captions_updated": captions_updated,
        "items_purged": items_purged,
        "field_summary": field_summary,
        "dataset": dataset.to_dict(),
    }

    # Include auto-detection result if detection was performed
    if structure_detection_result is not None:
        response["structure_detection"] = structure_detection_result

    return response
