"""
Dataset for SigLIP2 tagger training.

Loads images from sushiUI DatasetItems + DatasetCaption(is_tags_format=True),
produces (pixel_values, pixel_attention_mask, spatial_shapes, label, loss_mask).
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoProcessor

from .tag_vocabulary import (
    QUALITY_TAG_GROUPS,
    RATING_TAGS,
    TagVocabulary,
    normalize_tag,
)


def resolve_caption_tags(
    tag_data, content, comma_resolver, alias_resolver
) -> List[str]:
    """Turn a caption's raw ``tag_data`` / ``content`` into canonical tags.

    Shared by ``TaggerDataset._extract_tags`` (dataset build) and the live
    tag-refresh detector so a tag edited mid-training canonicalises exactly the
    same way it did at build time. ``tag_data`` (JSON list of ``{"tag": ...}``)
    takes precedence over the comma-separated ``content`` string.
    """
    raw_tags: List[str] = []
    if tag_data:
        try:
            raw = json.loads(tag_data) if isinstance(tag_data, str) else tag_data
            if isinstance(raw, list):
                raw_tags = [r["tag"] for r in raw if isinstance(r, dict) and "tag" in r]
        except (json.JSONDecodeError, TypeError):
            pass
    if not raw_tags and content:
        raw_tags = [t.strip() for t in content.split(",") if t.strip()]
    # Normalize first (order preserved) so the comma resolver matches the same
    # forms the vocabulary builder used.
    norm_tokens = [t for t in (normalize_tag(t) for t in raw_tags) if t]
    if comma_resolver is not None:
        norm_tokens = comma_resolver.resolve(norm_tokens)
    if alias_resolver:
        return [
            t if (comma_resolver is not None and comma_resolver.category_of(t) is not None)
            else alias_resolver.resolve(t)
            for t in norm_tokens
        ]
    return norm_tokens


class TaggerDataset(Dataset):
    """PyTorch Dataset for tagger training.

    Each item returns:
        pixel_values         : Tensor [num_patches, 768]     (NaFlex patch-flattened)
        pixel_attention_mask : Tensor [num_patches]           (int32)
        spatial_shapes       : Tensor [2]                     (int64, height x width patches)
        label                : Tensor [num_tags]              (float32, multi-hot)
        loss_mask            : Tensor [num_tags]              (float32, 0=ignore 1=use)
    """

    def __init__(
        self,
        dataset_ids: List[int],
        vocabulary: TagVocabulary,
        datasets_db,
        processor: AutoProcessor,
        caption_types: Optional[List[str]] = None,
        alias_resolver=None,
        comma_resolver=None,
        quality_masking_mode: str = "intra_group",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        dataset_ids    : list of Dataset.id to include
        vocabulary     : TagVocabulary built from the same datasets
        datasets_db    : SQLAlchemy session for datasets.db
        processor      : SigLIP2 AutoProcessor (handles NaFlex preprocessing)
        caption_types  : restrict to these caption_type values (None = all tags-format)
        alias_resolver : optional TagAliasResolver; when provided, deprecated
                         tags are resolved to canonical form during label construction
        comma_resolver : optional CommaTagResolver; when provided, comma-split tag
                         fragments are re-merged / aliased into comma-free canonical
                         tags per caption. MUST be the same resolver used to build
                         the vocabulary, or labels will not match vocab indices.
        quality_masking_mode : how to build loss_mask for Quality tags when at
                         least one Quality tag is present on a sample.
                         - "intra_group" (default, tagutl-style): the labelled
                           tag's group siblings are masked (treated as ignore);
                           tags in the *other* quality group remain unmasked
                           and train as negatives.  Safer when intra-group
                           label noise / prevalence imbalance is significant
                           (e.g. "best_quality" and "normal_quality" assigned
                           somewhat arbitrarily — training "best=0" on every
                           "normal_quality" sample creates noisy gradient).
                         - "cross_group" (previous SushiUI default): all non-
                           positive Quality tags train as negatives.  Correct
                           only when intra-group labels are truly mutually
                           exclusive and prevalences are balanced.
        """
        self.vocabulary = vocabulary
        self.processor = processor
        self.num_tags = vocabulary.num_tags
        self._alias_resolver = alias_resolver
        self._comma_resolver = comma_resolver
        if quality_masking_mode not in ("intra_group", "cross_group"):
            print(f"[TaggerDataset] Unknown quality_masking_mode={quality_masking_mode!r}, "
                  f"falling back to 'intra_group'")
            quality_masking_mode = "intra_group"
        self.quality_masking_mode = quality_masking_mode
        print(f"[TaggerDataset] Quality masking mode: {quality_masking_mode}")

        # Detect NaFlex vs standard by probing the processor output
        _probe = processor(images=[Image.new("RGB", (64, 64))], return_tensors="pt")
        self.is_naflex = "pixel_attention_mask" in _probe and "spatial_shapes" in _probe
        print(f"[TaggerDataset] Processor mode: {'NaFlex' if self.is_naflex else 'standard (fixed resolution)'}")

        self._samples: List[Tuple[str, List[str]]] = []  # (image_path, [tag, ...])
        # item_id aligned to _samples (sample idx -> item_id). Built during
        # _build_samples; used ONLY by the live tag-refresh detector in the main
        # process. Stripped from the worker pickle (see __getstate__) since
        # __getitem__ never needs it.
        self._item_ids: Optional["np.ndarray"] = None
        self._item_ids_list: List[int] = []
        # Live tag-refresh (Option B). Paths/flag are set by the trainer after
        # construction (before the first DataLoader spawn) and DO travel into the
        # worker pickle; the reader/mmap below are worker-local and rebuilt there.
        self._refresh_enabled: bool = False
        self._refresh_gen_path: Optional[str] = None
        self._refresh_payload_path: Optional[str] = None
        self._refresh_reader = None  # core.tagger.tag_refresh.TagRefreshReader (worker-local)
        # NOTE: progress_callback is a (closure) callable used only during
        # construction — do NOT store it on self. The dataset is pickled to
        # DataLoader worker processes (num_workers>0, Windows spawn) and a local
        # closure is unpicklable ("Can't pickle local object ..._ds_progress").
        self._build_samples(dataset_ids, datasets_db, caption_types, progress_callback)
        # The alias/comma resolvers are used ONLY during _build_samples (tag
        # canonicalisation); __getitem__ never touches them. Drop them so they are
        # not pickled into every DataLoader worker (Windows uses spawn, which
        # copies the whole dataset object per worker). The comma resolver in
        # particular is built from the full ~1.9M-tag category map, so keeping it
        # would waste roughly its size x num_workers of RAM for nothing.
        self._alias_resolver = None
        self._comma_resolver = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_samples(
        self,
        dataset_ids: List[int],
        datasets_db,
        caption_types: Optional[List[str]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        from database.models import DatasetItem, DatasetCaption

        # Throttled progress reporter (drives the frontend pre-training bar via
        # send_progress_sync upstream). Emits at most ~3x/sec so the WS/SSE
        # channel is not flooded while loading multi-million-item datasets.
        _last_emit = [0.0]
        n_datasets = len(dataset_ids)

        # Tag-string dedup pool. Tags repeat massively across samples (e.g. "1girl"
        # is on millions of images); without dedup each occurrence is a separate
        # str object, bloating _samples and — critically on Windows spawn — making
        # the per-worker pickle huge. Interning to one object per unique tag value
        # collapses millions of tag strings down to the few-hundred-thousand unique
        # ones (pickle also stores each shared object once).
        _tag_pool: Dict[str, str] = {}

        def _emit(done: int, total: int, label: str, force: bool = False) -> None:
            if progress_callback is None:
                return
            now = time.monotonic()
            if not force and now - _last_emit[0] < 0.3:
                return
            _last_emit[0] = now
            try:
                progress_callback(int(done), int(max(1, total)), label)
            except Exception:
                pass

        for _di, dataset_id in enumerate(dataset_ids):
            _ds_tag = f"dataset {dataset_id}" + (f" ({_di + 1}/{n_datasets})" if n_datasets > 1 else "")
            print(f"[TaggerDataset] Loading dataset_id={dataset_id}...")

            # Bulk-load all items for this dataset in one query
            items = (
                datasets_db.query(DatasetItem)
                .filter(DatasetItem.dataset_id == dataset_id)
                .all()
            )
            print(f"[TaggerDataset]   {len(items)} items loaded")

            # Collect item ids that have a valid image path.
            # Record a small sample of skipped paths (capped to keep the log readable).
            item_path_map: Dict[int, str] = {}
            _MAX_SHOW = 5
            skipped_examples: List[str] = []
            skipped_no_path = 0
            _n_items = len(items)
            for _ii, item in enumerate(tqdm(items, desc=f"  Checking files (dataset {dataset_id})", unit="item", leave=False)):
                if _ii % 2000 == 0:
                    _emit(_ii, _n_items, f"Loading {_ds_tag}: checking files {_ii:,}/{_n_items:,}")
                if item.image_path and os.path.isfile(item.image_path):
                    item_path_map[item.id] = item.image_path
                else:
                    if not item.image_path:
                        skipped_no_path += 1
                        if len(skipped_examples) < _MAX_SHOW:
                            skipped_examples.append(f"<item_id={item.id} (no image_path)>")
                    elif len(skipped_examples) < _MAX_SHOW:
                        skipped_examples.append(item.image_path)
            valid_item_ids = list(item_path_map.keys())
            skipped = len(items) - len(valid_item_ids)
            if skipped:
                print(f"[TaggerDataset]   {skipped} items skipped (missing image files)")
                for ex in skipped_examples:
                    print(f"[TaggerDataset]     - {ex}")
                if skipped > _MAX_SHOW:
                    print(f"[TaggerDataset]     ... and {skipped - _MAX_SHOW} more")

            if not valid_item_ids:
                continue

            # Bulk-load captions in chunks to stay within SQLite's 999-variable limit
            CHUNK = 500
            from collections import defaultdict
            captions_by_item: Dict[int, list] = defaultdict(list)
            total_captions = 0
            n_chunks = (len(valid_item_ids) + CHUNK - 1) // CHUNK
            for _ci, i in enumerate(tqdm(range(0, len(valid_item_ids), CHUNK), total=n_chunks,
                          desc=f"  Loading captions (dataset {dataset_id})", unit="chunk", leave=False)):
                _pct = int((_ci / max(1, n_chunks)) * 100)
                _emit(_ci, n_chunks, f"Loading {_ds_tag}: captions {_ci:,}/{n_chunks:,} ({_pct}%)")
                chunk_ids = valid_item_ids[i:i + CHUNK]
                q = (
                    datasets_db.query(DatasetCaption)
                    .filter(
                        DatasetCaption.item_id.in_(chunk_ids),
                        DatasetCaption.is_tags_format == True,  # noqa: E712
                    )
                )
                if caption_types:
                    q = q.filter(DatasetCaption.caption_type.in_(caption_types))
                for cap in q.all():
                    captions_by_item[cap.item_id].append(cap)
                    total_captions += 1
            print(f"[TaggerDataset]   {total_captions} tag captions loaded")

            # Build samples
            _n_valid = len(valid_item_ids)
            for _bi, item_id in enumerate(tqdm(valid_item_ids, desc=f"  Building samples (dataset {dataset_id})", unit="item", leave=False)):
                if _bi % 5000 == 0:
                    _emit(_bi, _n_valid, f"Loading {_ds_tag}: building samples {_bi:,}/{_n_valid:,}")
                item_captions = captions_by_item.get(item_id)
                if not item_captions:
                    continue
                tags: List[str] = []
                for caption in item_captions:
                    tags.extend(self._extract_tags(caption))
                if tags:
                    # Intern each tag to the shared pool object (dedup across all
                    # samples) before storing — see _tag_pool note above.
                    self._samples.append(
                        (item_path_map[item_id], [_tag_pool.setdefault(t, t) for t in tags])
                    )
                    self._item_ids_list.append(item_id)

            print(f"[TaggerDataset]   {len(self._samples)} samples so far")

        # Finalise the sample-idx -> item_id map for the live tag-refresh detector.
        self._item_ids = np.asarray(self._item_ids_list, dtype=np.int64)
        self._item_ids_list = []  # free the Python list; numpy array is the source

    def _extract_tags(self, caption) -> List[str]:
        return resolve_caption_tags(
            caption.tag_data, caption.content,
            self._comma_resolver, self._alias_resolver,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------
    # Worker pickle: drop main-process-only / unpicklable / worker-rebuilt
    # fields so each spawned worker stays small and rebuilds its own reader.
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_item_ids", None)        # detector-only (main process)
        state.pop("_item_ids_list", None)
        state["_refresh_reader"] = None     # rebuilt lazily in the worker
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._item_ids = None
        self._item_ids_list = []
        self._refresh_reader = None

    def _apply_tag_refresh(self, idx: int, tags: List[str]) -> List[str]:
        """Return live-edited tags for *idx* when a refresh override exists.

        Steady-state cost is one mmap memory read + one dict lookup (the reader
        only re-reads the override payload when the detector bumps its generation
        counter). Runs in the DataLoader worker, ahead of the GPU via prefetch, so
        it does not affect iteration time.
        """
        reader = self._refresh_reader
        if reader is None:
            from core.tagger.tag_refresh import TagRefreshReader
            reader = TagRefreshReader(self._refresh_gen_path, self._refresh_payload_path)
            self._refresh_reader = reader
        ov = reader.override(idx)
        return ov if ov is not None else tags

    def __getitem__(self, idx: int):
        image_path, tags = self._samples[idx]
        if self._refresh_enabled and self._refresh_gen_path:
            tags = self._apply_tag_refresh(idx, tags)
        try:
            image = _load_image(image_path)
            inputs = self.processor(images=[image], return_tensors="pt")
        except Exception as e:
            print(f"[TaggerDataset] Skipping corrupt image {image_path}: {e}")
            return None  # filtered out by tagger_collate_fn

        if self.is_naflex:
            pixel_values         = inputs["pixel_values"].squeeze(0)          # [num_patches, patch_dim]
            pixel_attention_mask = inputs["pixel_attention_mask"].squeeze(0)  # [num_patches]
            spatial_shapes       = inputs["spatial_shapes"].squeeze(0)        # [2]
        else:
            pixel_values         = inputs["pixel_values"].squeeze(0)          # [3, H, W]
            pixel_attention_mask = torch.zeros(0, dtype=torch.int32)          # sentinel → [B, 0] after collate
            spatial_shapes       = torch.zeros(0, dtype=torch.int64)          # sentinel

        label, loss_mask = self._build_label_and_mask(tags)
        return pixel_values, pixel_attention_mask, spatial_shapes, label, loss_mask

    # ------------------------------------------------------------------
    # Label / mask construction
    # ------------------------------------------------------------------

    def _build_label_and_mask(
        self, tags: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        voc = self.vocabulary
        # Size to the LIVE vocabulary, not the cached self.num_tags. Online
        # Danbooru vocab expansion grows the vocabulary during training; each
        # DataLoader worker re-pickles the vocab at its (grown) size at the epoch
        # boundary, so voc.tag_to_idx can exceed the stale self.num_tags captured
        # at dataset construction -> IndexError at label[idx]. Using voc.num_tags
        # (== len(tag_to_idx)) guarantees every index fits; the MixedDataLoader
        # still pads up to any further mid-epoch head growth.
        n_tags = voc.num_tags
        label     = torch.zeros(n_tags, dtype=torch.float32)
        loss_mask = torch.ones(n_tags,  dtype=torch.float32)

        # Set positive labels
        tag_set = set(tags)
        for tag in tag_set:
            if tag in voc.tag_to_idx:
                label[voc.tag_to_idx[tag]] = 1.0

        # Rating tags: mask out if none present in this sample
        has_rating = any(normalize_tag(r) in tag_set for r in RATING_TAGS)
        if not has_rating:
            for idx in voc.rating_indices:
                loss_mask[idx] = 0.0

        # Quality tags: behavior depends on quality_masking_mode.
        #
        # Detect which quality groups have at least one tag present on this sample.
        present_groups: set = set()
        for group_name, gtags in QUALITY_TAG_GROUPS.items():
            if any(normalize_tag(t) in tag_set for t in gtags):
                present_groups.add(group_name)

        if not present_groups:
            # No quality tag → mask ALL quality indices (both modes agree).
            for group_indices in voc.quality_indices.values():
                for idx in group_indices:
                    loss_mask[idx] = 0.0
        elif self.quality_masking_mode == "intra_group":
            # tagutl-style: within each present group, mask non-positive siblings.
            # Sibling masking avoids penalising "best=0" on a sample that an
            # annotator chose to label "high" instead — within-group distinctions
            # are often noisy / prevalence-imbalanced (e.g. normal_quality
            # dominates → ASL's high γ_neg would over-suppress best_quality).
            # Tags in the *other* group stay unmasked → trained as negatives
            # (high vs low is a meaningful, clean distinction).
            for group_name in present_groups:
                for idx in voc.quality_indices[group_name]:
                    if label[idx] == 0.0:
                        loss_mask[idx] = 0.0
        # else "cross_group": leave loss_mask[*]=1 — all non-positive quality
        # tags train as negatives (legacy behavior, assumes clean labels).

        return label, loss_mask


# ------------------------------------------------------------------
# Collate function — supports both NaFlex (variable patches) and standard
# ------------------------------------------------------------------

def tagger_collate_fn(batch):
    """Collate batch items.

    - Filters out None entries (corrupt/unreadable images).
    - Handles NaFlex (variable num_patches per image) by padding to max.
    - Handles standard fixed-resolution models (pam/ss are sentinel zero-tensors).
    Returns None if the entire batch is corrupt.
    """
    valid = [item for item in batch if item is not None]
    if not valid:
        return None

    pixel_values_list, masks_list, shapes_list, labels_list, loss_masks_list = zip(*valid)

    # Stack pixel_values; for NaFlex with varying patch counts, pad to max.
    try:
        pixel_values         = torch.stack(pixel_values_list)
        pixel_attention_mask = torch.stack(masks_list)
        spatial_shapes       = torch.stack(shapes_list)
    except RuntimeError:
        # NaFlex: different num_patches across images — pad to the maximum.
        max_patches = max(p.shape[0] for p in pixel_values_list)
        patch_dim   = pixel_values_list[0].shape[1]
        B = len(pixel_values_list)
        pixel_values         = torch.zeros(B, max_patches, patch_dim)
        pixel_attention_mask = torch.zeros(B, max_patches, dtype=torch.int32)
        for i, (pv, pm) in enumerate(zip(pixel_values_list, masks_list)):
            n = pv.shape[0]
            pixel_values[i, :n]         = pv
            pixel_attention_mask[i, :n] = pm
        spatial_shapes = torch.stack(shapes_list)

    return (
        pixel_values,
        pixel_attention_mask,
        spatial_shapes,
        torch.stack(labels_list),
        torch.stack(loss_masks_list),
    )


# ------------------------------------------------------------------
# Image loading helper
# ------------------------------------------------------------------

def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    img.load()  # force full decode here so truncation raises OSError before reaching the processor
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img
