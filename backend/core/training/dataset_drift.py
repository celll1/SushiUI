"""Pre-flight dataset drift detection for training runs.

When training starts, the dataset DB (datasets.db) may be out of sync
with the actual files on disk:

  * Files moved / deleted → DB has "phantom" rows whose image_path
    points nowhere.  Both the tagger trainer
    (``tagger_dataset.py``) and the LoRA / Full-FT loader
    (``train_runner.get_dataset_items_fast``) already skip these at
    load time, but the user has no visibility into the count.
  * Files added → not in DB, so they're INVISIBLE to training.  This
    is the more pernicious case: a user adds 1,000 new images and
    then wonders why their training doesn't see them.

This module provides:

  - ``detect_drift(dataset_id, db)`` — a read-only set-diff between
    on-disk files and DB rows.  Returns counts + small samples.
  - ``rescan_dataset_inline(dataset_id, datasets_db)`` — calls the
    existing ``scan_dataset`` route function in-process to fix drift.
  - ``cleanup_orphan_latent_cache(...)`` — removes ``.pt`` files in
    the cache whose source row is no longer in the DB (LoRA only;
    tagger doesn't latent-cache).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# Same extension set the rescan endpoint uses
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------------------------------------------------------------------------
# Pre-flight mode normalization
# ---------------------------------------------------------------------------

RESCAN_MODES = ("off", "path", "smart", "force")


def normalize_rescan_mode(value: Any) -> str:
    """Normalize a user-supplied rescan-before-training value to one of
    ``"off"`` / ``"path"`` / ``"smart"`` / ``"force"``.

    Accepts:
      - the new string enum values directly
      - legacy boolean ``True`` (maps to ``"path"`` — the original
        opt-in behavior before the 4-mode split) / ``False`` (``"off"``)
      - any other value: returns ``"off"`` (safe default)
    """
    if isinstance(value, bool):
        return "path" if value else "off"
    if value is None:
        return "off"
    try:
        s = str(value).strip().lower()
    except Exception:
        return "off"
    if s in RESCAN_MODES:
        return s
    return "off"


# ---------------------------------------------------------------------------
# Drift detection — read-only
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Result of a drift detection pass.

    ``has_drift`` is True iff EITHER items_missing > 0 (DB row → no
    file) OR items_new > 0 (file → no DB row) OR captions_stale > 0
    (caption sidecar mtime newer than the DB caption's updated_at).
    The caller decides whether to trigger a full rescan.

    ``captions_stale`` is only populated when ``detect_drift`` was
    called with ``check_caption_mtime=True`` (smart mode).
    """
    dataset_id: int
    dataset_name: str
    dataset_path: str
    items_in_db: int = 0
    items_missing: int = 0
    items_new: int = 0
    captions_stale: int = 0
    files_walked: int = 0
    elapsed_sec: float = 0.0
    missing_samples: List[str] = field(default_factory=list)
    new_samples: List[str] = field(default_factory=list)
    stale_caption_samples: List[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return (
            self.items_missing > 0
            or self.items_new > 0
            or self.captions_stale > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id":     self.dataset_id,
            "dataset_name":   self.dataset_name,
            "dataset_path":   self.dataset_path,
            "items_in_db":    self.items_in_db,
            "items_missing":  self.items_missing,
            "items_new":      self.items_new,
            "captions_stale": self.captions_stale,
            "files_walked":   self.files_walked,
            "elapsed_sec":    round(self.elapsed_sec, 2),
            "has_drift":      self.has_drift,
            "missing_samples":       self.missing_samples[:5],
            "new_samples":           self.new_samples[:5],
            "stale_caption_samples": self.stale_caption_samples[:5],
        }


# Sidecar extensions checked for caption-mtime drift.  Mirrors what the
# rescan endpoint reads when ingesting captions.
SIDECAR_EXTS = {".txt", ".json"}


def _walk_dataset_dir(
    root: str,
    *,
    recursive: bool,
    max_depth: Optional[int],
    extensions: Set[str],
    sidecar_extensions: Optional[Set[str]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    progress_every_files: int = 5000,
    progress_every_sec: float = 1.0,
) -> "Tuple[Set[str], Dict[str, float]]":
    """Walk ``root`` and return image paths + (optional) sidecar mtime map.

    Uses ``os.scandir`` which on Windows is several times faster than
    ``os.walk``.  Honours ``recursive`` + ``max_depth`` (mirrors the
    rescan endpoint's logic).

    Returns ``(image_paths, sidecar_mtime_by_stem)`` where:
      - ``image_paths``: absolute paths of files matching ``extensions``
      - ``sidecar_mtime_by_stem``: maps ``abs(splitext(image_path)[0])``
        to the *maximum* mtime seen across all sidecar files sharing that
        stem (empty dict when ``sidecar_extensions`` is None or empty).
        The mtime is the float ``st_mtime``; consumer compares to the
        DB caption row's ``updated_at`` to flag content-only drift.

    ``progress_callback`` (if given) is invoked with the cumulative image
    count every ``progress_every_files`` matched files OR every
    ``progress_every_sec`` seconds, whichever comes first.  Exceptions
    raised by the callback are caught and silenced so a bad reporter
    can never break the walk.
    """
    out: Set[str] = set()
    sidecar_mtime: Dict[str, float] = {}
    track_sidecars = bool(sidecar_extensions)
    root = os.path.abspath(root)
    # Stack of (path, depth_remaining) where None = unlimited
    stack: List[tuple] = [(root, max_depth)]
    last_report = time.monotonic()
    last_count = 0
    while stack:
        cur, remaining = stack.pop()
        try:
            it = os.scandir(cur)
        except OSError:
            continue
        with it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if recursive and (remaining is None or remaining > 0):
                            next_remaining = None if remaining is None else remaining - 1
                            stack.append((entry.path, next_remaining))
                    elif entry.is_file(follow_symlinks=False):
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in extensions:
                            out.add(os.path.abspath(entry.path))
                            # Emit progress on count OR time threshold
                            if progress_callback is not None:
                                cnt = len(out)
                                now = time.monotonic()
                                if (cnt - last_count >= progress_every_files
                                        or now - last_report >= progress_every_sec):
                                    try:
                                        progress_callback(cnt)
                                    except Exception:
                                        pass
                                    last_count = cnt
                                    last_report = now
                        elif track_sidecars and ext in sidecar_extensions:
                            # scandir's DirEntry caches stat on Windows so this
                            # is essentially free (no extra syscall).
                            stem = os.path.abspath(os.path.splitext(entry.path)[0])
                            try:
                                mt = entry.stat(follow_symlinks=False).st_mtime
                            except OSError:
                                continue
                            prev = sidecar_mtime.get(stem)
                            if prev is None or mt > prev:
                                sidecar_mtime[stem] = mt
                except OSError:
                    pass
    # Final report so the consumer sees the actual total
    if progress_callback is not None and len(out) != last_count:
        try:
            progress_callback(len(out))
        except Exception:
            pass
    return out, sidecar_mtime


def detect_drift(
    dataset_id: int,
    datasets_db,
    *,
    check_caption_mtime: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> DriftReport:
    """Compare on-disk files against DB rows for *dataset_id*.

    Read-only: never writes to ``datasets_db``.  Returns a populated
    ``DriftReport`` even when the dataset's root is missing (in which
    case all DB items are reported as ``items_missing``).

    When ``check_caption_mtime`` is True (smart mode), the walk also
    collects ``.txt`` / ``.json`` sidecar mtimes and compares them
    against the latest ``DatasetCaption.updated_at`` per item.  An
    image whose sidecar is newer than its caption row is counted in
    ``captions_stale``.  This catches the case where a tagging tool
    re-wrote captions in place without renaming the image.

    ``progress_callback`` (optional) receives the cumulative *image*
    file count as the directory walk progresses — used by the route
    handler to push WebSocket progress events to the training monitor.
    """
    from database.models import Dataset, DatasetItem, DatasetCaption
    from sqlalchemy import func

    ds = datasets_db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        return DriftReport(dataset_id=dataset_id, dataset_name="?",
                           dataset_path="", items_in_db=0)

    # Build extension set from the Dataset row (default fallback)
    exts = IMAGE_EXTS
    if getattr(ds, "file_extensions", None):
        try:
            exts = set(e.lower() if e.startswith(".") else "." + e.lower()
                       for e in ds.file_extensions)
        except Exception:
            exts = IMAGE_EXTS

    t0 = time.monotonic()

    # 1) Files actually present on disk (+ optional sidecar mtimes)
    on_disk: Set[str] = set()
    sidecar_mtime: Dict[str, float] = {}
    if ds.path and os.path.isdir(ds.path):
        on_disk, sidecar_mtime = _walk_dataset_dir(
            ds.path,
            recursive=bool(getattr(ds, "recursive", True)),
            max_depth=getattr(ds, "max_depth", None) or None,
            extensions=exts,
            sidecar_extensions=SIDECAR_EXTS if check_caption_mtime else None,
            progress_callback=progress_callback,
        )

    # 2) Files according to the DB.  Normalise to absolute paths so the
    #    set-diff is symmetric.
    db_paths: Set[str] = set()
    db_path_iter = (
        datasets_db.query(DatasetItem.image_path)
        .filter(DatasetItem.dataset_id == dataset_id)
        .yield_per(5000)
    )
    items_in_db = 0
    for (p,) in db_path_iter:
        items_in_db += 1
        if p:
            db_paths.add(os.path.abspath(p))

    missing = db_paths - on_disk
    new     = on_disk - db_paths

    # 3) Caption mtime drift (smart mode only)
    captions_stale = 0
    stale_caption_samples: List[str] = []
    if check_caption_mtime and sidecar_mtime:
        # Per item: (id, image_path, max(caption.updated_at)).  Outer-join so
        # items with zero captions still surface — a sidecar appearing for
        # an item with no DB caption row IS drift (new caption to ingest).
        rows = (
            datasets_db.query(
                DatasetItem.image_path,
                func.max(DatasetCaption.updated_at).label("last_updated"),
            )
            .outerjoin(DatasetCaption, DatasetCaption.item_id == DatasetItem.id)
            .filter(DatasetItem.dataset_id == dataset_id)
            .group_by(DatasetItem.id, DatasetItem.image_path)
            .yield_per(5000)
        )
        for image_path, last_updated in rows:
            if not image_path:
                continue
            stem = os.path.splitext(os.path.abspath(image_path))[0]
            mt = sidecar_mtime.get(stem)
            if mt is None:
                continue  # No sidecar for this item — nothing to compare
            if last_updated is None:
                # Sidecar exists but no caption row at all → definitely stale
                captions_stale += 1
                if len(stale_caption_samples) < 5:
                    stale_caption_samples.append(stem)
                continue
            try:
                updated_ts = last_updated.timestamp()
            except Exception:
                continue
            if mt > updated_ts + 1.0:  # 1-second tolerance for fs/db clock skew
                captions_stale += 1
                if len(stale_caption_samples) < 5:
                    stale_caption_samples.append(stem)

    return DriftReport(
        dataset_id=int(dataset_id),
        dataset_name=ds.name or f"dataset_{dataset_id}",
        dataset_path=ds.path or "",
        items_in_db=items_in_db,
        items_missing=len(missing),
        items_new=len(new),
        captions_stale=captions_stale,
        files_walked=len(on_disk),
        elapsed_sec=time.monotonic() - t0,
        missing_samples=sorted(missing)[:5],
        new_samples=sorted(new)[:5],
        stale_caption_samples=stale_caption_samples,
    )


# ---------------------------------------------------------------------------
# In-process rescan — wraps the existing /datasets/{id}/scan route handler
# ---------------------------------------------------------------------------

async def rescan_dataset_inline(
    dataset_id: int, datasets_db,
    *, progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run a full rescan of *dataset_id* by directly calling the
    existing ``scan_dataset`` FastAPI route function with a manually-
    provided db session.

    Returns the same JSON dict the endpoint would return.

    The route function is ``async`` and uses ``loop.run_in_executor``
    internally for the heavy walk; we call it from an async context
    so that all works seamlessly.
    """
    if progress_callback:
        try: progress_callback(f"Rescanning dataset {dataset_id}...")
        except Exception: pass
    # Lazy import to avoid circular dependency at module load time.
    from api.routes import scan_dataset
    result = await scan_dataset(dataset_id=dataset_id, db=datasets_db)
    return result


# ---------------------------------------------------------------------------
# Latent cache orphan cleanup (LoRA / Full-FT only)
# ---------------------------------------------------------------------------

def cleanup_orphan_latent_cache(
    dataset_unique_id: str,
    datasets_db,
    dataset_id: int,
    *,
    bucket_resolutions: Optional[List[tuple]] = None,
) -> int:
    """Remove latent cache ``.pt`` files whose source DatasetItem is no
    longer in the DB (i.e. removed by a fresh rescan).

    The cache filename is ``{md5(<abs_path>_<w>_<h>)}.pt`` — we cannot
    invert the hash, so the strategy is:

      1. Build a set of "expected" hashes by iterating every current
         DatasetItem × every bucket (width, height) the trainer might
         use.
      2. Delete any cache file whose stem (== hash) isn't in that set.

    Returns the number of files removed.

    For the bucket resolutions, the caller passes the same list it
    intends to train with (typically ``[(512,512), (768,768), (1024,1024)]``
    or finer bucketing).  When omitted, a sensible default list is used.
    """
    from database.models import DatasetItem
    from core.training.latent_cache import LatentCache, get_cache_base_dir

    cache_dir = Path(get_cache_base_dir()) / dataset_unique_id / "latents"
    if not cache_dir.is_dir():
        return 0

    # 1) Build set of expected hashes from current DB rows.
    if bucket_resolutions is None:
        # Conservative default — covers most SDXL/SD15 setups.
        bucket_resolutions = [
            (512, 512), (640, 640), (768, 768),
            (832, 1216), (1024, 1024), (1216, 832),
        ]

    expected: Set[str] = set()
    rows = (
        datasets_db.query(DatasetItem.image_path)
        .filter(DatasetItem.dataset_id == dataset_id)
        .yield_per(5000)
    )
    for (p,) in rows:
        if not p:
            continue
        abs_p = os.path.abspath(p)
        for (w, h) in bucket_resolutions:
            expected.add(LatentCache.compute_image_hash(abs_p, w, h))

    # 2) Iterate cache files; delete any not in expected.
    removed = 0
    for entry in cache_dir.glob("*.pt"):
        stem = entry.stem
        if stem not in expected:
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
    return removed
