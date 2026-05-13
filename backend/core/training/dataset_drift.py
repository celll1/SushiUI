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
# Drift detection — read-only
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Result of a drift detection pass.

    ``has_drift`` is True iff EITHER items_missing > 0 (DB row → no
    file) OR items_new > 0 (file → no DB row).  The caller decides
    whether to trigger a full rescan.
    """
    dataset_id: int
    dataset_name: str
    dataset_path: str
    items_in_db: int = 0
    items_missing: int = 0
    items_new: int = 0
    files_walked: int = 0
    elapsed_sec: float = 0.0
    missing_samples: List[str] = field(default_factory=list)
    new_samples: List[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return self.items_missing > 0 or self.items_new > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id":     self.dataset_id,
            "dataset_name":   self.dataset_name,
            "dataset_path":   self.dataset_path,
            "items_in_db":    self.items_in_db,
            "items_missing":  self.items_missing,
            "items_new":      self.items_new,
            "files_walked":   self.files_walked,
            "elapsed_sec":    round(self.elapsed_sec, 2),
            "has_drift":      self.has_drift,
            "missing_samples": self.missing_samples[:5],
            "new_samples":    self.new_samples[:5],
        }


def _walk_dataset_dir(
    root: str,
    *,
    recursive: bool,
    max_depth: Optional[int],
    extensions: Set[str],
) -> Set[str]:
    """Walk ``root`` and return absolute file paths matching extensions.

    Uses ``os.scandir`` which on Windows is several times faster than
    ``os.walk``.  Honours ``recursive`` + ``max_depth`` (mirrors the
    rescan endpoint's logic).
    """
    out: Set[str] = set()
    root = os.path.abspath(root)
    # Stack of (path, depth_remaining) where None = unlimited
    stack: List[tuple] = [(root, max_depth)]
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
                except OSError:
                    pass
    return out


def detect_drift(dataset_id: int, datasets_db) -> DriftReport:
    """Compare on-disk files against DB rows for *dataset_id*.

    Read-only: never writes to ``datasets_db``.  Returns a populated
    ``DriftReport`` even when the dataset's root is missing (in which
    case all DB items are reported as ``items_missing``).
    """
    from database.models import Dataset, DatasetItem

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

    # 1) Files actually present on disk
    on_disk: Set[str] = set()
    if ds.path and os.path.isdir(ds.path):
        on_disk = _walk_dataset_dir(
            ds.path,
            recursive=bool(getattr(ds, "recursive", True)),
            max_depth=getattr(ds, "max_depth", None) or None,
            extensions=exts,
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

    return DriftReport(
        dataset_id=int(dataset_id),
        dataset_name=ds.name or f"dataset_{dataset_id}",
        dataset_path=ds.path or "",
        items_in_db=items_in_db,
        items_missing=len(missing),
        items_new=len(new),
        files_walked=len(on_disk),
        elapsed_sec=time.monotonic() - t0,
        missing_samples=sorted(missing)[:5],
        new_samples=sorted(new)[:5],
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
