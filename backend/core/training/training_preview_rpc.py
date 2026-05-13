"""File-based RPC between the API (main process) and the training subprocess.

LoRA / Full-FT training runs in a child Python process spawned via
``asyncio.create_subprocess_exec`` ([`training_process.py`]).  The
API and the trainer therefore have separate memory spaces — we cannot
directly hand a model reference across.

To let an incoming generation request use the trainer's current
in-training UNet + LoRA, we exchange files in the run's
``output_dir``:

  API side (main process):
    1. write  ``<output_dir>/.preview_request_<id>.json``  (params)
    2. async wait for ``<output_dir>/.preview_result_<id>.meta.json``
    3. read    ``<output_dir>/.preview_result_<id>.png``
    4. delete  both result files

  Trainer side (subprocess):
    5. at every batch boundary, glob ``.preview_request_*.json``
    6. for each: read params, run preview generation, write result files
    7. delete the request file (idempotent — API may also delete)

All writes are atomic (write-to-``.tmp`` then ``os.replace``) so a
reader never sees a partial file.

This pattern mirrors the existing ``.stop_training`` flag-file the
trainer already polls — the polling cost is negligible (one
``Path.glob`` per batch).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUEST_PREFIX = ".preview_request_"
RESULT_PREFIX  = ".preview_result_"
RESULT_META_SUFFIX = ".meta.json"
RESULT_IMG_SUFFIX  = ".png"
# Auto-clean files older than this (orphan from a crashed trainer or
# API request that timed out).
STALE_TIMEOUT_SEC = 600   # 10 minutes


def make_request_id() -> str:
    """Short random identifier used in the request / result filenames."""
    return uuid.uuid4().hex[:16]


def request_path(output_dir: str | Path, request_id: str) -> Path:
    return Path(output_dir) / f"{REQUEST_PREFIX}{request_id}.json"


def result_image_path(output_dir: str | Path, request_id: str) -> Path:
    return Path(output_dir) / f"{RESULT_PREFIX}{request_id}{RESULT_IMG_SUFFIX}"


def result_meta_path(output_dir: str | Path, request_id: str) -> Path:
    return Path(output_dir) / f"{RESULT_PREFIX}{request_id}{RESULT_META_SUFFIX}"


def write_request(output_dir: str | Path, request_id: str, params: Dict[str, Any]) -> None:
    """Atomic-write a request file.  Trainer will pick it up at the next
    batch boundary."""
    p = request_path(output_dir, request_id)
    tmp = p.with_suffix(p.suffix + ".tmp")
    payload = {
        "request_id": request_id,
        "params": params,
        "ts": time.time(),
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, p)


def list_pending_requests(output_dir: str | Path) -> List[Path]:
    """All request files currently in *output_dir*, sorted by filename
    (which sorts by request_id — random order, but stable across the
    same run for fair scheduling)."""
    out = Path(output_dir)
    if not out.is_dir():
        return []
    return sorted(out.glob(f"{REQUEST_PREFIX}*.json"))


def read_request(req_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(req_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_result(
    output_dir: str | Path,
    request_id: str,
    image_bytes: Optional[bytes],
    meta: Dict[str, Any],
) -> None:
    """Atomic-write the result PNG (if any) + meta JSON.

    The meta file is written LAST so the API side can poll on its
    existence and be guaranteed the image is already complete.
    """
    out = Path(output_dir)
    if image_bytes is not None:
        img_path = result_image_path(out, request_id)
        img_tmp = img_path.with_suffix(img_path.suffix + ".tmp")
        with open(img_tmp, "wb") as f:
            f.write(image_bytes)
        os.replace(img_tmp, img_path)

    meta_path = result_meta_path(out, request_id)
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(meta_tmp, meta_path)


def cleanup_stale(output_dir: str | Path) -> int:
    """Remove request / result files older than ``STALE_TIMEOUT_SEC``.

    Returns the number of files removed.  Called periodically by the
    trainer (cheap — just iterates a small set of files).
    """
    out = Path(output_dir)
    if not out.is_dir():
        return 0
    now = time.time()
    removed = 0
    for pattern in (f"{REQUEST_PREFIX}*", f"{RESULT_PREFIX}*"):
        for p in out.glob(pattern):
            try:
                if now - p.stat().st_mtime > STALE_TIMEOUT_SEC:
                    p.unlink()
                    removed += 1
            except OSError:
                pass
    return removed
