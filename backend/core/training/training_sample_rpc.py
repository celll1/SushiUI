"""On-demand training-sample requests (API main process -> trainer subprocess).

Same file-RPC shape as ``training_preview_rpc`` (atomic write, delete before
processing, result written last), different contract: the trainer runs a claimed
request through its ORDINARY scheduled-sample block, so the PNG lands in
``<output_dir>/samples/`` under its own name and the result file here carries
only metadata.

  API side:
    1. write   ``<output_dir>/.sample_request_<id>.json``
    2. return 202 immediately -- a batch can take minutes, so nothing waits
    3. poll    ``GET /training/runs/{id}/sample-queue`` (+ the samples listing)

  Trainer side:
    4. at the scheduled-sample seam, claim AT MOST ONE request per batch
    5. delete the request file, generate, write ``.sample_result_<id>.json``

Request files carry no TTL: one queued during dataset scan / latent caching must
survive until the first batch, which can be a long time away. Stale files are
cleared before the next run is spawned instead (``training_process.py``).
"""
from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUEST_PREFIX = ".sample_request_"
RESULT_PREFIX = ".sample_result_"
SUFFIX = ".json"

# N queued requests would mean N full generations back to back with training
# stalled, so the queue is capped rather than unbounded.
MAX_PENDING_REQUESTS = 3
# Kept only so the queue endpoint can report what happened; pruned oldest-first.
MAX_KEPT_RESULTS = 20

# Architectures whose training-sample helper returns a blank white image instead
# of raising when generation fails (ops/sd_sdxl_ops.py). Stated in the result so
# a caller does not read "a PNG exists" as "generation succeeded".
BLANK_ON_FAILURE_ARCHS = ("sd15", "sdxl")
BLANK_ON_FAILURE_NOTE = (
    "On {arch} the training-sample helper returns a blank white image when "
    "generation fails, so a written PNG does not by itself establish success."
)


class SampleQueueFullError(RuntimeError):
    """Raised by :func:`queue_request` when the pending cap is already reached."""


def make_request_id() -> str:
    return uuid.uuid4().hex[:16]


def resolve_seed(configured_seed: Any) -> int:
    """A concrete seed, never the -1 sentinel.

    ``seed < 0`` reaches the arch ops as ``generator=None``, which consumes the
    global torch RNG (zimage_ops.py, sd_sdxl_ops.py ancestral samplers) — an
    on-demand sample must not perturb the training stream, so the sentinel is
    resolved here, in the API process, before the request is written.
    """
    try:
        value = int(configured_seed)
    except (TypeError, ValueError):
        value = -1
    return secrets.randbelow(2 ** 32) if value < 0 else value


def request_path(output_dir: str | Path, request_id: str) -> Path:
    return Path(output_dir) / f"{REQUEST_PREFIX}{request_id}{SUFFIX}"


def result_path(output_dir: str | Path, request_id: str) -> Path:
    return Path(output_dir) / f"{RESULT_PREFIX}{request_id}{SUFFIX}"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _sorted_by_age(paths: List[Path]) -> List[Path]:
    """Oldest first. mtime then name, so same-second requests still order."""
    def key(p: Path):
        try:
            return (p.stat().st_mtime, p.name)
        except OSError:
            return (0.0, p.name)
    return sorted(paths, key=key)


def owns(record: Optional[Dict[str, Any]], run_id: Optional[int]) -> bool:
    """Whether ``run_id`` may act on this request/result.

    Two runs that share a ``run_name`` share an ``output_dir`` (the same reason
    ``_step0_sample_done_for_this_run`` checks the marker's run id rather than
    the file's existence), so a request names the run it was queued for. A record
    with no ``run_id`` -- one written before this field existed -- is treated as
    ours rather than left to wedge the directory forever.
    """
    if run_id is None or record is None:
        return True
    owner = record.get("run_id")
    return owner is None or int(owner) == int(run_id)


def list_pending_requests(output_dir: str | Path,
                          run_id: Optional[int] = None) -> List[Path]:
    out = Path(output_dir)
    if not out.is_dir():
        return []
    paths = _sorted_by_age(list(out.glob(f"{REQUEST_PREFIX}*{SUFFIX}")))
    if run_id is None:
        return paths
    return [p for p in paths if owns(_read_json(p), run_id)]


def read_request(req_path: Path) -> Optional[Dict[str, Any]]:
    return _read_json(req_path)


def pending_requests(output_dir: str | Path,
                     run_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Parsed pending requests, oldest first (for the queue endpoint)."""
    out: List[Dict[str, Any]] = []
    for p in list_pending_requests(output_dir, run_id):
        req = _read_json(p)
        if req is not None:
            out.append(req)
    return out


def queue_request(
    output_dir: str | Path,
    *,
    seed: int,
    run_id: Optional[int] = None,
    request_id: Optional[str] = None,
    max_pending: int = MAX_PENDING_REQUESTS,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write one request file. Raises SampleQueueFullError at the cap."""
    out = Path(output_dir)
    existing = list_pending_requests(out, run_id)
    if len(existing) >= max_pending:
        raise SampleQueueFullError(
            f"{len(existing)} sample request(s) already queued for this run "
            f"(maximum {max_pending}); each one runs a full generation with "
            f"training stalled, so further requests are refused until the "
            f"trainer has worked through these."
        )
    rid = request_id or make_request_id()
    payload: Dict[str, Any] = {
        "request_id": rid,
        "run_id": None if run_id is None else int(run_id),
        "seed": int(seed),
        "queued_at": time.time(),
    }
    if extra:
        payload.update(extra)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(request_path(out, rid), payload)
    return payload


def claim_next_request(output_dir: str | Path,
                       run_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Claim the oldest pending request for ``run_id``, or None.

    A request belonging to another run sharing this directory is LEFT IN PLACE.
    The claimed file is deleted BEFORE the payload is returned, so a malformed or
    re-emitted request cannot be picked up twice and a crash mid-generation does
    not replay it into the next run.
    """
    for p in list_pending_requests(output_dir):
        req = _read_json(p)
        if req is not None and not owns(req, run_id):
            continue
        try:
            p.unlink()
        except OSError:
            continue
        if req is not None:
            return req
    return None


def write_result(output_dir: str | Path, request_id: str,
                 result: Dict[str, Any]) -> None:
    """Atomic-write the result record and prune the oldest ones.

    Single file, written last (the PNG it describes is already in ``samples/``
    under its own name), so a reader that sees this file sees a complete record.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload.setdefault("request_id", request_id)
    payload.setdefault("completed_at", time.time())
    _atomic_write_json(result_path(out, request_id), payload)
    stale = _sorted_by_age(list(out.glob(f"{RESULT_PREFIX}*{SUFFIX}")))
    for p in stale[:max(0, len(stale) - MAX_KEPT_RESULTS)]:
        try:
            p.unlink()
        except OSError:
            pass


def list_results(output_dir: str | Path,
                 run_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Recorded results, newest first."""
    out = Path(output_dir)
    if not out.is_dir():
        return []
    results: List[Dict[str, Any]] = []
    for p in reversed(_sorted_by_age(list(out.glob(f"{RESULT_PREFIX}*{SUFFIX}")))):
        rec = _read_json(p)
        if rec is not None and owns(rec, run_id):
            results.append(rec)
    return results


def clear_all(output_dir: str | Path) -> int:
    """Remove every request/result file. Called before a run is spawned so a
    request left pending by a stopped or crashed run cannot leak into the next
    one. Returns the number of files removed."""
    out = Path(output_dir)
    if not out.is_dir():
        return 0
    removed = 0
    for prefix in (REQUEST_PREFIX, RESULT_PREFIX):
        for p in out.glob(f"{prefix}*"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def sample_filename(step: int, sample_index: int,
                    request_id: Optional[str] = None) -> str:
    """The samples/ filename for a scheduled (request_id=None) or on-demand
    sample. An on-demand sample at the same global_step as a scheduled one must
    not overwrite it, and GET /training/runs/{id}/samples parses both forms."""
    base = f"step_{step:06d}_sample_{sample_index}"
    return f"{base}.png" if not request_id else f"{base}_ondemand_{request_id}.png"


def blank_on_failure_note(arch: Optional[str]) -> Optional[str]:
    if arch in BLANK_ON_FAILURE_ARCHS:
        return BLANK_ON_FAILURE_NOTE.format(arch=arch)
    return None
