"""On-demand training-sample requests (API main process -> trainer subprocess).

Same file-RPC shape as ``training_preview_rpc`` (atomic write, delete before
processing, result written last), different contract: the trainer runs a claimed
request through its ORDINARY scheduled-sample block, so the PNG lands in
``<output_dir>/samples/`` under its own name and the result file here carries
only metadata. The transport itself lives in ``training_file_rpc``, shared with
``training_control_rpc``.

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

import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Transport primitives shared with training_control_rpc. The private aliases
# keep this module's own call sites unchanged.
from core.training.training_file_rpc import (
    SUFFIX,
    atomic_write_json as _atomic_write_json,
    clear_prefixes as _clear_prefixes,
    list_records as _list_records,
    list_request_paths as _list_request_paths,
    make_request_id,
    owns,
    parse_records as _parse_records,
    read_json as _read_json,
    request_path as _request_path,
    write_record as _write_record,
)

REQUEST_PREFIX = ".sample_request_"
RESULT_PREFIX = ".sample_result_"

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
    return _request_path(output_dir, REQUEST_PREFIX, request_id)


def result_path(output_dir: str | Path, request_id: str) -> Path:
    return _request_path(output_dir, RESULT_PREFIX, request_id)


def list_pending_requests(output_dir: str | Path,
                          run_id: Optional[int] = None) -> List[Path]:
    return _list_request_paths(output_dir, REQUEST_PREFIX, run_id)


def read_request(req_path: Path) -> Optional[Dict[str, Any]]:
    return _read_json(req_path)


def pending_requests(output_dir: str | Path,
                     run_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Parsed pending requests, oldest first (for the queue endpoint)."""
    return _parse_records(list_pending_requests(output_dir, run_id))


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
    _write_record(output_dir, RESULT_PREFIX, request_id, result,
                  keep=MAX_KEPT_RESULTS)


def list_results(output_dir: str | Path,
                 run_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Recorded results, newest first."""
    return _list_records(output_dir, RESULT_PREFIX, run_id)


def clear_all(output_dir: str | Path) -> int:
    """Remove every request/result file. Called before a run is spawned so a
    request left pending by a stopped or crashed run cannot leak into the next
    one. Returns the number of files removed."""
    return _clear_prefixes(output_dir, (REQUEST_PREFIX, RESULT_PREFIX))


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
