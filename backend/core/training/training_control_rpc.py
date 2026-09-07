"""Runtime LR-schedule commands (API main process -> trainer subprocess).

§6 of ``docs/guides/LR_SCHEDULER_DESIGN.md``. Same file transport as
``training_sample_rpc`` (``training_file_rpc``), deliberately NOT the same
queue: every one of the sample queue's rules is about a request that runs a
full generation, and none of them holds for a command.

  API side:
    1. write   ``<output_dir>/.control_request_<id>.json``
    2. return 202 -- the trainer claims at its next batch boundary
    3. poll    ``GET /training/runs/{id}/lr-schedule``

  Trainer side:
    4. at the HEAD of the batch, before the forward, claim ALL pending commands
    5. apply each through ``ScheduleTimeline.add()``, write one
       ``.control_result_<id>.json`` per request carrying that call's §5.3
       result code, then refresh ``.lr_schedule.json``

Differences from the sample queue, each load-bearing:

* ALL pending commands are claimed per batch, not one. Coalescing happens in
  the timeline (a decay/cancel/decay burst folds in arrival order), so dribbling
  them out one batch at a time would only make the state depend on batch
  timing.
* Commands are claimed while a stop is pending too. A sample would delay the
  stop by a generation; a command is a dict append, and the checkpoint written
  on the way out carries it.
* The claim happens before the forward, not inside the sampling block: under
  the fused paths the optimizer hooks step during the BACKWARD, so a decay
  applied after the forward would not reach this batch's update (§5.6).

A ``retarget`` request additionally carries ``payload``: the schedule keys the
endpoint validated, in GLOBAL steps. The trainer resolves them into a
``ScheduleSpec`` at claim time, because the conversion to the scheduler axis
and the remaining span both need the run's own accumulation and totals.

``.lr_schedule.json`` is the trainer's display-only view of the timeline, read
by the GET endpoint. It is never an input to a resume -- the authoritative event
list is ``lr_schedule_events`` in ``training_state.json`` (D4).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.training.training_file_rpc import (
    atomic_write_json,
    clear_prefixes,
    list_records,
    list_request_paths,
    make_request_id,
    owns,
    parse_records,
    read_json,
    request_path as _request_path,
    write_record,
)

REQUEST_PREFIX = ".control_request_"
RESULT_PREFIX = ".control_result_"
STATUS_FILENAME = ".lr_schedule.json"

# The two parameterless commands, which is the enum of
# `POST /training/runs/{id}/lr-schedule`. `retarget` (§19) has its own endpoint
# because it carries a schedule; it shares this queue and nothing else.
COMMANDS = ("start_decay", "cancel_decay")
RETARGET_COMMAND = "retarget"
ALL_COMMANDS = COMMANDS + (RETARGET_COMMAND,)
# Mapped to the ScheduleTimeline event kinds they become. The trainer never
# invents a kind of its own from a command string.
COMMAND_EVENT_KINDS = {"start_decay": "decay", "cancel_decay": "cancel",
                       RETARGET_COMMAND: "retarget"}

# A command costs a dict append, so the cap is only there to stop an unbounded
# directory; it is not a throughput limit the way the sample queue's 3 is.
MAX_PENDING_REQUESTS = 20
# Above the pending cap, so the results of one full burst all survive to be read.
MAX_KEPT_RESULTS = 40


class ControlQueueFullError(RuntimeError):
    """Raised by :func:`queue_request` when the pending cap is already reached."""


def request_path(output_dir: str | Path, request_id: str) -> Path:
    return _request_path(output_dir, REQUEST_PREFIX, request_id)


def result_path(output_dir: str | Path, request_id: str) -> Path:
    return _request_path(output_dir, RESULT_PREFIX, request_id)


def status_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / STATUS_FILENAME


def list_pending_requests(output_dir: str | Path,
                          run_id: Optional[int] = None) -> List[Path]:
    return list_request_paths(output_dir, REQUEST_PREFIX, run_id)


def pending_requests(output_dir: str | Path,
                     run_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Parsed pending commands, oldest first."""
    return parse_records(list_pending_requests(output_dir, run_id))


def queue_request(
    output_dir: str | Path,
    *,
    command: str,
    run_id: Optional[int] = None,
    request_id: Optional[str] = None,
    max_pending: int = MAX_PENDING_REQUESTS,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write one command file. Raises ControlQueueFullError at the cap."""
    if command not in ALL_COMMANDS:
        raise ValueError(
            f"Unknown LR schedule command '{command}'. "
            f"Supported: {', '.join(ALL_COMMANDS)}")
    if command == RETARGET_COMMAND and not (extra or {}).get("payload"):
        raise ValueError(
            "A retarget command carries the schedule to switch to in "
            "extra={'payload': ...}; without it the trainer has nothing to "
            "retarget onto.")
    out = Path(output_dir)
    existing = list_pending_requests(out, run_id)
    if len(existing) >= max_pending:
        raise ControlQueueFullError(
            f"{len(existing)} LR schedule command(s) are already queued for "
            f"this run (maximum {max_pending}); the trainer applies all of them "
            f"at its next batch boundary."
        )
    rid = request_id or make_request_id()
    payload: Dict[str, Any] = {
        "request_id": rid,
        "run_id": None if run_id is None else int(run_id),
        "command": str(command),
        "queued_at": time.time(),
    }
    if extra:
        payload.update(extra)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(request_path(out, rid), payload)
    return payload


def claim_all(output_dir: str | Path,
              run_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Claim EVERY pending command for ``run_id``, oldest first.

    Commands for another run sharing this directory are left in place. Each file
    is deleted before its payload is returned, so a crash cannot replay it; a
    malformed file is deleted and dropped rather than left to be re-read forever.
    """
    claimed: List[Dict[str, Any]] = []
    for p in list_pending_requests(output_dir):
        record = read_json(p)
        if record is not None and not owns(record, run_id):
            continue
        try:
            p.unlink()
        except OSError:
            continue
        if record is not None:
            claimed.append(record)
    return claimed


def write_result(output_dir: str | Path, request_id: str,
                 result: Dict[str, Any]) -> None:
    """Record what ``ScheduleTimeline.add()`` answered for this request."""
    write_record(output_dir, RESULT_PREFIX, request_id, result,
                 keep=MAX_KEPT_RESULTS)


def list_results(output_dir: str | Path,
                 run_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Recorded results, newest first."""
    return list_records(output_dir, RESULT_PREFIX, run_id)


def write_status(output_dir: str | Path, status: Dict[str, Any]) -> None:
    """Atomic-write the display-only schedule state (D19)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(status_path(out), dict(status))


def read_status(output_dir: str | Path) -> Optional[Dict[str, Any]]:
    return read_json(status_path(output_dir))


def clear_all(output_dir: str | Path) -> int:
    """Remove every pending command and recorded result before a run is spawned.

    A "decay now" issued to a run that then stopped must NOT be applied by the
    next one; the commands that were already applied live in the checkpoint's
    event list instead. ``.lr_schedule.json`` is deliberately left alone -- it is
    the last known state, and the GET endpoint serves it for a stopped run.
    """
    return clear_prefixes(output_dir, (REQUEST_PREFIX, RESULT_PREFIX))
