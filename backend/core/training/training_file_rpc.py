"""File-RPC primitives shared by the trainer's inbound command queues.

The trainer is a subprocess with no stdin, no signal channel and no socket, so
every inbound control is a file in ``<output_dir>``: the API process writes a
request, the trainer claims it (delete first, so a crash cannot replay it) and
writes a result. ``training_sample_rpc`` and ``training_control_rpc`` are two
queues of that shape with different contracts; what they share lives here.

Naming convention both follow: ``<prefix><request_id>.json``, with the prefix
identifying the queue and the id echoed in the result file.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SUFFIX = ".json"


def make_request_id() -> str:
    return uuid.uuid4().hex[:16]


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def sorted_by_age(paths: List[Path]) -> List[Path]:
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


def request_path(output_dir: str | Path, prefix: str, request_id: str) -> Path:
    return Path(output_dir) / f"{prefix}{request_id}{SUFFIX}"


def list_request_paths(output_dir: str | Path, prefix: str,
                       run_id: Optional[int] = None) -> List[Path]:
    """Every ``<prefix>*`` file, oldest first, that ``run_id`` may act on."""
    out = Path(output_dir)
    if not out.is_dir():
        return []
    paths = sorted_by_age(list(out.glob(f"{prefix}*{SUFFIX}")))
    if run_id is None:
        return paths
    return [p for p in paths if owns(read_json(p), run_id)]


def parse_records(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """Readable records in the given path order; unreadable ones dropped."""
    records: List[Dict[str, Any]] = []
    for p in paths:
        record = read_json(p)
        if record is not None:
            records.append(record)
    return records


def write_record(output_dir: str | Path, prefix: str, request_id: str,
                 payload: Dict[str, Any], *, keep: Optional[int] = None) -> Path:
    """Atomic-write one ``<prefix><id>.json`` and prune the oldest siblings.

    Written last and in one piece, so a reader that sees the file sees a
    complete record.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record.setdefault("request_id", request_id)
    record.setdefault("completed_at", time.time())
    path = request_path(out, prefix, request_id)
    atomic_write_json(path, record)
    if keep is not None:
        stale = sorted_by_age(list(out.glob(f"{prefix}*{SUFFIX}")))
        for p in stale[:max(0, len(stale) - keep)]:
            try:
                p.unlink()
            except OSError:
                pass
    return path


def list_records(output_dir: str | Path, prefix: str,
                 run_id: Optional[int] = None,
                 newest_first: bool = True) -> List[Dict[str, Any]]:
    out = Path(output_dir)
    if not out.is_dir():
        return []
    paths = sorted_by_age(list(out.glob(f"{prefix}*{SUFFIX}")))
    if newest_first:
        paths = list(reversed(paths))
    return [r for r in parse_records(paths) if owns(r, run_id)]


def clear_prefixes(output_dir: str | Path, prefixes: Iterable[str]) -> int:
    """Remove every file under the given prefixes. Returns how many went.

    Globs ``<prefix>*`` rather than ``<prefix>*.json`` so an interrupted atomic
    write's ``.tmp`` leftover goes too.
    """
    out = Path(output_dir)
    if not out.is_dir():
        return 0
    removed = 0
    for prefix in prefixes:
        for p in out.glob(f"{prefix}*"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed
