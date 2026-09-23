"""Run-scoped inference export, separate from resumable training checkpoints."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path


_lock = threading.Lock()
_active_runs: set[int] = set()


def job_is_running(run_id: int) -> bool:
    with _lock:
        return run_id in _active_runs


def latest_complete_checkpoint(output_dir: str | Path, run_name: str) -> Path:
    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Training output directory not found: {root}")
    candidates = []
    pattern = re.compile(rf"^{re.escape(run_name)}_step_(\d+)\.safetensors(?:\.index\.json)?$")
    for candidate in root.iterdir():
        match = pattern.fullmatch(candidate.name)
        if match is None or not candidate.is_file():
            continue
        stem = f"{run_name}_step_{match.group(1)}"
        if not (root / f"{stem}_state.json").is_file():
            continue
        if not (root / f"{stem}_optimizer.pt").is_file():
            continue
        if candidate.name.endswith(".index.json"):
            try:
                members = set(json.loads(candidate.read_text(encoding="utf-8"))["weight_map"].values())
            except (OSError, ValueError, KeyError):
                continue
            if not members or any(not (root / member).is_file() for member in members):
                continue
        candidates.append((int(match.group(1)), candidate))
    if not candidates:
        raise FileNotFoundError("No complete floating checkpoint with optimizer state is available")
    return max(candidates, key=lambda item: item[0])[1]


def _status_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "exports" / "int8_convrot_status.json"


def _write_status(output_dir: str | Path, payload: dict) -> None:
    path = _status_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def export_status(output_dir: str | Path, run_id: int | None = None) -> dict:
    path = _status_path(output_dir)
    if not path.is_file():
        return {"state": "idle"}
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        if status.get("state") in {"queued", "running"}:
            if run_id is not None and not job_is_running(run_id):
                return {**status, "state": "failed", "error": "Export process ended before completion"}
        return status
    except (OSError, ValueError):
        return {"state": "unknown"}


def export_run(
    run_id: int, run_name: str, output_dir: str | Path, *, overwrite: bool = False,
) -> dict:
    with _lock:
        if run_id in _active_runs:
            raise RuntimeError("An export is already running for this run")
        _active_runs.add(run_id)
    try:
        return _execute_export(run_name, output_dir, overwrite=overwrite)
    finally:
        with _lock:
            _active_runs.discard(run_id)


def _execute_export(run_name: str, output_dir: str | Path, *, overwrite: bool) -> dict:
    try:
        source = latest_complete_checkpoint(output_dir, run_name)
        _write_status(output_dir, {
            "state": "running", "source": source.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        from core.training.qwen_full_export import export_qwen_full_checkpoint

        result = export_qwen_full_checkpoint(
            source, output_dir, overwrite=overwrite, device="cpu"
        )
        status = {
            "state": "completed", "source": source.name,
            "path": str(result),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_status(output_dir, status)
        return status
    except Exception as exc:
        _write_status(output_dir, {
            "state": "failed", "error": str(exc),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        raise


def start_export(run_id: int, run_name: str, output_dir: str | Path, *, overwrite: bool = False) -> dict:
    with _lock:
        if run_id in _active_runs:
            raise RuntimeError("An export is already running for this run")
        source = latest_complete_checkpoint(output_dir, run_name)
        published = Path(output_dir) / "exports" / "int8_convrot"
        previous = export_status(output_dir)
        if published.exists() and not overwrite:
            if previous.get("state") == "completed" and previous.get("source") == source.name:
                return previous
            raise FileExistsError("An INT8 ConvRot export already exists; choose overwrite")
        _active_runs.add(run_id)
    try:
        _write_status(output_dir, {"state": "queued", "started_at": datetime.now(timezone.utc).isoformat()})
    except BaseException:
        with _lock:
            _active_runs.discard(run_id)
        raise

    def worker() -> None:
        try:
            _execute_export(run_name, output_dir, overwrite=overwrite)
        except Exception as exc:
            print(f"[Training Export] Run {run_id} failed: {exc}")
        finally:
            with _lock:
                _active_runs.discard(run_id)

    thread = threading.Thread(target=worker, name=f"training-export-{run_id}", daemon=True)
    try:
        thread.start()
    except BaseException:
        with _lock:
            _active_runs.discard(run_id)
        raise
    return {"state": "queued"}
