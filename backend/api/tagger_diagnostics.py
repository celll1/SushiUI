"""Opt-in, checkpoint-isolated diagnostics for frozen FoveaTag comparisons."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid

from api.param_defaults import TAGGER_DIAGNOSTIC_DEFAULTS

_lock = threading.RLock()
_jobs: dict[str, subprocess.Popen] = {}


def directory(run):
    if (run.config or {}).get("architecture") not in ("foveatag_phase3", "foveatag_joint_review"):
        raise ValueError("Diagnostics currently support registered FoveaTag comparisons only")
    return Path(run.output_dir).resolve() / "diagnostics"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def status(run):
    root = directory(run)
    settings = dict(TAGGER_DIAGNOSTIC_DEFAULTS)
    if (root / "settings.json").is_file():
        settings.update(json.loads((root / "settings.json").read_text()))
    result = {"settings": settings, "items": [], "error": None}
    if (root / "index.json").is_file():
        result.update(json.loads((root / "index.json").read_text()))
    with _lock:
        job = _jobs.get(run.run_id)
        result["busy"] = job is not None and job.poll() is None
    return result


def configure(run, settings):
    with _lock:
        write_json(directory(run) / "settings.json", settings)
    return status(run)


def schedule(run, *, force=False):
    """Copy an atomic resume checkpoint before allowing training to overwrite it."""
    with _lock:
        state = status(run)
        if any(job.poll() is None for job in _jobs.values()) or (not force and not state["settings"]["enabled"]):
            return False
        checkpoint = Path(run.latest_checkpoint_path or "")
        root = directory(run)
        comparison = root.parent
        joint = run.config.get("architecture") == "foveatag_joint_review"
        if checkpoint.name != "latest.pt" or (checkpoint.parent.resolve() if joint else checkpoint.parent.parent.resolve()) != comparison:
            raise ValueError("A comparison resume checkpoint is required")
        mode = "routed" if joint else checkpoint.parent.name
        last = next((item for item in state["items"] if item["mode"] == mode), None)
        step = int(run.current_step)
        if not force and last and step - last["catalog_step"] < state["settings"]["interval_steps"]:
            return False
        job_dir = root / ("job-" + uuid.uuid4().hex)
        job_dir.mkdir(parents=True)
        try:
            shutil.copyfile(checkpoint, job_dir / "checkpoint.pt")
            request = {
                "comparison": str(comparison), "catalog_step": step,
                "arguments": run.config["joint_review"] if joint else run.config["foveatag"]["arguments"],
                "focus_preview": state["settings"]["focus_preview"],
            }
            write_json(job_dir / "request.json", request)
            worker = Path(__file__).resolve().parents[1] / "core/tagger/foveatag/diagnostic_worker.py"
            _jobs[run.run_id] = subprocess.Popen(
                [sys.executable, str(worker), str(job_dir)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception:
            shutil.rmtree(job_dir)
            raise
        return True


def image_path(run, name):
    root = directory(run)
    if name != Path(name).name or not name.endswith(".png"):
        raise ValueError("Invalid diagnostic image name")
    if not any(item["filename"] == name for item in status(run)["items"]):
        raise FileNotFoundError("Diagnostic is not published")
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise FileNotFoundError("Diagnostic image unavailable")
    return path
