"""Opt-in training-step diagnostics and legacy isolated comparison panels."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid
import hashlib

from api.param_defaults import TAGGER_DIAGNOSTIC_DEFAULTS

_lock = threading.RLock()
_jobs: dict[str, subprocess.Popen] = {}


def directory(run):
    if (run.config or {}).get("architecture", "siglip2") not in ("siglip2", "foveatag", "foveatag_phase3", "foveatag_joint_review"):
        raise ValueError("Unsupported tagger diagnostic architecture")
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
    if (root / "error.json").is_file():
        result.update(json.loads((root / "error.json").read_text()))
    result["attempts"] = json.loads((root / "attempts.json").read_text()) if (root / "attempts.json").is_file() else []
    with _lock:
        job = _jobs.get(run.run_id)
        result["busy"] = job is not None and job.poll() is None
    return result


def configure(run, settings):
    with _lock:
        write_json(directory(run) / "settings.json", settings)
    return status(run)


def schedule(run, *, force=False, filename=None):
    """Copy an atomic resume checkpoint before allowing training to overwrite it."""
    if run.config.get("architecture") != "foveatag_phase3":
        import torch
        path = directory(run) / "latest_training_snapshot.pt"
        if filename is not None:
            item = next((item for item in status(run)["items"] if item["filename"] == filename), None)
            snapshot_name = (item or {}).get("snapshot_filename")
            if not snapshot_name or Path(snapshot_name).name != snapshot_name:
                raise ValueError("Published diagnostic has no retained snapshot")
            path = directory(run) / snapshot_name
        if not path.is_file():
            raise ValueError("No captured training-step sample is available yet")
        return schedule_training(run, torch.load(path, map_location="cpu", weights_only=False), force=force, update_latest=filename is None)
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


def training_due(run, step):
    path = directory(run) / "settings.json"
    try:
        settings = json.loads(path.read_text()) if path.is_file() else TAGGER_DIAGNOSTIC_DEFAULTS
    except (OSError, ValueError):
        return False
    return settings["enabled"] and step % settings["interval_steps"] == 0


def schedule_training(run, snapshot, *, force=False, update_latest=True):
    """Publish the actual training forward, never substitute a fixed image."""
    import torch
    with _lock:
        root = directory(run)
        state = status(run)
        if not force and not state["settings"]["enabled"]:
            return False
        step = int(snapshot["step"])
        settings = state["settings"]
        threshold_settings = {key: settings[key] for key in ("threshold_mode", "threshold", "min_samples_for_per_tag", "min_best_thr")}
        revision = hashlib.sha256(json.dumps(threshold_settings, sort_keys=True).encode()).hexdigest()[:16]
        if any(item.get("prediction_origin") == snapshot.get("prediction_origin")
               and item["step"] == step and item.get("threshold_revision") == revision for item in state["items"]):
            return False
        reason = snapshot.get("skip_reason")
        if not reason and any(job.poll() is None for job in _jobs.values()):
            reason = "diagnostic_worker_busy"
        attempts = [*state["attempts"], {"step": step, "status": "skipped" if reason else "scheduled", "reason": reason}]
        write_json(root / "attempts.json", attempts)
        if reason:
            return False
        (root / "error.json").unlink(missing_ok=True)
        job_dir = root / ("job-" + uuid.uuid4().hex)
        job_dir.mkdir(parents=True)
        try:
            torch.save(snapshot, job_dir / "training_snapshot.pt")
            snapshot_name = "snapshot_" + job_dir.name.removeprefix("job-") + ".pt"
            shutil.copyfile(job_dir / "training_snapshot.pt", root / snapshot_name)
            if update_latest:
                temporary = root / "latest_training_snapshot.pt.tmp"
                shutil.copyfile(job_dir / "training_snapshot.pt", temporary)
                os.replace(temporary, root / "latest_training_snapshot.pt")
            write_json(job_dir / "request.json", {"comparison": str(root.parent), "catalog_step": step,
                "arguments": run.config.get("joint_review", {"max_source_pixels": run.config.get("fovea_max_source_pixels"),
                    "vocabulary_path": str(root.parent / "vocabulary.json")}), "training_snapshot": True,
                "snapshot_filename": snapshot_name, "threshold_settings": threshold_settings, "threshold_revision": revision})
            worker = Path(__file__).resolve().parents[1] / "core/tagger/foveatag/diagnostic_worker.py"
            _jobs[run.run_id] = subprocess.Popen([sys.executable, str(worker), str(job_dir)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
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


def schedule_training_safe(run, snapshot):
    try:
        return schedule_training(run, snapshot)
    except Exception as exc:
        try:
            write_json(directory(run) / "error.json", {"error": f"Diagnostic capture failed ({type(exc).__name__}); training continues"})
        except OSError:
            pass
        return False
