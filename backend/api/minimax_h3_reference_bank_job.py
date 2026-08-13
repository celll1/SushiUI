"""Background job that builds MiniMax-H3's text-encoder reference bank.

Backs ``GET /api/v1/models/minimax-h3/te-agreement`` and
``POST`` / ``DELETE .../te-agreement/reference-bank``.

WHY A POLLED JOB AND NOT THE GENERATION CHANNEL
-----------------------------------------------
The build encodes the tracked prompt suite with the loaded released encoder:
measured at ~5 min and 14-24 GiB of host RSS (``te_agreement.BUILD_COST``). That
is the shape ``quantized_export_job`` already answers -- one worker thread, a
polled job document, a hard refusal to start a second one -- so it uses the same
one. It is deliberately NOT on ``generation_status`` / the progress WebSocket:
``start_generation`` takes the coordinator's generation slot and sets the global
"a generation is running" state, which is exactly what this job must EXCLUDE
rather than claim, and the WS channel carries denoise steps with no completion
or failure message.

SERIALIZATION
-------------
The worker holds ``model_state_coordinator.mutation`` for the whole build, so a
model load, a component switch and a quantized export cannot swap the encoder
out from under it, and ``begin_generation`` refuses with 409 while it runs. The
START path additionally refuses when a generation or a training run is already
in flight -- they contend for the same GPU and the same host RAM.

CANCELLATION
------------
Cooperative, checked after each presentation, because the encode is a
single-presentation call this module cannot interrupt from outside. The bank
file is written only once every presentation is encoded, so a cancelled build
stores nothing; a cancelled REBUILD leaves the previous bank untouched.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.models.minimax_h3 import te_agreement as ta

_lock = threading.Lock()
_job: Optional[Dict[str, Any]] = None
_cancel = threading.Event()

# The start path already refused a running generation, so the worker only has to
# absorb one that began in the gap before it took the gate; queueing for longer
# would contradict the refusal the caller just saw.
_GATE_WAIT_SECONDS = 120.0


class BankBusyError(RuntimeError):
    """Something else is using the model, or a build is already running."""


class BankUnavailableError(ValueError):
    """The loaded model cannot produce a reference bank."""


class _BuildCancelled(RuntimeError):
    """Raised out of the progress callback to unwind the build."""


# ---------------------------------------------------------------------------
# The loaded model
# ---------------------------------------------------------------------------

def _loaded_components(pipeline_manager) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(components, reason_they_are_unavailable)`` for the loaded H3 model."""
    info = getattr(pipeline_manager, "current_model_info", None) or {}
    arch = str(info.get("type") or "").lower()
    if not arch:
        return None, "no model is loaded"
    if arch != "minimax_h3":
        return None, (f"the text-encoder reference bank is a MiniMax-H3 measurement; "
                      f"the loaded model is '{arch}'")
    components = getattr(pipeline_manager, "minimax_h3_components", None)
    if not components:
        return None, "the loaded MiniMax-H3 model exposes no components"
    return components, None


def _loaded_pairing(components: Dict[str, Any]) -> Dict[str, Any]:
    te_path = str(components.get("text_encoder_path") or "")
    projection_path = str((components.get("te_projection") or {}).get("path") or "")
    substitution = None
    if projection_path:
        # The one wording for a substituted pairing; it is what distinguishes a
        # local measurement from the compiled-in one measured elsewhere.
        from core.models.minimax_h3.te_projection import describe_te_substitution
        try:
            substitution = describe_te_substitution(te_path, projection_path)
        except Exception as exc:
            substitution = f"the substitution could not be described ({exc})"
    return {
        "text_encoder": os.path.basename(te_path) or None,
        "text_encoder_path": te_path or None,
        "projection": os.path.basename(projection_path) or None,
        "is_substitute": bool(projection_path),
        "substitution": substitution,
    }


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _job_snapshot() -> Dict[str, Any]:
    with _lock:
        if _job is None:
            return {"state": "idle"}
        return dict(_job)


def job_is_running() -> bool:
    with _lock:
        return _job is not None and _job.get("state") == "running"


def _bank_summary(manifest: Dict[str, Any], loaded_identity: Optional[str]) -> Dict[str, Any]:
    reference = manifest.get("reference") or {}
    presentations = manifest.get("presentations") or []
    return {
        "reference": reference.get("basename"),
        "suite_version": manifest.get("suite_version"),
        "presentations": len(presentations),
        "token_total": manifest.get("token_total"),
        "hidden_size": manifest.get("hidden_size"),
        "built_at": manifest.get("built_at"),
        "is_loaded_encoder": bool(loaded_identity) and
                             reference.get("identity") == loaded_identity,
    }


def _tree_identities(model_path: str) -> set:
    """Content identities of every text encoder and projection in one tree."""
    from core.models.minimax_h3.loader import describe_minimax_h3_text_encoder_choices

    choices = describe_minimax_h3_text_encoder_choices(model_path)
    paths = [entry["path"] for entry in choices.get("text_encoders") or []]
    paths += [entry["path"] for entry in choices.get("clip_projections") or []]
    identities = set()
    for path in paths:
        try:
            identities.add(ta.file_identity(path))
        except Exception:
            continue
    return identities


def _measurement_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    summary = ta.summarize_measurement(record)
    summary["encoder"] = (record.get("encoder") or {}).get("basename")
    summary["projection"] = (record.get("projection") or {}).get("basename")
    summary.pop("source", None)
    summary.pop("stage_b_reason", None)
    return summary


def agreement_status(pipeline_manager, model_path: Optional[str] = None, *,
                     root: Optional[str] = None) -> Dict[str, Any]:
    """The document ``GET /models/minimax-h3/te-agreement`` returns."""
    components, unavailable = _loaded_components(pipeline_manager)
    info = getattr(pipeline_manager, "current_model_info", None) or {}
    loaded = _loaded_pairing(components) if components else None
    tree = model_path or (str(info.get("source")) if components and info.get("source") else None)

    suite: Dict[str, Any] = {}
    suite_digest = None
    try:
        loaded_suite = ta.load_suite()
        suite_digest = loaded_suite["digest"]
        suite = {"version": loaded_suite["version"],
                 "prompts": len(loaded_suite["prompts"]),
                 "digest": suite_digest}
    except Exception as exc:
        suite = {"version": None, "prompts": 0, "digest": None, "error": str(exc)}

    loaded_identity = None
    if loaded and loaded.get("text_encoder_path"):
        try:
            loaded_identity = ta.file_identity(loaded["text_encoder_path"])
        except Exception:
            loaded_identity = None

    banks = [_bank_summary(manifest, loaded_identity)
             for manifest in ta.list_reference_banks(root=root)
             if manifest.get("suite_digest") == suite_digest]
    bank = next((entry for entry in banks if entry["is_loaded_encoder"]), None)

    measurements: List[Dict[str, Any]] = []
    measurements_reason = None
    if tree:
        try:
            identities = _tree_identities(tree)
        except Exception as exc:
            identities = None
            measurements_reason = f"{tree} could not be scanned: {exc}"
        if identities is not None:
            for record in ta.list_measurements(root=root):
                encoder = (record.get("encoder") or {}).get("identity")
                reference = (record.get("reference") or {}).get("identity")
                if encoder in identities or reference in identities:
                    measurements.append(_measurement_summary(record))
    else:
        measurements_reason = "no model tree to filter stored measurements by"

    # Naming the loaded encoder to the engine's own gate leaves exactly the
    # refusals that are about the encoder itself -- above all "a substitute
    # cannot be a reference" -- in the engine's wording.
    refusal = ta.reference_bank_refusal(
        components, reference_basename=str(loaded["text_encoder"] or "")) if loaded else None
    reason = unavailable or _busy_reason() or refusal
    return {
        "supported": components is not None,
        "can_build": reason is None,
        "reason": reason,
        "model_path": tree,
        "loaded": loaded,
        "suite": suite,
        "cost": dict(ta.BUILD_COST),
        "bank": bank,
        "banks": banks,
        "measurements": measurements,
        "measurements_reason": measurements_reason,
        "job": _job_snapshot(),
    }


# ---------------------------------------------------------------------------
# Start / cancel
# ---------------------------------------------------------------------------

def _busy_reason() -> Optional[str]:
    """Why a build must not run right now, or ``None``.

    A generation and this build contend for the same GPU and the same host RAM,
    and the build additionally needs the loaded encoder to hold still, so every
    lifecycle mutation blocks it too.
    """
    if job_is_running():
        return "a reference-bank build is already running"
    from api.routes import _fp8_scaled_mm_busy_reason

    busy = _fp8_scaled_mm_busy_reason()
    if busy:
        return busy
    from core.model_state_coordinator import model_state_coordinator

    mutation = model_state_coordinator.snapshot().get("mutation")
    if mutation:
        return f"{mutation} is changing the loaded model"
    return None


def start_bank_build(pipeline_manager, text_encoder_path: str, *,
                     root: Optional[str] = None) -> Dict[str, Any]:
    """Validate, then start the worker thread. Returns the initial job document."""
    global _job

    components, unavailable = _loaded_components(pipeline_manager)
    if components is None:
        raise BankUnavailableError(unavailable or "no MiniMax-H3 model is loaded")
    reference_basename = os.path.basename(str(text_encoder_path or "").strip())
    if not reference_basename:
        raise BankUnavailableError(
            "name the released text encoder the bank is to be built from.")
    refusal = ta.reference_bank_refusal(components, reference_basename=reference_basename)
    if refusal is not None:
        raise BankUnavailableError(refusal)

    # Outside the lock: `_busy_reason` takes it through `job_is_running`, and
    # `_lock` is not reentrant.
    busy = _busy_reason()
    if busy:
        raise BankBusyError(busy)

    with _lock:
        if _job is not None and _job.get("state") == "running":
            raise BankBusyError(
                f"a reference-bank build for {_job.get('reference')} is already running")
        job_id = uuid.uuid4().hex[:12]
        _cancel.clear()
        _job = {
            "job_id": job_id,
            "state": "running",
            "reference": reference_basename,
            "processed": 0,
            "total": 0,
            "message": "starting",
            "error": None,
            "result": None,
            "started_at": time.time(),
            "finished_at": None,
        }
        snapshot = dict(_job)

    threading.Thread(
        target=_run_build,
        args=(pipeline_manager, job_id, components, reference_basename, root),
        name=f"h3-reference-bank-{job_id}",
        daemon=True,
    ).start()
    return snapshot


def cancel_bank_build() -> Dict[str, Any]:
    """Ask a running build to stop after the presentation it is encoding."""
    with _lock:
        running = _job is not None and _job.get("state") == "running"
        if running:
            _cancel.set()
            _job["message"] = "cancelling"
        return dict(_job) if _job is not None else {"state": "idle"}


def _update(job_id: str, **fields) -> None:
    with _lock:
        if _job is None or _job.get("job_id") != job_id:
            return
        _job.update(fields)


def _discard_pending_bank(te_path: str, root: Optional[str]) -> None:
    """Remove the build's directory unless it holds a readable bank.

    A cancelled first build leaves an empty directory the engine created; a
    cancelled REBUILD leaves the previous bank there, and that one is kept.
    """
    try:
        if ta.find_reference_bank(te_path, root=root) is not None:
            return
        directory = (ta.store_dir(root) / "banks"
                     / ta.bank_key(ta.load_suite()["digest"], ta.file_identity(te_path)))
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
    except Exception as exc:
        print(f"[MiniMaxH3ReferenceBank] pending bank directory not removed: {exc}")


def _run_build(pipeline_manager, job_id, components, reference_basename, root):
    te_path = str(components.get("text_encoder_path") or "")

    def progress(done: int, total: int, name: str) -> None:
        if _cancel.is_set():
            raise _BuildCancelled(f"cancelled after {done} of {total} presentations")
        _update(job_id, processed=done, total=total, message=f"encoding presentation {name}")

    try:
        from core.model_state_coordinator import model_state_coordinator

        load_lock = getattr(pipeline_manager, "_load_model_lock", None)
        _update(job_id, message="waiting for the model lifecycle gate")

        def _report_wait(reasons):
            _update(job_id, message="waiting for " + ", ".join(reasons))

        # Exclusive for the whole build: it reads the live encoder for minutes,
        # and holding the gate is also what makes a generation started meanwhile
        # refuse. Subprocess training never touches these modules, so it is not
        # a blocker here (the start path already refused if one was running).
        with model_state_coordinator.mutation(
            "MiniMax-H3 reference bank build",
            wait_timeout=_GATE_WAIT_SECONDS,
            wait_for_activities=False,
            on_wait=_report_wait,
        ):
            if load_lock is not None:
                load_lock.acquire()
            try:
                _update(job_id, message="encoding the prompt suite")
                manifest = ta.build_reference_bank(
                    components,
                    reference_basename=reference_basename,
                    root=root,
                    progress=progress,
                )
            finally:
                if load_lock is not None:
                    load_lock.release()

        manifest.pop("dir", None)
        _update(job_id, state="completed", result=manifest, message="completed",
                finished_at=time.time())
    except _BuildCancelled as exc:
        _discard_pending_bank(te_path, root)
        _update(job_id, state="cancelled", message=str(exc), finished_at=time.time())
    except BaseException as exc:  # noqa: BLE001 - reported verbatim to the user
        import traceback
        traceback.print_exc()
        _discard_pending_bank(te_path, root)
        _update(job_id, state="failed", error=f"{type(exc).__name__}: {exc}",
                message="failed", finished_at=time.time())
    finally:
        _cancel.clear()
