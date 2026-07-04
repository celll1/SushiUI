"""In-memory image-generation status store.

This is a polling-friendly alternative to the WebSocket ``/api/v1/ws/progress``
channel (see ``backend/api/WS_PROTOCOL.md``). The WS channel is a global,
unfiltered broadcast with no ``complete``/``error`` message type — a client
can only infer completion/failure from the REST POST response returning.
This module keeps a small snapshot of the latest known state so a
frontend-less script can instead poll ``GET /generation/status`` without
holding a WebSocket connection open.

Single-process, single FastAPI worker assumption — same as
``ConnectionManager`` (the WS manager this mirrors). Not persisted across
backend restarts, and not multi-worker safe (this project runs one worker).

Hooked from (additive only, no change to WS behavior or generation logic):
  - ``create_progress_callback_factory()`` in ``backend/api/generation_utils.py``
    (per-step callback) -> ``update_progress()``
  - ``generate_txt2img`` / ``generate_img2img`` / ``generate_inpaint`` in
    ``backend/api/routes.py`` -> ``start_generation()`` at the top of the
    ``try:`` block, ``complete_generation()`` right before the success
    ``return``, and ``fail_generation()`` in the ``except`` blocks.
"""
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_lock = threading.Lock()

_state: Dict[str, Any] = {
    "status": "idle",  # "idle" | "running" | "error"
    "generation_type": None,  # "txt2img" | "img2img" | "inpaint"
    "current_step": None,
    "total_steps": None,
    "phase": None,  # human-readable status text, e.g. "Step 3/28"
    "started_at": None,
    "updated_at": None,
    "last_error": None,
    "last_result": None,  # {"image_id", "filename", "seed"} of the last successful generation
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_generation(generation_type: str) -> None:
    """Call at the very start of a generation request (top of the try: block)."""
    with _lock:
        now = _now()
        _state["status"] = "running"
        _state["generation_type"] = generation_type
        _state["current_step"] = None
        _state["total_steps"] = None
        _state["phase"] = None
        _state["started_at"] = now
        _state["updated_at"] = now
        _state["last_error"] = None


def update_progress(current_step: int, total_steps: int, phase: Optional[str] = None) -> None:
    """Call from the per-step progress callback (same callback that feeds the WS broadcast)."""
    with _lock:
        if _state["status"] != "running":
            # A stray callback firing after completion/failure should not
            # resurrect a "running" snapshot.
            return
        _state["current_step"] = current_step
        _state["total_steps"] = total_steps
        if phase is not None:
            _state["phase"] = phase
        _state["updated_at"] = _now()


def complete_generation(last_result: Optional[Dict[str, Any]] = None) -> None:
    """Call right before the success return of a generation endpoint."""
    with _lock:
        _state["status"] = "idle"
        _state["current_step"] = None
        _state["total_steps"] = None
        _state["phase"] = None
        _state["updated_at"] = _now()
        _state["last_error"] = None
        if last_result is not None:
            _state["last_result"] = last_result


def fail_generation(error_message: str) -> None:
    """Call from a generation endpoint's exception handler."""
    with _lock:
        _state["status"] = "error"
        _state["updated_at"] = _now()
        _state["last_error"] = error_message


def get_snapshot() -> Dict[str, Any]:
    """Return a shallow copy of the current status snapshot."""
    with _lock:
        return dict(_state)
