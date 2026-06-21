"""Shared generation-cancellation check for inference denoise loops.

The `/cancel` endpoint sets `pipeline_manager.cancel_requested = True`. Denoise
loops call `raise_if_cancelled()` once per step; it raises so the in-flight
generation unwinds promptly. The flag is reset at the start of the next
generation (routes call `reset_cancel_flag()`), so no reset is needed here.

Imported lazily inside loops so model pipeline ops stay import-light and work in
contexts where the pipeline singleton is absent (e.g. training subprocesses).
"""


def is_cancelled() -> bool:
    try:
        from core.pipeline import pipeline_manager
    except ImportError:
        return False
    return bool(getattr(pipeline_manager, "cancel_requested", False))


def raise_if_cancelled() -> None:
    if is_cancelled():
        raise RuntimeError("Generation cancelled by user")
