from __future__ import annotations

from typing import Callable, Optional


def callback_requests(
    callback: Optional[Callable], capability: str, step: int, total_steps: int
) -> bool:
    """Return False only when a callback explicitly declines optional work."""
    if callback is None:
        return False
    predicate = getattr(callback, capability, None)
    if predicate is None:
        return True
    try:
        return bool(predicate(step, total_steps))
    except Exception:
        return True


def compose_sampler_callbacks(
    progress_callback: Optional[Callable],
    diffusers_step_callback: Optional[Callable],
) -> Optional[Callable]:
    """Adapt a Diffusers step callback to the internal five-argument sampler API."""
    if diffusers_step_callback is None:
        return progress_callback
    if progress_callback is None:
        def combined(step, total_steps, latents, cfg_metrics=None, pred_x0=None):
            diffusers_step_callback(None, step, None, {"latents": latents})
    else:
        def combined(step, total_steps, latents, cfg_metrics=None, pred_x0=None):
            diffusers_step_callback(None, step, None, {"latents": latents})
            return progress_callback(step, total_steps, latents, cfg_metrics, pred_x0)

    for capability in ("wants_predicted_x0", "wants_cfg_metrics"):
        predicate = getattr(progress_callback, capability, None)
        if predicate is not None:
            setattr(combined, capability, predicate)
        elif progress_callback is None:
            setattr(combined, capability, lambda _step, _total: False)
    return combined
