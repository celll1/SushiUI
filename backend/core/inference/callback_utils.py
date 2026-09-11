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
