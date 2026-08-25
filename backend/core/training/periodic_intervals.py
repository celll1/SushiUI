"""Periodic-action intervals for the training loop.

Every optional periodic action (checkpointing, sampling, debug dumps) is
scheduled by ``step % interval``. An interval of 0 -- the obvious spelling of
"never do this" -- used to reach that modulo and raise ``ZeroDivisionError`` on
the first step. Here 0 means "never", matching ``tagger_trainer`` and
``vae_trainer``, and ``due()`` is the only place the modulo is written.
"""

from typing import Optional


def normalize_interval(value: Optional[int], minimum: int = 0) -> int:
    """Clamp a periodic interval to ``minimum``.

    ``minimum=0`` for optional actions (0 = never); ``minimum=1`` for intervals
    that are not optional, where 0 is meaningless rather than disabling.
    """
    try:
        n = int(value or 0)
    except (TypeError, ValueError):
        n = 0
    return max(minimum, n)


def due(step: int, interval: int) -> bool:
    """True when ``step`` lands on a multiple of a positive ``interval``."""
    return interval > 0 and step % interval == 0
