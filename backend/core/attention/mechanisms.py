"""Semantic attention methods, kept separate from kernel selection."""

from enum import Enum
from typing import Optional


class AttentionMechanism(str, Enum):
    DENSE = "dense"
    H3_VIDEO_WINDOW = "h3_video_window"
    H3_SOL_ATTN = "h3_sol_attn"


def known_mechanisms() -> tuple[str, ...]:
    return tuple(mechanism.value for mechanism in AttentionMechanism)


def validate_mechanism(
    mechanism: Optional[str], *, default: str = AttentionMechanism.DENSE.value
) -> str:
    if mechanism is None or not str(mechanism).strip():
        return default
    key = str(mechanism).strip().lower()
    if key not in known_mechanisms():
        raise ValueError(
            f"attention mechanism must be one of {', '.join(known_mechanisms())}; "
            f"got {mechanism!r}"
        )
    return key
