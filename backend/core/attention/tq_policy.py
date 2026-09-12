"""Process-local TQ backward policy for the isolated training worker."""

from dataclasses import dataclass
from typing import Optional


TQ_BACKWARD_MODES = ("auto", "triton", "fa2", "fa2_deterministic")


@dataclass(frozen=True)
class TQBackwardPolicy:
    mode: str
    fa2_deterministic: bool


_policy = TQBackwardPolicy(mode="triton", fa2_deterministic=False)


def resolve_tq_backward_mode(value: Optional[str], *, resuming: bool) -> str:
    """Preserve Triton for old resumes while new runs adopt TQ's auto policy."""
    resolved = value if value is not None else ("triton" if resuming else "auto")
    if resolved not in TQ_BACKWARD_MODES:
        raise ValueError(
            f"tq_backward_mode must be one of {TQ_BACKWARD_MODES}; got {resolved!r}"
        )
    return resolved


def set_tq_backward_policy(value: str) -> TQBackwardPolicy:
    """Set the worker-wide policy and return its resolved TQ arguments."""
    global _policy
    value = resolve_tq_backward_mode(value, resuming=False)
    deterministic = value == "fa2_deterministic"
    _policy = TQBackwardPolicy(
        mode="fa2" if deterministic else value,
        fa2_deterministic=deterministic,
    )
    return _policy


def get_tq_backward_policy() -> TQBackwardPolicy:
    return _policy
