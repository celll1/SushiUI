"""Shared attention call contracts.

Kernel selection and attention semantics are separate concerns.  This module
owns the execution-level invariants that every dense kernel must satisfy; sparse
or otherwise non-dense mechanisms add their own semantic plan on top.
"""

from enum import Enum
from typing import Optional

import torch


class AttentionMode(str, Enum):
    INFERENCE = "inference"
    TRAINING = "training"


class AttentionFallbackPolicy(str, Enum):
    ERROR = "error"
    WARN = "warn"


VALID_LAYOUTS = frozenset({"BSHD", "BHSD"})


def is_training_mode(mode) -> bool:
    return mode == "training" or getattr(mode, "value", None) == "training"


def resolve_fallback_policy(
    policy: Optional[AttentionFallbackPolicy | str], mode
) -> AttentionFallbackPolicy:
    if policy is None:
        return (
            AttentionFallbackPolicy.ERROR
            if is_training_mode(mode)
            else AttentionFallbackPolicy.WARN
        )
    try:
        return AttentionFallbackPolicy(policy)
    except ValueError as exc:
        accepted = ", ".join(item.value for item in AttentionFallbackPolicy)
        raise ValueError(f"fallback_policy must be one of {accepted}; got {policy!r}") from exc


def validate_layout(layout: str) -> None:
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"layout must be 'BSHD' or 'BHSD'; got {layout!r}")


def validate_dense_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: str,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
) -> None:
    """Reject ambiguous shapes before a backend can fail or misroute them."""
    validate_layout(layout)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("dense attention requires rank-4 q/k/v tensors")

    head_axis, seq_axis = ((2, 1) if layout == "BSHD" else (1, 2))
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("q/k/v batch dimensions must match")
    if key.shape[seq_axis] != value.shape[seq_axis]:
        raise ValueError("key/value sequence lengths must match")
    if key.shape[head_axis] != value.shape[head_axis]:
        raise ValueError("key/value head counts must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key head dimensions must match")
    if query.device != key.device or key.device != value.device:
        raise ValueError("q/k/v must be on the same device")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("q/k/v must have the same dtype")

    q_heads = query.shape[head_axis]
    kv_heads = key.shape[head_axis]
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError(
            f"query heads ({q_heads}) must be a positive multiple of key/value heads ({kv_heads})"
        )
    if is_causal and attn_mask is not None:
        raise ValueError("attn_mask and is_causal cannot be set together")
