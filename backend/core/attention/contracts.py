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


def validate_varlen_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> None:
    """Validate the packed ``[total, heads, dim]`` representation."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("varlen attention requires rank-3 packed q/k/v tensors")
    if key.shape != value.shape:
        raise ValueError("packed key/value shapes must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("packed query/key head dimensions must match")
    if query.device != key.device or key.device != value.device:
        raise ValueError("packed q/k/v must be on the same device")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("packed q/k/v must have the same dtype")

    q_heads, kv_heads = query.shape[1], key.shape[1]
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError(
            f"query heads ({q_heads}) must be a positive multiple of key/value heads ({kv_heads})"
        )
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be one-dimensional")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must hold n_segments + 1 offsets")
    if cu_seqlens_q.dtype not in (torch.int32, torch.int64) or cu_seqlens_k.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must contain integer offsets")
    if cu_seqlens_q.device != query.device or cu_seqlens_k.device != query.device:
        raise ValueError("packed q/k/v and cumulative offsets must be on the same device")
    if not cu_seqlens_q.is_contiguous() or not cu_seqlens_k.is_contiguous():
        raise ValueError("cumulative offsets must be contiguous")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("maximum sequence lengths must be positive")

    # CPU offsets are cheap to inspect. Avoid a hidden CUDA synchronization in
    # the shared boundary; CUDA kernels validate the terminal offset themselves.
    if cu_seqlens_q.device.type == "cpu":
        offsets_q = cu_seqlens_q.tolist()
        offsets_k = cu_seqlens_k.tolist()
        if offsets_q[0] != 0 or offsets_k[0] != 0:
            raise ValueError("cumulative offsets must start at zero")
        if offsets_q[-1] != query.shape[0] or offsets_k[-1] != key.shape[0]:
            raise ValueError("terminal cumulative offsets must equal packed token counts")
        if any(b < a for a, b in zip(offsets_q, offsets_q[1:])) or any(
            b < a for a, b in zip(offsets_k, offsets_k[1:])
        ):
            raise ValueError("cumulative offsets must be nondecreasing")
        if max((b - a for a, b in zip(offsets_q, offsets_q[1:])), default=0) > max_seqlen_q:
            raise ValueError("max_seqlen_q is smaller than a packed query segment")
        if max((b - a for a, b in zip(offsets_k, offsets_k[1:])), default=0) > max_seqlen_k:
            raise ValueError("max_seqlen_k is smaller than a packed key/value segment")
