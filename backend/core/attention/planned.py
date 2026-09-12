"""Dispatch for explicit semantic attention plans."""

from typing import Optional

import torch

from .contracts import AttentionMode, is_training_mode, validate_dense_qkv
from .observed import note_backend

_compiled_flex_attention = None


def _flex_attention_callable():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention

        _compiled_flex_attention = torch.compile(flex_attention, fullgraph=True, dynamic=False)
    return _compiled_flex_attention


def dispatch_planned_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan,
    *,
    scale: Optional[float] = None,
    mode: AttentionMode = AttentionMode.INFERENCE,
) -> torch.Tensor:
    """Evaluate a BSHD connectivity plan with compiled FlexAttention."""
    validate_dense_qkv(query, key, value, layout="BSHD", is_causal=False, attn_mask=None)
    if is_training_mode(mode) or torch.is_grad_enabled():
        raise RuntimeError("planned sparse attention is inference-only until gradient gates are complete")
    if query.shape[1] != plan.position_ids.shape[0] or key.shape[1] != query.shape[1]:
        raise ValueError("attention plan sequence length does not match q/k")
    if query.device != plan.position_ids.device:
        raise ValueError("attention plan and q/k/v must be on the same device")

    flex_attention = _flex_attention_callable()
    output = flex_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        block_mask=plan.block_mask(),
        scale=scale,
        enable_gqa=query.shape[2] != key.shape[2],
    )
    note_backend("flex")
    return output.transpose(1, 2).contiguous()
