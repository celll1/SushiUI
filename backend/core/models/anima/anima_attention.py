"""Minimal attention dispatcher for vendored Anima model code.

The upstream sd-scripts implementation uses library.attention which dispatches
between PyTorch SDPA, xformers, flash-attn and sageattn. For SushiUI inference
we only need PyTorch SDPA, but we keep the AttentionParams / attention()
interface identical so the vendored anima_models.py works unchanged.
"""

from dataclasses import dataclass
from typing import Optional, Union, List

import torch
import torch.nn.functional as F


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None

    @property
    def supports_fp32(self) -> bool:
        return True

    @property
    def requires_same_dtype(self) -> bool:
        return False

    @staticmethod
    def create_attention_params(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        return AttentionParams(attn_mode=attn_mode, split_attn=split_attn)

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str], split_attn: bool, img_len: Optional[int], attention_mask: Optional[torch.Tensor]
    ) -> "AttentionParams":
        if attention_mask is None:
            return AttentionParams(attn_mode, split_attn)
        seqlens = attention_mask.sum(dim=1).to(torch.int32) + (img_len or 0)
        max_seqlen = attention_mask.shape[1] + (img_len or 0)
        # Expand attention mask to include image tokens (image tokens are always valid)
        expanded_mask = torch.nn.functional.pad(attention_mask, (img_len or 0, 0), value=1)
        expanded_mask = expanded_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]
        return AttentionParams(
            attn_mode=attn_mode,
            split_attn=split_attn,
            img_len=img_len,
            attention_mask=expanded_mask,
            seqlens=seqlens,
            cu_seqlens=None,
            max_seqlen=max_seqlen,
        )


def attention(
    qkv_or_q: Union[torch.Tensor, List[torch.Tensor]],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Input layout (matches sd-scripts): q/k/v are [B, L, H, D] (bshd).
    Returns [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        qkv_or_q.clear()
    else:
        q = qkv_or_q
        assert k is not None and v is not None

    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    # bshd -> bhsd for SDPA
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    attn_mask = attn_params.attention_mask if attn_params is not None else None
    x = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, dropout_p=drop_rate)

    x = x.transpose(1, 2)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]
    return x
