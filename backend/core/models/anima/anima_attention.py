"""Anima adapter from the vendored call shape to SushiUI's attention conduit."""

from dataclasses import dataclass
from typing import Optional, Union, List

import torch

from core.attention import AttentionMode, dispatch_attention, normalize_backend


def set_attention_backend(model, backend: Optional[str], mode: AttentionMode) -> int:
    """Stamp Anima's root and LLM-adapter attention without process globals."""
    canonical = normalize_backend(backend)
    count = 0
    for module in model.modules():
        if hasattr(module, "attn_mode"):
            module.attn_mode = canonical
            count += 1
        if module.__class__.__name__ == "LLMAdapterAttention":
            module._attn_backend = canonical
            module._attn_mode = mode
            count += 1
    return count


def _resolve_backend(attn_params: Optional["AttentionParams"]) -> str:
    mode = attn_params.attn_mode if attn_params is not None else None
    return "native" if mode in (None, "torch") else normalize_backend(mode)


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None
    mode: AttentionMode = AttentionMode.INFERENCE

    @property
    def supports_fp32(self) -> bool:
        return True

    @property
    def requires_same_dtype(self) -> bool:
        return False

    @staticmethod
    def create_attention_params(
        attn_mode: Optional[str], split_attn: bool, mode: AttentionMode = AttentionMode.INFERENCE
    ) -> "AttentionParams":
        return AttentionParams(attn_mode=attn_mode, split_attn=split_attn, mode=mode)

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

    attn_mask = attn_params.attention_mask if attn_params is not None else None

    # Route the kernel through the unified attention conduit. q/k/v are
    # [B, L, H, D] (bshd) == the conduit's canonical BSHD layout, so no boundary
    # transpose is needed here (the conduit does BHSD<->BSHD itself).
    #
    # Backend selection: the inference module-global (set by
    # pipeline_backends/anima.py) wins; otherwise the attn_mode carried in
    # attn_params (training path) decides. The conduit normalizes the string,
    # applies capability guards, and falls back to native on any kernel failure.
    #
    # split_attn (per-sample varlen) and the varlen block-diagonal / additive
    # attention_mask cannot be honored by FlashAttention/SageAttention. The
    # conduit's mask guard already downgrades any masked call to native; we
    # additionally force native when split_attn is set so those paths stay
    # native-only (matching the vendored SDPA numerics exactly).
    backend = _resolve_backend(attn_params)
    if attn_params is not None and attn_params.split_attn:
        backend = "native"

    x = dispatch_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=drop_rate,
        backend=backend,
        mode=attn_params.mode,
        layout="BSHD",
    )  # [B, L, H, D]

    return x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]
