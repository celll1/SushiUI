"""
Custom Attention Processor for accelerated inference (SDXL / SD1.5 UNet).

A single :class:`UnifiedAttnProcessor` handles every backend by routing the core
attention region through the unified conduit (:func:`core.attention.dispatch_attention`).
The processor keeps only the diffusers norm / residual / reshape boilerplate; the
backend selection, capability guards (head_dim / mask / dtype / GQA), and native
fallback all live in the conduit.

Backends (selected by the ``attention_type`` string, normalized inside the conduit):
- "normal"/"none"/"sdpa"/None: PyTorch scaled_dot_product_attention (native)
- "flash":                     FlashAttention-2 (falls back to native on any failure)
- "sage":                      SageAttention INT8 (auto-downgrades to native when the
                               head_dim is unsupported -- e.g. SD1.5 40/80/160)
"""

import torch
from typing import Optional
from diffusers.models.attention_processor import Attention

from core.attention import AttentionMode, dispatch_attention


class UnifiedAttnProcessor:
    """
    Unified attention processor for the diffusers UNet (SDXL / SD1.5).

    Replaces the former hand-written SageAttnProcessor / FlashAttnProcessor and
    the default AttnProcessor2_0. The QKV projections are reshaped to
    ``[batch, heads, seq_len, head_dim]`` (BHSD) and handed to the conduit with
    ``layout="BHSD"``; the conduit adapts the layout and dispatches to the
    selected kernel, falling back to native SDPA when the kernel is unavailable
    or unsupported for the given shapes.

    Args:
        backend: Backend selector string ("normal", "sage", or "flash"). The
            string is normalized and capability-gated inside the conduit, so no
            per-processor availability probing is needed here.
    """

    def __init__(self, backend: str = "normal"):
        self.backend = backend

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape to BHSD == [batch, heads, seq_len, head_dim].
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Core attention region -> unified conduit (BHSD in/out).
        hidden_states = dispatch_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self.backend,
            mode=AttentionMode.INFERENCE,
            layout="BHSD",
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def set_attention_processor(unet, attention_type: str = "normal"):
    """
    Set the unified attention processor on every attention layer of the UNet.

    Args:
        unet: The UNet model
        attention_type: Backend selector ("normal", "sage", "flash"). Normalized
            inside the conduit.

    Returns:
        dict: Original processors for restoration
    """
    # Store original processors
    original_processors = unet.attn_processors.copy()

    num_processors = len(unet.attn_processors)

    print(f"[AttentionProcessor] Setting UnifiedAttnProcessor (backend={attention_type}) for {num_processors} attention layers")
    new_processors = {name: UnifiedAttnProcessor(attention_type) for name in unet.attn_processors.keys()}
    unet.set_attn_processor(new_processors)
    print(f"[AttentionProcessor] [OK] UnifiedAttnProcessor ACTIVE for all {num_processors} layers")

    return original_processors


def restore_processors(unet, original_processors: dict):
    """Restore original attention processors"""
    if original_processors:
        unet.set_attn_processor(original_processors)
        print("[AttentionProcessor] Restored original processors")
