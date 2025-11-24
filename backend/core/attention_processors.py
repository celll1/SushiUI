"""
Custom Attention Processors for accelerated inference

Supports:
- Normal: PyTorch 2.0+ scaled_dot_product_attention (Flash Attention automatically enabled)
- SageAttention: Quantized attention for 2-5x speedup
- FlashAttention: Explicit Flash Attention 2 (when available)
"""

import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention


class SageAttnProcessor:
    """
    SageAttention Processor - Quantized attention for accelerated inference

    Uses INT8 quantization for QK^T and FP16/FP8 for PV to achieve 2-5x speedup
    over standard attention while maintaining accuracy.

    Requires: pip install sageattention
    """

    def __init__(self):
        try:
            from sageattention import sageattn
            self.sageattn = sageattn
            self._available = True
        except ImportError:
            print("[SageAttention] Warning: sageattention not installed. Falling back to normal attention.")
            print("[SageAttention] Install with: pip install sageattention")
            self._available = False

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

        # Reshape to (batch, heads, seq_len, head_dim) for SageAttention
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply SageAttention or fallback to normal
        if self._available:
            # SageAttention expects (batch, heads, seq_len, head_dim) - "HND" layout
            hidden_states = self.sageattn(query, key, value, tensor_layout="HND", is_causal=False)
        else:
            # Fallback to standard attention
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
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


class FlashAttnProcessor:
    """
    FlashAttention-2 Processor - Explicit Flash Attention acceleration

    PyTorch 2.0+ automatically uses Flash Attention when available via SDPA,
    but this processor can be used to explicitly ensure Flash Attention is used.

    Note: PyTorch 2.0+ with CUDA will automatically use Flash Attention in SDPA
    """

    def __init__(self):
        self._flash_available = False
        try:
            # Check if flash_attn is installed
            import flash_attn
            self._flash_available = True
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            print("[FlashAttention] Using explicit flash_attn package")
        except ImportError:
            # Fallback to PyTorch SDPA (which uses Flash Attention automatically on supported hardware)
            print("[FlashAttention] flash_attn package not found, using PyTorch SDPA")
            print("[FlashAttention] PyTorch 2.0+ will automatically use Flash Attention when available")
            self._flash_available = False

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

        # Reshape for attention
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if self._flash_available:
            # Use explicit flash_attn package
            # flash_attn_func expects (batch, seqlen, nheads, headdim)
            hidden_states = self.flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
        else:
            # Use PyTorch SDPA (automatically uses Flash Attention when available)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
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
    Set attention processor type for the UNet

    Args:
        unet: The UNet model
        attention_type: Type of attention to use ("normal", "sage", "flash")

    Returns:
        dict: Original processors for restoration
    """
    # Store original processors
    original_processors = unet.attn_processors.copy()

    if attention_type == "sage":
        print("[AttentionProcessor] Setting SageAttention processors")
        processor = SageAttnProcessor()
        new_processors = {name: processor for name in unet.attn_processors.keys()}
        unet.set_attn_processor(new_processors)

    elif attention_type == "flash":
        print("[AttentionProcessor] Setting FlashAttention processors")
        processor = FlashAttnProcessor()
        new_processors = {name: processor for name in unet.attn_processors.keys()}
        unet.set_attn_processor(new_processors)

    else:  # "normal"
        print("[AttentionProcessor] Using default PyTorch 2.0 SDPA (auto Flash Attention)")
        # Reset to default processors - PyTorch 2.0+ automatically uses Flash Attention
        from diffusers.models.attention_processor import AttnProcessor2_0
        processor = AttnProcessor2_0()
        new_processors = {name: processor for name in unet.attn_processors.keys()}
        unet.set_attn_processor(new_processors)

    return original_processors


def restore_processors(unet, original_processors: dict):
    """Restore original attention processors"""
    if original_processors:
        unet.set_attn_processor(original_processors)
        print("[AttentionProcessor] Restored original processors")
