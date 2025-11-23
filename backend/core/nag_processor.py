"""Custom Attention Processors for NAG (Normalized Attention Guidance)

This module provides custom attention processors that enable NAG for SD1.5 and SDXL models.
NAG applies guidance in attention space by computing separate attention outputs for positive
and negative prompts, then combining them with normalized extrapolation.

Based on diffusers' attention processor architecture with NAG extensions.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention

from .nag import nag, split_nag_context_output


class NAGAttnProcessor:
    """Attention processor with NAG support for SD1.5

    This processor computes attention twice:
    1. For positive context (normal attention)
    2. For NAG negative context (additional attention)
    Then applies NAG to combine the outputs.
    """

    def __init__(
        self,
        nag_scale: float = 5.0,
        nag_tau: float = 3.5,
        nag_alpha: float = 0.25,
    ):
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process attention with NAG

        Args:
            attn: Attention module
            hidden_states: Input hidden states [B, N, C]
            encoder_hidden_states: Context from text encoder [B or 2*B, seq_len, dim]
                If NAG is active, this contains [positive_context, nag_negative_context] concatenated
            attention_mask: Optional attention mask
            temb: Optional time embeddings

        Returns:
            Processed hidden states with NAG applied
        """
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape

        # Check if NAG is active (context is doubled)
        if encoder_hidden_states is not None:
            context_batch = encoder_hidden_states.shape[0]
            apply_nag = context_batch == 2 * batch_size
            # Debug: Log context sizes
            if not hasattr(self, '_context_logged'):
                print(f"[NAG Processor] Context check: batch_size={batch_size}, context_batch={context_batch}, apply_nag={apply_nag}")
                self._context_logged = True
        else:
            apply_nag = False

        # Prepare query
        query = attn.to_q(hidden_states)

        # Use encoder_hidden_states if provided (cross-attention), else use hidden_states (self-attention)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # For cross-attention with NAG, split context into positive and negative
        if apply_nag and attn.to_k.in_features == encoder_hidden_states.shape[-1]:
            # This is cross-attention and NAG is active
            positive_context = encoder_hidden_states[:batch_size]
            nag_negative_context = encoder_hidden_states[batch_size:]

            # Compute attention for positive context
            key_positive = attn.to_k(positive_context)
            value_positive = attn.to_v(positive_context)

            query = attn.head_to_batch_dim(query)
            key_positive = attn.head_to_batch_dim(key_positive)
            value_positive = attn.head_to_batch_dim(value_positive)

            attention_probs_positive = attn.get_attention_scores(query, key_positive, attention_mask)
            hidden_states_positive = torch.bmm(attention_probs_positive, value_positive)
            hidden_states_positive = attn.batch_to_head_dim(hidden_states_positive)

            # Compute attention for NAG negative context
            key_negative = attn.to_k(nag_negative_context)
            value_negative = attn.to_v(nag_negative_context)

            key_negative = attn.head_to_batch_dim(key_negative)
            value_negative = attn.head_to_batch_dim(value_negative)

            attention_probs_negative = attn.get_attention_scores(query, key_negative, attention_mask)
            hidden_states_negative = torch.bmm(attention_probs_negative, value_negative)
            hidden_states_negative = attn.batch_to_head_dim(hidden_states_negative)

            # Apply NAG
            # Debug: Log NAG parameters being used
            if not hasattr(self, '_nag_logged'):
                print(f"[NAG Processor] Applying NAG with scale={self.nag_scale}, tau={self.nag_tau}, alpha={self.nag_alpha}")
                self._nag_logged = True
            hidden_states = nag(
                hidden_states_positive,
                hidden_states_negative,
                self.nag_scale,
                self.nag_tau,
                self.nag_alpha,
            )
        else:
            # Normal attention (self-attention or NAG not active)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class NAGAttnProcessor2_0:
    """Attention processor with NAG support using torch 2.0 scaled_dot_product_attention

    This is an optimized version for PyTorch 2.0+ that uses flash attention when available.
    """

    def __init__(
        self,
        nag_scale: float = 5.0,
        nag_tau: float = 3.5,
        nag_alpha: float = 0.25,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("NAGAttnProcessor2_0 requires PyTorch 2.0+ for scaled_dot_product_attention")
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape

        # Check if NAG is active
        if encoder_hidden_states is not None:
            context_batch = encoder_hidden_states.shape[0]
            apply_nag = context_batch == 2 * batch_size
            # Debug: Log context sizes
            if not hasattr(self, '_context_logged'):
                print(f"[NAG Processor2_0] Context check: batch_size={batch_size}, context_batch={context_batch}, apply_nag={apply_nag}")
                self._context_logged = True
        else:
            apply_nag = False

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # For cross-attention with NAG
        if apply_nag and attn.to_k.in_features == encoder_hidden_states.shape[-1]:
            positive_context = encoder_hidden_states[:batch_size]
            nag_negative_context = encoder_hidden_states[batch_size:]

            # Positive attention
            key_positive = attn.to_k(positive_context)
            value_positive = attn.to_v(positive_context)

            inner_dim = key_positive.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_positive = key_positive.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_positive = value_positive.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states_positive = F.scaled_dot_product_attention(
                query, key_positive, value_positive, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states_positive = hidden_states_positive.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            # Negative attention
            key_negative = attn.to_k(nag_negative_context)
            value_negative = attn.to_v(nag_negative_context)

            key_negative = key_negative.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_negative = value_negative.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Reuse query (already computed)
            hidden_states_negative = F.scaled_dot_product_attention(
                query, key_negative, value_negative, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states_negative = hidden_states_negative.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            # Apply NAG
            # Debug: Log NAG parameters being used
            if not hasattr(self, '_nag_logged'):
                print(f"[NAG Processor] Applying NAG with scale={self.nag_scale}, tau={self.nag_tau}, alpha={self.nag_alpha}")
                self._nag_logged = True
            hidden_states = nag(
                hidden_states_positive,
                hidden_states_negative,
                self.nag_scale,
                self.nag_tau,
                self.nag_alpha,
            )
        else:
            # Normal attention
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

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


def set_nag_processors(
    unet,
    nag_scale: float = 5.0,
    nag_tau: float = 3.5,
    nag_alpha: float = 0.25,
    use_torch2: bool = True,
):
    """Set NAG attention processors on all cross-attention layers in UNet

    Args:
        unet: UNet2DConditionModel
        nag_scale: NAG extrapolation scale
        nag_tau: NAG normalization threshold
        nag_alpha: NAG blending factor
        use_torch2: Use optimized torch 2.0 processor if available
    """
    processor_cls = NAGAttnProcessor2_0 if use_torch2 and hasattr(F, "scaled_dot_product_attention") else NAGAttnProcessor

    # Iterate through all attention layers
    nag_processor_count = 0
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            # Only set NAG processor for cross-attention layers
            # Cross-attention has different input dimensions for query and key/value
            if hasattr(module, 'to_k') and module.to_k is not None:
                # Check if this is cross-attention (not self-attention)
                # In cross-attention, encoder_hidden_states is used
                # We can identify this by checking if the attention is typically cross-attn
                # A simple heuristic: cross-attention layers usually have "attn2" in their name in SD
                if "attn2" in name or "cross" in name.lower():
                    module.set_processor(
                        processor_cls(
                            nag_scale=nag_scale,
                            nag_tau=nag_tau,
                            nag_alpha=nag_alpha,
                        )
                    )
                    nag_processor_count += 1
    print(f"[NAG] Set {nag_processor_count} NAG processors (scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha})")


def restore_original_processors(unet):
    """Restore original attention processors (remove NAG)

    Args:
        unet: UNet2DConditionModel
    """
    # Import default processors
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0, AttnProcessor

        # Try to use torch 2.0 processor if available
        if hasattr(F, "scaled_dot_product_attention"):
            default_processor = AttnProcessor2_0()
        else:
            default_processor = AttnProcessor()

        # Restore all attention processors
        for name, module in unet.named_modules():
            if isinstance(module, Attention):
                module.set_processor(default_processor)
    except ImportError:
        # Fallback: use UNet's built-in method
        unet.set_default_attn_processor()
