"""
NAG (Normalized Attention Guidance) Attention Processor

Implements attention-space guidance by computing:
1. hidden_states_positive = Attn(Q, K_pos, V_pos) - positive attention output
2. hidden_states_negative = Attn(Q, K_neg, V_neg) - negative attention output
3. guidance = positive * φ - negative * (φ - 1) - NAG guidance formula
4. guidance_normalized = guidance * min(||guidance||/||positive||, tau) - L1 normalization with tau threshold
5. result = guidance_normalized * α + positive * (1 - α) - alpha blending

Where:
- φ (phi) = nag_scale (guidance strength, typical: 3-7)
- τ (tau) = nag_tau (normalization threshold, typical: 2.5-3.5)
- α (alpha) = nag_alpha (blending factor, typical: 0.25-0.5)

Reference: Official implementation from ChenDarYen/Normalized-Attention-Guidance
"""

import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention


class NAGAttnProcessor2_0:
    """
    NAG Attention Processor for PyTorch 2.0+ (scaled_dot_product_attention)

    Processes cross-attention with NAG guidance in attention space.

    Args:
        nag_scale: Extrapolation scale φ (default: 5.0)
        nag_tau: L1 normalization threshold (default: 3.5)
        nag_alpha: Blending factor α (default: 0.25)
        attention_type: Attention backend - "normal", "sage", or "flash" (default: "normal")
    """

    def __init__(
        self,
        nag_scale: float = 5.0,
        nag_tau: float = 3.5,
        nag_alpha: float = 0.25,
        attention_type: str = "normal",
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("NAGAttnProcessor2_0 requires PyTorch 2.0+")
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.attention_type = attention_type
        self._call_count = 0

        # Initialize SageAttention if requested
        if attention_type == "sage":
            try:
                from sageattention import sageattn
                self.sageattn = sageattn
                self._sage_available = True
                print(f"[NAG-SageAttention] Successfully loaded SageAttention module")
            except ImportError:
                print(f"[NAG-SageAttention] Warning: sageattention not installed, falling back to normal")
                self._sage_available = False
                self.attention_type = "normal"
        else:
            self._sage_available = False

    def _compute_attention(self, query, key, value, attention_mask=None):
        """
        Compute attention using the selected backend (normal/sage/flash)

        Args:
            query: [batch, heads, seq_len, head_dim]
            key: [batch, heads, seq_len, head_dim]
            value: [batch, heads, seq_len, head_dim]

        Returns:
            [batch, heads, seq_len, head_dim]
        """
        self._call_count += 1

        if self.attention_type == "sage" and self._sage_available:
            # Log first call
            if self._call_count == 1:
                print(f"[NAG-SageAttention] First attention call - using SageAttention backend")

            # SageAttention expects HND layout
            try:
                output = self.sageattn(query, key, value, tensor_layout="HND", is_causal=False)
                if self._call_count == 1:
                    print(f"[NAG-SageAttention] sageattn call succeeded")
                return output
            except Exception as e:
                if self._call_count == 1:
                    print(f"[NAG-SageAttention] Error, falling back to SDPA: {e}")
                return F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        else:
            # Normal or Flash (both use PyTorch SDPA which auto-selects Flash if available)
            if self._call_count == 1:
                backend = "FlashAttention-2" if self.attention_type == "flash" else "PyTorch SDPA"
                print(f"[NAG-{backend}] First attention call - using {backend} backend")

            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

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
        # Log that processor is being called
        import sys
        if not hasattr(self, '_called_logged'):
            print(f"[NAG CALL] NAG processor called! encoder_hidden_states is None: {encoder_hidden_states is None}", file=sys.stderr)
            self._called_logged = True

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

        # Prepare query from hidden_states (image features)
        query = attn.to_q(hidden_states)

        # Check if this is cross-attention with NAG-formatted embeddings
        is_cross_attention_original = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            # Self-attention: use original processing
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # NAG mode detection: cross-attention with batch_size=2, context_batch=2
        # This will be triggered when NAG processors are set
        context_batch = encoder_hidden_states.shape[0]

        # NAG marker: Check if this processor was set as NAG processor
        # IMPORTANT: Only apply NAG to cross-attention (not self-attention)
        is_nag_mode = is_cross_attention_original and hasattr(self, 'nag_scale')

        # Debug logging (only log first few times to avoid spam)
        import sys
        if is_nag_mode and not hasattr(self, '_debug_logged'):
            print(f"[NAG DEBUG] is_cross={is_cross_attention}, batch_size={batch_size}, context_batch={context_batch}, has_nag_scale={hasattr(self, 'nag_scale')}", file=sys.stderr)
            self._debug_logged = True

        if is_nag_mode and batch_size == 2 and context_batch == 2:
            # NAG mode: encoder_hidden_states = [negative, positive]
            # Query = [negative_batch, positive_batch]
            # Apply NAG only to positive batch

            # Log NAG execution (only once)
            if not hasattr(self, '_nag_executed_logged'):
                print(f"[NAG EXEC] Applying NAG guidance: scale={self.nag_scale}, tau={self.nag_tau}, alpha={self.nag_alpha}", file=sys.stderr)
                self._nag_executed_logged = True

            # In NAG mode, encoder_hidden_states contains [nag_negative, positive]
            # But for CFG batch 0, we should use regular processing (not NAG negative)
            nag_negative_context = encoder_hidden_states[0:1]
            positive_context = encoder_hidden_states[1:2]

            # Split query
            query_negative = query[0:1]
            query_positive = query[1:2]

            inner_dim = attn.to_k(positive_context).shape[-1]
            head_dim = inner_dim // attn.heads

            # Batch 0: Standard attention for CFG unconditioned batch
            # Use nag_negative_context (which is typically empty or negative prompt)
            key_cfg_uncond = attn.to_k(nag_negative_context)
            value_cfg_uncond = attn.to_v(nag_negative_context)

            query_neg_heads = query_negative.view(1, -1, attn.heads, head_dim).transpose(1, 2)
            key_cfg_uncond_heads = key_cfg_uncond.view(1, -1, attn.heads, head_dim).transpose(1, 2)
            value_cfg_uncond_heads = value_cfg_uncond.view(1, -1, attn.heads, head_dim).transpose(1, 2)

            A_uncond = self._compute_attention(query_neg_heads, key_cfg_uncond_heads, value_cfg_uncond_heads)
            A_uncond = A_uncond.transpose(1, 2).reshape(1, -1, attn.heads * head_dim).to(query.dtype)

            # Batch 1: NAG guidance on positive batch
            # Step 1: Compute positive attention - Ap = Attn(Q_positive, K_positive, V_positive)
            key_positive = attn.to_k(positive_context)
            value_positive = attn.to_v(positive_context)

            query_pos_heads = query_positive.view(1, -1, attn.heads, head_dim).transpose(1, 2)
            key_pos_heads = key_positive.view(1, -1, attn.heads, head_dim).transpose(1, 2)
            value_pos_heads = value_positive.view(1, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states_positive = self._compute_attention(query_pos_heads, key_pos_heads, value_pos_heads)
            hidden_states_positive = hidden_states_positive.transpose(1, 2).reshape(1, -1, attn.heads * head_dim).to(query.dtype)

            # Step 2: Compute negative attention - An = Attn(Q_positive, K_nag_negative, V_nag_negative)
            # Use NAG negative context (not CFG negative)
            key_nag_neg = attn.to_k(nag_negative_context)
            value_nag_neg = attn.to_v(nag_negative_context)
            key_nag_neg_heads = key_nag_neg.view(1, -1, attn.heads, head_dim).transpose(1, 2)
            value_nag_neg_heads = value_nag_neg.view(1, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states_negative_attn = self._compute_attention(query_pos_heads, key_nag_neg_heads, value_nag_neg_heads)
            hidden_states_negative_attn = hidden_states_negative_attn.transpose(1, 2).reshape(1, -1, attn.heads * head_dim).to(query.dtype)

            # Step 3: NAG guidance formula (official implementation)
            # guidance = positive * φ - negative * (φ - 1)
            phi = self.nag_scale
            hidden_states_guidance = hidden_states_positive * phi - hidden_states_negative_attn * (phi - 1)

            # Step 4: L1 Normalization with tau threshold
            eps = 1e-6
            norm_positive = torch.norm(hidden_states_positive, p=1, dim=-1, keepdim=True).clamp_min(eps)
            norm_guidance = torch.norm(hidden_states_guidance, p=1, dim=-1, keepdim=True).clamp_min(eps)
            scale = norm_guidance / norm_positive

            # Clamp scale to tau threshold
            scale_clamped = torch.minimum(scale, torch.full_like(scale, self.nag_tau))
            hidden_states_guidance = hidden_states_guidance * scale_clamped / scale

            # Step 5: Alpha blending (interpolation between guidance and positive)
            alpha = self.nag_alpha
            A_cond = hidden_states_guidance * alpha + hidden_states_positive * (1 - alpha)

            # Combine: [uncond_batch, cond_batch_with_nag]
            hidden_states = torch.cat([A_uncond, A_cond], dim=0)

        else:
            # Standard attention (not NAG mode)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(context_batch, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(context_batch, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states = self._compute_attention(query, key, value, attention_mask)

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

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


def set_nag_processors(unet, nag_scale: float, nag_tau: float, nag_alpha: float, attention_type: str = "normal"):
    """
    Set NAG attention processors on cross-attention layers (attn2)

    Args:
        unet: The UNet model
        nag_scale: NAG scale parameter
        nag_tau: NAG tau parameter
        nag_alpha: NAG alpha parameter
        attention_type: Attention backend - "normal", "sage", or "flash"

    Returns:
        dict: Original processors for restoration
    """
    # Get current processors
    original_processors = unet.attn_processors.copy()

    # Create new processor dict with NAG processors for attn2
    new_processors = {}
    for name, processor in unet.attn_processors.items():
        if "attn2" in name:  # Cross-attention only
            new_processors[name] = NAGAttnProcessor2_0(
                nag_scale=nag_scale,
                nag_tau=nag_tau,
                nag_alpha=nag_alpha,
                attention_type=attention_type,
            )
        else:
            new_processors[name] = processor

    # Set processors using diffusers' method
    unet.set_attn_processor(new_processors)

    # Verify processors were set
    nag_count = sum(1 for proc in unet.attn_processors.values() if isinstance(proc, NAGAttnProcessor2_0))
    attn2_count = len([n for n in unet.attn_processors.keys() if 'attn2' in n])
    print(f"[NAG] Set {attn2_count} NAG processors (scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}, attention={attention_type})")
    print(f"[NAG] Verification: {nag_count} NAGAttnProcessor2_0 instances found in unet.attn_processors")

    return original_processors


def restore_original_processors(unet, original_processors: dict):
    """Restore original attention processors"""
    if not original_processors:
        return

    # Use diffusers' method to restore
    unet.set_attn_processor(original_processors)

    print("[NAG] Restored original attention processors")
