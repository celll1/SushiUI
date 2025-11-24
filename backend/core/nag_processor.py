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
            print(f"[NAG DEBUG] is_cross={is_cross_attention_original}, batch_size={batch_size}, context_batch={context_batch}, has_nag_scale={hasattr(self, 'nag_scale')}", file=sys.stderr)
            self._debug_logged = True

        if is_nag_mode and batch_size == 2 and context_batch == 3:
            # NAG mode (official implementation):
            # encoder_hidden_states = [cfg_negative, cfg_positive, nag_negative] (batch=3)
            # Query from hidden_states = [uncond_query, cond_query] (batch=2, from CFG latent duplication)
            #
            # Official implementation:
            # 1. Tile query: [uncond, cond] → [uncond, uncond, cond, cond] (batch=4)
            # 2. Compute attention with 3 contexts to get results
            # 3. Extract positive and negative for NAG guidance
            # 4. Apply NAG formula, normalization, and blending
            # 5. Return final result

            # Log NAG execution (only once)
            if not hasattr(self, '_nag_executed_logged'):
                print(f"[NAG EXEC] Applying NAG guidance: scale={self.nag_scale}, tau={self.nag_tau}, alpha={self.nag_alpha}", file=sys.stderr)
                print(f"[NAG EXEC] batch_size={batch_size}, context_batch={context_batch}", file=sys.stderr)
                self._nag_executed_logged = True

            # Origin batch size (number of images being generated, usually 1)
            origin_batch_size = 1

            # Compute key and value for ALL 3 contexts at once
            key = attn.to_k(encoder_hidden_states)  # [3, seq, dim] - cfg_neg, cfg_pos, nag_neg
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            # Official implementation: tile query to match batch expansion
            # For batch_size=2*origin_batch_size (CFG), tile each query:
            # [uncond, cond] → [uncond, uncond, cond, cond]
            query_tiled = query.tile(2, 1, 1)  # [4, seq, dim]

            # Reshape for multi-head attention
            query_tiled = query_tiled.view(batch_size * 2, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(context_batch, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(context_batch, -1, attn.heads, head_dim).transpose(1, 2)

            # Compute attention: [4 queries] × [3 contexts] via broadcasting
            # Result indices (batch=4):
            # 0: uncond→cfg_negative, 1: uncond→cfg_positive
            # 2: cond→cfg_positive, 3: cond→nag_negative
            hidden_states_all = self._compute_attention(query_tiled, key, value, attention_mask)
            hidden_states_all = hidden_states_all.transpose(1, 2).reshape(batch_size * 2, -1, attn.heads * head_dim).to(query.dtype)

            # Extract results following official implementation:
            # For NAG: use results from cond queries (indices 2 and 3)
            hidden_states_negative = hidden_states_all[-origin_batch_size:]  # cond→nag_negative (index 3)
            hidden_states_positive = hidden_states_all[batch_size:batch_size + origin_batch_size]  # cond→cfg_positive (index 2)

            # Debug logging - tensor statistics (first cross-attention call only)
            if not hasattr(self, '_tensor_stats_logged'):
                pos_norm = torch.norm(hidden_states_positive, p=2).item()
                neg_norm = torch.norm(hidden_states_negative, p=2).item()
                pos_mean = hidden_states_positive.mean().item()
                neg_mean = hidden_states_negative.mean().item()
                print(f"[NAG STATS] Positive: norm={pos_norm:.4f}, mean={pos_mean:.6f}, shape={hidden_states_positive.shape}", file=sys.stderr)
                print(f"[NAG STATS] Negative: norm={neg_norm:.4f}, mean={neg_mean:.6f}, shape={hidden_states_negative.shape}", file=sys.stderr)
                self._tensor_stats_logged = True

            # NAG guidance formula (official implementation)
            # guidance = positive * φ - negative * (φ - 1)
            phi = self.nag_scale
            hidden_states_guidance = hidden_states_positive * phi - hidden_states_negative * (phi - 1)

            # Debug: guidance after formula
            if not hasattr(self, '_guidance_stats_logged'):
                guid_norm = torch.norm(hidden_states_guidance, p=2).item()
                guid_mean = hidden_states_guidance.mean().item()
                print(f"[NAG STATS] Guidance (φ={phi}): norm={guid_norm:.4f}, mean={guid_mean:.6f}", file=sys.stderr)
                self._guidance_stats_logged = True

            # L1 Normalization with tau threshold (official implementation)
            norm_positive = torch.norm(hidden_states_positive, p=1, dim=-1, keepdim=True).expand(*hidden_states_positive.shape)
            norm_guidance = torch.norm(hidden_states_guidance, p=1, dim=-1, keepdim=True).expand(*hidden_states_guidance.shape)

            scale = norm_guidance / norm_positive

            # Debug: scale statistics
            if not hasattr(self, '_scale_stats_logged'):
                scale_mean = scale.mean().item()
                scale_max = scale.max().item()
                scale_min = scale.min().item()
                print(f"[NAG STATS] Scale before clamp: mean={scale_mean:.4f}, min={scale_min:.4f}, max={scale_max:.4f}, tau={self.nag_tau}", file=sys.stderr)
                self._scale_stats_logged = True

            # Clamp and normalize
            hidden_states_guidance = hidden_states_guidance * torch.minimum(scale, scale.new_ones(1) * self.nag_tau) / scale

            # Debug: guidance after normalization
            if not hasattr(self, '_normalized_stats_logged'):
                guid_norm_after = torch.norm(hidden_states_guidance, p=2).item()
                guid_mean_after = hidden_states_guidance.mean().item()
                print(f"[NAG STATS] Guidance after norm: norm={guid_norm_after:.4f}, mean={guid_mean_after:.6f}", file=sys.stderr)
                self._normalized_stats_logged = True

            # Alpha blending (official implementation)
            alpha = self.nag_alpha
            A_cond = hidden_states_guidance * alpha + hidden_states_positive * (1 - alpha)

            # Debug: final output
            if not hasattr(self, '_final_stats_logged'):
                final_norm = torch.norm(A_cond, p=2).item()
                final_mean = A_cond.mean().item()
                print(f"[NAG STATS] Final A_cond (α={alpha}): norm={final_norm:.4f}, mean={final_mean:.6f}", file=sys.stderr)
                self._final_stats_logged = True

            # Reconstruct final output following official implementation
            # For batch_size == 2 * origin_batch_size (CFG with NAG):
            # hidden_states = guidance (batch=2)
            # We need to extract uncond result and combine with NAG guidance result

            # Extract uncond result (index 0 or 1)
            A_uncond = hidden_states_all[0:origin_batch_size]  # uncond→cfg_negative

            # Debug: compare uncond vs cond
            if not hasattr(self, '_uncond_cond_logged'):
                uncond_norm = torch.norm(A_uncond, p=2).item()
                uncond_mean = A_uncond.mean().item()
                diff_norm = torch.norm(A_cond - A_uncond, p=2).item()
                print(f"[NAG STATS] A_uncond: norm={uncond_norm:.4f}, mean={uncond_mean:.6f}", file=sys.stderr)
                print(f"[NAG STATS] A_cond: norm={torch.norm(A_cond, p=2).item():.4f}, mean={A_cond.mean().item():.6f}", file=sys.stderr)
                print(f"[NAG STATS] Difference (A_cond - A_uncond): norm={diff_norm:.4f}", file=sys.stderr)
                self._uncond_cond_logged = True

            # Official implementation: For batch_size == 2 * origin_batch_size, return guidance directly (batch=1)
            # But we need batch=2 for CFG [uncond, cond]
            # Actually, looking at official code more carefully:
            # "if batch_size == 2 * origin_batch_size: hidden_states = hidden_states_guidance"
            # This returns ONLY the guidance (batch=1), not [uncond, cond]
            # Then CFG is applied in the pipeline on the noise prediction, not here

            # So the final output should be: [uncond_result, nag_guidance_result]
            hidden_states = torch.cat([A_uncond, A_cond], dim=0)  # [2, seq, dim]

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
