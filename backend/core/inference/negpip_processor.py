"""NegPip Attention Processor.

NegPip lets you push *away* from a concept written inside an ordinary prompt by
giving it a NEGATIVE emphasis weight, e.g. ``(worst quality:-1)``. A token with a
negative weight has its attention VALUE (V) negated, so attending to it SUBTRACTS
its concept from the output instead of adding it. Positive weights scale V up as
usual. Because this is a single elementwise scale of V inside the existing
cross-attention, it adds NO extra forward pass -- iter speed is unchanged (unlike
NAG, which computes a second attention).

It applies per-token in BOTH prompt contexts: a negative weight in the POSITIVE
prompt removes the concept; a negative weight in the NEGATIVE prompt is a
double-negative that re-affirms it (negate the value in the unconditional branch).

The signed per-token weights are supplied as a [batch, seq] tensor aligned with
the encoder_hidden_states the U-Net receives (batch order [negative, positive] for
CFG). V is scaled as V[b, :, t, :] *= weights[b, t]; Q and K are left untouched so
the attention pattern is the normal one for the (unweighted) token embedding.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention

from core.attention import AttentionMode, dispatch_attention


class NegPipAttnProcessor2_0:
    """Cross-attention processor that scales V by signed per-token weights.

    Args:
        token_weights: [batch, kv_seq] signed weights, or None (acts as identity).
            Self-attention calls (encoder_hidden_states is None) are never weighted.
        attention_type: "normal" | "sage" | "flash" (SDPA backend selection).
    """

    def __init__(self, token_weights: Optional[torch.Tensor] = None, attention_type: str = "normal"):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("NegPipAttnProcessor2_0 requires PyTorch 2.0+")
        self.token_weights = token_weights
        # Backend selector ("normal"/"sage"/"flash"); normalized and
        # capability-gated inside the unified conduit.
        self.attention_type = attention_type

    def _attend(self, query, key, value, attention_mask=None):
        # q/k/v are [B, H, S, D] (BHSD); flash is now honored (was flash->SDPA).
        return dispatch_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self.attention_type,
            mode=AttentionMode.INFERENCE,
            layout="BHSD",
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

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        context_batch = encoder_hidden_states.shape[0]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(context_batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(context_batch, -1, attn.heads, head_dim).transpose(1, 2)

        # NegPip: scale V by signed per-token weights (cross-attention only).
        if is_cross and self.token_weights is not None:
            w = self.token_weights.to(device=value.device, dtype=value.dtype)  # [Bw, seq_w]
            kv_seq = value.shape[2]
            # Align batch: weights may be [context_batch, seq] (per-context) or [seq].
            if w.dim() == 1:
                w = w.unsqueeze(0).expand(context_batch, -1)
            if w.shape[0] != context_batch:
                # Broadcast a single row, or take the first context_batch rows.
                if w.shape[0] == 1:
                    w = w.expand(context_batch, -1)
                else:
                    w = w[:context_batch]
            # Align sequence length (chunking/padding may differ): pad with 1.0
            # (identity) or truncate to the actual kv sequence.
            if w.shape[1] != kv_seq:
                if w.shape[1] < kv_seq:
                    pad = torch.ones(w.shape[0], kv_seq - w.shape[1], device=w.device, dtype=w.dtype)
                    w = torch.cat([w, pad], dim=1)
                else:
                    w = w[:, :kv_seq]
            value = value * w[:, None, :, None]   # [B, heads, seq, head_dim] *= [B,1,seq,1]

        hidden_states = self._attend(query, key, value, attention_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def set_negpip_processors(unet, token_weights: torch.Tensor, attention_type: str = "normal"):
    """Install NegPip processors on cross-attention (attn2) layers.

    token_weights: [batch, kv_seq] signed per-token weights aligned with the
    encoder_hidden_states the U-Net receives (batch order [negative, positive]).
    Returns the original processors for restoration.
    """
    original = unet.attn_processors.copy()
    new = {}
    for name, proc in unet.attn_processors.items():
        if "attn2" in name:   # cross-attention only
            new[name] = NegPipAttnProcessor2_0(token_weights=token_weights, attention_type=attention_type)
        else:
            new[name] = proc
    unet.set_attn_processor(new)
    n = sum(1 for p in unet.attn_processors.values() if isinstance(p, NegPipAttnProcessor2_0))
    print(f"[NegPip] Installed {n} processors (signed V weighting, attention={attention_type})")
    return original


def restore_original_processors(unet, original_processors: dict):
    if not original_processors:
        return
    unet.set_attn_processor(original_processors)
    print("[NegPip] Restored original attention processors")
