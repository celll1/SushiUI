"""
Conduit-routed FLUX.2 attention processors.

These are line-for-line clones of diffusers' default FLUX.2 attention processors
(``Flux2AttnProcessor`` dual-stream and ``Flux2ParallelSelfAttnProcessor``
single-stream, diffusers 0.38.0 transformer_flux2.py:325 / :568). The ONLY change
versus the diffusers bodies is the kernel call: instead of diffusers'
``dispatch_attention_fn`` they call the unified conduit
``core.attention.dispatch_attention``. Everything else — the QKV projections,
per-head RMSNorm (norm_q/norm_k, norm_added_q/norm_added_k), the joint text+image
concat, 4-axis RoPE (apply_rotary_emb, sequence_dim=1), split-back, and to_out —
is preserved exactly, so the native path is numerically identical to diffusers.

This lets conduit-only backends (notably ``tq``) run on FLUX.2 while keeping the
diffusers path selectable (``attention_impl='diffusers'``) for byte-identical
reversion. Installed by ``pipeline_backends/flux2.py::set_flux2_attention_backend``
and the training hook only on the NON-KV default attention modules; the
reference-image KV-cache processors stay on the diffusers registry.

Tensors are BSHD == ``[batch, seq_len, num_heads, head_dim]`` throughout (the
layout the diffusers processors and ``dispatch_attention_fn`` already use); FLUX.2
head_dim is 128, no GQA (H_kv == H). attn_mask is None in the standard txt2img/CFG
path (the ref-image causal path is handled by the KV processors, not here).
"""

import torch

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import _get_qkv_projections

from core.attention import AttentionMode, dispatch_attention


class ConduitFlux2AttnProcessor:
    """Dual-stream (joint text+image) FLUX.2 attention via the unified conduit.

    Clone of diffusers ``Flux2AttnProcessor.__call__`` with the kernel call routed
    through ``dispatch_attention``.

    Args:
        backend: Canonical conduit backend string ("native"|"flash"|"sage"|"tq").
        mode: INFERENCE or TRAINING (sage is refused in TRAINING by the conduit).
    """

    # Kept for parity with the diffusers processor (some diffusers code paths read
    # these attrs); unused by the conduit call.
    _attention_backend = None
    _parallel_config = None

    def __init__(self, backend: str = "native", mode: AttentionMode = AttentionMode.INFERENCE):
        self._conduit_backend = backend
        self._conduit_mode = mode

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # ONLY change vs diffusers Flux2AttnProcessor: kernel -> unified conduit.
        hidden_states = dispatch_attention_conduit(
            query, key, value, attention_mask, self._conduit_backend, self._conduit_mode
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class ConduitFlux2ParallelSelfAttnProcessor:
    """Single-stream (fused QKV+MLP) FLUX.2 attention via the unified conduit.

    Clone of diffusers ``Flux2ParallelSelfAttnProcessor.__call__`` with the kernel
    call routed through ``dispatch_attention``.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, backend: str = "native", mode: AttentionMode = AttentionMode.INFERENCE):
        self._conduit_backend = backend
        self._conduit_mode = mode

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        # Parallel in (QKV + MLP in) projection
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        # Handle the attention logic
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # ONLY change vs diffusers Flux2ParallelSelfAttnProcessor: kernel -> conduit.
        hidden_states = dispatch_attention_conduit(
            query, key, value, attention_mask, self._conduit_backend, self._conduit_mode
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Handle the feedforward (FF) logic
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        # Concatenate and parallel output projection
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


def dispatch_attention_conduit(query, key, value, attention_mask, backend, mode):
    """Shared conduit kernel call for the FLUX.2 conduit processors (BSHD in/out).

    Mirrors what diffusers' ``dispatch_attention_fn`` returns for the default
    backends (a BSHD tensor) so the surrounding processor body is unchanged.
    FLUX.2 has no GQA (H_kv == H) so ``enable_gqa`` stays False; masks are None on
    the standard path and the conduit downgrades flash/sage/tq -> native if one is
    ever present.
    """
    return dispatch_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        backend=backend,
        mode=mode,
        layout="BSHD",
        enable_gqa=False,
    )
