"""Training-free reference-style transfer (StyleAligned/VSP-style KV injection) for
FLUX.2, wired through the arch-agnostic ``core.inference.reference_style`` module
(see that module's docstring for the shared math: ``inject_kv``,
``cross_batch_adain_qk``, ``make_ref_value``, ``frequency_scale_vector``).

Design (chosen over reusing FLUX.2's existing Image-Edit reference-image
infrastructure -- see the module-level note below): these are Capture/Inject
variants of the unified-conduit FLUX.2 attention processors
(``core.inference.conduit_flux2.ConduitFlux2AttnProcessor`` /
``ConduitFlux2ParallelSelfAttnProcessor``), following the exact same pattern as
Krea2's ``Krea2Attention`` style hook (``core/models/krea2/vendor/transformer.py``):
after qk-RMSNorm + RoPE, before the attention kernel call, a per-block
``StyleContext`` (capture/inject) either stashes the post-RoPE image-token
Query/Key/Value, or reads a previously-stashed reference Q/K/V and calls
``inject_kv``.

Why NOT FLUX.2's existing Image-Edit reference-KV infra: FLUX.2's Image-Edit
feature (``pipeline_backends/flux2.py::encode_flux2_image_refs`` /
``ref_tokens``/``ref_ids``) works by literally concatenating the *clean*
(never re-noised) reference-image tokens onto the *real* sequence and letting
them participate in ordinary joint attention every step, at their OWN rope
"time" axis offset (10, 20, ...) -- then slicing the extra tokens back off the
output. There is no RoPE-frequency suppression, no AdaIN, no strength control,
and no per-step re-noising to the current sigma: it is closer to instruction
conditioning than to a StyleAligned-style attention-sharing transfer. diffusers
does ALSO ship a genuine attention-side KV-cache pair for this feature
(``Flux2KVAttnProcessor`` / ``Flux2KVParallelSelfAttnProcessor``, imported only
by name in ``pipeline_backends/flux2.py``'s docstring as a "do not clobber"
guard) but SushiUI's Image-Edit wiring does not actually install them anywhere
(grep-verified) -- it uses the simpler token-concatenation approach instead.
Neither path reuses ``reference_style.py``'s math, so style transfer needed its
own Capture/Inject processor pair; the Krea2 precedent (attention-forward hook
with an externally-stamped context) was the closest architecture to mirror,
NOT the Image-Edit ref-token-concat approach.

Interop: style transfer and FLUX.2 Image-Edit (``ref_images``) are mutually
EXCLUSIVE for a given generation (see ``pipeline_backends/flux2.py``'s
``_flux2_style_config`` caller) -- combining them would require deciding
whether Image-Edit's ref tokens count as "image" tokens for the capture/inject
img_start/img_end bookkeeping, which is not a well-defined StyleAligned
operation. Style transfer is likewise mutually exclusive with NAG, NegPip and
FBCache for the step(s) it is active on (all of these also rewrite the
attention-time token layout or cache attention outputs); it composes cleanly
with block swap (``Flux2BlockSwapWrapper``) since block swap only changes WHERE
compute happens, not what attention sees, and with quantization/autocast.

Tensors are BSHD throughout, matching ``conduit_flux2.py``; FLUX.2 has no GQA
(H_kv == H) so ``adain``'s GQA-broadcast path is a no-op here.
"""

from typing import Any, List, Optional, Tuple

import torch

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import _get_qkv_projections

from core.attention import AttentionMode
from core.inference.conduit_flux2 import dispatch_attention_conduit


class StyleConduitFlux2AttnProcessor:
    """Dual-stream (joint text+image) FLUX.2 attention, capture/inject variant.

    Byte-identical to ``ConduitFlux2AttnProcessor`` when ``_style_ctx`` is
    ``None`` (the class default) -- only installed on the transformer for the
    duration of a style-active generation (see ``install_flux2_style_processors``)
    and restored afterwards, so a generation without a style reference never
    touches this class at all.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, backend: str = "native", mode: AttentionMode = AttentionMode.INFERENCE):
        self._conduit_backend = backend
        self._conduit_mode = mode
        self._style_ctx = None
        self.block_idx: Optional[int] = None

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

        # --- Reference-style KV injection (training-free) ---
        # Must run strictly AFTER qk-RMSNorm and AFTER RoPE (same reasoning as
        # Krea2Attention: RMSNorm would erase pre-norm scaling, and RoPE must
        # already carry the token position before K is stashed/injected).
        query, key, value = _apply_style_hook(self, attn, query, key, value)

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


class StyleConduitFlux2ParallelSelfAttnProcessor:
    """Single-stream (fused QKV+MLP) FLUX.2 attention, capture/inject variant.

    The hidden_states this processor receives are ALREADY the concatenated
    ``[txt, img]`` sequence (``Flux2SingleTransformerBlock`` concatenates before
    calling ``attn``), so the image-token slice uses the SAME absolute
    ``img_start``/``img_end`` offsets as the dual-stream processor above.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, backend: str = "native", mode: AttentionMode = AttentionMode.INFERENCE):
        self._conduit_backend = backend
        self._conduit_mode = mode
        self._style_ctx = None
        self.block_idx: Optional[int] = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states_proj = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states_proj, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        query, key, value = _apply_style_hook(self, attn, query, key, value)

        attn_output = dispatch_attention_conduit(
            query, key, value, attention_mask, self._conduit_backend, self._conduit_mode
        )
        attn_output = attn_output.flatten(2, 3)
        attn_output = attn_output.to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


def _apply_style_hook(
    proc, attn, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared capture/inject body for both FLUX.2 processor variants above.

    ``query``/``key``/``value`` are already post-qk-norm, post-RoPE, and (for
    the dual-stream processor) already the full ``[txt, img]`` concatenated
    tensors. No-op (returns the inputs unchanged) whenever ``_style_ctx`` is
    ``None``, the processor was never assigned a ``block_idx`` (shouldn't
    happen once installed, but guards against a partially-wired module), or
    the context's ``block_range`` excludes this block.
    """
    ctx = proc._style_ctx
    if ctx is None or proc.block_idx is None or not ctx.active_for_block(proc.block_idx):
        return query, key, value

    img_start, img_end = ctx.img_start, ctx.img_end
    if ctx.mode == "capture":
        ctx.store[proc.block_idx] = (
            query[:, img_start:img_end].detach().clone(),
            key[:, img_start:img_end].detach().clone(),
            value[:, img_start:img_end].detach().clone(),
        )
        return query, key, value

    # mode == "inject"
    ref_qkv = ctx.store.get(proc.block_idx)
    if ref_qkv is None:
        return query, key, value

    from core.inference.reference_style import inject_kv, make_ref_value

    ref_q, ref_k, ref_v = ref_qkv
    cfg = ctx.config
    if cfg.ref_k_strength == 0.0 and cfg.adain_strength <= 0.0:
        return query, key, value

    freq_vec = cfg.get_freq_scale_vector(attn.head_dim, ctx.progress, key.device, key.dtype)
    target_v_img = value[:, img_start:img_end]
    ref_v_final = make_ref_value(
        target_v_img, ref_v, cfg.value_mode, cfg.value_adain_strength, cfg.ref_value_mix
    )
    key, value, query = inject_kv(
        key, value, ref_k, ref_v_final, img_start, img_end,
        cfg.ref_k_strength, freq_vec, cfg.adain_strength, q=query, ref_q=ref_q,
    )
    return query, key, value


def install_flux2_style_processors(
    transformer, canonical_backend: str, mode: AttentionMode,
) -> Tuple[List[Any], List[Tuple[Any, Any]]]:
    """Replace the DEFAULT dual/single-stream attention processors (diffusers'
    ``Flux2AttnProcessor``/``Flux2ParallelSelfAttnProcessor`` OR SushiUI's
    conduit-routed ``ConduitFlux2AttnProcessor``/``ConduitFlux2ParallelSelfAttnProcessor``
    -- whichever ``set_flux2_attention_backend`` left installed) with the
    style-aware Capture/Inject variants above. Deliberately does NOT touch any
    module whose current processor is neither of those (i.e. any reserved
    KV-cache processor some future code path might install) -- mirrors the
    same defensive gating as ``_install_flux2_conduit_processors``.

    Returns ``(processors, saved)``:
      - ``processors``: the installed style processor instances, in unified
        block order (dual blocks ``0..num_layers-1`` then single blocks
        ``num_layers..num_layers+num_single_layers-1``) -- this is the SAME
        unified numbering the block-swap wrapper uses for its swap index, and
        is what ``StyleTransferConfig.block_range`` gates against.
      - ``saved``: ``(module, original_processor)`` pairs for
        ``restore_flux2_style_processors`` to undo after the generation.
    """
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2Attention,
        Flux2AttnProcessor,
        Flux2ParallelSelfAttention,
        Flux2ParallelSelfAttnProcessor,
    )
    from core.inference.conduit_flux2 import ConduitFlux2AttnProcessor, ConduitFlux2ParallelSelfAttnProcessor

    processors: List[Any] = []
    saved: List[Tuple[Any, Any]] = []

    dual_blocks = list(transformer.transformer_blocks)
    single_blocks = list(transformer.single_transformer_blocks)

    for idx, block in enumerate(dual_blocks):
        module = block.attn
        proc = getattr(module, "processor", None)
        if isinstance(module, Flux2Attention) and isinstance(proc, (Flux2AttnProcessor, ConduitFlux2AttnProcessor)):
            style_proc = StyleConduitFlux2AttnProcessor(canonical_backend, mode)
            style_proc.block_idx = idx
            saved.append((module, proc))
            module.set_processor(style_proc)
            processors.append(style_proc)

    base = len(dual_blocks)
    for idx, block in enumerate(single_blocks):
        module = block.attn
        proc = getattr(module, "processor", None)
        if isinstance(module, Flux2ParallelSelfAttention) and isinstance(
            proc, (Flux2ParallelSelfAttnProcessor, ConduitFlux2ParallelSelfAttnProcessor)
        ):
            style_proc = StyleConduitFlux2ParallelSelfAttnProcessor(canonical_backend, mode)
            style_proc.block_idx = base + idx
            saved.append((module, proc))
            module.set_processor(style_proc)
            processors.append(style_proc)

    return processors, saved


def restore_flux2_style_processors(saved: List[Tuple[Any, Any]]) -> None:
    """Undo ``install_flux2_style_processors`` -- puts back whatever processor
    (diffusers default or Conduit*) was installed before style transfer ran,
    so a subsequent non-style generation is unaffected."""
    for module, proc in saved:
        module.set_processor(proc)


def set_flux2_style_context(processors: List[Any], ctx) -> None:
    """Stamp the SAME ``StyleContext`` (or ``None`` to disarm) onto every
    installed style processor. Called once per capture/inject/disarm phase per
    active step by the pipeline's denoise loop (mirrors what Krea2's
    ``_stamp_style_context`` does per-forward, except here there is no single
    vendored model forward to hook -- FLUX.2 attention lives behind the
    diffusers processor pattern, so the context is pushed explicitly instead)."""
    for p in processors:
        p._style_ctx = ctx
