"""Training-free reference-style transfer (StyleAligned/VSP-style KV injection) for
Ideogram 4, wired through the arch-agnostic ``core.inference.reference_style`` module
(see that module's docstring for the shared math: ``inject_kv``, ``cross_batch_adain_qk``,
``make_ref_value``, ``frequency_scale_vector``).

Design (mirrors ``core.inference.style_flux2``'s processor-swap pattern -- the closest
architectural precedent, since Ideogram 4's self-attention already lives behind a
diffusers-style ``Ideogram4Attention.set_processor`` object, unlike MiniT2I's inlined
block forward): a ``StyleIdeogram4AttnProcessor`` is a byte-identical duplicate of
``Ideogram4AttnProcessor.__call__`` (same q/k/v projection, qk-RMSNorm, MRoPE order) with
one extra call to ``_apply_style_hook`` inserted right after MRoPE and right before
``ideogram4_dispatch_attention`` -- the same insertion point Krea2/FLUX.2's style
processors use (after qk-norm + RoPE, before the attention kernel). It is installed ONLY
on ``Ideogram4Attention`` modules of the CONDITIONAL transformer, and ONLY for the
duration of a style-active generation (``install_ideogram4_style_processors`` /
``restore_ideogram4_style_processors``); a generation without a style reference never
touches this module at all.

Dual-transformer scope: Ideogram 4 runs CFG as two SEPARATE transformer objects every
step (the conditional ``transformer`` and the ``unconditional_transformer``, blended by
``ideogram4_pipeline_ops._blend_guidance``). Style transfer only ever installs processors
on the CONDITIONAL transformer -- capture (the style reference re-noised to the current
sigma, run through the SAME conditional transformer with the target's own positive-prompt
conditioning) and inject (the target's own conditional forward) both happen there; the
unconditional transformer is never touched, matching the Lens/Krea2/FLUX.2 precedent of
leaving the negative/unconditional branch's style context disarmed.

Packed-sequence image-token span: Ideogram 4's packed layout is
``[left-pad][text][image]`` with ``total_seq_len == max_text_tokens + num_image_tokens``
(``ideogram4_pipeline_ops._prepare_ids`` / ``build_training_conditioning``). Crucially,
regardless of how much LEFT-padding a given sample's text needs (``offset = max_text_tokens
- num_text``), the image region always starts at ``position_ids[b, offset + num_text:]``
where ``offset + num_text == max_text_tokens`` for EVERY sample in the batch (the text +
its left-pad always exactly fill ``max_text_tokens``) -- so the image-token span for the
CONDITIONAL forward is always the fixed contiguous suffix
``[max_text_tokens : max_text_tokens + num_image_tokens]`` of the packed sequence,
independent of per-sample text length. This is exactly what
``ideogram4_pipeline_ops._dual_branch_velocity`` already relies on when it slices
``pos_out[:, max_text:]`` for the velocity output. The caller (``_ideogram4_style_step``)
computes this span once per step and stamps it onto the ``StyleContext`` as
``img_start``/``img_end`` (mirroring Krea2's single-main-stack convention) -- no
gather/scatter over a boolean indicator mask is needed since the span is a provably fixed
slice, not data-dependent.

Attention-mask extension under the block-diagonal segment mask (THE crux): Ideogram 4's
self-attention mask is not additive but a full ``(B, 1, L, L)`` BOOLEAN block-diagonal
mask built once per forward from ``segment_ids`` (``segment_ids.unsqueeze(2) ==
segment_ids.unsqueeze(1)``) and shared by every block. When ``inject_kv`` appends
``extra`` reference-K/V columns onto the END of the target's own K/V (which, since the
image region is the sequence'S OWN suffix here, lands the appended columns immediately
after the real image tokens and before nothing else -- there is no trailing text/pad
after the image region in this packed layout), the mask's KEY axis (last dim) must grow
by the same ``extra`` and get columns whose VISIBILITY differs by QUERY row: only image
QUERY rows (``[img_start:img_end)`` on the query axis, i.e. the SAME range used for
K/V capture) may attend the appended ref-K columns; text/pad query rows get ``False``
(invisible) for those columns, leaving their existing block-diagonal visibility over the
ORIGINAL keys completely unchanged. Implementation: build a
``(1, 1, seq_len_q, extra)`` boolean column block that is ``True`` only for rows in
``[img_start:img_end)`` and ``False`` elsewhere, expand it to the mask's batch size, and
``torch.cat`` it onto the mask's last dim. This is the correct semantics for the
block-diagonal case (as opposed to SDXL/Krea2/Lens's simpler "all rows may attend the
extra columns" additive-mask pad, which is valid there only because those masks are
purely padding-exclusion, not a multi-segment block-diagonal partition).

Multi-reference (N-ref) transfer: when ``ctx.refs`` is set (2+ simultaneous style
references, see ``core.inference.reference_style.StyleContext``/``inject_kv_multi``),
``_apply_style_hook`` takes a separate branch that calls ``collect_block_refs`` +
``inject_kv_multi`` instead of the single-ref ``inject_kv``. The SAME mask-extension
recipe above applies unchanged -- it derives ``extra`` from the actual post-injection
key length rather than assuming a fixed per-ref column count, so it is correct whether
``inject_kv_multi`` appended ONE column block per active reference ("stack" mode) or a
single averaged consensus block ("common_concept" mode). The single-ref branch
(``ctx.refs is None``) is completely untouched by this addition.

Because the mask extension above is only implemented for the DENSE ``(B,1,L,L)``
boolean-mask path (the ``ideogram4_dispatch_attention`` "native"/default branch, which
passes ``attn_mask`` straight through to ``dispatch_attention_fn``), style-active
generations force the attention backend to ``native`` for the whole generation (see
``pipeline_backends/ideogram4.py``'s style-active branch): the ``flash`` backend
converts the block-diagonal mask to ``cu_seqlens`` and bypasses ``attention_mask``
entirely, which would silently miss the appended ref-K columns; ``sage`` is already
native-only above head_dim=128 (Ideogram 4 is 256), so only ``flash`` needed the guard.

RoPE note: Ideogram 4's MRoPE (``Ideogram4MRoPE``) is INTERLEAVED -- H/W frequencies are
spliced into every-3rd channel (``idx = arange(offset, length, 3)``), NOT the
``repeat_interleave_real=True`` concatenated-per-axis-block layout
``frequency_scale_vector`` assumes for Krea2/Z-Image/FLUX.2. Re-deriving the equivalent
per-channel frequency curve for this 3-way interleave is out of scope for v1 (mirrors
MiniT2I's ``rotate_half`` interleave decision in ``style_minit2i.py``) -- this wiring
passes an all-ones vector straight to ``inject_kv`` instead of calling
``StyleTransferConfig.get_freq_scale_vector`` (which requires ``axes_dims``; it is never
set for Ideogram 4). Frequency suppression is a quality knob only -- ``ref_k_strength``
and AdaIN (the load-bearing mechanisms) apply in full regardless.

Interop: mutually exclusive with NAG and NegPip for the WHOLE generation (both rewrite
the attention-time token/value layout the same way FBCache/style would conflict) --
gated off by the ``style_active`` branch in ``pipeline_backends/ideogram4.py`` before
either is ever set up. FBCache is disabled for the whole generation whenever style
transfer is active (mirrors Lens/Z-Image/FLUX.2: a cache hit skips ``layers[1:]``, which
would desync the per-block style capture/inject store across steps) -- enforced by
``ideogram4_pipeline_ops._build_ideogram4_fbcache``'s ``style_active`` parameter. Block
Swap composes unchanged (it only changes WHERE compute happens, not what attention sees).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch


class StyleIdeogram4AttnProcessor:
    """Byte-identical duplicate of ``Ideogram4AttnProcessor.__call__`` with a
    capture/inject hook after MRoPE, before ``ideogram4_dispatch_attention``.
    ``_style_ctx`` defaults to ``None`` at the class level (never touched by a
    non-style generation) -- only installed on the CONDITIONAL transformer's
    ``Ideogram4Attention`` modules for the duration of a style-active generation
    (see ``install_ideogram4_style_processors``) and restored afterwards.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        self._style_ctx = None
        self.block_idx: Optional[int] = None

    def __call__(
        self,
        attn: "Any",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb,
        segment_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        from .vendor.transformer import _rotate_half, ideogram4_dispatch_attention

        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        cos, sin = image_rotary_emb
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        # Reference-style KV injection (training-free) -- see module docstring for the
        # img-token-span derivation and the block-diagonal mask extension. Must run
        # strictly AFTER qk-RMSNorm and AFTER MRoPE (RMSNorm would erase pre-norm
        # scaling, and RoPE must already carry the token position before K is
        # stashed/injected).
        query, key, value, attention_mask = _apply_style_hook(
            self, query, key, value, attention_mask
        )

        hidden_states = ideogram4_dispatch_attention(
            query,
            key,
            value,
            attention_mask,
            self._attention_backend,
            self._parallel_config,
            segment_ids,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.to_out[0](hidden_states)


def _apply_style_hook(
    proc: "StyleIdeogram4AttnProcessor",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Shared capture/inject body. ``query``/``key``/``value`` are ``[B, L, H, D]``
    (post-qk-norm, post-MRoPE, the FULL packed ``[left-pad][text][image]`` sequence
    for the conditional forward). No-op (returns inputs unchanged) whenever
    ``_style_ctx`` is ``None``, the processor was never assigned a ``block_idx``, or
    the context's ``block_range`` excludes this block."""
    ctx = proc._style_ctx
    if ctx is None or proc.block_idx is None or not ctx.active_for_block(proc.block_idx):
        return query, key, value, attention_mask

    img_start, img_end = ctx.img_start, ctx.img_end
    if ctx.mode == "capture":
        ctx.store[proc.block_idx] = (
            query[:, img_start:img_end].detach().clone(),
            key[:, img_start:img_end].detach().clone(),
            value[:, img_start:img_end].detach().clone(),
        )
        return query, key, value, attention_mask

    if ctx.mode == "inject" and ctx.refs is not None:
        # Multi-reference ("stack" / "common_concept"): centralizes the
        # per-ref active/freq/make_ref_value logic in
        # ``StyleContext.collect_block_refs`` so this hook stays thin. The
        # single-ref branch below (reached only when ``ctx.refs is None``) is
        # completely untouched -- this branch is only ever reached for 2+
        # refs (see ``ideogram4_pipeline_ops._ideogram4_style_step_multi`` and
        # the ``style_refs``/``StyleContext(refs=...)`` wiring in
        # ``pipeline_backends.ideogram4``).
        from core.inference.reference_style import inject_kv_multi

        target_v_img = value[:, img_start:img_end]
        block_refs = ctx.collect_block_refs(proc.block_idx, target_v_img, key.device, key.dtype)
        if not block_refs:
            return query, key, value, attention_mask

        seq_len_before = key.shape[1]
        key, value, query = inject_kv_multi(
            key, value, query, img_start, img_end, block_refs, ctx.combine_mode
        )

        if attention_mask is not None and key.shape[1] != seq_len_before:
            # Same block-diagonal mask extension as the single-ref path below,
            # generalized to N appended reference-K/V blocks: in "stack" mode
            # ``inject_kv_multi`` appends ONE column block PER active
            # reference (each independently scaled), and in "common_concept"
            # mode it appends a SINGLE averaged consensus block -- either way
            # every appended block lands contiguously at the END of the key
            # axis (the image region is already the sequence's own suffix, so
            # there is nothing trailing it to displace) and needs the exact
            # same visibility rule as the single-ref case: visible ONLY to
            # image QUERY rows ([img_start:img_end) on the query axis, i.e.
            # the same range used for K/V capture); text/pad query rows get
            # False, leaving their existing block-diagonal visibility over the
            # ORIGINAL keys unchanged. Computing ``extra`` from the actual
            # post-injection key length (rather than assuming a fixed
            # per-ref column count) means this is correct regardless of how
            # many refs contributed or which combine_mode was used. See the
            # module docstring's "Attention-mask extension" section for why
            # this differs from the simpler "all rows visible" additive-mask
            # padding used by SDXL/Krea2/Lens.
            extra = key.shape[1] - seq_len_before
            batch = attention_mask.shape[0]
            seq_len_q = attention_mask.shape[2]
            img_rows = torch.zeros(seq_len_q, dtype=torch.bool, device=attention_mask.device)
            img_rows[img_start:img_end] = True
            col_block = img_rows.view(1, 1, seq_len_q, 1).expand(batch, 1, seq_len_q, extra)
            attention_mask = torch.cat([attention_mask, col_block], dim=-1)

        return query, key, value, attention_mask

    # mode == "inject" (single-ref, ctx.refs is None) -- untouched
    ref_qkv = ctx.store.get(proc.block_idx)
    if ref_qkv is None:
        return query, key, value, attention_mask

    from core.inference.reference_style import inject_kv, make_ref_value

    ref_q, ref_k, ref_v = ref_qkv
    cfg = ctx.config
    if cfg.ref_k_strength == 0.0 and cfg.adain_strength <= 0.0:
        return query, key, value, attention_mask

    # Interleaved MRoPE: frequency_scale_vector's concatenated-per-axis-block layout
    # does not match Ideogram4MRoPE's every-3rd-channel interleave (see module
    # docstring) -- pass an all-ones vector (documented quality no-op, mirrors
    # MiniT2I's style_minit2i.py). ref_k_strength + AdaIN (load-bearing) still apply.
    freq_vec = torch.ones(query.shape[-1], device=key.device, dtype=key.dtype)

    target_v_img = value[:, img_start:img_end]
    ref_v_final = make_ref_value(
        target_v_img, ref_v, cfg.value_mode, cfg.value_adain_strength, cfg.ref_value_mix
    )

    seq_len_before = key.shape[1]
    key, value, query = inject_kv(
        key, value, ref_k, ref_v_final, img_start, img_end,
        cfg.ref_k_strength, freq_vec, cfg.adain_strength, q=query, ref_q=ref_q,
    )

    if attention_mask is not None and key.shape[1] != seq_len_before:
        # Appended `extra` ref-K/V columns land at the END of the key axis (the image
        # region is already the sequence's own suffix in this packed layout, so "end
        # of image region" == "end of full sequence"). Extend the block-diagonal mask's
        # key axis with a column block visible ONLY to image-query rows
        # ([img_start:img_end) on the QUERY axis) -- text/pad query rows get False,
        # leaving their existing visibility over the ORIGINAL keys unchanged. See the
        # module docstring for why this differs from the simpler "all rows visible"
        # padding used by SDXL/Krea2/Lens (those masks are pure padding-exclusion, not
        # a multi-segment block-diagonal partition).
        extra = key.shape[1] - seq_len_before
        batch = attention_mask.shape[0]
        seq_len_q = attention_mask.shape[2]
        img_rows = torch.zeros(seq_len_q, dtype=torch.bool, device=attention_mask.device)
        img_rows[img_start:img_end] = True
        col_block = img_rows.view(1, 1, seq_len_q, 1).expand(batch, 1, seq_len_q, extra)
        attention_mask = torch.cat([attention_mask, col_block], dim=-1)

    return query, key, value, attention_mask


def install_ideogram4_style_processors(transformer) -> Tuple[List[StyleIdeogram4AttnProcessor], List[Tuple[Any, Any]]]:
    """Replace the DEFAULT ``Ideogram4AttnProcessor`` on every ``layers[i].attention``
    module of the (CONDITIONAL) ``transformer`` with a style-aware capture/inject
    variant. Only ever called on the conditional transformer -- the unconditional
    transformer is never touched (see module docstring).

    Returns ``(processors, saved)``:
      - ``processors``: the installed style processor instances, indexed by block
        (this is the SAME numbering ``StyleTransferConfig.block_range`` gates against).
      - ``saved``: ``(module, original_processor)`` pairs for
        ``restore_ideogram4_style_processors`` to undo after the generation.
    """
    from .vendor.transformer import Ideogram4AttnProcessor

    processors: List[StyleIdeogram4AttnProcessor] = []
    saved: List[Tuple[Any, Any]] = []

    for idx, block in enumerate(transformer.layers):
        module = block.attention
        proc = getattr(module, "processor", None)
        if isinstance(proc, Ideogram4AttnProcessor):
            style_proc = StyleIdeogram4AttnProcessor()
            style_proc.block_idx = idx
            style_proc._attention_backend = getattr(proc, "_attention_backend", None)
            style_proc._parallel_config = getattr(proc, "_parallel_config", None)
            saved.append((module, proc))
            module.set_processor(style_proc)
            processors.append(style_proc)

    return processors, saved


def restore_ideogram4_style_processors(saved: List[Tuple[Any, Any]]) -> None:
    """Undo ``install_ideogram4_style_processors`` -- puts back the original
    ``Ideogram4AttnProcessor`` instance on every module, so a subsequent
    non-style generation is byte-identical to before style transfer ran."""
    for module, proc in saved:
        module.set_processor(proc)


def set_ideogram4_style_context(processors: List[StyleIdeogram4AttnProcessor], ctx) -> None:
    """Stamp the SAME ``StyleContext`` (or ``None`` to disarm) onto every installed
    style processor. Called once per capture/inject/disarm phase per active step by
    ``ideogram4_pipeline_ops._ideogram4_style_step``."""
    if ctx is not None:
        ctx.config.resolve_default_block_range(len(processors))
    for p in processors:
        p._style_ctx = ctx
