"""Training-free reference-style transfer (StyleAligned/VSP-style KV injection) for
LTX-2.3 VIDEO, wired through the arch-agnostic ``core.inference.reference_style``
module (see that module's docstring for the shared math: ``inject_kv``,
``cross_batch_adain_qk``, ``make_ref_value``).

Scope: video self-attention ONLY. LTX-2.3's per-block attention is split into five
separate ``LTX2Attention`` modules (``attn1`` video self-attn, ``audio_attn1`` audio
self-attn, ``attn2`` video-text cross-attn, ``audio_attn2`` audio-text cross-attn,
``audio_to_video_attn`` / ``video_to_audio_attn`` cross-modality attn). Style
transfer patches ONLY ``attn1`` (pure VIDEO self-attention over video tokens,
``encoder_hidden_states=None`` in the stock block forward -- see
``transformer_ltx2.LTX2VideoTransformerBlock.forward``, section "1.1 Video
Self-Attention"). The audio stream and the text cross-attentions are never
touched, matching the task's "audio is a separate stream; text is cross-attn"
scoping.

Design (mirrors Ideogram 4's ``StyleIdeogram4AttnProcessor`` -- the closest
precedent for a diffusers-`Attention`-module-style ``set_processor`` API,
``core.models.ideogram4.style_ideogram4``): ``StyleLtx2Attn1Processor`` is a
byte-identical duplicate of the stock ``LTX2AudioVideoAttnProcessor.__call__``
(diffusers ``transformer_ltx2.py``) with one extra call to ``_apply_style_hook``
inserted right after qk-RMSNorm + RoPE, right before ``dispatch_attention_fn`` --
the same insertion point every other arch's style processor uses. It is installed
ONLY on ``attn1`` modules of ``transformer.transformer_blocks`` (the INNER,
unwrapped ``LTX2VideoTransformer3DModel``), and ONLY for the duration of a
style-active generation (``install_ltx2_style_processors`` /
``restore_ltx2_style_processors``); a generation without a style reference never
touches this module (``_style_ctx`` defaults to ``None`` at construction, and the
hook is a no-op whenever it is ``None``).

Video-token span: attn1's ``hidden_states`` is the FULL video-token sequence (no
text, no audio, no left-pad -- unlike Ideogram 4's packed
``[left-pad][text][image]`` layout). ``img_start=0`` / ``img_end=seq_len`` for
EVERY call (the "image" region *is* the whole self-attention sequence), so no
per-sample span bookkeeping is needed (unlike Ideogram 4's packed sequence).

Still -> single-frame video-latent reference: the user-provided still image is
VAE-encoded (LTX-2.3's own video VAE, ``num_frames=1``) at the SAME
height/width as the target, producing a ``[1, H*W, C]`` packed one-frame video
latent (patch_size=1 in this codebase's LTX-2.3 config, so packing is a plain
reshape/permute -- see ``diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.
_pack_latents``). Because LTX-2.3 patchifies frame-major (token index
``f*H*W + h*W + w``, see ``_pack_latents``'s permute/flatten), the target's OWN
frame-0 spatial tokens occupy exactly the first ``H*W`` rows of
``video_rotary_emb`` -- the ref capture pass reuses THOSE rope rows verbatim
(``Ltx2BlockLoopWrapper._gather_video_rope`` with ``idx = arange(H*W)``) instead
of computing an independent RoPE grid for the reference. This is the natural
StyleAligned choice (reference at the SAME positions as the target content it
stylizes) and requires ``ref token count == H*W == the target's own frame-0 token
count``, asserted at ref-preparation time.

Because attn1 is FULL spatiotemporal self-attention (every video token, every
frame, attends every other video token in the SAME call), injecting the single
still's ``H*W`` Key/Value tokens ONCE per block means every frame's queries can
attend the SAME reference tokens -- temporally consistent by construction, no
per-frame repetition of the reference needed.

CFG composition (THE crux for a diffusers-native pipeline with no custom denoise
loop): the LTX-2.3 pipeline batches CFG by concatenating
``[uncond_0..uncond_{N-1}, cond_0..cond_{N-1}]`` along batch dim BEFORE calling
the transformer once per step (``pipeline_ltx2.py``: ``prompt_embeds =
torch.cat([negative_prompt_embeds, prompt_embeds])``, ``latent_model_input =
torch.cat([latents] * 2)``). So a single attn1 call sees the WHOLE batch (both
branches) at once; style must be injected into the COND rows only, leaving the
UNCOND rows' attention untouched (mirrors FLUX.2's separate uncond forward,
which never touches the style context). ``_apply_style_hook`` derives the
cond/uncond row split purely from the batch size it is handed: if the batch is
even and >= 2, the SECOND half (``[batch//2 : batch]``) is treated as the cond
rows (matching the concat order above, generalized to any ``num_videos_per_prompt``
== N, since the halves are each N rows regardless of N); if the batch is odd or
1, there is no CFG doubling and the WHOLE batch is cond. The appended ref-K/V
columns are given an additive attention-mask bias of ``-inf`` for uncond rows
(and 0 for cond rows) so uncond attention is bit-for-bit as if the columns did
not exist. UNCERTAINTY: this heuristic has not been exercised against
``num_videos_per_prompt > 1`` combined with CFG (cannot run cold / no GPU); the
halves-split logic is structurally correct for any N given the concat order
above, but is unverified end-to-end.

Spatio-Temporal Guidance (STG) / audio-modality-guidance EXTRA transformer calls
(``pipeline_ltx2.py``'s ``noise_pred_..._uncond_stg`` / ``..._uncond_modality``
branches) issue ADDITIONAL ``self.transformer(...)`` calls per step with their
own (non-doubled) batch composition that this module's row-split heuristic does
NOT model correctly (an uncond_stg call's single-row batch would be
misidentified as "all cond" and incorrectly stylized). Style transfer is
therefore mutually exclusive with STG / audio-STG (mirrors Spectrum's existing
guard on ``stg_scale`` / ``audio_stg_scale`` in ``pipeline_backends/ltx2.py``);
enforced by the caller before ever building a style config.

RoPE note: LTX-2.3's default ``rope_type="interleaved"`` RoPE is NOT the
``repeat_interleave_real=True`` concatenated-per-axis-block layout
``frequency_scale_vector`` assumes (mirrors Ideogram 4 / MiniT2I's identical
decision) -- this wiring passes an all-ones vector straight to ``inject_kv``
(``axes_dims`` is never set on the ``StyleTransferConfig``; per the task's
explicit constraint, frequency suppression is skipped entirely for LTX-2.3).
Frequency suppression is a quality knob only -- ``ref_k_strength`` and AdaIN
(the load-bearing mechanisms) apply in full regardless.

Value-mode deviation: ``reference_style.make_ref_value``'s "target_adain" value
blend requires the target's own image-token region and the reference to share
the SAME token count (true for every other arch: their style image is encoded
at the target's own canvas size). That invariant never holds for LTX-2.3 (the
still reference has ``H*W`` tokens; the target video has ``num_frames*H*W``
tokens), so this wiring ALWAYS uses the raw reference Value directly (the
"ref_raw" mode's result) regardless of the user's requested ``value_mode`` /
``value_adain_strength`` / ``ref_value_mix`` -- those three knobs are no-ops
for this arch (see ``_apply_style_hook``). The single-ref branch bypasses
``make_ref_value`` entirely (``ref_v_final = ref_v``); the multi-reference
branch instead forces every LTX-2.3 ``StyleTransferConfig.value_mode`` to
``"ref_raw"`` (in ``pipeline_backends.ltx2._ltx2_style_triple``) so the
shared ``StyleContext.collect_block_refs``/``make_ref_value`` call (part of
the arch-agnostic core, not modified for this port) resolves to the same
raw-reference-Value result without ever computing the shape-mismatched
"target_adain" blend.

Multi-reference (N>1) support: mirrors the Anima/Krea2 N-ref wiring
(``core.inference.reference_style.inject_kv_multi`` / ``StyleContext.refs`` /
``collect_block_refs``). ``_apply_style_hook``'s inject branch dispatches on
``ctx.refs is not None``; single-ref (``ctx.refs is None``) is byte-identical
to the pre-multi-ref implementation. The multi-ref branch reproduces the
SAME CFG row-split / uncond-column-masking / cond-rows-only AdaIN handling
described above, applied to the MEAN of the active references' Q/K instead of
one reference's own. ``Ltx2BlockLoopWrapper._style_run_capture_and_arm_inject``
drives one capture sub-pass PER reference (each independently step-gated by
its own config) whenever ``len(style_refs) > 1``; a single reference (via
either ``style_transfer`` or a length-1 ``style_transfers``) always resolves
through the untouched single-ref ``(cfg, ref_x0, eps_ref)`` triple in
``pipeline_backends.ltx2._ltx2_style_configs``.

FBCache/Spectrum interop: FBCache and Spectrum MUST be disabled for the whole
generation whenever style transfer is active (a cache hit / forecast skip
bypasses the real block loop entirely, which would desync the per-block style
capture/inject store across steps -- mirrors every other arch's audited
finding); enforced in ``pipeline_backends/ltx2.py`` BEFORE ``_ltx2_build_fbcache``
/ ``_ltx2_build_spectrum`` are even called.

Block Swap interop: style transfer forces Block Swap OFF for the whole
generation (the reverse precedence from FBCache/Spectrum, which are disabled
WHEN Block Swap is on) -- the ref-capture sub-pass this module drives
(``Ltx2BlockLoopWrapper``'s style branch in ``_custom_forward``) does not thread
the block offloader's per-block wait/submit calls through the ref loop, so
running it concurrently with real block-swap prefetch would desync the swap
rotation exactly like an FBCache hit does. UNCERTAIN whether composing them
(threading offloader.wait_for_block/submit_move_blocks_forward through the ref
sub-pass too) is safe/correct -- not attempted; enforced by the caller
(``pipeline_backends/ltx2.py``) forcing ``blocks_to_swap = 0`` whenever style is
requested, with a printed reason.

TREAD / DiT-BlockSkip: both are training-only (gated on
``self.training and torch.is_grad_enabled()`` in the wrapper); style transfer is
inference-only (asserted in ``Ltx2BlockLoopWrapper.attach_style``), so they are
structurally mutually exclusive and never need an explicit guard.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch


class StyleLtx2Attn1Processor:
    """Byte-identical duplicate of diffusers'
    ``transformer_ltx2.LTX2AudioVideoAttnProcessor.__call__`` (video self-attn
    call signature: ``encoder_hidden_states`` is always ``None`` for ``attn1``)
    with a capture/inject hook inserted after qk-RMSNorm + RoPE, before
    ``dispatch_attention_fn``. ``_style_ctx`` defaults to ``None`` at
    construction -- never touched by a non-style generation. Installed ONLY on
    ``attn1`` modules (see ``install_ltx2_style_processors``); never on
    ``audio_attn1`` / ``attn2`` / ``audio_attn2`` / the cross-modality attns.
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_ltx2 import (
            apply_interleaved_rotary_emb,
            apply_split_rotary_emb,
        )

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if query_rotary_emb is not None:
            if attn.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )
            elif attn.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Reference-style KV injection (training-free) -- see module docstring
        # for the CFG row-split derivation. Must run strictly AFTER qk-RMSNorm
        # and AFTER RoPE (mirrors every other arch's style hook placement).
        query, key, value, attention_mask = _apply_style_hook(self, query, key, value, attention_mask)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if attn.to_gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _apply_style_hook(
    proc: "StyleLtx2Attn1Processor",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Shared capture/inject body. ``query``/``key``/``value`` are ``[B, S, H, D]``
    (post-qk-norm, post-RoPE) -- the FULL video-token self-attention sequence
    (``img_start=0``, ``img_end=S`` for every call; see module docstring). No-op
    (returns inputs unchanged) whenever ``_style_ctx`` is ``None``, the processor
    was never assigned a ``block_idx``, or the context's ``block_range`` excludes
    this block."""
    ctx = proc._style_ctx
    if ctx is None or proc.block_idx is None or not ctx.active_for_block(proc.block_idx):
        return query, key, value, attention_mask

    if ctx.mode == "capture":
        # Ref forward is always batch=1 (a single reference image); the WHOLE
        # sequence is the ref's own H*W video tokens.
        ctx.store[proc.block_idx] = (
            query.detach().clone(),
            key.detach().clone(),
            value.detach().clone(),
        )
        return query, key, value, attention_mask

    # mode == "inject"
    if ctx.refs is not None:
        # Multi-reference ("stack" / "common_concept"): centralizes the
        # per-ref active/freq/make_ref_value logic in
        # ``StyleContext.collect_block_refs`` so this hook stays thin. The
        # single-ref branch below (reached only when ``ctx.refs is None``) is
        # completely untouched -- this branch is only ever reached for 2+
        # refs (see ``Ltx2BlockLoopWrapper._style_run_capture_and_arm_inject``'s
        # multi-ref branch and the ``style_refs``/``StyleContext(refs=...)``
        # wiring in ``pipeline_backends.ltx2``). Preserves the SAME CFG
        # row-split / uncond-column-masking / cond-rows-only AdaIN handling as
        # the single-ref branch below -- only the reference tensors and the
        # injection call (``inject_kv_multi`` instead of ``inject_kv``) differ.
        #
        # ``value_mode`` is forced to "ref_raw" for every LTX-2.3 style config
        # by ``pipeline_backends.ltx2._ltx2_style_triple`` (mirrors this
        # module's "Value-mode deviation" docstring section), so
        # ``StyleContext.collect_block_refs``'s unconditional ``make_ref_value``
        # call always resolves to the raw reference Value here too, matching
        # the single-ref branch's explicit ``ref_v_final = ref_v`` bypass
        # below (``make_ref_value``'s "target_adain" blend would otherwise
        # raise a shape mismatch: ref = H*W tokens, target = num_frames*H*W).
        from core.inference.reference_style import inject_kv_multi, cross_batch_adain_qk

        m_batch = query.shape[0]
        m_seq = query.shape[1]
        m_half = m_batch // 2 if (m_batch >= 2 and m_batch % 2 == 0) else 0

        target_v_img = value[:, 0:m_seq]
        block_refs = ctx.collect_block_refs(proc.block_idx, target_v_img, key.device, key.dtype)
        if not block_refs:
            return query, key, value, attention_mask

        # AdaIN: cond rows ONLY, never uncond (identical rationale to the
        # single-ref branch below) -- toward the MEAN of the active refs'
        # Q/K, applied ONCE here rather than letting ``inject_kv_multi``'s own
        # "stack"/"common_concept" AdaIN fire against the FULL batch (which
        # would stylize the uncond rows' Q/K too). ``adain_strength`` is then
        # zeroed in the ``block_refs`` handed to ``inject_kv_multi`` so its
        # internal AdaIN branch is a no-op (avoids a redundant second align).
        max_adain = max(r[4] for r in block_refs)
        if max_adain > 0.0:
            mean_ref_k = torch.stack([r[0] for r in block_refs], dim=0).mean(dim=0)
            if all(r[2] is not None for r in block_refs):
                mean_ref_q = torch.stack([r[2] for r in block_refs], dim=0).mean(dim=0)
            else:
                mean_ref_q = None
            q_cond = query[m_half:]
            k_cond = key[m_half:]
            if mean_ref_q is not None:
                q_cond_aligned, k_cond_aligned = cross_batch_adain_qk(
                    q_cond, k_cond, mean_ref_q, mean_ref_k, max_adain
                )
            else:
                q_cond_aligned, k_cond_aligned = cross_batch_adain_qk(
                    q_cond, k_cond, mean_ref_k, mean_ref_k, max_adain
                )
            if m_half > 0:
                query = torch.cat([query[:m_half], q_cond_aligned], dim=0)
                key = torch.cat([key[:m_half], k_cond_aligned], dim=0)
            else:
                query, key = q_cond_aligned, k_cond_aligned
            block_refs = [(rk, rv, rq, rs, 0.0, rf) for (rk, rv, rq, rs, _ra, rf) in block_refs]

        seq_len_before = key.shape[1]
        key, value, query = inject_kv_multi(key, value, query, 0, m_seq, block_refs, ctx.combine_mode)

        extra = key.shape[1] - seq_len_before
        if extra > 0 and m_half > 0:
            # Same uncond-masking as the single-ref branch below: the
            # appended ref-K/V columns land at the end of the key axis;
            # invisible (large negative bias) to uncond rows, visible (bias 0)
            # to cond rows.
            neg = torch.finfo(key.dtype).min
            extra_bias = torch.zeros(m_batch, 1, 1, extra, device=key.device, dtype=key.dtype)
            extra_bias[:m_half] = neg
            if attention_mask is None:
                base_bias = torch.zeros(m_batch, 1, 1, seq_len_before, device=key.device, dtype=key.dtype)
                attention_mask = torch.cat([base_bias, extra_bias], dim=-1)
            else:
                attention_mask = torch.cat([attention_mask, extra_bias], dim=-1)

        return query, key, value, attention_mask

    ref_qkv = ctx.store.get(proc.block_idx)
    if ref_qkv is None:
        return query, key, value, attention_mask

    from core.inference.reference_style import inject_kv, cross_batch_adain_qk

    ref_q, ref_k, ref_v = ref_qkv
    cfg = ctx.config
    if cfg.ref_k_strength == 0.0 and cfg.adain_strength <= 0.0:
        return query, key, value, attention_mask

    batch = query.shape[0]
    seq = query.shape[1]

    # CFG row split (see module docstring "CFG composition"): the LTX-2.3
    # pipeline always concatenates [uncond rows..., cond rows...] along batch
    # dim before calling the transformer, in two equal halves, whenever CFG is
    # active. half==0 means "no CFG doubling this call" -> the whole batch is
    # cond (inject for every row, no mask needed).
    half = batch // 2 if (batch >= 2 and batch % 2 == 0) else 0

    # Interleaved RoPE: frequency_scale_vector's concatenated-per-axis-block
    # layout does not match LTX-2.3's interleaved RoPE (see module docstring) --
    # pass an all-ones vector (documented quality no-op, mirrors Ideogram 4 /
    # MiniT2I). ref_k_strength + AdaIN (load-bearing) still apply.
    freq_vec = torch.ones(query.shape[-1], device=key.device, dtype=key.dtype)

    # NOTE (LTX-2.3-specific deviation from every other arch's style wiring):
    # ``reference_style.make_ref_value``'s "target_adain" value-blend mode ends
    # with a plain elementwise combine of the (target-shaped) blended base and
    # the (ref-shaped) raw reference Value -- valid ONLY when the target's own
    # image-token region and the reference occupy the SAME token count (true
    # for every other arch: the style image is encoded at the target's own
    # canvas size). That invariant NEVER holds here: the reference is a single
    # still (H*W tokens) while the target is a full video (num_frames*H*W
    # tokens) -- calling make_ref_value with "target_adain" would raise a shape
    # mismatch. LTX-2.3 therefore ALWAYS uses the raw reference Value directly
    # (equivalent to make_ref_value's "ref_raw" mode with ref_value_mix=1.0),
    # regardless of the user's requested value_mode/value_adain_strength/
    # ref_value_mix -- those three knobs are no-ops for this arch (documented
    # limitation, mirrors the frequency-scale ones-vector no-op above).
    ref_v_final = ref_v

    # AdaIN: apply to the COND rows ONLY, never the uncond rows -- reference_
    # style.inject_kv's own AdaIN branch (cross_batch_adain_qk) mutates Q/K for
    # EVERY row it is given, so calling inject_kv directly with the FULL batch
    # and a nonzero adain_strength would stylize the uncond rows' own Q/K too
    # (only the APPENDED ref-K columns are masked off for uncond below -- that
    # does not undo an AdaIN mutation of the uncond rows' pre-existing Q/K).
    # Fix: manually AdaIN-align query/key for the cond rows (img_start=0,
    # img_end=seq -- the WHOLE sequence is the "image" region for attn1), write
    # the aligned rows back, and leave uncond rows ([0:half)) byte-for-byte
    # untouched. When there is no CFG doubling (half == 0) every row is cond,
    # so the "cond slice" is the whole batch. inject_kv is then called with
    # adain_strength=0.0 (its own AdaIN branch never fires) so it ONLY appends
    # the (scaled) reference K/V -- the uncond-column mask below then makes
    # that appended-columns-only change invisible to uncond rows as well,
    # leaving uncond attention output bit-identical to a no-style forward.
    if cfg.adain_strength > 0.0:
        q_cond = query[half:]
        k_cond = key[half:]
        q_cond_aligned, k_cond_aligned = cross_batch_adain_qk(
            q_cond, k_cond, ref_q, ref_k, cfg.adain_strength
        )
        if half > 0:
            query = torch.cat([query[:half], q_cond_aligned], dim=0)
            key = torch.cat([key[:half], k_cond_aligned], dim=0)
        else:
            query, key = q_cond_aligned, k_cond_aligned

    seq_len_before = key.shape[1]
    key, value, query = inject_kv(
        key, value, ref_k, ref_v_final, 0, seq, cfg.ref_k_strength, freq_vec, 0.0,
        q=query, ref_q=ref_q,
    )

    extra = key.shape[1] - seq_len_before
    if extra > 0 and half > 0:
        # Appended ref-K/V columns land at the END of the key axis (img_end ==
        # seq == the full sequence here, so "end of image region" == "end of
        # full sequence", same simplification Krea2/SDXL/Lens rely on).
        # Visible (bias 0) to cond rows [half:batch); invisible (large negative
        # bias) to uncond rows [0:half) -- so uncond attention is bit-for-bit
        # as if the ref columns were never appended. The ORIGINAL seq_len_before
        # columns keep whatever mask they already had (always None here --
        # self_attention_mask is never set by the wrapper's block loop).
        neg = torch.finfo(key.dtype).min
        extra_bias = torch.zeros(batch, 1, 1, extra, device=key.device, dtype=key.dtype)
        extra_bias[:half] = neg
        if attention_mask is None:
            base_bias = torch.zeros(batch, 1, 1, seq_len_before, device=key.device, dtype=key.dtype)
            attention_mask = torch.cat([base_bias, extra_bias], dim=-1)
        else:
            attention_mask = torch.cat([attention_mask, extra_bias], dim=-1)

    return query, key, value, attention_mask


def install_ltx2_style_processors(inner_transformer) -> Tuple[List[StyleLtx2Attn1Processor], List[Tuple[Any, Any]]]:
    """Replace the DEFAULT ``LTX2AudioVideoAttnProcessor`` on every
    ``transformer_blocks[i].attn1`` module of the (INNER, unwrapped)
    ``LTX2VideoTransformer3DModel`` with a style-aware capture/inject variant.
    Blocks whose ``attn1`` uses a different processor (e.g.
    ``LTX2PerturbedAttnProcessor``, installed only when
    ``perturbed_attn=True`` for Spatio-Temporal-Guidance blocks -- not used
    anywhere in this codebase's LTX-2.3 wiring, see module docstring) are
    skipped with a printed warning rather than silently mis-wrapped.

    Returns ``(processors, saved)``:
      - ``processors``: the installed style processor instances, indexed by
        block (the SAME numbering ``StyleTransferConfig.block_range`` gates
        against).
      - ``saved``: ``(module, original_processor)`` pairs for
        ``restore_ltx2_style_processors`` to undo after the generation.
    """
    from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoAttnProcessor

    processors: List[StyleLtx2Attn1Processor] = []
    saved: List[Tuple[Any, Any]] = []

    for idx, block in enumerate(inner_transformer.transformer_blocks):
        module = block.attn1
        proc = getattr(module, "processor", None)
        if isinstance(proc, LTX2AudioVideoAttnProcessor):
            style_proc = StyleLtx2Attn1Processor()
            style_proc.block_idx = idx
            style_proc._attention_backend = getattr(proc, "_attention_backend", None)
            style_proc._parallel_config = getattr(proc, "_parallel_config", None)
            saved.append((module, proc))
            module.set_processor(style_proc)
            processors.append(style_proc)
        else:
            print(f"[StyleLTX2] block {idx} attn1 processor is {type(proc).__name__}, not the "
                  "stock LTX2AudioVideoAttnProcessor -- style transfer skipped for this block")

    return processors, saved


def restore_ltx2_style_processors(saved: List[Tuple[Any, Any]]) -> None:
    """Undo ``install_ltx2_style_processors`` -- puts back the original
    ``LTX2AudioVideoAttnProcessor`` instance on every module, so a subsequent
    non-style generation is byte-identical to before style transfer ran."""
    for module, proc in saved:
        module.set_processor(proc)


def set_ltx2_style_context(processors: List[StyleLtx2Attn1Processor], ctx) -> None:
    """Stamp the SAME ``StyleContext`` (or ``None`` to disarm) onto every
    installed style processor. Called once per capture/inject/disarm phase per
    active step by ``Ltx2BlockLoopWrapper``'s style branch in
    ``_custom_forward``."""
    for p in processors:
        p._style_ctx = ctx
