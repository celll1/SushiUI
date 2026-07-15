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
        mode: Conduit dispatch mode. Defaults to ``AttentionMode.INFERENCE`` so
            inference callers are unchanged; SDXL/SD1.5 training installs this
            same processor with ``AttentionMode.TRAINING`` (autograd-safe path,
            training-only backend guards).
    """

    def __init__(self, backend: str = "normal", mode: AttentionMode = AttentionMode.INFERENCE):
        self.backend = backend
        self.mode = mode
        # --- Training-free reference-style transfer (StyleAligned/VSP-style KV
        # injection, see core.inference.reference_style) ---
        # `block_idx` is the self-attention layer's ordinal position (assigned by
        # `ensure_style_block_indices`); stays `None` for cross-attention ("attn2")
        # processors, which are never touched by style transfer (SD1.5/SDXL U-Net
        # self-attention has no text tokens, so injection only targets self-attn).
        # `_style_ctx` is a `StyleContext` or `None`; both default to `None` so a
        # processor that is never stamped takes the byte-identical original path.
        self.block_idx: Optional[int] = None
        self._style_ctx = None
        # --- Regional additional prompt (STAGE R2, method "attention"; see
        # RegionalPromptContext below) --- `_region_ctx` is a sibling of
        # `_style_ctx` for CROSS-attention ("attn2") only: `set_region_context`
        # stamps it exclusively onto attn2-named processors (disjoint from
        # `set_style_context`'s attn1-only targeting), and the `__call__` hook
        # below only ever reads it when `is_self_attn` is False -- so a self-attn
        # style/B3 context and a cross-attn region context can coexist on the
        # same processor SET without collision (never the same attribute on the
        # same processor instance, since attn1/attn2 are always separate
        # instances). Defaults to `None` so an unstamped processor (region
        # inactive) takes the byte-identical original path.
        self._region_ctx = None

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
        # Self-attention: diffusers UNet attn1 is always called with
        # encoder_hidden_states=None (cross-attention/attn2 always passes the text
        # embeddings). Must be captured BEFORE the "encoder_hidden_states = hidden_states"
        # fallback below overwrites it.
        is_self_attn = encoder_hidden_states is None

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

        # BSHD == [batch, seq_len, heads, head_dim] (pre-transpose) -- this is the
        # layout `core.inference.reference_style` expects for the KV-injection hook.
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # --- Regional additional prompt (training-free, cross-attention ONLY;
        # STAGE R2 method "attention" -- scratchpad/regional_prompt_synthesis.md) ---
        # Attention-Couple-style token-append + per-query spatial log-bias: the
        # image queries Q attend jointly over [main K/V (unchanged) (+) region
        # K/V (appended)], with an additive bias that is 0 on the main columns
        # and log(strength * m_l(query_position)) on the region columns (-inf
        # where the downsampled region mask is exactly 0 there) -- so a query
        # OUTSIDE the region attends to the region tokens with EXACTLY zero
        # softmax weight, numerically IDENTICAL to the un-regioned baseline for
        # that query. Cross-attention only: the self-attention sequence has no
        # text axis to append region tokens onto (region conditioning needs the
        # TEXT axis), so this is gated on `not is_self_attn`. `region_ctx.side`
        # is `None` for any forward that must not see region tokens at all
        # (feature inactive, or an internal capture-only forward such as
        # OUTPAINT B3's reference composite) -- in that case this whole block
        # is a no-op and `attention_mask` stays exactly as passed in (byte-
        # identical to the pre-region code path).
        region_ctx = self._region_ctx
        if not is_self_attn and region_ctx is not None and region_ctx.side is not None:
            region_embeds = region_ctx.pos_embeds if region_ctx.side == "pos" else region_ctx.neg_embeds
            if region_embeds is not None:
                from core.inference.custom_sampling import _outpaint_reference_block_hw

                cache_key = (id(self), region_ctx.side)
                cached_kv = region_ctx.kv_cache.get(cache_key)
                if cached_kv is None:
                    # K_r/V_r are STATIC across denoise steps (the region text
                    # never changes) -- computed once per (processor, side) and
                    # cached on the (persistent, loop-lifetime) context.
                    region_embeds_t = region_embeds.to(device=key.device, dtype=key.dtype)
                    r_k = attn.to_k(region_embeds_t).view(1, -1, attn.heads, head_dim)
                    r_v = attn.to_v(region_embeds_t).view(1, -1, attn.heads, head_dim)
                    cached_kv = (r_k, r_v)
                    region_ctx.kv_cache[cache_key] = cached_kv
                r_k, r_v = cached_kv

                # Recover this layer's own image-token grid (Hb, Wb) from its
                # query sequence length by mirroring the U-Net's actual stride-2
                # down-block chain -- reused VERBATIM from the OUTPAINT B3
                # reference-KV masking (`_outpaint_reference_filter_store`);
                # returns None (drop-don't-guess) for a block whose seq_len
                # can't be mapped back to a grid, exactly like B3.
                seq_len_img = query.shape[1]
                hw = _outpaint_reference_block_hw(region_ctx.mask_h, region_ctx.mask_w, seq_len_img)
                if hw is not None:
                    bias = region_ctx.bias_cache.get(hw)
                    if bias is None:
                        # Per-attention-resolution spatial bias, shared between
                        # the pos/neg sides (same mask + strength for both) --
                        # cached once per (Hb, Wb) since it never changes across
                        # steps or sides.
                        hb, wb = hw
                        m = torch.nn.functional.interpolate(
                            region_ctx.mask_latent, size=(hb, wb), mode="nearest"
                        ).reshape(1, 1, seq_len_img, 1).to(dtype=torch.float32)
                        wm = (region_ctx.strength * m).clamp(min=0.0)
                        bias = torch.where(
                            wm > 0.0, torch.log(wm.clamp(min=1e-38)), torch.full_like(wm, float("-inf"))
                        )
                        region_ctx.bias_cache[hw] = bias

                    s_main = key.shape[1]
                    s_region = r_k.shape[1]
                    if r_k.shape[0] != key.shape[0]:
                        r_k = r_k.expand(key.shape[0], -1, -1, -1)
                        r_v = r_v.expand(key.shape[0], -1, -1, -1)
                    key = torch.cat([key, r_k], dim=1)
                    value = torch.cat([value, r_v], dim=1)

                    region_bias = bias.expand(1, 1, seq_len_img, s_region)
                    main_bias = torch.zeros((1, 1, seq_len_img, s_main), dtype=torch.float32, device=bias.device)
                    full_bias = torch.cat([main_bias, region_bias], dim=-1)
                    if attention_mask is not None:
                        # Existing mask only covers the ORIGINAL (main) columns
                        # -- extend it with zeros for the newly appended region
                        # columns before adding our bias on top, so main-column
                        # contributions are preserved and the region-column bias
                        # is purely additive.
                        pad = torch.zeros(
                            attention_mask.shape[:-1] + (s_region,),
                            dtype=torch.float32, device=attention_mask.device,
                        )
                        attention_mask = torch.cat([attention_mask.to(dtype=torch.float32), pad], dim=-1) + full_bias
                    else:
                        attention_mask = full_bias
                    attention_mask = attention_mask.to(device=query.device, dtype=query.dtype)
                # else: this block's token grid couldn't be recovered -- drop
                # region conditioning for this block only (attention_mask stays
                # unchanged), exactly like B3's per-block store filtering.

        # --- Reference-style KV injection (training-free, self-attention only) ---
        # The whole self-attention sequence IS the image-token sequence (no text
        # prefix, unlike DiT archs), so img_start=0 / img_end=sequence_length always
        # -- computed locally per call since every U-Net resolution (down/mid/up
        # block) has a DIFFERENT sequence_length, unlike a DiT's fixed token count.
        # No RoPE in this U-Net: the frequency-scale vector is a constant `ones`
        # (no per-frequency content suppression), relying on block selection +
        # AdaIN + ref_k_strength for content/style control instead (StyleAligned's
        # original recipe).
        ctx = self._style_ctx
        if is_self_attn and self.block_idx is not None and ctx is not None and ctx.active_for_block(self.block_idx):
            from core.inference.reference_style import inject_kv, make_ref_value

            img_start, img_end = 0, query.shape[1]
            if ctx.mode == "capture":
                ctx.store[self.block_idx] = (
                    query[:, img_start:img_end].detach().clone(),
                    key[:, img_start:img_end].detach().clone(),
                    value[:, img_start:img_end].detach().clone(),
                )
            elif ctx.mode == "inject" and ctx.refs is not None:
                # Multi-reference ("stack" / "common_concept"): centralizes the
                # per-ref active/freq/make_ref_value logic in
                # StyleContext.collect_block_refs so this hook stays thin. The
                # single-ref path below (``ctx.refs is None``) is completely
                # untouched -- this branch is only ever reached for 2+ refs
                # (see custom_sampling.py's multi-ref capture/inject branches and
                # the ``style_refs``/``StyleContext(refs=...)`` wiring in
                # pipeline.py).
                from core.inference.reference_style import inject_kv_multi

                target_v_img = value[:, img_start:img_end]
                block_refs = ctx.collect_block_refs(self.block_idx, target_v_img, key.device, key.dtype)
                if block_refs:
                    key, value, query = inject_kv_multi(key, value, query, img_start, img_end, block_refs, ctx.combine_mode)
            elif ctx.mode == "inject":
                ref_qkv = ctx.store.get(self.block_idx)
                if ref_qkv is not None:
                    ref_q, ref_k, ref_v = ref_qkv
                    cfg = ctx.config
                    if cfg.ref_k_strength != 0.0 or cfg.adain_strength > 0.0:
                        freq_vec = torch.ones(head_dim, device=key.device, dtype=key.dtype)
                        target_v_img = value[:, img_start:img_end]
                        ref_v_final = make_ref_value(
                            target_v_img, ref_v, cfg.value_mode, cfg.value_adain_strength, cfg.ref_value_mix
                        )
                        key, value, query = inject_kv(
                            key, value, ref_k, ref_v_final, img_start, img_end,
                            cfg.ref_k_strength, freq_vec, cfg.adain_strength, q=query, ref_q=ref_q,
                        )
                        # attention_mask is not extended: SD1.5/SDXL U-Net self-attention
                        # is called with attention_mask=None (no padding to account for).

        # Reshape to BHSD == [batch, heads, seq_len, head_dim].
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Core attention region -> unified conduit (BHSD in/out).
        hidden_states = dispatch_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self.backend,
            mode=self.mode,
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


def set_attention_processor(
    unet,
    attention_type: str = "normal",
    mode: AttentionMode = AttentionMode.INFERENCE,
):
    """
    Set the unified attention processor on every attention layer of the UNet.

    Args:
        unet: The UNet model
        attention_type: Backend selector ("normal", "sage", "flash"). Normalized
            inside the conduit.
        mode: Conduit dispatch mode forwarded to each processor. Defaults to
            ``AttentionMode.INFERENCE`` so inference callers are unchanged; SDXL/
            SD1.5 training passes ``AttentionMode.TRAINING``.

    Returns:
        dict: Original processors for restoration
    """
    # Store original processors
    original_processors = unet.attn_processors.copy()

    num_processors = len(unet.attn_processors)

    print(f"[AttentionProcessor] Setting UnifiedAttnProcessor (backend={attention_type}, mode={mode.name}) for {num_processors} attention layers")
    new_processors = {name: UnifiedAttnProcessor(attention_type, mode=mode) for name in unet.attn_processors.keys()}
    unet.set_attn_processor(new_processors)
    print(f"[AttentionProcessor] [OK] UnifiedAttnProcessor ACTIVE for all {num_processors} layers")

    return original_processors


def restore_processors(unet, original_processors: dict):
    """Restore original attention processors"""
    if original_processors:
        unet.set_attn_processor(original_processors)
        print("[AttentionProcessor] Restored original processors")


def ensure_style_block_indices(unet) -> int:
    """Assign a stable self-attention block index (0..N-1) to every self-attn
    ("attn1") processor, in ``unet.attn_processors`` traversal order (whatever
    order this diffusers version's UNet2DConditionModel registers its blocks in
    -- e.g. down_blocks / up_blocks / mid_block, each in module-registration
    order). This is the layer-index space that a style-transfer request's
    ``style_blocks`` (``StyleTransferConfig.block_range``) selects into.

    Idempotent: safe to call every generation (re-numbers identically as long
    as the processor set/order hasn't changed -- e.g. after
    ``set_attention_processor`` swaps in new instances). Cross-attention
    ("attn2") processors are left with ``block_idx=None`` -- style transfer
    only targets self-attention, so they are silently never touched.

    Returns the total self-attention layer count (the valid ``[0, N-1]`` range
    for ``style_blocks``); ``None``/unset ``block_range`` (the default) applies
    style transfer to ALL of them, matching StyleAligned's original full-U-Net
    shared-self-attention setting.
    """
    idx = 0
    for name, proc in unet.attn_processors.items():
        if name.endswith("attn1.processor") and hasattr(proc, "block_idx"):
            proc.block_idx = idx
            idx += 1
    return idx


def set_style_context(unet, ctx) -> None:
    """Stamp a ``core.inference.reference_style.StyleContext`` (or ``None`` to
    clear) onto every self-attention processor that has already been assigned a
    ``block_idx`` by ``ensure_style_block_indices``. No-op (nothing to stamp)
    if that hasn't been called yet."""
    for name, proc in unet.attn_processors.items():
        if hasattr(proc, "block_idx") and proc.block_idx is not None:
            proc._style_ctx = ctx


# ---------------------------------------------------------------------------
# Regional additional prompt (STAGE R2, method "attention") -- cross-attention
# ONLY context, sibling of StyleContext (see reference_style.py). Lives here
# (not reference_style.py) since it is intrinsically tied to this processor's
# cross-attention hook (token-append + spatial log-bias) rather than the
# arch-agnostic self-attention KV-injection math reference_style.py provides.
# ---------------------------------------------------------------------------

class RegionalPromptContext:
    """Per-generation (loop-lifetime) runtime state for the regional additional
    prompt's "attention" method (scratchpad/regional_prompt_synthesis.md).
    Unlike ``StyleContext`` (recreated fresh for every capture/inject pass),
    this context is created ONCE before the denoise loop starts and persists
    across every step/pass -- only ``side`` is mutated per U-Net call (by the
    SAME call sites in ``custom_inpaint_sampling_loop`` that toggle
    ``set_style_context`` for B3/style), so every OUTPAINT B2 time-travel
    resample visit automatically re-applies it with no extra wiring.

    ``side``:
      - ``"pos"``: the conditional (positive) U-Net pass -- appends
        ``pos_embeds`` (no-ops if the region-positive prompt field was left
        empty, i.e. ``pos_embeds is None``).
      - ``"neg"``: the unconditional (negative) U-Net pass -- appends
        ``neg_embeds`` (no-op if the region-negative prompt field was left
        empty). The region-negative rides the UNCOND context so the existing
        CFG combine does both steer (region-positive on the cond side) AND
        suppress (region-negative on the uncond side) with no extra forward.
      - ``None``: region conditioning is off for this forward entirely (e.g.
        an internal capture-only forward such as OUTPAINT B3's reference
        composite pass, which must never see region tokens).

    ``kv_cache``/``bias_cache`` memoize the STATIC region K/V (per
    (processor identity, side) -- the region text never changes across
    steps) and the per-attention-resolution spatial bias (per (Hb, Wb) --
    shared between the pos/neg sides, since both use the same mask/strength),
    so steady-state cost after the first use is negligible (no re-projection,
    no re-interpolation).
    """

    __slots__ = (
        "side", "pos_embeds", "neg_embeds", "mask_latent", "mask_h", "mask_w",
        "strength", "kv_cache", "bias_cache",
    )

    def __init__(self, pos_embeds, neg_embeds, mask_latent, strength: float, side=None):
        self.side = side
        self.pos_embeds = pos_embeds  # [1, S_pos, D] region-positive text embeds, or None
        self.neg_embeds = neg_embeds  # [1, S_neg, D] region-negative text embeds, or None
        self.mask_latent = mask_latent  # [1, 1, H, W] float (feathered region mask; 1 = generate region)
        self.mask_h = int(mask_latent.shape[-2])
        self.mask_w = int(mask_latent.shape[-1])
        self.strength = float(strength)  # region_prompt_strength ("w" in b = log(w * m))
        self.kv_cache: dict = {}
        self.bias_cache: dict = {}


def set_region_context(unet, ctx) -> None:
    """Stamp a ``RegionalPromptContext`` (or ``None`` to clear) onto every
    CROSS-attention ("attn2") processor. Deliberately disjoint from
    ``set_style_context``'s self-attention ("attn1") targeting -- a self-attn
    style/B3 context and a cross-attn region context are never the same
    attribute on the same processor instance (attn1/attn2 are always separate
    diffusers-registered instances), so the two coexist on the same processor
    set without collision."""
    for name, proc in unet.attn_processors.items():
        if name.endswith("attn2.processor") and hasattr(proc, "_region_ctx"):
            proc._region_ctx = ctx
