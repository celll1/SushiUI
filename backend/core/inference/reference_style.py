"""Arch-agnostic training-free reference-style transfer (StyleAligned / VSP-style
attention KV-injection).

Mechanism: at each denoise step, a style reference image (VAE-encoded and
forward-noised to the current sigma) is run through the transformer to capture
the per-block post-RoPE image-token Key/Value tensors; the target's conditional
forward then reads those stashed K/V, applies a RoPE-frequency-aware scale
(``frequency_scale_vector``) + a user strength to the reference Key, AdaIN-aligns
the target's own Query/Key to the reference Key statistics, builds a controlled
reference Value (``make_ref_value``), and concatenates the (scaled) reference
K/V onto the image-token region of the target's own K/V before attention
(``inject_kv``). This is training-free: no weights change, only the runtime
K/V sequence seen by self-attention.

This module intentionally carries NO architecture-specific assumptions (layout
is BSHD ``[B, S, H, D]``, GQA-aware since key/value may have fewer heads than
query) so that Krea2, SDXL and Flux2 backends can all route through the exact
same math. Per-arch wiring (where in the attention forward to hook, how to
locate the image-token slice, how to build position_ids for the reference
forward) lives in each arch's own transformer / pipeline_ops module.

Defaults mirror the ComfyUI-Krea2-StyleTransfer reference node constants. Only
``ref_k_strength``, ``block range`` and ``adain_strength`` (plus the frozen
frequency-scale constants and the value-mode path) are exercised by the v1
Krea2 wiring; ``late_release``, ``rope_offset`` and multi-reference support are
carried as no-op-by-default knobs for later exposure.

Reference: StyleAligned (Hertz et al.) / VSP-style attention sharing; the exact
node this port matches is the community ComfyUI-Krea2-StyleTransfer custom node
(``nodes.py``: ``_build_frequency_scale_vector``, ``_adain``,
``_cross_batch_adain_qk``, ``_make_controlled_ref_value``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StyleTransferConfig:
    """Every style-transfer knob, arch-agnostic. Defaults = reference-node
    constants. ``axes_dims`` must be filled in by the arch-specific wiring
    (e.g. Krea2's ``transformer.config.axes_dims_rope``) before
    ``get_freq_scale_vector`` is called -- there is no universal default since
    the RoPE axis split is architecture-specific.
    """

    # --- transfer type (R&D) --- "style" = appearance/texture (default); "character"
    # = identity transfer (early blocks + raw reference Value + minimal AdaIN). Carried
    # here so the arch injection path stays shared; the recipe is applied via the
    # knobs below (block_range/value_mode/ref_value_mix/value_adain_strength/adain_
    # strength/freq curve). "character" does not (yet) auto-preset those — the caller
    # sets them — but the field lets us codify a preset once experiments settle it.
    transfer_type: str = "style"

    # --- v1 (exercised) --- tuned defaults (validated by GPU experiments across
    # SDXL/SD1.5/Anima): the verbatim ComfyUI-Krea2-StyleTransfer constants
    # (1.06 / all-blocks / 0.85) over-copy the reference's CONTENT (and on
    # freq-suppressing DiT archs like Anima, destabilize into artifacts).
    ref_k_strength: float = 0.75
    block_range: Optional[Tuple[int, int]] = None  # inclusive (lo, hi); None = resolve to the
    # proportional mid-late default via ``resolve_default_block_range`` (below)
    # once the arch's total block count is known -- NOT "all blocks".
    adain_strength: float = 0.6

    # Fraction of an arch's total style-injectable blocks (lo_frac, hi_frac)
    # used to build the default ``block_range`` when the caller leaves it
    # unset (None). Default = inject only in the LAST 50% of blocks (mid-late),
    # matching the reference node's own "7-27 of 28" (~25-96%) mid-late subset
    # rather than truly all blocks. Scales automatically per-arch (SDXL ~70
    # self-attn layers, Krea2 28, etc.) via ``resolve_default_block_range``.
    block_range_frac: Tuple[float, float] = (0.5, 1.0)

    # --- frequency-scale: STEP-PROGRESS dependent (interpolated start->end over
    # the denoise progress, progress=0 at the first step -> 1 at the last).
    # ``curve = high + (low - high) * x**beta`` per-axis, where at a given
    # progress ``p``: ``high_scale = lerp(high_scale_start, high_scale_end, p)``,
    # ``low_scale = lerp(low_scale_start, low_scale_end, p)``.
    high_scale_start: float = 1.04
    high_scale_end: float = 0.0
    low_scale_start: float = 1.0
    low_scale_end: float = 1.10
    beta: float = 2.5
    axes_dims: Optional[Tuple[int, ...]] = None

    # --- value-mode path (implemented) ---
    value_mode: str = "target_adain"  # "target_adain" | "ref_raw"
    value_adain_strength: float = 0.65
    ref_value_mix: float = 1.0

    # --- deferred / stubbed (no-op at these defaults; carried for parity) ---
    late_release: float = 0.0   # fraction of steps after which injection fades out; 0 = never
    rope_offset: int = 0        # positional offset for the reference RoPE grid; 0 = none

    # --- step gating: arrives in the ControlNet 0-1000 convention from the
    # frontend (NOT literal step indices) and is mapped to the denoise
    # progress fraction by ``is_step_active``.
    start_step: int = 0
    end_step: int = 1000

    # --- CFG-decoupled style guidance (SDXL/SD1.5 prototype only) --- a SECOND
    # guidance axis (lambda) that dials style strength independently of the
    # prompt CFG scale. None or <= 0 = DISABLED (the classic 2-pass: style
    # rides on the cond-uncond delta and is scaled by cfg like the prompt is,
    # so it weakens at low cfg). > 0 enables a 3rd (cond, no-style) forward
    # per style-active step and rewrites the two preds so the SAME shared CFG
    # combine yields: uncond + cfg*(cond_noStyle - uncond) + lambda*(cond_style
    # - cond_noStyle) -- i.e. style strength no longer depends on cfg.
    style_guidance_scale: Optional[float] = None

    # internal cache: (device, dtype, rounded progress) -> frequency scale vector
    _freq_cache: Dict[Tuple[Any, Any, float], torch.Tensor] = field(default_factory=dict, repr=False, compare=False)

    def resolve_default_block_range(self, total_blocks: int) -> None:
        """Fill in ``block_range`` from ``block_range_frac`` scaled to this
        arch's ``total_blocks``, but ONLY if the caller left ``block_range``
        unset (None) -- an explicit user/config ``block_range`` always wins
        and this is a no-op. Idempotent: once ``block_range`` is set (by this
        call or by the user), subsequent calls do nothing. No-op when
        ``total_blocks <= 0`` (arch couldn't report a count)."""
        if self.block_range is not None:
            return
        if total_blocks <= 0:
            return
        lo_frac, hi_frac = self.block_range_frac
        lo = round(lo_frac * total_blocks)
        hi = total_blocks - 1
        self.block_range = (lo, hi)

    def is_block_active(self, block_idx: int) -> bool:
        if self.block_range is None:
            return True
        lo, hi = self.block_range
        return lo <= block_idx <= hi

    def is_step_active(self, step_idx: int, num_steps: int) -> bool:
        """``start_step``/``end_step`` are the ControlNet 0-1000 convention
        (fraction of the FULL diffusion progress * 1000), not literal step
        indices. Map ``step_idx`` (0-based, out of ``num_steps`` total steps)
        to the same 0-1000 scale and gate against that."""
        if num_steps <= 1:
            progress_1000 = 0.0
        else:
            progress_1000 = 1000.0 * step_idx / (num_steps - 1)
        return self.start_step <= progress_1000 <= self.end_step

    def step_progress(self, step_idx: int, num_steps: int) -> float:
        """0 at the first denoise step, 1 at the last (``num_steps - 1``)."""
        if num_steps <= 1:
            return 0.0
        return step_idx / (num_steps - 1)

    def get_freq_scale_vector(
        self, head_dim: int, progress: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self.axes_dims is None:
            raise ValueError(
                "StyleTransferConfig.axes_dims must be set by the arch-specific wiring "
                "before requesting the frequency scale vector."
            )
        progress = max(0.0, min(1.0, float(progress)))
        key = (device, dtype, round(progress, 6))
        cached = self._freq_cache.get(key)
        if cached is None:
            high_scale = self.high_scale_start + (self.high_scale_end - self.high_scale_start) * progress
            low_scale = self.low_scale_start + (self.low_scale_end - self.low_scale_start) * progress
            cached = frequency_scale_vector(
                head_dim, self.axes_dims, high_scale, low_scale, self.beta, device, dtype
            )
            self._freq_cache[key] = cached
        return cached


# ---------------------------------------------------------------------------
# Frequency-scale vector (RoPE-frequency-content suppression on the ref Key)
# ---------------------------------------------------------------------------

def frequency_scale_vector(
    head_dim: int,
    axes_dims: Tuple[int, ...],
    high_scale: float,
    low_scale: float,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-head-dim scale curve, shape ``(head_dim,)``.

    Within each RoPE axis (e.g. Krea2's ``(32, 48, 48)`` t/h/w split), builds a
    ``high -> low`` power curve over the axis' *frequency* index (low index =
    high frequency / fine detail, matching ``get_1d_rotary_pos_embed``'s
    descending-frequency layout) and duplicates each value once to match the
    ``repeat_interleave_real=True`` RoPE convention used by
    ``apply_rotary_emb`` (each rotary pair occupies 2 adjacent head-dim slots).
    ``curve = high + (low - high) * x**beta``, ``x`` in ``[0, 1]``.
    """
    if sum(axes_dims) != head_dim:
        raise ValueError(f"sum(axes_dims)={sum(axes_dims)} must equal head_dim={head_dim}")

    curves = []
    for dim in axes_dims:
        half = dim // 2
        if half == 0:
            continue
        x = torch.linspace(0.0, 1.0, half, device=device, dtype=torch.float32)
        curve = high_scale + (low_scale - high_scale) * x.pow(beta)
        curve = curve.repeat_interleave(2)
        curves.append(curve)
    vec = torch.cat(curves, dim=0)
    if vec.shape[0] != head_dim:
        # Odd per-axis dim (shouldn't happen for Krea2/Flux-style RoPE, but guard
        # rather than silently mis-shape).
        raise ValueError(f"frequency_scale_vector produced {vec.shape[0]} dims, expected {head_dim}")
    return vec.to(dtype)


# ---------------------------------------------------------------------------
# AdaIN
# ---------------------------------------------------------------------------

def _stats_over_tokens(x: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-(batch,head,dim) mean/std over the token axis (``dim=-3`` for BSHD
    ``[B, S, H, D]`` tensors -- i.e. the S axis, one axis before the head axis)."""
    token_dim = -3
    mean = x.mean(dim=token_dim, keepdim=True)
    std = x.std(dim=token_dim, keepdim=True) + eps
    return mean, std


def _broadcast_heads(stat: torch.Tensor, target_heads: int) -> torch.Tensor:
    """Expand a per-KV-head stat (``[..., H_kv, D]``) to ``target_heads``
    (``H_q``) by repeating each KV-head's stat across its GQA group, mirroring
    ``enable_gqa``'s implicit KV-head repetition (group size = H_q // H_kv).
    No-op when the head counts already match."""
    src_heads = stat.shape[-2]
    if src_heads == target_heads:
        return stat
    if target_heads % src_heads != 0:
        raise ValueError(f"target_heads={target_heads} is not a multiple of stat heads={src_heads}")
    group = target_heads // src_heads
    return stat.repeat_interleave(group, dim=-2)


def adain(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Align ``target``'s per-(batch,head,dim) mean/std to ``style``'s, computed
    over the token axis. GQA-aware: when ``style`` has fewer heads than
    ``target`` (e.g. aligning Q against a KV-head-count reference), the
    reference's per-head stats are broadcast across each query-head group."""
    t_mean, t_std = _stats_over_tokens(target, eps)
    s_mean, s_std = _stats_over_tokens(style, eps)
    s_mean = _broadcast_heads(s_mean, target.shape[-2])
    s_std = _broadcast_heads(s_std, target.shape[-2])
    return (target - t_mean) / t_std * s_std + s_mean


def cross_batch_adain_qk(
    q: torch.Tensor, k: torch.Tensor, ref_q: torch.Tensor, ref_k: torch.Tensor, strength: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blend target Q/K toward their AdaIN-aligned version by ``strength``
    (0 = no-op, 1 = fully aligned): target Q is stylized by the REFERENCE
    QUERY (``ref_q``), target K is stylized by the REFERENCE KEY (``ref_k``) --
    each aligned to its own-kind reference statistics (verbatim
    ComfyUI-Krea2-StyleTransfer ``_cross_batch_adain_qk``). This requires the
    reference Query to have been captured alongside K/V in the capture pass.
    GQA-aware: Q typically has more heads than ``ref_q``/``ref_k`` only if the
    reference capture used a different head count (it doesn't for Krea2 --
    ref_q has the same H_q as q); see ``adain``'s head-broadcast for the
    (arch-agnostic) case where they differ."""
    if strength <= 0.0:
        return q, k
    q_aligned = adain(q, ref_q)
    k_aligned = adain(k, ref_k)
    q_out = q * (1.0 - strength) + q_aligned * strength
    k_out = k * (1.0 - strength) + k_aligned * strength
    return q_out, k_out


# ---------------------------------------------------------------------------
# Controlled reference Value
# ---------------------------------------------------------------------------

def make_ref_value(
    target_v_img: torch.Tensor,
    ref_v_raw: torch.Tensor,
    value_mode: str,
    value_adain_strength: float,
    ref_value_mix: float,
) -> torch.Tensor:
    """Build the reference Value actually injected, blending between "target's
    own Value AdaIN-aligned to the reference" (``value_mode="target_adain"``,
    reduces harsh color/texture pop-in) and "raw reference Value"
    (``value_mode="ref_raw"``), then mixing that base with the raw reference
    Value by ``ref_value_mix`` (default 1.0 == use raw reference Value
    entirely, matching the reference node's default behavior)."""
    if value_mode == "target_adain":
        base = target_v_img * (1.0 - value_adain_strength) + adain(target_v_img, ref_v_raw) * value_adain_strength
    elif value_mode == "ref_raw":
        base = ref_v_raw
    else:
        raise ValueError(f"Unknown value_mode: {value_mode!r} (expected 'target_adain' or 'ref_raw')")
    return base * (1.0 - ref_value_mix) + ref_v_raw * ref_value_mix


# ---------------------------------------------------------------------------
# KV injection
# ---------------------------------------------------------------------------

def inject_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    ref_k: torch.Tensor,
    ref_v: torch.Tensor,
    img_start: int,
    img_end: int,
    ref_k_strength: float,
    freq_scale_vec: torch.Tensor,
    adain_strength: float,
    q: Optional[torch.Tensor] = None,
    ref_q: Optional[torch.Tensor] = None,
):
    """Concatenate the (scaled) reference K/V onto the END of the target's own
    K/V sequence, and AdaIN-align the target's own Q (toward ``ref_q``) / K
    (toward ``ref_k``) (image-token region only).

    True no-op fast path: when ``ref_k_strength == 0`` AND ``adain_strength ==
    0`` the style transfer contributes nothing (scaled ref-K would be exactly
    zero and Q/K would be untouched) -- in that case we skip the concat
    entirely and return ``k``/``v``/``q`` UNCHANGED, so a disabled/zeroed
    style is bit-identical to no injection at all (no extra zero-Key columns
    inflating the softmax denominator).

    Shapes (BSHD): ``k``/``v``/``q`` are ``[B, S, H_kv or H_q, D]`` (the full
    text+image sequence); ``ref_k``/``ref_v``/``ref_q`` are
    ``[1, S_img, H_kv or H_q, D]`` (image tokens only, already
    post-norm/post-RoPE from the capture forward). GQA note: concatenation
    happens on the KV-head tensors BEFORE ``dispatch_attention``'s
    ``enable_gqa`` broadcast, so no manual head-repeat is needed here -- the
    conduit's native SDPA path repeats KV heads for us.

    ``ref_k``/``ref_v``'s batch dim (1) broadcasts against ``k``'s batch dim
    (B) via ``expand`` when B > 1.

    Returns ``(k_out, v_out)`` or ``(k_out, v_out, q_out)`` if ``q`` was given.
    """
    if ref_k_strength == 0.0 and adain_strength <= 0.0:
        if q is not None:
            return k, v, q
        return k, v

    batch = k.shape[0]

    img_q = q[:, img_start:img_end] if q is not None else None
    img_k = k[:, img_start:img_end]

    if adain_strength > 0.0:
        if img_q is not None and ref_q is not None:
            img_q, img_k = cross_batch_adain_qk(img_q, img_k, ref_q, ref_k, adain_strength)
        elif img_q is not None:
            # No reference Query captured (arch/caller opted out): fall back to
            # aligning Q against ref_k as the sole available anchor.
            img_q, img_k = cross_batch_adain_qk(img_q, img_k, ref_k, ref_k, adain_strength)
        else:
            _, img_k = cross_batch_adain_qk(img_k, img_k, ref_k, ref_k, adain_strength)

    if q is not None and img_q is not None:
        q = torch.cat([q[:, :img_start], img_q, q[:, img_end:]], dim=1)
    k = torch.cat([k[:, :img_start], img_k, k[:, img_end:]], dim=1)

    scaled_ref_k = ref_k * freq_scale_vec.view(1, 1, 1, -1) * ref_k_strength
    if scaled_ref_k.shape[0] != batch:
        scaled_ref_k = scaled_ref_k.expand(batch, -1, -1, -1)
    if ref_v.shape[0] != batch:
        ref_v = ref_v.expand(batch, -1, -1, -1)

    k_out = torch.cat([k, scaled_ref_k], dim=1)
    v_out = torch.cat([v, ref_v], dim=1)

    if q is not None:
        return k_out, v_out, q
    return k_out, v_out


# ---------------------------------------------------------------------------
# Multi-reference KV injection ("stack" / "common_concept" combine modes)
# ---------------------------------------------------------------------------

# One active reference's already-finalized per-block state, as produced by
# ``StyleContext.collect_block_refs``: ``(ref_k, ref_v, ref_q, ref_k_strength,
# adain_strength, freq_scale_vec)``. ``ref_v`` is expected to already be the
# FINAL value to inject (i.e. ``make_ref_value`` has already been applied by
# the caller against this forward's own target Value) so both combine modes
# below only need to combine/concat tensors, mirroring exactly what a
# single-ref ``inject_kv`` call receives.
BlockRef = Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float, float, torch.Tensor]


def inject_kv_multi(
    k: torch.Tensor,
    v: torch.Tensor,
    q: Optional[torch.Tensor],
    img_start: int,
    img_end: int,
    block_refs: List[BlockRef],
    combine_mode: str,
):
    """Multi-reference generalization of ``inject_kv``.

    ``combine_mode="common_concept"``: average every active ref's K/V/Q (and
    ``ref_k_strength``/``adain_strength``) into ONE consensus reference, then
    perform a SINGLE ``inject_kv`` call with the averaged tensors -- shared
    directions across refs survive the mean, idiosyncratic per-ref directions
    partially cancel.

    ``combine_mode="stack"``: AdaIN the target Q/K ONCE toward the MEAN
    reference (avoids compounding N sequential AdaIN passes), then append
    EACH ref's own (independently) scaled K/V as its own extra block of
    image-token columns, so the target attends to every reference's tokens
    simultaneously with its own ``ref_k_strength``/frequency curve.

    For ``len(block_refs) == 1`` BOTH modes are mathematically identical to a
    plain ``inject_kv`` call with that ref's tensors (this is asserted by the
    scratch verification script, not by this function -- callers should
    prefer the single-ref ``inject_kv`` fast path directly and only reach
    this function when there are 2+ refs).
    """
    if not block_refs:
        if q is not None:
            return k, v, q
        return k, v

    if combine_mode == "common_concept":
        ref_k = torch.stack([r[0] for r in block_refs], dim=0).mean(dim=0)
        ref_v = torch.stack([r[1] for r in block_refs], dim=0).mean(dim=0)
        if all(r[2] is not None for r in block_refs):
            ref_q = torch.stack([r[2] for r in block_refs], dim=0).mean(dim=0)
        else:
            ref_q = None
        strength = sum(r[3] for r in block_refs) / len(block_refs)
        adain_strength = sum(r[4] for r in block_refs) / len(block_refs)
        freq_vec = block_refs[0][5]
        return inject_kv(
            k, v, ref_k, ref_v, img_start, img_end,
            strength, freq_vec, adain_strength, q=q, ref_q=ref_q,
        )

    if combine_mode == "stack":
        # A ref whose ref_k_strength==0 AND adain_strength<=0 contributes
        # nothing (same no-op test as inject_kv's own fast path) -- drop it
        # rather than appending a zero-scaled (but still softmax-visible)
        # extra K/V block.
        active = [r for r in block_refs if not (r[3] == 0.0 and r[4] <= 0.0)]
        if not active:
            if q is not None:
                return k, v, q
            return k, v

        batch = k.shape[0]
        img_q = q[:, img_start:img_end] if q is not None else None
        img_k = k[:, img_start:img_end]

        mean_ref_k = torch.stack([r[0] for r in active], dim=0).mean(dim=0)
        if all(r[2] is not None for r in active):
            mean_ref_q = torch.stack([r[2] for r in active], dim=0).mean(dim=0)
        else:
            mean_ref_q = None
        max_adain = max(r[4] for r in active)

        if max_adain > 0.0:
            if img_q is not None and mean_ref_q is not None:
                img_q, img_k = cross_batch_adain_qk(img_q, img_k, mean_ref_q, mean_ref_k, max_adain)
            elif img_q is not None:
                img_q, img_k = cross_batch_adain_qk(img_q, img_k, mean_ref_k, mean_ref_k, max_adain)
            else:
                _, img_k = cross_batch_adain_qk(img_k, img_k, mean_ref_k, mean_ref_k, max_adain)

        if q is not None and img_q is not None:
            q = torch.cat([q[:, :img_start], img_q, q[:, img_end:]], dim=1)
        k = torch.cat([k[:, :img_start], img_k, k[:, img_end:]], dim=1)

        # Per-ref strength normalization: stacking appends N reference blocks of
        # image-token columns, so at full per-ref strength the target attends to
        # N x as many reference tokens as a single ref and the reference mass
        # overwhelms the target's own content (empirically: 2-3 stacked refs at
        # default 0.75 collapse the target into an abstract blob). Divide each
        # ref's strength by N so the TOTAL appended reference influence stays
        # comparable to a single-ref injection; the relative weighting between
        # refs (their own ref_k_strength ratios) is preserved. For N==1 this is a
        # 1.0 factor -> identical to inject_kv (byte-identity property intact).
        strength_norm = 1.0 / len(active)

        k_out, v_out = k, v
        for ref_k_i, ref_v_i, _ref_q_i, strength_i, _adain_i, freq_i in active:
            scaled_ref_k = ref_k_i * freq_i.view(1, 1, 1, -1) * (strength_i * strength_norm)
            if scaled_ref_k.shape[0] != batch:
                scaled_ref_k = scaled_ref_k.expand(batch, -1, -1, -1)
            ref_v_i_b = ref_v_i if ref_v_i.shape[0] == batch else ref_v_i.expand(batch, -1, -1, -1)
            k_out = torch.cat([k_out, scaled_ref_k], dim=1)
            v_out = torch.cat([v_out, ref_v_i_b], dim=1)

        if q is not None:
            return k_out, v_out, q
        return k_out, v_out

    raise ValueError(f"Unknown combine_mode: {combine_mode!r} (expected 'stack' or 'common_concept')")


# ---------------------------------------------------------------------------
# Runtime context (capture / inject) shared by the per-block attention hook
# ---------------------------------------------------------------------------

class StyleContext:
    """Per-forward-pass runtime state threaded through the transformer's
    attention blocks. ``mode="capture"`` stashes post-norm/post-RoPE image-token
    K/V per block into ``store``; ``mode="inject"`` reads ``store`` (populated
    by a prior capture forward on the SAME context's ``store`` dict) and
    performs the injection. ``img_start``/``img_end`` are set once per forward
    by the arch-specific transformer (they depend on that call's text sequence
    length) -- NOT per-block.

    Multi-reference (N-ref) support: when ``refs`` is left ``None`` (default)
    the context behaves EXACTLY as before -- single-ref ``mode``/``config``/
    ``store`` drive ``active_for_block`` and the arch hook reads ``store``
    directly and calls ``inject_kv``. When ``refs`` is a non-empty list of
    ``(per_ref_store, per_ref_config)`` tuples (one entry per reference image,
    each with its OWN ``StyleTransferConfig``), the arch hook should instead
    call ``collect_block_refs`` + ``inject_kv_multi`` with ``combine_mode``.
    ``config``/``store`` are ignored by ``inject_kv_multi``-based callers in
    this mode but are still required by the constructor (a harmless unused
    placeholder, e.g. the first ref's config) since ``StyleContext`` is a
    single dataclass-like carrier for both paths."""

    __slots__ = ("mode", "config", "store", "img_start", "img_end", "progress", "refs", "combine_mode")

    def __init__(
        self, mode: str, config: StyleTransferConfig, store: Optional[Dict[int, Any]] = None,
        progress: float = 0.0,
        refs: Optional[List[Tuple[Dict[int, Any], StyleTransferConfig]]] = None,
        combine_mode: str = "stack",
    ):
        if mode not in ("capture", "inject"):
            raise ValueError(f"StyleContext.mode must be 'capture' or 'inject', got {mode!r}")
        self.mode = mode
        self.config = config
        self.store: Dict[int, Any] = store if store is not None else {}
        self.img_start: Optional[int] = None
        self.img_end: Optional[int] = None
        # Denoise progress (0 at first step -> 1 at last) for THIS forward call;
        # drives the step-dependent frequency-scale curve (fix #2). Only
        # meaningful in "inject" mode (capture doesn't scale anything).
        self.progress: float = progress
        # Multi-ref state (see class docstring). None => single-ref (unchanged
        # behavior); a non-empty list => multi-ref inject via
        # collect_block_refs()/inject_kv_multi().
        self.refs: Optional[List[Tuple[Dict[int, Any], StyleTransferConfig]]] = refs
        self.combine_mode: str = combine_mode

    def active_for_block(self, block_idx: int) -> bool:
        if self.refs is None:
            return self.config.is_block_active(block_idx)
        return any(cfg.is_block_active(block_idx) for _store, cfg in self.refs)

    def collect_block_refs(
        self, block_idx: int, target_v_img: torch.Tensor, device: torch.device, dtype: torch.dtype,
    ) -> list:
        """Multi-ref only (``self.refs`` must be set): for every
        ``(store_i, config_i)`` where ``config_i.is_block_active(block_idx)``
        AND ``block_idx`` was actually captured into ``store_i`` (a ref's
        capture forward may skip blocks outside its own active range),
        compute that ref's per-block state and return it as a
        ``reference_style.BlockRef`` tuple ready for ``inject_kv_multi``:
        ``(ref_k, ref_v, ref_q, ref_k_strength, adain_strength,
        freq_scale_vec)``.

        ``ref_v`` is already the FINAL value to inject -- ``make_ref_value``
        is applied here (per-ref ``value_mode``/``value_adain_strength``/
        ``ref_value_mix``, against the CURRENT forward's own ``target_v_img``)
        so ``inject_kv_multi`` only has to combine/concat tensors, exactly
        mirroring what a single-ref hook does before calling ``inject_kv``.

        The frequency-scale vector is derived the same way a single-ref hook
        would: ``config_i.get_freq_scale_vector(head_dim, self.progress,
        device, dtype)`` when ``config_i.axes_dims`` was set by the arch
        wiring, and an all-ones vector otherwise (mirrors archs like Anima
        whose RoPE layout is incompatible with the interleave-real frequency
        curve and always use an all-ones vector regardless of ``axes_dims``).
        ``head_dim`` is read from the captured ``ref_k`` tensor's last
        dimension so no extra arch-specific parameter is needed here.

        Returns ``[]`` when ``self.refs`` is None/empty or no ref is active
        for this block."""
        if not self.refs:
            return []
        out = []
        for store_i, config_i in self.refs:
            if not config_i.is_block_active(block_idx):
                continue
            entry = store_i.get(block_idx)
            if entry is None:
                continue
            ref_q, ref_k, ref_v_raw = entry
            head_dim = ref_k.shape[-1]
            if config_i.axes_dims is not None:
                freq_vec = config_i.get_freq_scale_vector(head_dim, self.progress, device, dtype)
            else:
                freq_vec = torch.ones(head_dim, device=device, dtype=dtype)
            ref_v_final = make_ref_value(
                target_v_img, ref_v_raw, config_i.value_mode, config_i.value_adain_strength, config_i.ref_value_mix,
            )
            out.append((ref_k, ref_v_final, ref_q, config_i.ref_k_strength, config_i.adain_strength, freq_vec))
        return out


# ---------------------------------------------------------------------------
# Config construction from a plain params dict (API/frontend boundary)
# ---------------------------------------------------------------------------

def style_config_from_dict(d: Dict[str, Any]) -> StyleTransferConfig:
    """Build a ``StyleTransferConfig`` from the plain dict assembled by
    ``generation_utils.process_controlnet_configs`` (arch-agnostic; does not
    set ``axes_dims`` -- the arch wiring must call
    ``config.axes_dims = transformer_axes_dims`` before first use)."""

    def _block_range(raw) -> Optional[Tuple[int, int]]:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return int(raw[0]), int(raw[1])
        if isinstance(raw, str) and raw.strip():
            parts = raw.split("-")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        return None

    return StyleTransferConfig(
        transfer_type=str(d.get("transfer_type", "style") or "style"),
        ref_k_strength=float(d.get("ref_k_strength", 0.75) if d.get("ref_k_strength") is not None else 0.75),
        block_range=_block_range(d.get("block_range")),
        adain_strength=float(d.get("adain_strength", 0.6) if d.get("adain_strength") is not None else 0.6),
        high_scale_start=float(d.get("high_scale_start", 1.04) if d.get("high_scale_start") is not None else 1.04),
        high_scale_end=float(d.get("high_scale_end", 0.0) if d.get("high_scale_end") is not None else 0.0),
        low_scale_start=float(d.get("low_scale_start", 1.0) if d.get("low_scale_start") is not None else 1.0),
        low_scale_end=float(d.get("low_scale_end", 1.10) if d.get("low_scale_end") is not None else 1.10),
        beta=float(d.get("beta", 2.5) if d.get("beta") is not None else 2.5),
        value_mode=str(d.get("value_mode", "target_adain") or "target_adain"),
        value_adain_strength=float(d.get("value_adain_strength", 0.65) if d.get("value_adain_strength") is not None else 0.65),
        ref_value_mix=float(d.get("ref_value_mix", 1.0) if d.get("ref_value_mix") is not None else 1.0),
        late_release=float(d.get("late_release", 0.0) or 0.0),
        rope_offset=int(d.get("rope_offset", 0) or 0),
        start_step=int(d.get("start_step", 0) or 0),
        end_step=int(d.get("end_step", 1000) if d.get("end_step") is not None else 1000),
        style_guidance_scale=(float(d["style_guidance_scale"]) if d.get("style_guidance_scale") is not None else None),
    )
