"""Two-pass global-GroupNorm statistics for a tiled VAE decode (Phase 4A-2).

WHY THIS EXISTS
---------------
A tiled decode splits the latent, and every ``nn.GroupNorm`` inside the decoder
then normalises each tile with THAT TILE's own per-group mean/variance instead of
the whole image's. The result is a per-tile TINT: a spatially coherent, whole-tile
colour offset. ``context_tiled_decode.py`` (Phase 4A-1) removed the boundary term
but explicitly left this one; its docstring names the fix as "a statistics pass
over the tiles and a forced-statistics pass during each tile decode". This is that
fix.

Pass 1 runs today's tiled decode unchanged while RECORDING each GroupNorm's
per-group sum / sumsq / count across every tile call. Pass 2 re-runs the same
decode while FORCING the accumulated whole-image statistics. Because the two
passes wrap the WHOLE decode, this is mode-agnostic: pass 1 and pass 2 each run
whatever tiled decode is installed (diffusers' internal blend tiling, or 4A-1's
context loop) with no knowledge of the tile geometry.

MEASURED (2026-07-28, 3 images x 2 budgets x 2 join modes x {sdxl, flux1, qwen}
in fp32, plus the full check matrix re-run in fp16 and bf16 on sdxl; write-up in
an untracked scratch directory, so the numbers are restated here rather than
referenced):

* Per-tile tint peak-to-peak on SDXL, blend join, 512px budget:
  **1.32 -> 0.037 /255** (35x) in fp32, and **1.35 -> 0.038 (35x) in fp16** --
  the dtype production actually runs. Across 24 fp32 cells the two-pass result
  recovers a median ~91 % of the tint gap to an exact whole-image statistics
  transplant, and 47-89 % of the whole-image mean gap. 24/24 cells improve; none
  regress.
* **bf16 is materially weaker: 1.36 -> 0.18 (7.4x) at the same setting**, and
  2.6x-16x across the four sdxl cells (still an improvement in every cell, no
  regressions). bf16 has 8 mantissa bits, and ``F.group_norm`` requires the
  folded weight/bias to be cast to the activation dtype, so the correction
  itself is quantised at ~4e-3 relative -- that, not the statistics, is the
  floor. Stated because bf16 is what several architectures decode in.
* Peak VRAM +0.00003 GB, i.e. only the retained state (~23-30 KB of scalars:
  30 modules x <=32 groups x 3 fp64 numbers). The tiling memory contract is
  preserved -- that is the entire reason this is two passes over the decode
  rather than a layer-interleaved (Tiled-VAE style) implementation, which
  returns the peak to whole-image scale.
* Wall time is exactly 2.0x the decode -- EVERY decode of the request. On
  SD1.5/SDXL ``_apply_vae_tiling`` runs before the sampling loop, so the
  in-loop ``vae.decode`` calls made by ``flatten_in_loop`` (one per injected
  step) and ``vae_drift_correction`` are doubled as well, not just the final
  decode.

THREE CONSTRAINTS, each of which the measurement made non-negotiable
--------------------------------------------------------------------
1. EXACTLY TWO PASSES. NEVER ITERATE. See the "DO NOT ADD AN 'ITERATIONS'
   PARAMETER" comment block below the imports.
2. Skip entirely when the decoder contains no ``nn.GroupNorm``. On the Qwen-family
   autoencoder (Anima / Krea2 -- RMSNorm over channels, ZERO GroupNorms) every
   arm was bit-exact identical to the plain decode while still costing 2x decode
   time (3.03 s -> 6.05 s for a byte-identical image). ``supports_global_group_norm``
   is that gate.
3. Apply the forced statistics by FOLDING them into ``F.group_norm``'s own
   per-channel weight/bias and calling the fused kernel once. See ``_gn_hook``.

SCOPE
-----
Anything whose ``decode`` runs a decoder containing ``nn.GroupNorm``: diffusers
``AutoencoderKL`` (SD1.5 / SDXL / Z-Image / Lens / Ideogram4) and
``AutoencoderKLFlux2``. ``AutoencoderKLQwenImage`` has none and is skipped by the
gate. Wrapper objects (SDXLVAEWrapper / FluxVAEWrapper / PidVaeWrapper) have no
``.decoder`` of their own and are rejected, exactly as in
``context_tiled_decode.supports_context_tiling`` -- the install lands on the inner
autoencoder instead.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.inference.context_tiled_decode import (
    _MARK_ATTR as _CTX_MARK_ATTR,
    _as_tensor,
    _wrap_result,
    spatial_compression_of,
)

# ---------------------------------------------------------------------------
# DO NOT ADD AN "ITERATIONS" PARAMETER. READ THIS FIRST.
# ---------------------------------------------------------------------------
# Two-pass is ONE step of a fixed-point iteration ("decode under statistics S,
# re-accumulate S"). It is tempting to iterate to convergence. That was measured,
# and it DIVERGES.
#
# MEASURED, SDXL, whole-image mean |delta| /255 against an un-tiled decode:
#
#   budget/mode      2 passes   3 passes   4 passes   (exact-transplant ceiling)
#   1024 blend         0.285      0.433      3.094              0.149
#   1024 context       0.247      0.262      2.400              0.158
#    512 blend         0.440      5.642     14.466              0.274
#    512 context       0.445      3.336     11.057              0.260
#
# At 4 passes, SDXL @512 reaches mean 14.47 and max 357.75 /255 -- far worse than
# not correcting at all. A third pass is worse than the second in 5 of 6 SDXL
# cells measured across 3 images. On FLUX.1 the iteration is stable but converges
# to a BIASED point ~10-17x above the ceiling, so extra passes buy nothing there.
#
# The reason (inferred, but tightly constrained by the data): the fixed point of
# that map is not the whole-image statistics, because the union of per-tile
# activation fields is not the whole-image activation field. Iterating therefore
# walks toward a wrong attractor, and 30 GroupNorms in series is a composition
# whose gain exceeds 1 on SDXL. It is NOT blend's overlapping tiles
# double-counting the shared band: the zero-overlap `context` mode diverges just
# as hard.
#
# So the pass count is a constant 2. A user-facing "iterations" dial would be a
# footgun that produces a worse image at 3 and a broken one at 4.
# ---------------------------------------------------------------------------

# Instance-attribute names for the decode override, mirroring the
# ``_sushi_ctx_*`` convention in context_tiled_decode.py.
_ORIG_ATTR = "_sushi_gn_orig_decode"
_ORIG_IN_DICT_ATTR = "_sushi_gn_orig_in_dict"
_MARK_ATTR = "_sushi_gn_global"


# ---------------------------------------------------------------------------
# discovery / gating
# ---------------------------------------------------------------------------

def decoder_group_norms(vae) -> list:
    """``(name, module)`` of every ``nn.GroupNorm`` inside ``vae.decoder``."""
    dec = getattr(vae, "decoder", None)
    if dec is None or not isinstance(dec, nn.Module):
        return []
    return [(n, m) for n, m in dec.named_modules() if isinstance(m, nn.GroupNorm)]


def supports_global_group_norm(vae) -> bool:
    """True for objects whose own ``decode`` runs a decoder containing GroupNorms.

    The ``len(groupnorms) > 0`` half of this is CONSTRAINT 2 from the module
    docstring, not an optimisation: on a GroupNorm-free decoder the second pass
    is provably a bit-exact no-op that still costs a full extra decode. It is
    simply not applicable there, so it is skipped silently (no warning).
    """
    if vae is None:
        return False
    if not (callable(getattr(vae, "decode", None))
            and getattr(vae, "decoder", None) is not None
            and getattr(vae, "config", None) is not None):
        return False
    return len(decoder_group_norms(vae)) > 0


def _tiling_engaged(vae, z: torch.Tensor, threshold_px: int,
                    inner_is_context: bool) -> bool:
    """True if this decode will actually be split into more than one tile.

    CONSTRAINT 3 of the recipe: if the latent already fits the budget there is a
    single tile, the accumulated statistics ARE that tile's own, and pass 2 is a
    no-op up to float rounding. Running it anyway would break 4A-1's
    below-threshold bit-identity guarantee and cost 2x for nothing.

    The real trigger has TWO halves and both are checked here, because the
    "tiling is on" half can be false even though this override is installed:
    ``_apply_vae_tiling`` turns diffusers' ``use_tiling`` OFF for context mode,
    and if the context install then fails (or ``enable_tiling`` itself raised)
    the decode is a single WHOLE decode. Running two of those corrects nothing
    and doubles a decode that was never tiled.
      * ``AutoencoderKL._decode``:          ``use_tiling and lat > tile_latent_min_size``
      * ``AutoencoderKLQwenImage._decode``: ``use_tiling and lat > tile_sample_min_h // ratio``
      * ``context_tiled_decode``:           tiles unless ``lat <= budget_cells``
    ``inner_is_context`` is captured at install time rather than probed here:
    once this override is installed it is the object in ``vae.__dict__["decode"]``,
    so ``is_context_tiled(vae)`` would look at THIS wrapper and answer False.

    ``threshold_px`` is the resolved decode budget the caller already computed
    (``tile_sample_min_size`` / ``tile_sample_min_height``), so the extent half
    is one formula for all three paths: ``lat > threshold_px // scale``.
    """
    if threshold_px <= 0 or z.ndim < 2:
        return False
    if not (inner_is_context or bool(getattr(vae, "use_tiling", False))):
        return False
    scale = max(1, spatial_compression_of(vae))
    budget_cells = int(threshold_px) // scale
    if budget_cells <= 0:
        return False
    return int(z.shape[-2]) > budget_cells or int(z.shape[-1]) > budget_cells


# ---------------------------------------------------------------------------
# accumulator
# ---------------------------------------------------------------------------

class _Accum:
    """Per-(module, group) sum / sumsq / count, accumulated across tile decodes.

    Stored as sum/sumsq rather than a running mean so tiles of DIFFERENT sizes
    combine correctly -- edge tiles are smaller, and blend's overlap band is
    decoded twice. ``mean = sum/n``, ``var = sumsq/n - mean**2``.

    Widened to fp64 only AFTER the reduction: a ``.double()`` copy of the
    activation would double the transient, which is the one thing this whole
    feature exists to avoid.
    """

    __slots__ = ("d",)

    def __init__(self):
        self.d = {}

    def add(self, name: str, mu: torch.Tensor, var: torch.Tensor, n: int) -> None:
        """``mu``/``var``: ``[B, G]`` this call's per-group moments.
        ``n``: elements per group in this call."""
        mu64 = mu.double()
        s = mu64 * n
        ss = (var.double() + mu64 ** 2) * n
        e = self.d.get(name)
        if e is None:
            self.d[name] = [s, ss, float(n)]
        else:
            e[0] = e[0] + s
            e[1] = e[1] + ss
            e[2] = e[2] + float(n)

    def stats(self) -> dict:
        """-> ``{name: (mean[B, G], var[B, G])}`` as float32."""
        out = {}
        for name, (s, ss, n) in self.d.items():
            mean = s / n
            var = (ss / n) - mean ** 2
            out[name] = (mean.float(), var.clamp_min(0).float())
        return out


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------

@contextmanager
def _gn_hook(mods, record: "Optional[_Accum]" = None,
             force: "Optional[dict]" = None):
    """Temporarily replace each GroupNorm's ``forward``.

    ``record``: accumulate the input activation's per-group statistics. A pass
    that only records is a BIT-EXACT no-op (measured: ``torch.equal`` True and
    max deviation exactly 0.0 in all 28 cells tested) -- pass 1's output *is*
    today's output, so the statistics carry no observer effect.

    ``force``: normalise with the supplied ``(mean, var)`` instead of the
    module's own reduction, then apply the module's weight/bias.

    HOW THE FORCING IS DONE, AND WHY IT MUST STAY THIS WAY
    -----------------------------------------------------
    ``F.group_norm`` always subtracts the tile's OWN ``(mu_t, sd_t)``. But its
    weight/bias are per-CHANNEL and free, so the forced result is obtained from
    the UNMODIFIED FUSED KERNEL by re-expressing it:

        (x - mu_g)/sd_g * w + b
          = [(x - mu_t)/sd_t] * (w * sd_t/sd_g)  +  (b + w*(mu_t - mu_g)/sd_g)
          = F.group_norm(x, G, w', b')    with  w' = w * sd_t/sd_g
                                                b' = b + w*(mu_t - mu_g)/sd_g

    so forcing costs one extra per-CHANNEL vector and NOTHING at activation size.

    *** DO NOT "SIMPLIFY" THIS INTO EXPLICIT ELEMENTWISE OPS. *** Writing the
    naive, mathematically identical form ``(x - mu)/sd * w + b`` MEASURED
    **+1.76 GB peak VRAM** on a single SDXL 1024px tile (3.32 -> 5.08 GB) purely
    as transients of the unfused arithmetic -- more memory than tiling saves,
    i.e. it silently destroys the feature. ``torch.addcmul``, ``mul().add_()``
    and ``empty_like`` were all measured and were all equally bad; per-GroupNorm
    allocation checkpoints were identical between the two implementations at all
    120 call sites, so it is an intra-op transient, not a retained tensor.

    DTYPE HANDLING (the production VAE runs in fp16/bf16, not fp32)
    ---------------------------------------------------------------
    Every statistic and every step of the fold is computed in fp32 (accumulated
    in fp64 across tiles), and only the finished ``w'``/``b'`` are cast down to
    ``x.dtype`` for the kernel call. ``F.group_norm`` requires weight/bias to
    match the input dtype -- an fp32 ``w'`` against an fp16 ``x`` raises
    ``RuntimeError: expected scalar type Half but found Float``, which is exactly
    how this was found: fp32-only offline verification and a code audit both
    passed while the feature could not execute at all on the live backend.

    The reductions are written as ``mean(dtype=torch.float32)`` plus
    ``linalg.vector_norm(..., dtype=torch.float32)`` rather than the obvious
    ``xg.var()``/``(xg*xg).mean()``, for three MEASURED reasons:
      * ``xg.float()`` first would materialise a second, wider copy of the
        activation: +512 MB on one [1,512,512,512] fp16 tile. That is the memory
        contract this feature exists to protect.
      * ``xg * xg`` squares IN fp16 and overflows: a constant-300 fp16 tensor
        gives ``inf``; the ``dtype=`` form gives the correct 90000. SDXL VAEs are
        known to be fp16-marginal, so this is not hypothetical.
      * ``torch.var_mean`` is stable and copy-free but RETURNS fp16, i.e. ~4.5e-4
        relative error on the very statistics being corrected. The ``dtype=``
        form measured 5e-6 (mu) / 1.4e-7 (var) against an fp64 reference.
    ``dtype=torch.float64`` was also tried and is NOT usable: on CUDA it
    materialises the widened copy (+1024 MB measured).

    ``var = E[x^2] - mu^2`` can cancel when ``mu >> sd``. MEASURED over every
    GroupNorm call of a real tiled SDXL decode, ``max(mu^2/var) = 7.8`` (same in
    fp16/bf16/fp32), so the amplification is ~8x on an fp32 relative error of
    ~1e-6. Re-check this if the module is ever pointed at a different decoder.

    Both flags may be active at once; that is what makes pass 2 cost exactly one
    extra decode rather than two.
    """
    # Populated BEFORE anything is patched, so the finally below can restore a
    # PARTIAL install too -- see the try: that the install loop lives inside.
    saved = []
    for name, m in mods:
        saved.append((m, "forward" in m.__dict__, m.__dict__.get("forward")))

    try:
        for name, m in mods:
            def _make(mm=m, nm=name):
                def _forward(x):
                    B, C = x.shape[0], x.shape[1]
                    G = mm.num_groups
                    forcing = force is not None and nm in force
                    mu_t = var_t = None
                    if record is not None or forcing:
                        # reshape() COPIES when the activation is channels_last (one
                        # SDXL decoder GroupNorm receives such an input), so only pay
                        # for it when the reduction is actually needed.
                        xg = x.reshape(B, G, -1)
                        n_elem = xg.shape[-1]
                        # fp32 accumulation for EVERY input dtype, with no widened
                        # copy of the activation and no fp16 squaring -- see the
                        # DTYPE HANDLING section of this docstring.
                        mu_t = xg.mean(-1, dtype=torch.float32)
                        nrm = torch.linalg.vector_norm(
                            xg, 2, dim=-1, dtype=torch.float32)
                        var_t = (nrm * nrm / n_elem - mu_t * mu_t).clamp_min(0)
                        del xg, nrm
                        if record is not None:
                            record.add(nm, mu_t, var_t, n_elem)
                    if not forcing:
                        return F.group_norm(x, G, mm.weight, mm.bias, mm.eps)

                    mu_g, var_g = force[nm]                    # fp32 by construction
                    if mu_g.shape[0] != B:
                        mu_g = mu_g.expand(B, -1)
                        var_g = var_g.expand(B, -1)
                    rep = C // G
                    # The fold runs entirely in fp32; the module's own affine
                    # parameters follow x.dtype (fp16/bf16 in production) and are
                    # widened for it.
                    w = (mm.weight.detach().float().view(1, C)
                         if mm.weight is not None
                         else torch.ones(1, C, dtype=torch.float32, device=x.device))
                    b = (mm.bias.detach().float().view(1, C)
                         if mm.bias is not None
                         else torch.zeros(1, C, dtype=torch.float32, device=x.device))
                    sd_t = torch.sqrt(var_t + mm.eps).repeat_interleave(rep, dim=1)
                    inv_g = torch.rsqrt(var_g + mm.eps).repeat_interleave(rep, dim=1)
                    dmu = (mu_t - mu_g).repeat_interleave(rep, dim=1)
                    wp = w * sd_t * inv_g                      # [B, C] fp32
                    bp = b + w * dmu * inv_g                   # [B, C] fp32
                    # Cast down only at the call: F.group_norm demands
                    # weight/bias.dtype == x.dtype.
                    wp = wp.to(x.dtype)
                    bp = bp.to(x.dtype)
                    if B == 1:
                        return F.group_norm(x, G, wp[0], bp[0], mm.eps)
                    # F.group_norm's weight/bias have no batch dimension, and the
                    # folded values are per-sample. This branch is a safety net that
                    # should be unreachable: two_pass_global_group_norm_decode splits
                    # a B > 1 latent and recurses per sample, precisely so that each
                    # image gets its OWN global statistics. (Do NOT assume diffusers'
                    # `use_slicing` makes B > 1 safe -- it splits the decode into B
                    # calls of B == 1, which would make one accumulator hold the
                    # batch-average and force it onto every image.) It still calls
                    # only the fused kernel -- the transient stays at one sample,
                    # never the naive elementwise form warned about above.
                    out = torch.empty_like(x)
                    for i in range(B):
                        out[i:i + 1] = F.group_norm(
                            x[i:i + 1], G, wp[i], bp[i], mm.eps)
                    return out
                return _forward
            m.forward = _make()

        yield
    finally:
        # A leaked hook would corrupt EVERY later decode in the process, which is
        # far worse than the artifact this module removes. Hence the finally, and
        # hence restoring the exact prior state (usually: no instance attribute at
        # all, so the class method is reached again) rather than assigning a bound
        # method back.
        for m, had_own, prior in saved:
            if had_own:
                m.__dict__["forward"] = prior
            else:
                m.__dict__.pop("forward", None)


# ---------------------------------------------------------------------------
# the two-pass decode
# ---------------------------------------------------------------------------

def two_pass_global_group_norm_decode(
    vae,
    z: torch.Tensor,
    inner_decode,
    threshold_px: int,
    inner_is_context: bool = False,
    return_dict: bool = True,
    **decode_kwargs,
):
    """Run ``inner_decode`` twice: once recording, once forcing.

    ``inner_decode`` is whatever decode was installed underneath this override --
    the VAE's own bound ``decode`` (diffusers blend tiling) or 4A-1's
    context-tiled override. This function never inspects the tile geometry, which
    is exactly why it composes with both.

    Every path that is not "a real tiled decode of a still-image latent through a
    GroupNorm-bearing decoder" falls through to a single ordinary decode, so the
    flag can never change a below-threshold result.
    """
    if not isinstance(z, torch.Tensor) or z.ndim not in (4, 5):
        return inner_decode(z, return_dict=return_dict, **decode_kwargs)
    if not _tiling_engaged(vae, z, threshold_px, inner_is_context):
        return inner_decode(z, return_dict=return_dict, **decode_kwargs)

    mods = decoder_group_norms(vae)
    if not mods:
        return inner_decode(z, return_dict=return_dict, **decode_kwargs)

    # ---- a batch is B INDEPENDENT images: give each its own statistics ------
    # The accumulator is keyed by module only, so one pass over a batch would
    # pool every image's moments into a single entry and pass 2 would force that
    # batch AVERAGE onto all of them -- normalising the images to each other
    # instead of each to itself. Note that diffusers' `use_slicing` does NOT
    # save us here: it splits the decode into B calls of B == 1, which pool
    # exactly the same way. So split here, before pass 1, and recurse.
    if int(z.shape[0]) > 1:
        out = None
        for i in range(int(z.shape[0])):
            part = _as_tensor(two_pass_global_group_norm_decode(
                vae, z[i:i + 1], inner_decode, threshold_px,
                inner_is_context=inner_is_context, return_dict=False,
                **decode_kwargs))
            if out is None:
                shape = list(part.shape)
                shape[0] = int(z.shape[0])
                out = torch.empty(shape, dtype=part.dtype, device=part.device)
            out[i:i + 1] = part
            # Written into the canvas and dropped, so the peak is one canvas +
            # one sample, not two canvases (which torch.cat would cost).
            del part
        return _wrap_result(out, return_dict)

    # ---- pass 1: today's decode, recording only (bit-exact no-op) ----------
    acc = _Accum()
    with _gn_hook(mods, record=acc):
        first = inner_decode(z, return_dict=False, **decode_kwargs)
    # Drop pass 1's canvas before pass 2 allocates its own, so the peak stays at
    # one canvas + one tile rather than two canvases.
    del first

    stats = acc.stats()
    acc.d.clear()
    if not stats:
        # No GroupNorm was actually reached (unexpected, given the gate above).
        # Returning the second pass unforced keeps behaviour identical to today.
        return inner_decode(z, return_dict=return_dict, **decode_kwargs)

    # ---- pass 2: the same decode, forcing the accumulated statistics -------
    # EXACTLY TWO PASSES. Do not turn this into a loop -- see the DO NOT ADD AN
    # "ITERATIONS" PARAMETER block at the top of this module.
    with _gn_hook(mods, force=stats):
        return inner_decode(z, return_dict=return_dict, **decode_kwargs)


# ---------------------------------------------------------------------------
# install / uninstall on a VAE object
# ---------------------------------------------------------------------------

def install_global_group_norm_decode(vae, threshold_px: int) -> bool:
    """Install (or re-point) the two-pass ``decode`` override on ``vae``.

    Idempotent and reversible: an existing override is removed first, so the
    wrapper never stacks and a later request's threshold replaces an earlier one.

    ORDERING CONTRACT (enforced by ``PipelineManager._apply_vae_tiling``): this
    override wraps whatever ``decode`` is current at install time, so it must be
    installed AFTER the 4A-1 context override and uninstalled BEFORE it. Doing it
    the other way round would let the context module snapshot this wrapper as its
    "original" decode, and the two would stack.
    """
    if not supports_global_group_norm(vae):
        return False

    uninstall_global_group_norm_decode(vae)

    inner_in_dict = "decode" in vae.__dict__
    inner = vae.__dict__["decode"] if inner_in_dict else vae.decode
    # Whether the decode being wrapped is 4A-1's context override. Captured HERE
    # because once this wrapper is in vae.__dict__["decode"], is_context_tiled()
    # would inspect this wrapper and answer False. Used by _tiling_engaged: in
    # context mode diffusers' own use_tiling is off, so the "is tiling on" half
    # of the gate has to come from this flag instead.
    inner_is_context = bool(getattr(inner, _CTX_MARK_ATTR, False))

    def _decode(z, return_dict=True, **kwargs):
        return two_pass_global_group_norm_decode(
            vae, z, inner,
            threshold_px=threshold_px,
            inner_is_context=inner_is_context,
            return_dict=return_dict,
            **kwargs,
        )

    setattr(_decode, _MARK_ATTR, True)
    vae.__dict__[_ORIG_ATTR] = inner
    vae.__dict__[_ORIG_IN_DICT_ATTR] = inner_in_dict
    vae.__dict__["decode"] = _decode
    return True


def uninstall_global_group_norm_decode(vae) -> bool:
    """Restore the decode this override wrapped. Safe to call unconditionally."""
    if vae is None:
        return False
    current = vae.__dict__.get("decode")
    if current is not None and getattr(current, _MARK_ATTR, False):
        inner = vae.__dict__.get(_ORIG_ATTR)
        if vae.__dict__.get(_ORIG_IN_DICT_ATTR) and inner is not None:
            vae.__dict__["decode"] = inner
        else:
            vae.__dict__.pop("decode", None)
    vae.__dict__.pop(_ORIG_ATTR, None)
    vae.__dict__.pop(_ORIG_IN_DICT_ATTR, None)
    return True


def is_global_group_norm(vae) -> bool:
    """True if ``vae`` currently carries this override."""
    if vae is None:
        return False
    return getattr(vae.__dict__.get("decode"), _MARK_ATTR, False)
