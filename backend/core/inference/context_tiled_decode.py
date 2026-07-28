"""Context-padded, discard-margin tiled VAE decode (Phase 4A-1).

WHY THIS EXISTS
---------------
diffusers' own ``tiled_decode`` splits the latent into tiles that OVERLAP by
``tile_overlap_factor`` (0.25) and cross-fades the overlap with a linear blend.
A cross-fade hides a seam; it does not remove the error that causes it. A tile
decoded without its neighbours is genuinely *wrong* near its border, because the
decoder has a finite receptive field and the missing neighbour is replaced by
zero padding.

MEASURED FACTS THIS IS BUILT ON (2026-07-28, fp32, n = 1 image per VAE, on the
SDXL / FLUX.1 / Qwen-Image autoencoders this repo actually loads). Stated inline
rather than by reference, because the measurement write-ups live in an untracked
scratch directory and will not exist in a fresh clone:

* An independently decoded latent crop disagrees with the corresponding region
  of a whole-image decode by 12-25 /255 at the boundary pixel (individual pixels
  peaking at 71-218 /255), decaying to a floor by ~32-48 px inward.
* Giving the tile k latent cells of REAL neighbouring context and DISCARDING
  that margin after the decode extinguishes that boundary term exactly at
  k = 14-16 cells (measured 0.0000 /255 once the two non-local terms below are
  ablated away). k = 8 already reduces it 38x on SDXL and 190x on Qwen.
* Holding the output tile fixed and sweeping the margin, the error in a 16 px
  band around each tile join falls from 5.95 to 0.79 /255 (SDXL) and 3.12 to
  0.051 (Qwen) and, from k = 8 onward, EQUALS the tile-interior error: the join
  is gone, and what is left is not boundary-local.

So: decode with a real-context margin, then throw the margin away. Tiles then
tile the output exactly, with no overlap and no blending.

WHAT IS *NOT* FIXED HERE
------------------------
What remains after the margin is discarded is a whole-tile term: each tile's
GroupNorm normalises with its own per-tile mean/var instead of the whole
image's, tinting the tile. Measured residual 0.4-1.1 /255 on the SDXL-family
decoders (30 GroupNorms each) and 0.05-0.11 /255 on the Qwen-family ones (zero
GroupNorms — the residual there is the mid-block attention). Measured as a
per-tile signed tint, the tiles of a tiled decode differ from each other by up
to 1.8 /255 peak-to-peak on SDXL at a 512px tile.

That term is why ``vae_tile_mode`` defaults to ``"blend"`` and not to this
module: diffusers' cross-fade ramps that tint step across the overlap instead of
stepping it, so at equal threshold blend measures slightly LOWER whole-image
error on the GroupNorm-bearing decoders even though it has no exactness
guarantee at the join. Removing the tint term needs a statistics pass over the
tiles and a forced-statistics pass during each tile decode; ``iter_tiles`` below
is re-iterable specifically so that second pass can be added over the same plan.

SCOPE
-----
Applies to diffusers ``AutoencoderKL`` / ``AutoencoderKLFlux2`` (4-D
``[B, C, h, w]`` latents) and ``AutoencoderKLQwenImage`` (5-D
``[B, C, T, h, w]`` with ``T = 1`` for stills — Anima / Krea2). PiD is a 4-step
pixel-diffusion decoder with its own tiling, not an autoencoder decode, and is
deliberately excluded (see ``PipelineManager._apply_vae_tiling``).
"""

from __future__ import annotations

from typing import Optional

import torch

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# Latent cells of REAL neighbouring context decoded around each output tile and
# discarded afterwards. 16 is the measured exact-extinction point of the
# receptive-field term (it hits 0.0000 /255 at k = 14-16 cells on the SDXL,
# FLUX.1 and Qwen-Image decoders alike; the theoretical receptive field is
# 17.25 cells, so 16 measured is the effective figure, not a guess).
DEFAULT_MARGIN_CELLS = 16

# Smallest output tile worth producing. Below this the tile count explodes.
MIN_OUTPUT_TILE_CELLS = 8

# If the budget is so small that the margin has to shrink below this, a
# context-tiled decode is not worth doing: the margin no longer reaches far
# enough to extinguish the boundary term (measured: k = 4 still leaves the join
# band 16% above the interior on SDXL), and the tile count is at its worst. The
# caller falls back to diffusers' blend tiling instead.
MIN_USEFUL_MARGIN_CELLS = 4


def _warn(message: str, code: str) -> None:
    """Best-effort feature-degradation notice for the current generation.

    Lazily imported so this inference module never hard-depends on the api
    package at import time. Never raises. Mirrors
    ``custom_sampling._add_generation_warning``.
    """
    try:
        from api.generation_status import add_warning
        add_warning(message, code=code)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def spatial_compression_of(vae) -> int:
    """Pixels per latent cell for ``vae``.

    Mirrors the derivation ``PipelineManager._apply_vae_tiling`` already uses:
    the explicit ``spatial_compression_ratio`` when the class exposes one
    (Qwen family), otherwise ``2 ** (len(block_out_channels) - 1)`` (the number
    of upsample stages in a diffusers ``Decoder``), defaulting to 8.
    """
    ratio = getattr(vae, "spatial_compression_ratio", None)
    try:
        ratio = int(ratio or 0)
    except (TypeError, ValueError):
        ratio = 0
    if ratio > 0:
        return ratio
    cfg = getattr(vae, "config", None)
    boc = getattr(cfg, "block_out_channels", None) if cfg is not None else None
    if boc:
        return 2 ** (len(boc) - 1)
    return 8


class TileRect:
    """One output tile plus the padded latent window that produces it.

    ``(y0, y1, x0, x1)`` is the OUTPUT footprint in latent cells; ``(py0, py1,
    px0, px1)`` is the padded window actually pushed through the decoder. The
    margin to discard in pixel space is ``(y0 - py0) * scale`` on the top and
    ``(x0 - px0) * scale`` on the left.
    """

    __slots__ = ("y0", "y1", "x0", "x1", "py0", "py1", "px0", "px1")

    def __init__(self, y0, y1, x0, x1, py0, py1, px0, px1):
        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1
        self.py0, self.py1, self.px0, self.px1 = py0, py1, px0, px1


def iter_tiles(lat_h: int, lat_w: int, out_cells: int, margin_cells: int):
    """Yield the tile plan, in row-major order.

    Factored out of the decode loop deliberately: the plan is re-iterable, so a
    second pass over the SAME tiles (Phase 4A-2's statistics-gathering pass) can
    be added without reshaping the decode loop at all.
    """
    n_rows = (lat_h + out_cells - 1) // out_cells
    n_cols = (lat_w + out_cells - 1) // out_cells
    for r in range(n_rows):
        y0 = r * out_cells
        y1 = min(lat_h, y0 + out_cells)
        # Clamp the margin at the latent edge. At the true canvas border there
        # IS no neighbour, and zero padding is then exactly what a whole-image
        # decode does there too -- so clamping is correct, not an approximation.
        py0 = max(0, y0 - margin_cells)
        py1 = min(lat_h, y1 + margin_cells)
        for c in range(n_cols):
            x0 = c * out_cells
            x1 = min(lat_w, x0 + out_cells)
            px0 = max(0, x0 - margin_cells)
            px1 = min(lat_w, x1 + margin_cells)
            yield TileRect(y0, y1, x0, x1, py0, py1, px0, px1)


def resolve_geometry(threshold_px: int, margin_cells: int, scale: int) -> dict:
    """Resolve the decode budget into (output tile, margin) in latent cells.

    ``threshold_px`` is interpreted as the **decode-area budget**: the size of
    the latent block actually pushed through the decoder, which is what sets
    peak VRAM. The requested OUTPUT tile is therefore ``threshold - 2*margin``,
    NOT ``threshold``. Defining the budget on the decode area is what keeps this
    mode VRAM-neutral against the existing blend mode: a naive implementation
    that kept the output tile at ``threshold`` and decoded ``threshold +
    2*margin`` would raise the decode peak ~1.5x at a 1536px budget and could
    OOM exactly the users who turned tiling on to avoid an OOM.

    INVARIANT (asserted by the caller's geometry and by the harness):
    ``out_cells + 2 * margin_cells <= budget_cells`` -- ALWAYS. When the budget
    is too small to hold ``MIN_OUTPUT_TILE_CELLS`` plus two full margins it is
    the MARGIN that shrinks, never the decode window that grows. Clamping the
    output tile instead would make the padded window ``8 + 2*16 = 40`` cells for
    a 32-cell budget, i.e. bigger than the budget the user lowered precisely
    because they were running out of memory -- the exact inversion this
    docstring's whole argument forbids. It also matters on the AUTO default for
    Qwen-family VAEs, whose class default ``tile_sample_min_*`` is 256px = 32
    cells.

    A margin below ``MIN_USEFUL_MARGIN_CELLS`` is reported as ``degenerate``;
    the caller then declines to context-tile at all and hands the decode back to
    diffusers' blend tiling rather than paying an exploded tile count for a
    margin too short to do its job.
    """
    scale = max(1, int(scale))
    margin_cells = max(0, int(margin_cells))
    budget_cells = max(1, int(threshold_px) // scale)
    out_cells = budget_cells - 2 * margin_cells
    floored = False
    if out_cells < MIN_OUTPUT_TILE_CELLS:
        floored = True
        # Shrink the MARGIN to fit the budget, keeping the padded decode window
        # at (or under) the budget. max(0, ...) covers a budget smaller than
        # MIN_OUTPUT_TILE_CELLS itself, where the margin goes to 0 and the
        # output tile is simply the whole budget.
        margin_cells = max(0, (budget_cells - MIN_OUTPUT_TILE_CELLS) // 2)
        out_cells = max(1, budget_cells - 2 * margin_cells)
    return {
        "scale": scale,
        "margin_cells": margin_cells,
        "budget_cells": budget_cells,
        "out_cells": out_cells,
        "floored": floored,
        "degenerate": margin_cells < MIN_USEFUL_MARGIN_CELLS,
    }


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def _as_tensor(out) -> torch.Tensor:
    """Normalise whatever a decode returned into a plain tensor."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    sample = getattr(out, "sample", None)
    if sample is not None:
        return sample
    raise TypeError(f"unexpected VAE decode return type: {type(out)!r}")


def _wrap_result(sample: torch.Tensor, return_dict: bool):
    """Honour ``AutoencoderKL.decode``'s return contract exactly.

    Call sites use both forms: ``decode(z, return_dict=True).sample``
    (custom_sampling.py) and ``decode(z, return_dict=False)[0]``
    (flux2.py / zimage.py).
    """
    if not return_dict:
        return (sample,)
    from diffusers.models.autoencoders.vae import DecoderOutput
    return DecoderOutput(sample=sample)


def _fallback_to_blend(vae, z, orig_decode, return_dict, message, code,
                       **decode_kwargs):
    """Hand this decode back to diffusers, keeping it memory-bounded.

    Context mode turns diffusers' own ``use_tiling`` OFF (so the latent is not
    tiled twice), which means a bare ``orig_decode(z)`` here would run a WHOLE
    un-tiled decode -- on the exact code path a user enabled tiling to avoid an
    OOM. So re-enable diffusers tiling first: the fallback is then blend-tiled,
    not unbounded. The next ``_apply_vae_tiling`` call re-establishes whichever
    mode the next request asks for.
    """
    print(f"[VAE Tiling] context mode: {message}")
    _warn(message, code)
    try:
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
    except Exception as e:
        print(f"[VAE Tiling] could not re-enable blend tiling for the fallback: {e}")
    return orig_decode(z, return_dict=return_dict, **decode_kwargs)


def context_tiled_decode(
    vae,
    z: torch.Tensor,
    orig_decode,
    threshold_px: int,
    margin_cells: int = DEFAULT_MARGIN_CELLS,
    return_dict: bool = True,
    log: bool = True,
    **decode_kwargs,
):
    """Decode ``z`` tile-by-tile with a discarded real-context margin.

    ``orig_decode`` is the VAE's own (unpatched) bound ``decode``; it is used
    both for the whole-latent fast path and for each padded tile.
    """
    if not isinstance(z, torch.Tensor):
        return _fallback_to_blend(
            vae, z, orig_decode, return_dict,
            "decode input is not a tensor; using diffusers tiling for this decode",
            "vae_tile_context_unsupported", **decode_kwargs)

    # ---- layout -----------------------------------------------------------
    # 4-D [B, C, h, w] (AutoencoderKL / AutoencoderKLFlux2) or
    # 5-D [B, C, T, h, w] (AutoencoderKLQwenImage; T = 1 for stills).
    if (z.ndim == 5 and int(z.shape[2]) != 1) or z.ndim not in (4, 5):
        # A real video latent (T > 1) or an unrecognised layout: temporal tiling
        # is a different problem and this path makes no claim about it.
        return _fallback_to_blend(
            vae, z, orig_decode, return_dict,
            f"latent layout {tuple(z.shape)} is not a still image; using "
            "diffusers tiling for this decode",
            "vae_tile_context_unsupported", **decode_kwargs)

    lat_h, lat_w = int(z.shape[-2]), int(z.shape[-1])
    scale = spatial_compression_of(vae)
    geo = resolve_geometry(threshold_px, margin_cells, scale)
    out_cells = geo["out_cells"]
    margin = geo["margin_cells"]
    budget_cells = geo["budget_cells"]

    # ---- whole-image fast path -------------------------------------------
    # The latent already fits the decode budget, so there is nothing to tile and
    # a whole decode is exactly what blend mode would run too (diffusers does
    # not tile below its own threshold either). Delegating keeps the result
    # bit-identical to the un-patched path -- and it is memory-bounded by
    # definition, so this is NOT a degraded fallback and carries no warning.
    if lat_h <= budget_cells and lat_w <= budget_cells:
        return orig_decode(z, return_dict=return_dict, **decode_kwargs)

    # ---- budget too small to context-tile usefully -----------------------
    # Two independent ways to get here:
    #   floored     -- the budget cannot hold MIN_OUTPUT_TILE_CELLS plus two
    #                  full margins, so the margin had to shrink. That is also
    #                  exactly the regime where the decode-work ratio explodes:
    #                  at a 256px threshold (which is AutoencoderKLQwenImage's
    #                  own class default, i.e. the AUTO path on Anima/Krea2) the
    #                  geometry is a 32-cell window around an 8-cell output
    #                  tile -- 16x the decoder work per output cell, against
    #                  blend's 1.78x, and 144 tiles on a 96-cell latent.
    #   degenerate  -- the caller passed a margin below MIN_USEFUL_MARGIN_CELLS
    #                  outright, so the margin cannot extinguish the boundary
    #                  term it exists for.
    # In both cases context tiling costs a great deal and delivers little, so
    # hand the decode to diffusers' blend tiling instead of tiling badly.
    if geo["floored"] or geo["degenerate"]:
        code = "vae_tile_budget_floored" if geo["floored"] else "vae_tile_budget_too_small"
        return _fallback_to_blend(
            vae, z, orig_decode, return_dict,
            f"tile threshold {budget_cells * scale}px is too small for a "
            f"{DEFAULT_MARGIN_CELLS}-cell context margin (it would leave a "
            f"{out_cells}-cell output tile inside a {out_cells + 2 * margin}-cell "
            f"decode window); using diffusers tiling for this decode",
            code, **decode_kwargs)

    n_rows = (lat_h + out_cells - 1) // out_cells
    n_cols = (lat_w + out_cells - 1) // out_cells

    if log:
        print(
            f"[VAE Tiling] context mode: decode budget {budget_cells * scale}px "
            f"({budget_cells} cells), margin {margin} cells ({margin * scale}px, "
            f"discarded), output tile {out_cells * scale}px, grid {n_rows}x{n_cols} "
            f"over a {lat_h}x{lat_w} latent (scale {scale})"
        )

    out: Optional[torch.Tensor] = None

    for rect in iter_tiles(lat_h, lat_w, out_cells, margin):
        tile = z[..., rect.py0:rect.py1, rect.px0:rect.px1]
        dec = _as_tensor(orig_decode(tile, return_dict=False, **decode_kwargs))

        # Verify the decoder's actual spatial ratio before trusting the
        # geometry. If a VAE ever disagrees with the derived scale, bail out
        # rather than writing a wrong canvas. This is the safety net, so the
        # fallback must stay memory-bounded.
        if (dec.shape[-2] != (rect.py1 - rect.py0) * scale
                or dec.shape[-1] != (rect.px1 - rect.px0) * scale):
            del tile, dec, out
            return _fallback_to_blend(
                vae, z, orig_decode, return_dict,
                f"decoder output does not match the derived spatial ratio "
                f"{scale}; using diffusers tiling for this decode",
                "vae_tile_scale_mismatch", **decode_kwargs)

        if out is None:
            shape = list(dec.shape)
            shape[-2] = lat_h * scale
            shape[-1] = lat_w * scale
            out = torch.empty(shape, dtype=dec.dtype, device=dec.device)

        # Discard the margin in PIXEL space and write the interior.
        # Tiles abut exactly: no overlap, no blend.
        ty0 = (rect.y0 - rect.py0) * scale
        tx0 = (rect.x0 - rect.px0) * scale
        th = (rect.y1 - rect.y0) * scale
        tw = (rect.x1 - rect.x0) * scale
        out[..., rect.y0 * scale:rect.y1 * scale,
            rect.x0 * scale:rect.x1 * scale] = \
            dec[..., ty0:ty0 + th, tx0:tx0 + tw]

        # Drop the padded tile before the next one is decoded so the peak
        # stays bounded by a single padded tile (+ the canvas).
        del tile, dec

    return _wrap_result(out, return_dict)


# ---------------------------------------------------------------------------
# install / uninstall on a VAE object
# ---------------------------------------------------------------------------

_ORIG_ATTR = "_sushi_ctx_orig_decode"
_MARK_ATTR = "_sushi_ctx_tiled"


def supports_context_tiling(vae) -> bool:
    """True for objects whose own ``decode`` actually runs an AE decoder.

    Wrapper objects (``SDXLVAEWrapper`` / ``FluxVAEWrapper``) delegate to an
    inner ``.vae`` and have no ``.decoder`` of their own, so they are rejected
    here and the install lands on the inner autoencoder instead — exactly one
    install per real decoder. ``PidVaeWrapper`` likewise has no ``.decoder``
    (and no ``.vae``), so it is never touched.
    """
    if vae is None:
        return False
    return (
        hasattr(vae, "decode")
        and callable(getattr(vae, "decode", None))
        and getattr(vae, "decoder", None) is not None
        and getattr(vae, "config", None) is not None
    )


def install_context_tiled_decode(
    vae,
    threshold_px: int,
    margin_cells: int = DEFAULT_MARGIN_CELLS,
) -> bool:
    """Install (or re-point) the context-tiled ``decode`` override on ``vae``.

    Idempotent: the ORIGINAL bound ``decode`` is snapshotted once (the
    ``_sushi_tile_defaults`` convention in ``_apply_vae_tiling``) and every
    subsequent install rebuilds the wrapper from that snapshot, so wrappers
    never stack and a later request's threshold replaces an earlier one.
    """
    if not supports_context_tiling(vae):
        return False

    orig = vae.__dict__.get(_ORIG_ATTR)
    if orig is None:
        current = vae.__dict__.get("decode")
        if current is not None and getattr(current, _MARK_ATTR, False):
            # Defensive: a marked override with no snapshot should not happen.
            return False
        orig = vae.decode  # bound method off the class
        vae.__dict__[_ORIG_ATTR] = orig

    def _decode(z, return_dict=True, **kwargs):
        return context_tiled_decode(
            vae, z, orig,
            threshold_px=threshold_px,
            margin_cells=margin_cells,
            return_dict=return_dict,
            **kwargs,
        )

    setattr(_decode, _MARK_ATTR, True)
    vae.__dict__["decode"] = _decode
    return True


def uninstall_context_tiled_decode(vae) -> bool:
    """Restore the original bound ``decode``. Safe to call unconditionally."""
    if vae is None:
        return False
    current = vae.__dict__.get("decode")
    if current is not None and getattr(current, _MARK_ATTR, False):
        vae.__dict__.pop("decode", None)
    vae.__dict__.pop(_ORIG_ATTR, None)
    return True


def is_context_tiled(vae) -> bool:
    """True if ``vae`` currently carries our override."""
    if vae is None:
        return False
    return getattr(vae.__dict__.get("decode"), _MARK_ATTR, False)
