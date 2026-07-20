"""Harmonic boundary-offset membrane for outpaint seam correction (post-decode).

Implements Option 1 ("offset-membrane" reformulation of Poisson/gradient-domain
blending, Farbman et al. 2009 "Convolution Pyramids"/instant-cloning form) from
``scratchpad/outpaint_seam_redesign.md``. The strict-preservation outpaint
contract pastes the original placed rect over the decoded result unconditionally
(``paste_preserved_region``), which creates a visible seam because the decoded
GENERATED pixels adjacent to the seam are consistent with the VAE
RECONSTRUCTION of the known region, not the original (non-round-tripped) known
pixels the paste swaps in. This module computes a smooth correction field ``h``
over the generate region (the exact geometric complement of the placed rect)
such that ``g + h`` matches the original preserved pixels at the seam (C0
continuity) while staying harmonic (``Delta h = 0``) everywhere else, so all of
the generated content's own gradients/detail are preserved unchanged away from
the boundary.

    Delta h = 0            on Omega (generate region = canvas \\ rect)
    h  = p - g             on the seam ring (1px inside the rect, restricted to
                            edges bordering generated content)
    dh/dn = 0               on the canvas outer border (Neumann, mirrored ghost)
    output = g + w(d) * h   on Omega only; rect untouched

``w(d)`` is a distance taper (1 at the seam, raised-cosine to 0 at the taper
band edge) that bounds any color bleed from the correction to a fixed band
near the boundary; far-field tone drift remains ``match_generated_exposure``'s
job (this module is a purely local seam fix).

Pure numpy + scipy (scipy is already a hard dependency of the inference stack;
see ``core.inference.custom_sampling._outpaint_collar_weight``'s docstring).
No PIL, no torch, no ``api.*``/``core.pipeline``/other backend package import
(mirrors ``outpaint_utils.py``'s import-time decoupling policy) -- this module
stays trivially unit-testable and side-effect-free at import time. Callers
(``outpaint_utils.reconcile_and_paste``) convert to/from PIL and own any
``add_warning`` reporting using the metadata this module returns.

Byte-exactness: this module writes ONLY pixels where ``gen_mask == True`` (the
exact geometric complement of ``rect``); the seam ring used for the Dirichlet
boundary condition and the fine Gauss-Seidel refinement lies INSIDE the rect
(``gen_mask == False`` there) and is never copied to the output, so ``rect``
pixels are guaranteed unmodified by construction -- independent of, and in
addition to, the caller's own final unconditional paste.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized

# Internal constants (not user-exposed -- see design doc section 2.4).
RING_SMOOTH_SIGMA_DEFAULT = 1.5
CLAMP_DEFAULT = 48.0
GS_SWEEPS_DEFAULT = 20
REFINE_BAND_PX_DEFAULT = 16
COARSE_MAX_UNKNOWNS_DEFAULT = 120_000
# F1 detection threshold: mean(|h|) in the far half of the taper band, above
# which the caller should surface a "large correction" warning (content
# mismatch at the boundary, not a seam-continuity issue).
FAR_BAND_WARN_THRESHOLD = 8.0

# Boundary-offset propagation ("G_prop16") internal constants -- see
# ``scratchpad/outpaint_seamless_vae_native.md`` section 3.3/5 and the
# validated sweep in the session scratchpad's ``vae_native_ab/g_arms3.py``
# (variant ``lf16_hf4``, crossing ratio 2.03 -> 0.585, preserved region
# byte-exact by construction). Not user-exposed -- only the overall
# ``strength`` is a caller parameter.
OFFSET_PROP_LF_SIGMA_DEFAULT = 8.0
OFFSET_PROP_LF_ROWS_DEFAULT = 16
OFFSET_PROP_HF_ROWS_DEFAULT = 4


def _edge_flags(rect: Tuple[int, int, int, int], canvas_size: Tuple[int, int]) -> Dict[str, bool]:
    """Which rect edges border GENERATED content (mirrors
    ``outpaint_utils.build_paste_alpha``'s edge-adjacency rule: an edge that
    coincides with the canvas boundary borders nothing generated)."""
    x0, y0, x1, y1 = rect
    W, H = canvas_size
    return {"left": x0 > 0, "right": x1 < W, "top": y0 > 0, "bottom": y1 < H}


def _build_gen_mask(rect: Tuple[int, int, int, int], canvas_size: Tuple[int, int]) -> np.ndarray:
    """(H, W) bool, True everywhere OUTSIDE ``rect`` -- the exact geometric
    complement, NOT the blurred outpaint mask."""
    x0, y0, x1, y1 = rect
    W, H = canvas_size
    mask = np.ones((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = False
    return mask


def _build_ring_local_mask(rect: Tuple[int, int, int, int], canvas_size: Tuple[int, int]) -> np.ndarray:
    """Rect-local (rh, rw) bool mask: the outermost 1px ring of the rect,
    restricted to edges bordering generated content. All-False iff every edge
    is flush with the canvas (F4, defensive -- upstream placement validation
    already rejects a rect that fully covers the canvas)."""
    x0, y0, x1, y1 = rect
    rw, rh = x1 - x0, y1 - y0
    flags = _edge_flags(rect, canvas_size)
    ring = np.zeros((rh, rw), dtype=bool)
    if flags["left"]:
        ring[:, 0] = True
    if flags["right"]:
        ring[:, rw - 1] = True
    if flags["top"]:
        ring[0, :] = True
    if flags["bottom"]:
        ring[rh - 1, :] = True
    return ring


def _compute_h_ring_local(
    result_arr: np.ndarray,
    placed_arr: np.ndarray,
    rect: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    ring_local: np.ndarray,
    ring_smooth_sigma: float,
    clamp: float,
) -> np.ndarray:
    """Dirichlet data on the ring, rect-local coords: (rh, rw, 3) float32,
    ``placed - g`` (``g`` = the decoded result at the SAME rect-interior
    location, i.e. the VAE reconstruction the generated side was made
    consistent with), smoothed along each active edge segment independently
    (suppresses per-pixel VAE noise so it doesn't stamp a ghost of original
    high-freq detail into the generated side -- failure mode F2), then
    clamped. Zero outside ``ring_local`` (unused there).
    """
    x0, y0, x1, y1 = rect
    rh, rw = ring_local.shape
    g = result_arr[y0:y1, x0:x1, :3].astype(np.float32)
    p = placed_arr[:, :, :3].astype(np.float32)
    h_local = p - g

    flags = _edge_flags(rect, canvas_size)
    smoothed = h_local.copy()
    sigma = float(ring_smooth_sigma)
    if sigma > 0:
        if flags["top"]:
            smoothed[0, :, :] = ndimage.gaussian_filter1d(h_local[0, :, :], sigma=sigma, axis=0, mode="nearest")
        if flags["bottom"]:
            smoothed[rh - 1, :, :] = ndimage.gaussian_filter1d(h_local[rh - 1, :, :], sigma=sigma, axis=0, mode="nearest")
        if flags["left"]:
            smoothed[:, 0, :] = ndimage.gaussian_filter1d(h_local[:, 0, :], sigma=sigma, axis=0, mode="nearest")
        if flags["right"]:
            smoothed[:, rw - 1, :] = ndimage.gaussian_filter1d(h_local[:, rw - 1, :], sigma=sigma, axis=0, mode="nearest")

    smoothed = np.clip(smoothed, -float(clamp), float(clamp))
    return np.where(ring_local[:, :, None], smoothed, 0.0).astype(np.float32)


def _solve_coarse(
    gen_mask: np.ndarray,
    ring_mask_full: np.ndarray,
    h_ring_full: np.ndarray,
    coarse_max_unknowns: int,
) -> Optional[Tuple[np.ndarray, int]]:
    """Coarse harmonic solve. Downsamples by factor ``s`` chosen so the
    coarse grid has <= ``coarse_max_unknowns`` cells; builds a 5-point
    Laplacian over the coarse domain (generate cells union ring cells), with
    identity (Dirichlet) rows for ring cells and Neumann (mirrored-ghost,
    i.e. reduced-degree) rows at the canvas border; factorizes the matrix
    ONCE (channel-independent) and solves the 3 RHS vectors via back-
    substitution. Returns ``(coarse_h_grid[coarse_h, coarse_w, 3], s)`` or
    ``None`` if the domain is empty (degenerate).
    """
    H, W = gen_mask.shape
    s = max(1, int(np.ceil(np.sqrt((W * H) / float(coarse_max_unknowns)))))
    coarse_h = int(np.ceil(H / s))
    coarse_w = int(np.ceil(W / s))
    pad_h = coarse_h * s - H
    pad_w = coarse_w * s - W

    def _block_reduce_mean(arr2d: np.ndarray, pad_value: float) -> np.ndarray:
        if pad_h or pad_w:
            arr2d = np.pad(arr2d, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=pad_value)
        arr2d = arr2d.reshape(coarse_h, s, coarse_w, s)
        return arr2d.mean(axis=(1, 3))

    gen_frac = _block_reduce_mean(gen_mask.astype(np.float64), 0.0)
    ring_any = _block_reduce_mean(ring_mask_full.astype(np.float64), 0.0) > 0.0

    domain = (gen_frac >= 0.5) | ring_any
    is_ring = ring_any & domain
    if not domain.any():
        return None

    ring_val = np.zeros((coarse_h, coarse_w, 3), dtype=np.float64)
    if ring_mask_full.any():
        ring_pad = np.pad(ring_mask_full, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
        h_pad = np.pad(h_ring_full, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0.0)
        ring_pad_r = ring_pad.reshape(coarse_h, s, coarse_w, s)
        h_pad_r = h_pad.reshape(coarse_h, s, coarse_w, s, 3)
        counts = ring_pad_r.sum(axis=(1, 3)).astype(np.float64)
        sums = np.where(ring_pad_r[..., None], h_pad_r, 0.0).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            ring_val = np.divide(sums, counts[..., None], out=np.zeros_like(sums), where=counts[..., None] > 0)

    idx_map = -np.ones((coarse_h, coarse_w), dtype=np.int64)
    domain_coords = np.argwhere(domain)
    idx_map[domain_coords[:, 0], domain_coords[:, 1]] = np.arange(len(domain_coords))
    n = len(domain_coords)

    rows = []
    cols = []
    vals = []
    rhs = np.zeros((n, 3), dtype=np.float64)

    for k in range(n):
        cy, cx = int(domain_coords[k, 0]), int(domain_coords[k, 1])
        if is_ring[cy, cx]:
            rows.append(k); cols.append(k); vals.append(1.0)
            rhs[k, :] = ring_val[cy, cx, :]
            continue
        neighbors = ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1))
        diag = 0
        for ny, nx in neighbors:
            if 0 <= ny < coarse_h and 0 <= nx < coarse_w and idx_map[ny, nx] >= 0:
                rows.append(k); cols.append(int(idx_map[ny, nx])); vals.append(-1.0)
                diag += 1
        rows.append(k); cols.append(k); vals.append(float(diag) if diag > 0 else 1.0)
        # rhs[k, :] stays 0 -- harmonic interior row.

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    solve = factorized(A.tocsc())
    h_flat = np.zeros((n, 3), dtype=np.float64)
    for c in range(3):
        h_flat[:, c] = solve(rhs[:, c])

    # SuperLU (factorized) does NOT raise on a singular/degenerate matrix -- it
    # returns inf/nan, which would then propagate through the taper (0*nan=nan)
    # and black out GENERATED pixels. Fall back to no correction (safe no-op).
    if not np.isfinite(h_flat).all():
        h_flat = np.zeros_like(h_flat)

    coarse_h_grid = np.zeros((coarse_h, coarse_w, 3), dtype=np.float64)
    coarse_h_grid[domain_coords[:, 0], domain_coords[:, 1], :] = h_flat
    return coarse_h_grid, s


def _bilinear_upsample(coarse: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear-upsample a (coarse_h, coarse_w, 3) array to (out_h, out_w, 3),
    guarding the exact output shape against ``scipy.ndimage.zoom``'s rounding.
    """
    ch, cw = coarse.shape[:2]
    zoom_y = out_h / ch
    zoom_x = out_w / cw
    up = ndimage.zoom(coarse, (zoom_y, zoom_x, 1.0), order=1, mode="nearest")
    if up.shape[0] != out_h or up.shape[1] != out_w:
        fixed = np.zeros((out_h, out_w, up.shape[2]), dtype=up.dtype)
        h = min(out_h, up.shape[0])
        w = min(out_w, up.shape[1])
        fixed[:h, :w] = up[:h, :w]
        if h < out_h:
            fixed[h:, :w] = up[-1:, :w]
        if w < out_w:
            fixed[:, w:] = fixed[:, w - 1:w]
        up = fixed
    return up


def _gauss_seidel_refine(
    h_full: np.ndarray,
    band_mask: np.ndarray,
    sweeps: int,
) -> np.ndarray:
    """~``sweeps`` vectorized red-black Gauss-Seidel sweeps, restricted to
    ``band_mask`` (generate pixels within the fine refinement band of the
    seam), converging the fine-scale structure of ``h`` near the boundary
    that the coarse solve + bilinear prolongation approximates only coarsely.
    Neumann (mirrored-ghost/replicate) boundary at the canvas border; ring
    pixels (outside ``band_mask`` by construction, since they lie inside the
    rect) stay fixed as Dirichlet source values throughout.
    """
    H, W = band_mask.shape
    parity = (np.add.outer(np.arange(H), np.arange(W)) % 2)
    for c in range(h_full.shape[2]):
        hp = np.pad(h_full[:, :, c], 1, mode="edge")
        for _ in range(max(0, int(sweeps))):
            for color in (0, 1):
                sel = band_mask & (parity == color)
                ys, xs = np.nonzero(sel)
                if ys.size == 0:
                    continue
                newval = 0.25 * (
                    hp[ys, xs + 1] + hp[ys + 2, xs + 1] +
                    hp[ys + 1, xs] + hp[ys + 1, xs + 2]
                )
                hp[ys + 1, xs + 1] = newval
            hp[0, :] = hp[1, :]
            hp[-1, :] = hp[-2, :]
            hp[:, 0] = hp[:, 1]
            hp[:, -1] = hp[:, -2]
        h_full[:, :, c] = hp[1:-1, 1:-1]
    return h_full


def _distance_taper(gen_mask: np.ndarray, dist: np.ndarray, band: float) -> np.ndarray:
    """``w(d)``: 1 for ``d <= band/4``, raised-cosine to 0 at ``d = band``,
    0 outside ``gen_mask``. ``w(0) == 1`` so C0 exactness at the seam holds.
    """
    inner = band / 4.0
    span = max(band - inner, 1e-6)
    w = np.where(
        dist <= inner, 1.0,
        np.where(dist < band, 0.5 * (1.0 + np.cos(np.pi * (dist - inner) / span)), 0.0),
    )
    return np.where(gen_mask, w, 0.0)


def _auto_band(canvas_size: Tuple[int, int]) -> float:
    W, H = canvas_size
    return float(min(256, max(64, max(W, H) // 8)))


def apply_seam_membrane(
    result_arr: np.ndarray,
    placed_arr: np.ndarray,
    rect: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    band: int = 0,
    ring_smooth_sigma: float = RING_SMOOTH_SIGMA_DEFAULT,
    clamp: float = CLAMP_DEFAULT,
    gs_sweeps: int = GS_SWEEPS_DEFAULT,
    refine_band_px: int = REFINE_BAND_PX_DEFAULT,
    coarse_max_unknowns: int = COARSE_MAX_UNKNOWNS_DEFAULT,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the harmonic boundary-offset membrane to the GENERATED region of
    a decoded outpaint result, bending it to meet the preserved rect's pixels
    at the seam (C0 continuity) while staying harmonic elsewhere.

    Args:
        result_arr: (H, W, 3) uint8, the DECODED generate result BEFORE the
            final preserved-rect paste (its rect-region content is still the
            pipeline's own reconstruction of the known region -- this is what
            makes the ring Dirichlet data measurable without a re-encode).
        placed_arr: (rh, rw, 3) uint8, the preserved input content (rect-local
            coords, i.e. ``placed_arr[0, 0]`` corresponds to canvas position
            ``(rect[0], rect[1])``).
        rect: ``(x0, y0, x1, y1)`` half-open, canvas pixel coords.
        canvas_size: ``(W, H)``, matching ``result_arr.shape[1], shape[0]``.
        band: taper band B in px; 0 = auto (``clamp(max(W,H)//8, 64, 256)``).

    Returns:
        ``(out_arr, info)`` where ``out_arr`` is (H, W, 3) uint8 (a copy of
        ``result_arr`` with ONLY ``gen_mask`` pixels modified) and ``info`` is
        a metadata dict for the caller's own ``add_warning`` reporting:
        ``applied`` (bool), ``ring_pixel_count`` (int), ``band_px`` (int),
        ``coarse_factor`` (int), ``mean_abs_h_far_band`` (float, F1 signal),
        ``large_correction`` (bool, F1 threshold hit).

    Never writes ``rect`` pixels: ``gen_mask`` (the only mask ever assigned
    to) is the exact geometric complement of ``rect`` by construction, and
    the rect crop is additionally restored from ``result_arr`` unconditionally
    before returning (belt-and-suspenders, mirrors
    ``match_generated_exposure``'s defensive final-crop restore).
    """
    x0, y0, x1, y1 = rect
    W, H = canvas_size
    info: Dict[str, Any] = {
        "applied": False,
        "ring_pixel_count": 0,
        "band_px": 0,
        "coarse_factor": 0,
        "mean_abs_h_far_band": 0.0,
        "large_correction": False,
    }
    out = np.array(result_arr, copy=True)

    flags = _edge_flags(rect, canvas_size)
    if not any(flags.values()):
        # F4: every rect edge is flush with the canvas -- no generated
        # content borders the rect. Defensive (upstream placement validation
        # already rejects a fully-covering rect); h == 0 identically.
        return out, info

    ring_local = _build_ring_local_mask(rect, canvas_size)
    if not ring_local.any():
        return out, info

    gen_mask = _build_gen_mask(rect, canvas_size)
    if not gen_mask.any():
        return out, info

    h_ring_local = _compute_h_ring_local(
        result_arr, placed_arr, rect, canvas_size, ring_local, ring_smooth_sigma, clamp,
    )

    ring_mask_full = np.zeros((H, W), dtype=bool)
    ring_mask_full[y0:y1, x0:x1] = ring_local
    h_ring_full = np.zeros((H, W, 3), dtype=np.float32)
    h_ring_full[y0:y1, x0:x1, :] = h_ring_local
    info["ring_pixel_count"] = int(ring_local.sum())

    coarse_result = _solve_coarse(gen_mask, ring_mask_full, h_ring_full, coarse_max_unknowns)
    if coarse_result is None:
        return out, info
    coarse_h_grid, s = coarse_result
    info["coarse_factor"] = int(s)

    h_full = _bilinear_upsample(coarse_h_grid, H, W).astype(np.float32)
    # Re-impose the exact fine ring data (bilinear prolongation only
    # approximates it) -- this is the value the fine Gauss-Seidel refinement
    # reads as its fixed Dirichlet source at the boundary.
    h_full[ring_mask_full] = h_ring_full[ring_mask_full]

    dist = ndimage.distance_transform_edt(gen_mask.astype(np.float64))
    band_px = max(1, int(refine_band_px))
    band_mask = gen_mask & (dist <= band_px)
    h_full = _gauss_seidel_refine(h_full, band_mask, gs_sweeps)

    B = float(band) if band and band > 0 else _auto_band(canvas_size)
    B = max(8.0, B)
    w_taper = _distance_taper(gen_mask, dist, B)
    info["band_px"] = int(round(B))

    far_mask = gen_mask & (dist >= B / 2.0) & (dist < B)
    if far_mask.any():
        # F1 (halo / color-bleed) must measure the correction ACTUALLY APPLIED to
        # the far field -- the tapered w*h -- not the raw h. A uniform VAE tone
        # step (~16/255, the benign defect the membrane exists to fix) gives
        # h~=16 everywhere, so raw mean|h| would warn on every normal success;
        # the taper suppresses that to near-0 in the far band, so mean|w*h| only
        # rises when a genuinely large/structured correction bleeds outward.
        _applied_far = np.abs(w_taper[far_mask][:, None] * h_full[far_mask])
        info["mean_abs_h_far_band"] = float(np.mean(_applied_far))
        info["large_correction"] = info["mean_abs_h_far_band"] > FAR_BAND_WARN_THRESHOLD

    g = out.astype(np.float32)
    corrected = np.clip(np.round(g + w_taper[:, :, None] * h_full), 0, 255).astype(np.uint8)
    out2 = out.copy()
    out2[gen_mask] = corrected[gen_mask]
    # Defensive guard (belt-and-suspenders, mirrors match_generated_exposure):
    # restore the rect crop from the pre-membrane result unconditionally, even
    # though gen_mask already excludes it from every write above.
    out2[y0:y1, x0:x1] = out[y0:y1, x0:x1]
    info["applied"] = True
    return out2, info


# --- R2: cross-seam low-frequency tone membrane -----------------------------
# ``scratchpad/outpaint_seam_redesign_v2.md`` section 2 (measurement) and
# section 4 Phase 1 (design). Unlike ``apply_seam_membrane`` above (the C0
# harmonic membrane, which is a proven no-op here: its ring Dirichlet data is
# ``placed - result-at-rect-interior``, and a well-trained ControlNet
# reconstructs the known region faithfully, so that residual is ~0), this
# targets a DIFFERENT, measured defect: a low-frequency TONE STEP baked into
# the decoded canvas ACROSS the seam between the preserved content and the
# generated surroundings (measured: row-diff ~5.8x background, B-channel step
# ~4.6/255). It measures ``placed`` (the true preserved pixels, unbiased)
# against the GENERATED side immediately across the seam (not the rect
# interior), subtracts the legitimate local content gradient (so a genuine
# ramp across the boundary is not flattened), low-passes the residual along
# the seam axis, and writes a decaying offset into the GENERATED side only.
TONE_BAND_DEFAULT_PX = 16.0
TONE_CAP_DEFAULT = 6.0
TONE_SIGMA_DEFAULT = 16.0
TONE_GRAD_ROWS_DEFAULT = 8


def _cosine_decay(d: np.ndarray, band: float) -> np.ndarray:
    """Raised-cosine decay: 1 at ``d == 0``, 0 at ``d >= band``. Same taper
    shape as ``_distance_taper``'s falloff segment, applied here over the
    full ``[0, band)`` range (no flat inner plateau -- the tone offset is
    meant to be strongest exactly at the seam and fade out smoothly)."""
    band = max(float(band), 1e-6)
    dn = np.clip(d.astype(np.float64) / band, 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(np.pi * dn))).astype(np.float32)


def _tone_edge_profile(
    near_seam_placed: np.ndarray,
    first_gen_result: np.ndarray,
    grad_stack_deep_to_near: np.ndarray,
    sigma: float,
    cap: float,
) -> np.ndarray:
    """Per-edge tone-step profile along the seam axis, corrected for the
    legitimate local content gradient and low-passed/clamped.

    Args:
        near_seam_placed: (L, 3) float32, the preserved pixel row/column
            immediately inside the rect at the seam.
        first_gen_result: (L, 3) float32, the decoded result's first
            generated row/column immediately outside the rect.
        grad_stack_deep_to_near: (gr, L, 3) float32, the last ``gr`` preserved
            rows/columns, ORDERED from deepest-inside-the-rect to nearest-the
            -seam (index 0 = deepest); used to estimate the local gradient
            toward the seam so a genuine ramp is not treated as an artifact.
            May have ``gr < 2`` (degenerate -- gradient term is then 0).
        sigma: Gaussian low-pass sigma (px) along the seam axis.
        cap: symmetric clamp (/255) applied to the corrected, low-passed step.

    Returns:
        ``s_low``: (L, 3) float32, the offset to ADD to the generated side at
        the seam (``d == 0``) to cancel the tone discontinuity while leaving
        a legitimate content gradient untouched.
    """
    observed_step = near_seam_placed.astype(np.float32) - first_gen_result.astype(np.float32)
    if grad_stack_deep_to_near.shape[0] >= 2:
        diffs = np.diff(grad_stack_deep_to_near, axis=0)
        slope = np.median(diffs, axis=0)
    else:
        slope = np.zeros_like(observed_step)
    # expected_step_from_preserved_gradient = -slope (the value observed_step
    # would take if the generated side purely continued the preserved side's
    # own local gradient, with no additional discontinuity -- see the module
    # docstring derivation in the design doc).
    s_corrected = observed_step + slope
    s_low = ndimage.gaussian_filter1d(s_corrected, sigma=float(sigma), axis=0, mode="nearest")
    return np.clip(s_low, -float(cap), float(cap)).astype(np.float32)


def apply_cross_seam_tone(
    result_arr: np.ndarray,
    placed_arr: np.ndarray,
    rect: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    strength: float = 1.0,
    band: int = 0,
    cap: float = TONE_CAP_DEFAULT,
    sigma: float = TONE_SIGMA_DEFAULT,
    grad_rows: int = TONE_GRAD_ROWS_DEFAULT,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the cross-seam low-frequency tone membrane (R2) to the GENERATED
    region of a decoded outpaint result.

    For each rect edge that borders generated content (``_edge_flags``), this
    measures the per-channel tone step between the preserved pixels
    immediately inside the rect and the decoded generated pixels immediately
    outside it, subtracts the local content gradient estimated from the last
    ``grad_rows`` preserved rows/columns (so a legitimate ramp across the
    boundary is not flattened), low-passes the residual along the seam axis
    (Gaussian, ``sigma``), clamps it to ``+/- cap``, and adds it -- scaled by
    ``strength`` and a raised-cosine distance decay -- to the generated pixels
    within ``band`` px of the seam. Each edge's band strip spans only that
    edge's own extent along the seam axis (bottom/top over cols ``[x0,x1)``,
    left/right over rows ``[y0,y1)``), so the four strips are geometrically
    disjoint; the diagonal corner regions beyond the rect corners are not
    tone-corrected (they border no straight seam edge). The ``+=`` into a
    zeroed full-canvas accumulator is therefore never a double write.

    Args:
        result_arr: (H, W, 3) uint8, the DECODED generate result BEFORE the
            final preserved-rect paste (and, when combined with
            ``apply_seam_membrane`` in ``reconcile_and_paste``, after that
            membrane's own correction -- this function's own edge sampling is
            unaffected either way since it never reads the rect interior).
        placed_arr: (rh, rw, 3) uint8, the preserved input content, rect-local
            coords.
        rect: ``(x0, y0, x1, y1)`` half-open, canvas pixel coords.
        canvas_size: ``(W, H)``, matching ``result_arr.shape[1], shape[0]``.
        strength: scales the applied offset; ``0`` = exact no-op.
        band: decay band width in px; ``0`` = ``TONE_BAND_DEFAULT_PX`` (16).
        cap: symmetric per-channel clamp (/255) on the corrected tone step.
        sigma: Gaussian low-pass sigma (px) along the seam axis.
        grad_rows: number of preserved rows/columns used to estimate the
            local content gradient near the seam.

    Returns:
        ``(out_arr, info)`` where ``out_arr`` is (H, W, 3) uint8 (a copy of
        ``result_arr`` with ONLY ``gen_mask`` pixels modified) and ``info``:
        ``applied`` (bool), ``band_px`` (int), ``max_abs_offset`` (float),
        ``edges`` (list[str], edges that bordered generated content and were
        processed), ``mean_abs_step`` (float, mean |corrected tone step|
        across all processed edges -- diagnostic magnitude signal),
        ``max_abs_step`` (float, pre-strength post-clamp peak |tone step| --
        the strength-independent "clamp saturated" signal).

    Never writes ``rect`` pixels: only band strips OUTSIDE the rect (the exact
    geometric complement) are ever written, and the rect crop is additionally
    restored from ``result_arr`` unconditionally before returning
    (belt-and-suspenders, mirrors ``apply_seam_membrane``).
    """
    x0, y0, x1, y1 = rect
    W, H = canvas_size
    info: Dict[str, Any] = {
        "applied": False,
        "band_px": 0,
        "max_abs_offset": 0.0,
        "edges": [],
        "mean_abs_step": 0.0,
        "max_abs_step": 0.0,
    }
    out = np.array(result_arr, copy=True)

    if strength is None or float(strength) <= 0.0:
        return out, info

    flags = _edge_flags(rect, canvas_size)
    if not any(flags.values()):
        # F4-equivalent: every rect edge is flush with the canvas.
        return out, info

    gen_mask = _build_gen_mask(rect, canvas_size)
    if not gen_mask.any():
        return out, info

    rw, rh = x1 - x0, y1 - y0
    B = float(band) if band and band > 0 else TONE_BAND_DEFAULT_PX
    B = max(1.0, B)
    strength_f = float(strength)
    gr_v = max(0, int(grad_rows))

    total_offset = np.zeros((H, W, 3), dtype=np.float32)
    step_samples = []
    edges_applied = []

    if flags["bottom"] and y1 < H:
        band_rows = min(int(np.ceil(B)), H - y1)
        if band_rows > 0:
            near_seam_placed = placed_arr[rh - 1, :, :3]
            first_gen_result = result_arr[y1, x0:x1, :3]
            gr = min(gr_v, rh)
            grad_stack = placed_arr[rh - gr:rh, :, :3].astype(np.float32) if gr >= 2 else np.zeros((0, rw, 3), dtype=np.float32)
            s_low = _tone_edge_profile(near_seam_placed, first_gen_result, grad_stack, sigma, cap)
            d = np.arange(band_rows, dtype=np.float32)
            decay = _cosine_decay(d, B)
            offset = decay[:, None, None] * s_low[None, :, :] * strength_f
            total_offset[y1:y1 + band_rows, x0:x1, :] += offset
            step_samples.append(s_low)
            edges_applied.append("bottom")

    if flags["top"] and y0 > 0:
        band_rows = min(int(np.ceil(B)), y0)
        if band_rows > 0:
            near_seam_placed = placed_arr[0, :, :3]
            first_gen_result = result_arr[y0 - 1, x0:x1, :3]
            gr = min(gr_v, rh)
            grad_stack = placed_arr[0:gr, :, :3].astype(np.float32)[::-1] if gr >= 2 else np.zeros((0, rw, 3), dtype=np.float32)
            s_low = _tone_edge_profile(near_seam_placed, first_gen_result, grad_stack, sigma, cap)
            d = np.arange(band_rows, dtype=np.float32)
            decay = _cosine_decay(d, B)
            offset = decay[:, None, None] * s_low[None, :, :] * strength_f
            # offset[0] (d=0) belongs at row y0-1 (nearest the seam); ascending
            # row index in the target slice moves AWAY from the seam, so the
            # slice is written in reverse distance order.
            total_offset[y0 - band_rows:y0, x0:x1, :] += offset[::-1]
            step_samples.append(s_low)
            edges_applied.append("top")

    if flags["left"] and x0 > 0:
        band_rows = min(int(np.ceil(B)), x0)
        if band_rows > 0:
            near_seam_placed = placed_arr[:, 0, :3]
            first_gen_result = result_arr[y0:y1, x0 - 1, :3]
            gr = min(gr_v, rw)
            grad_stack = np.moveaxis(placed_arr[:, 0:gr, :3], 1, 0).astype(np.float32)[::-1] if gr >= 2 else np.zeros((0, rh, 3), dtype=np.float32)
            s_low = _tone_edge_profile(near_seam_placed, first_gen_result, grad_stack, sigma, cap)
            d = np.arange(band_rows, dtype=np.float32)
            decay = _cosine_decay(d, B)
            offset = decay[:, None, None] * s_low[None, :, :] * strength_f  # (band_rows, rh, 3)
            offset_t = np.moveaxis(offset, 0, 1)  # (rh, band_rows, 3), col index ascends with d
            total_offset[y0:y1, x0 - band_rows:x0, :] += offset_t[:, ::-1, :]
            step_samples.append(s_low)
            edges_applied.append("left")

    if flags["right"] and x1 < W:
        band_rows = min(int(np.ceil(B)), W - x1)
        if band_rows > 0:
            near_seam_placed = placed_arr[:, rw - 1, :3]
            first_gen_result = result_arr[y0:y1, x1, :3]
            gr = min(gr_v, rw)
            grad_stack = np.moveaxis(placed_arr[:, rw - gr:rw, :3], 1, 0).astype(np.float32) if gr >= 2 else np.zeros((0, rh, 3), dtype=np.float32)
            s_low = _tone_edge_profile(near_seam_placed, first_gen_result, grad_stack, sigma, cap)
            d = np.arange(band_rows, dtype=np.float32)
            decay = _cosine_decay(d, B)
            offset = decay[:, None, None] * s_low[None, :, :] * strength_f  # (band_rows, rh, 3)
            offset_t = np.moveaxis(offset, 0, 1)  # (rh, band_rows, 3), col index ascends with d
            total_offset[y0:y1, x1:x1 + band_rows, :] += offset_t
            step_samples.append(s_low)
            edges_applied.append("right")

    if not edges_applied:
        return out, info

    g = out.astype(np.float32)
    corrected = np.clip(np.round(g + total_offset), 0, 255).astype(np.uint8)
    out2 = out.copy()
    out2[gen_mask] = corrected[gen_mask]
    # Defensive guard (belt-and-suspenders, mirrors apply_seam_membrane):
    # restore the rect crop from the pre-correction result unconditionally,
    # even though gen_mask already excludes it from every write above.
    out2[y0:y1, x0:x1] = out[y0:y1, x0:x1]

    info["applied"] = True
    info["band_px"] = int(round(B))
    info["edges"] = edges_applied
    info["max_abs_offset"] = float(np.max(np.abs(total_offset))) if edges_applied else 0.0
    if step_samples:
        info["mean_abs_step"] = float(np.mean([np.mean(np.abs(s)) for s in step_samples]))
        # Pre-strength, post-clamp peak: measures whether the +/-cap clamp bound
        # (content disagreed enough to saturate), independent of `strength` --
        # the faithful signal for the caller's "saturated its clamp" warning.
        info["max_abs_step"] = float(np.max([np.max(np.abs(s)) for s in step_samples]))
    return out2, info


# --- G_prop16: generated-side-only boundary-offset propagation --------------
# ``scratchpad/outpaint_seamless_vae_native.md`` sections 2-5. Distinct from
# BOTH mechanisms above. ``apply_seam_membrane`` solves a harmonic (Poisson)
# field over the whole generate region from the same ring Dirichlet data this
# function uses, but the coarse-grid + bilinear-prolongation + limited
# Gauss-Seidel refinement numerics attenuate short-wavelength (< ~19px)
# structure in that data before it reaches the generated pixels a row or two
# from the seam (measured: crossing ratio only 2.03 -> 1.28, half the
# achievable reduction). ``apply_cross_seam_tone`` measures a DIFFERENT
# quantity (the step BETWEEN the preserved edge and the first generated
# pixel ACROSS the seam) and only ever writes a slowly-varying (sigma=16),
# capped (+/-6) low-frequency term, so it cannot repay the high-frequency
# half of the measured discontinuity either (2.03 -> 1.85).
#
# This function instead measures the SAME quantity ``apply_seam_membrane``'s
# ring data does -- ``placed - result`` at the rect-interior boundary
# row/column (the decoded reconstruction of the known region, still sitting
# in ``result_arr`` before the final paste) -- but propagates it DIRECTLY
# into the generated band with a simple two-term construction instead of a
# PDE solve: a Gaussian-smoothed low-frequency term carried over
# ``lf_rows`` generated rows/columns (raised-cosine taper to 0), plus the
# high-frequency residual (the part the low-pass removed) carried over the
# much shorter ``hf_rows`` (its own raised-cosine taper) -- both terms are
# added together, not chosen exclusively, so the innermost generated
# row/column receives the full (clamped) measured offset. The validated
# sweep (``vae_native_ab/g_arms3.py``) found the high-frequency term
# necessary (omitting it, "lf-only", only reaches 1.66): roughly half the
# measured offset's energy is short-wavelength content the Poisson solve
# above cannot deliver but a direct copy easily can.
#
# Byte-exactness: identical contract to ``apply_seam_membrane`` /
# ``apply_cross_seam_tone`` above -- writes ONLY ``gen_mask`` pixels (the
# geometric complement of ``rect``), plus an unconditional final rect-crop
# restore as a second, independent guarantee.
def _offset_prop_taper_weights(n: int) -> np.ndarray:
    """Raised-cosine taper over ``n`` discrete steps: ``1.0`` at step 0 (the
    step immediately adjacent to the seam), decaying to ``0.0`` at step
    ``n - 1``. Mirrors ``g_arms3.py``'s ``cos_w`` exactly (the validated
    arithmetic this module replicates) -- NOT the same shape as
    ``_cosine_decay`` above (that one is a continuous distance-band taper
    that reaches exactly 0 only at ``d >= band``; this one is discrete-step
    and reaches exactly 0 at its last included step). ``n <= 1`` degenerates
    to a single full-weight step (avoids a ``0/0`` division)."""
    if n <= 1:
        return np.ones((max(n, 0),), dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    return 0.5 * (1.0 + np.cos(np.pi * i / (n - 1)))


def _corner_edge_profile(
    dlf_edge: np.ndarray,
    dhf_edge: np.ndarray,
    corner_idx: int,
    w_lf_padded: np.ndarray,
    w_hf_padded: np.ndarray,
) -> np.ndarray:
    """The 1D per-distance profile an edge strip WOULD have if it were
    evaluated at a single fixed column/row (``corner_idx``, the sample
    nearest the vertex shared with the corner quadrant) instead of varying
    along the seam axis -- i.e. "the vertical/horizontal contribution that
    edge strip would carry if extended past the rect corner". ``dlf_edge``/
    ``dhf_edge`` are the edge's own full-length smoothed low/high-frequency
    arrays (shape ``(rw_or_rh, 3)``); ``corner_idx`` selects the single
    sample nearest the corner. ``w_lf_padded``/``w_hf_padded`` are
    ``_offset_prop_taper_weights(n_lf)``/``(n_hf)`` zero-padded out to the
    shared corner band length ``band_max = max(n_lf, n_hf)`` (mirrors the
    edge strip's own ``w_lf[i]``/``w_hf[i]`` weighting, just reused at a
    fixed sample instead of the full per-column/row array). Returns
    ``(band_max, 3)`` float64.
    """
    dlf_c = dlf_edge[corner_idx].astype(np.float64)
    dhf_c = dhf_edge[corner_idx].astype(np.float64)
    return w_lf_padded[:, None] * dlf_c[None, :] + w_hf_padded[:, None] * dhf_c[None, :]


def _coons_corner_grid(off_h: np.ndarray, off_v: np.ndarray, clamp_f: float) -> np.ndarray:
    """Bilinear transfinite (Coons patch) interpolation filling a
    ``(band_c, band_c, 3)`` diagonal corner quadrant from two 1D boundary
    profiles: ``off_h(i)`` (the vertical continuation of the horizontal
    edge -- top or bottom -- along the quadrant's shared row, indexed by
    row-distance ``i`` from the vertex) and ``off_v(j)`` (the horizontal
    continuation of the vertical edge -- left or right -- along the
    quadrant's shared column, indexed by column-distance ``j`` from the
    vertex).

    Construction (standard Coons-patch closed form for a unit square with
    two of its four boundary curves known and the other two/the far corner
    approximated as 0, since both ``off_h``/``off_v`` already decay to
    EXACTLY 0 at their last index by ``_offset_prop_taper_weights``'s
    raised-cosine construction):

        u = i / (band_c - 1), v = j / (band_c - 1)   (in [0, 1])
        C0 = 0.5 * (off_h[0] + off_v[0])              (blended vertex value
                                                         -- off_h[0] and
                                                         off_v[0] are two
                                                         INDEPENDENT
                                                         measurements of the
                                                         same physical
                                                         corner, from the
                                                         two different edge
                                                         strips; C0 is their
                                                         average)
        off_h_adj[0], off_v_adj[0] := C0               (override the two
                                                         edges' own corner
                                                         sample with the
                                                         shared blend, so
                                                         the two boundary
                                                         curves AGREE at the
                                                         vertex -- a Coons
                                                         patch only
                                                         reproduces its
                                                         boundary curves
                                                         exactly when they
                                                         agree at shared
                                                         corners)
        grid[i, j] = (1-v)*off_h_adj[i] + (1-u)*off_v_adj[j]
                     - (1-u)*(1-v)*C0

    This satisfies, BY CONSTRUCTION (elementary substitution, independent of
    the actual data): ``grid[i, 0] == off_h_adj[i]`` for every ``i`` (exact
    continuity with the horizontal edge's own continuation along the
    quadrant's shared row) and ``grid[0, j] == off_v_adj[j]`` for every
    ``j`` (exact continuity with the vertical edge's own continuation along
    the quadrant's shared column) -- i.e. C0-continuous with BOTH adjacent
    strips everywhere along the quadrant's two inner edges, with the single
    unavoidable ambiguity (the two edges' independently-measured corner
    samples disagreeing) resolved by averaging exactly at that one shared
    vertex. As ``u, v -> 1`` (the outer corner of the quadrant, farthest
    from the vertex) every term vanishes since ``off_h_adj``/``off_v_adj``
    themselves reach 0 there, so the field decays to 0 outward, matching the
    edges' own taper-to-0 behavior. A defensive final clamp to
    ``+/- clamp_f`` is applied (the Coons blend is an affine, not convex,
    combination of already-clamped inputs, so it is not algebraically
    guaranteed to stay within bounds for all possible input configurations).

    ``band_c == 1`` degenerates to a single cell equal to ``C0`` (the ``u``/
    ``v`` normalization would divide by zero otherwise).
    """
    band_c = int(off_h.shape[0])
    c0 = 0.5 * (off_h[0].astype(np.float64) + off_v[0].astype(np.float64))
    if band_c <= 1:
        grid = np.broadcast_to(c0, (max(band_c, 0), max(band_c, 0), 3)).copy()
        return np.clip(grid, -clamp_f, clamp_f).astype(np.float32)

    off_h_adj = off_h.astype(np.float64).copy()
    off_h_adj[0] = c0
    off_v_adj = off_v.astype(np.float64).copy()
    off_v_adj[0] = c0

    u = (np.arange(band_c, dtype=np.float64) / (band_c - 1))[:, None, None]  # (band_c, 1, 1), varies with i
    v = (np.arange(band_c, dtype=np.float64) / (band_c - 1))[None, :, None]  # (1, band_c, 1), varies with j

    grid = (
        (1.0 - v) * off_h_adj[:, None, :]
        + (1.0 - u) * off_v_adj[None, :, :]
        - (1.0 - u) * (1.0 - v) * c0[None, None, :]
    )
    return np.clip(grid, -clamp_f, clamp_f).astype(np.float32)


def apply_seam_offset_propagation(
    result_arr: np.ndarray,
    placed_arr: np.ndarray,
    rect: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    strength: float = 1.0,
    lf_sigma: float = OFFSET_PROP_LF_SIGMA_DEFAULT,
    lf_rows: int = OFFSET_PROP_LF_ROWS_DEFAULT,
    hf_rows: int = OFFSET_PROP_HF_ROWS_DEFAULT,
    clamp: float = CLAMP_DEFAULT,
    fill_corners: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the G_prop16 boundary-offset propagation to the GENERATED region
    of a decoded outpaint result.

    For each rect edge that borders generated content (``_edge_flags``),
    measures the per-channel offset ``placed - result`` at the rect-interior
    boundary row/column (``result_arr`` there is still the pipeline's own
    decoded reconstruction of the known region -- this is measurable without
    a re-encode, exactly like ``apply_seam_membrane``'s ring data), clamps it
    to ``+/- clamp``, splits it into a Gaussian-smoothed low-frequency term
    (``lf_sigma`` along the seam axis) and the high-frequency residual, and
    ADDS both -- independently raised-cosine-tapered over ``lf_rows`` and
    ``hf_rows`` generated rows/columns respectively, scaled by ``strength``
    -- into the generated pixels adjacent to that edge. Each edge's band
    strip spans only that edge's own extent along the seam axis (bottom/top
    over cols ``[x0, x1)``, left/right over rows ``[y0, y1)``), so the four
    strips are geometrically disjoint and, by default (``fill_corners=False``,
    the byte-identical-preserving default), diagonal corners beyond the rect
    corners are not touched (mirrors ``apply_cross_seam_tone``'s edge
    generalization).

    When ``fill_corners`` is True (opt-in), each of the (up to) four diagonal
    corner quadrants beyond a rect vertex -- present only when BOTH of that
    vertex's adjacent edges border generated content -- is ALSO filled, using
    ``_coons_corner_grid`` to bilinearly blend the two adjacent edges' own
    per-distance offset profiles (each edge's profile evaluated at the fixed
    sample nearest the vertex, i.e. the vertical/horizontal contribution that
    edge strip would carry if extended past the corner -- see
    ``_corner_edge_profile``) so the quadrant is C0-continuous with BOTH
    adjacent strips along its two inner edges and decays to 0 at its outer
    corner. Each quadrant is the geometric complement of the four straight
    edge strips (disjoint rows/cols from every strip and from the other three
    quadrants by construction -- see the corner placement code below), so the
    ``+=`` accumulation is never a double write. ``fill_corners=False`` (the
    default) leaves this function's output bit-identical to its behavior
    before this parameter existed.

    Args:
        result_arr: (H, W, 3) uint8, the DECODED generate result BEFORE the
            final preserved-rect paste (and after any earlier seam
            corrections in the caller's pipeline, e.g. ``apply_seam_membrane``
            / ``apply_cross_seam_tone`` -- this function only ever reads the
            rect-interior boundary and the generated band, so running after
            those is safe, though the design doc recommends NOT stacking all
            three at once to avoid over-correction).
        placed_arr: (rh, rw, 3) uint8, the preserved input content (rect-local
            coords).
        rect: ``(x0, y0, x1, y1)`` half-open, canvas pixel coords.
        canvas_size: ``(W, H)``, matching ``result_arr.shape[1], shape[0]``.
        strength: scales the applied offset; ``0`` (or negative) = exact
            no-op.
        lf_sigma, lf_rows, hf_rows, clamp: internal constants (see module
            top), exposed as arguments only for the self-test / offline
            sweep reproducibility -- not caller-configurable via the public
            API.
        fill_corners: opt-in, default False. When True, also fills the
            diagonal corner-quadrant tonal-step wedge left untreated by the
            four straight edge strips (see above). Has no effect unless
            ``strength > 0`` and at least one corner is present (both its
            adjacent edges border generated content).

    Returns:
        ``(out_arr, info)`` where ``out_arr`` is (H, W, 3) uint8 (a copy of
        ``result_arr`` with ONLY ``gen_mask`` pixels modified) and ``info``:
        ``applied`` (bool), ``edges`` (list[str]), ``max_abs_delta`` (float,
        pre-strength post-clamp peak |offset| across all processed edges --
        the strength-independent "clamp saturated" signal, mirrors
        ``apply_cross_seam_tone``'s ``max_abs_step``), ``large_correction``
        (bool, ``max_abs_delta >= 0.9 * clamp``), ``corners`` (list[str], the
        corner quadrants that were filled -- always empty when
        ``fill_corners`` is False).

    Never writes ``rect`` pixels: only band strips OUTSIDE the rect (the
    exact geometric complement) are ever written, and the rect crop is
    additionally restored from ``result_arr`` unconditionally before
    returning (belt-and-suspenders, mirrors both mechanisms above).
    """
    x0, y0, x1, y1 = rect
    W, H = canvas_size
    info: Dict[str, Any] = {
        "applied": False,
        "edges": [],
        "max_abs_delta": 0.0,
        "large_correction": False,
        "corners": [],
    }
    out = np.array(result_arr, copy=True)

    if strength is None or float(strength) <= 0.0:
        return out, info

    flags = _edge_flags(rect, canvas_size)
    if not any(flags.values()):
        # F4-equivalent: every rect edge is flush with the canvas.
        return out, info

    gen_mask = _build_gen_mask(rect, canvas_size)
    if not gen_mask.any():
        return out, info

    n_lf = max(0, int(lf_rows))
    n_hf = max(0, int(hf_rows))
    if n_lf == 0 and n_hf == 0:
        return out, info

    rw, rh = x1 - x0, y1 - y0
    sigma = float(lf_sigma)
    strength_f = float(strength)
    clamp_f = float(clamp)
    w_lf = _offset_prop_taper_weights(n_lf)
    w_hf = _offset_prop_taper_weights(n_hf)

    def _split(delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        delta_c = np.clip(delta, -clamp_f, clamp_f)
        # No explicit `mode=` -- matches g_arms3.py's `gaussian_filter1d`
        # call exactly (scipy's default, `mode="reflect"`), reproducing the
        # validated arithmetic rather than this module's other functions'
        # `mode="nearest"` convention.
        dlf = ndimage.gaussian_filter1d(delta_c, sigma=sigma, axis=0)
        dhf = delta_c - dlf
        return dlf, dhf

    total_offset = np.zeros((H, W, 3), dtype=np.float32)
    clamped_deltas = []
    edges_applied = []
    # Per-edge (dlf, dhf) full-length arrays, stashed only when
    # `fill_corners` is set -- consumed below by the corner-quadrant fill,
    # which needs each edge's own smoothed low/high-frequency profile at the
    # single sample nearest a given vertex (see `_corner_edge_profile`).
    # `None` unless populated, so a corner whose adjacent edge never applied
    # (e.g. an edge flush with the canvas) is correctly skipped.
    edge_dlf_dhf: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if flags["bottom"] and y1 < H:
        band = min(max(n_lf, n_hf), H - y1)
        if band > 0:
            delta = placed_arr[rh - 1, :, :3].astype(np.float32) - result_arr[y1 - 1, x0:x1, :3].astype(np.float32)
            delta_c = np.clip(delta, -clamp_f, clamp_f)
            dlf, dhf = _split(delta)
            for i in range(min(n_lf, band)):
                total_offset[y1 + i, x0:x1, :] += w_lf[i] * dlf * strength_f
            for i in range(min(n_hf, band)):
                total_offset[y1 + i, x0:x1, :] += w_hf[i] * dhf * strength_f
            clamped_deltas.append(delta_c)
            edges_applied.append("bottom")
            if fill_corners:
                edge_dlf_dhf["bottom"] = (dlf, dhf)

    if flags["top"] and y0 > 0:
        band = min(max(n_lf, n_hf), y0)
        if band > 0:
            delta = placed_arr[0, :, :3].astype(np.float32) - result_arr[y0, x0:x1, :3].astype(np.float32)
            delta_c = np.clip(delta, -clamp_f, clamp_f)
            dlf, dhf = _split(delta)
            for i in range(min(n_lf, band)):
                total_offset[y0 - 1 - i, x0:x1, :] += w_lf[i] * dlf * strength_f
            for i in range(min(n_hf, band)):
                total_offset[y0 - 1 - i, x0:x1, :] += w_hf[i] * dhf * strength_f
            clamped_deltas.append(delta_c)
            edges_applied.append("top")
            if fill_corners:
                edge_dlf_dhf["top"] = (dlf, dhf)

    if flags["left"] and x0 > 0:
        band = min(max(n_lf, n_hf), x0)
        if band > 0:
            delta = placed_arr[:, 0, :3].astype(np.float32) - result_arr[y0:y1, x0, :3].astype(np.float32)
            delta_c = np.clip(delta, -clamp_f, clamp_f)
            dlf, dhf = _split(delta)
            for i in range(min(n_lf, band)):
                total_offset[y0:y1, x0 - 1 - i, :] += w_lf[i] * dlf * strength_f
            for i in range(min(n_hf, band)):
                total_offset[y0:y1, x0 - 1 - i, :] += w_hf[i] * dhf * strength_f
            clamped_deltas.append(delta_c)
            edges_applied.append("left")
            if fill_corners:
                edge_dlf_dhf["left"] = (dlf, dhf)

    if flags["right"] and x1 < W:
        band = min(max(n_lf, n_hf), W - x1)
        if band > 0:
            delta = placed_arr[:, rw - 1, :3].astype(np.float32) - result_arr[y0:y1, x1 - 1, :3].astype(np.float32)
            delta_c = np.clip(delta, -clamp_f, clamp_f)
            dlf, dhf = _split(delta)
            for i in range(min(n_lf, band)):
                total_offset[y0:y1, x1 + i, :] += w_lf[i] * dlf * strength_f
            for i in range(min(n_hf, band)):
                total_offset[y0:y1, x1 + i, :] += w_hf[i] * dhf * strength_f
            clamped_deltas.append(delta_c)
            edges_applied.append("right")
            if fill_corners:
                edge_dlf_dhf["right"] = (dlf, dhf)

    if not edges_applied:
        return out, info

    corners_applied: list = []
    if fill_corners:
        # Corner-quadrant fill (opt-in): the diagonal wedge beyond a rect
        # vertex, present only when BOTH of that vertex's adjacent edges
        # applied above. Geometrically disjoint from all four edge strips
        # and from the other three quadrants by construction (each quadrant's
        # row range is entirely >= y1 or entirely < y0, and its col range is
        # entirely >= x1 or entirely < x0, whereas every edge strip's row OR
        # col range is [y0, y1) / [x0, x1) -- the rect's own extent), so the
        # `+=` below never double-writes a cell any edge strip (or another
        # corner) already wrote.
        band_max = max(n_lf, n_hf)
        w_lf_padded = np.zeros(band_max, dtype=np.float64)
        w_lf_padded[:n_lf] = w_lf
        w_hf_padded = np.zeros(band_max, dtype=np.float64)
        w_hf_padded[:n_hf] = w_hf

        def _fill_one_corner(
            name: str,
            h_edge: str,
            v_edge: str,
            h_col_idx: int,
            v_row_idx: int,
            avail_rows: int,
            avail_cols: int,
            row_base: int,
            row_step: int,
            col_base: int,
            col_step: int,
        ) -> None:
            if h_edge not in edge_dlf_dhf or v_edge not in edge_dlf_dhf:
                return
            band_c = min(band_max, avail_rows, avail_cols)
            if band_c <= 0:
                return
            dlf_h, dhf_h = edge_dlf_dhf[h_edge]
            dlf_v, dhf_v = edge_dlf_dhf[v_edge]
            off_h = _corner_edge_profile(dlf_h, dhf_h, h_col_idx, w_lf_padded[:band_c], w_hf_padded[:band_c])
            off_v = _corner_edge_profile(dlf_v, dhf_v, v_row_idx, w_lf_padded[:band_c], w_hf_padded[:band_c])
            corner_grid = _coons_corner_grid(off_h, off_v, clamp_f)  # (band_c, band_c, 3), [i, j] = row/col distance
            # Row `i` (distance from the vertex row) maps to canvas row
            # `row_base + row_step * i`; row_step is +1 (bottom corners,
            # rows increase downward away from y1) or -1 (top corners, rows
            # decrease upward away from y0). Same pattern for columns.
            if row_step >= 0:
                rows = slice(row_base, row_base + band_c)
                grid_rows = corner_grid
            else:
                rows = slice(row_base - band_c + 1, row_base + 1)
                grid_rows = corner_grid[::-1, :, :]
            if col_step >= 0:
                cols = slice(col_base, col_base + band_c)
                grid_full = grid_rows
            else:
                cols = slice(col_base - band_c + 1, col_base + 1)
                grid_full = grid_rows[:, ::-1, :]
            total_offset[rows, cols, :] += grid_full * strength_f
            corners_applied.append(name)

        _fill_one_corner(
            "bottom_right", "bottom", "right", rw - 1, rh - 1,
            avail_rows=H - y1, avail_cols=W - x1,
            row_base=y1, row_step=1, col_base=x1, col_step=1,
        )
        _fill_one_corner(
            "bottom_left", "bottom", "left", 0, rh - 1,
            avail_rows=H - y1, avail_cols=x0,
            row_base=y1, row_step=1, col_base=x0 - 1, col_step=-1,
        )
        _fill_one_corner(
            "top_right", "top", "right", rw - 1, 0,
            avail_rows=y0, avail_cols=W - x1,
            row_base=y0 - 1, row_step=-1, col_base=x1, col_step=1,
        )
        _fill_one_corner(
            "top_left", "top", "left", 0, 0,
            avail_rows=y0, avail_cols=x0,
            row_base=y0 - 1, row_step=-1, col_base=x0 - 1, col_step=-1,
        )

    g = out.astype(np.float32)
    corrected = np.clip(np.round(g + total_offset), 0, 255).astype(np.uint8)
    out2 = out.copy()
    out2[gen_mask] = corrected[gen_mask]
    # Defensive guard (belt-and-suspenders, mirrors apply_seam_membrane /
    # apply_cross_seam_tone): restore the rect crop from the pre-correction
    # result unconditionally, even though gen_mask already excludes it from
    # every write above.
    out2[y0:y1, x0:x1] = out[y0:y1, x0:x1]

    info["applied"] = True
    info["edges"] = edges_applied
    info["corners"] = corners_applied
    if clamped_deltas:
        info["max_abs_delta"] = float(np.max([np.max(np.abs(d)) for d in clamped_deltas]))
        info["large_correction"] = info["max_abs_delta"] >= 0.9 * clamp_f
    return out2, info


if __name__ == "__main__":
    # Minimal embedded self-test (no backend/GPU): synthetic canvas with a
    # known tone step across the seam. Run directly:
    #   venv/Scripts/python.exe -m core.inference.seam_membrane   (from backend/)
    # or:
    #   venv/Scripts/python.exe backend/core/inference/seam_membrane.py
    rng = np.random.default_rng(0)
    Wc, Hc = 256, 256
    rect_t = (64, 64, 192, 192)
    x0t, y0t, x1t, y1t = rect_t

    # The whole DECODED result (both the rect-interior reconstruction and the
    # generated surroundings) carries a uniform VAE tone bias (-16) relative
    # to the ORIGINAL preserved content (`placed`, unbiased) -- this is the
    # structural defect the membrane targets: generated pixels near the seam
    # are consistent with the reconstruction, not the original.
    result = (112 + rng.normal(0, 3, size=(Hc, Wc, 3))).clip(0, 255).astype(np.uint8)
    placed = (128 + rng.normal(0, 3, size=(y1t - y0t, x1t - x0t, 3))).clip(0, 255).astype(np.uint8)

    out_arr, meta = apply_seam_membrane(result, placed, rect_t, (Wc, Hc))

    assert np.array_equal(out_arr[y0t:y1t, x0t:x1t], result[y0t:y1t, x0t:x1t]), \
        "seam_membrane must never modify rect pixels"
    assert meta["applied"], "expected the membrane to apply on a well-formed rect"

    pre_jump = float(np.mean(np.abs(
        result[y0t - 1, x0t:x1t, :].astype(np.float32) - placed[0, :, :].astype(np.float32)
    )))
    post_jump = float(np.mean(np.abs(
        out_arr[y0t - 1, x0t:x1t, :].astype(np.float32) - placed[0, :, :].astype(np.float32)
    )))
    assert post_jump < pre_jump, f"expected boundary jump reduced: pre={pre_jump} post={post_jump}"

    print(f"[seam_membrane self-test] OK: rect byte-exact, jump {pre_jump:.3f} -> {post_jump:.3f}, "
          f"meta={meta}")

    # --- R2 cross-seam tone self-test ---------------------------------------
    ct_rng = np.random.default_rng(1)
    ct_W, ct_H = 256, 256
    ct_rect = (64, 64, 192, 192)
    ct_x0, ct_y0, ct_x1, ct_y1 = ct_rect

    ct_placed = (128 + ct_rng.normal(0, 2, size=(ct_y1 - ct_y0, ct_x1 - ct_x0, 3))).clip(0, 255).astype(np.uint8)
    # The whole GENERATED surround carries a uniform low-frequency tone offset
    # (-6) relative to `placed` -- the defect apply_cross_seam_tone targets.
    ct_result = (122 + ct_rng.normal(0, 2, size=(ct_H, ct_W, 3))).clip(0, 255).astype(np.uint8)
    # Rect-interior content is irrelevant to apply_cross_seam_tone (it never
    # reads inside the rect except via `placed_arr`), but fill it plausibly.
    ct_result[ct_y0:ct_y1, ct_x0:ct_x1] = ct_placed

    ct_out, ct_meta = apply_cross_seam_tone(ct_result, ct_placed, ct_rect, (ct_W, ct_H))

    assert np.array_equal(ct_out[ct_y0:ct_y1, ct_x0:ct_x1], ct_result[ct_y0:ct_y1, ct_x0:ct_x1]), \
        "cross-seam tone membrane must never modify rect pixels"
    assert ct_meta["applied"], "expected the tone membrane to apply on a well-formed rect"

    ct_pre_jump = float(np.mean(np.abs(
        ct_result[ct_y1, ct_x0:ct_x1, :].astype(np.float32) - ct_placed[-1, :, :].astype(np.float32)
    )))
    ct_post_jump = float(np.mean(np.abs(
        ct_out[ct_y1, ct_x0:ct_x1, :].astype(np.float32) - ct_placed[-1, :, :].astype(np.float32)
    )))
    assert ct_post_jump < ct_pre_jump, \
        f"expected cross-seam tone jump reduced: pre={ct_pre_jump} post={ct_post_jump}"

    ct_out_zero, ct_meta_zero = apply_cross_seam_tone(ct_result, ct_placed, ct_rect, (ct_W, ct_H), strength=0.0)
    assert np.array_equal(ct_out_zero, ct_result), "strength=0 must be an exact no-op"
    assert not ct_meta_zero["applied"]

    print(f"[cross_seam_tone self-test] OK: rect byte-exact, seam jump {ct_pre_jump:.3f} -> {ct_post_jump:.3f}, "
          f"meta={ct_meta}")

    # --- G_prop16 boundary-offset propagation self-test ---------------------
    op_rng = np.random.default_rng(2)
    op_W, op_H = 256, 256
    op_rect = (64, 64, 192, 192)
    ox0, oy0, ox1, oy1 = op_rect

    op_placed = (128 + op_rng.normal(0, 2, size=(oy1 - oy0, ox1 - ox0, 3))).clip(0, 255).astype(np.uint8)
    # The whole decoded canvas (rect interior AND generated surroundings)
    # sits at a uniform bias (-10) relative to `placed` -- exactly the
    # structural defect this function targets: the decoded reconstruction of
    # the known region (still sitting in `op_result`'s rect interior) is
    # self-consistent with its own generated surroundings but NOT with the
    # true preserved pixels the final paste will swap in.
    op_result = (118 + op_rng.normal(0, 2, size=(op_H, op_W, 3))).clip(0, 255).astype(np.uint8)
    op_result[oy0:oy1, ox0:ox1] = np.clip(op_placed.astype(np.int16) - 10, 0, 255).astype(np.uint8)

    op_out, op_meta = apply_seam_offset_propagation(op_result, op_placed, op_rect, (op_W, op_H))

    assert np.array_equal(op_out[oy0:oy1, ox0:ox1], op_result[oy0:oy1, ox0:ox1]), \
        "seam offset propagation must never modify rect pixels"
    assert op_meta["applied"], "expected offset propagation to apply on a well-formed rect"
    assert set(op_meta["edges"]) == {"top", "bottom", "left", "right"}

    op_pre_jump = float(np.mean(np.abs(
        op_result[oy1, ox0:ox1, :].astype(np.float32) - op_placed[-1, :, :].astype(np.float32)
    )))
    op_post_jump = float(np.mean(np.abs(
        op_out[oy1, ox0:ox1, :].astype(np.float32) - op_placed[-1, :, :].astype(np.float32)
    )))
    assert op_post_jump < op_pre_jump, \
        f"expected boundary-offset jump reduced: pre={op_pre_jump} post={op_post_jump}"

    op_out_zero, op_meta_zero = apply_seam_offset_propagation(op_result, op_placed, op_rect, (op_W, op_H), strength=0.0)
    assert np.array_equal(op_out_zero, op_result), "strength=0 must be an exact no-op"
    assert not op_meta_zero["applied"]

    print(f"[seam_offset_propagation self-test] OK: rect byte-exact, boundary jump "
          f"{op_pre_jump:.3f} -> {op_post_jump:.3f}, meta={op_meta}")

    # --- G_prop16 corner-quadrant fill self-test (opt-in) -------------------
    # (a) rect byte-exact with corners on; (b) fill_corners=False (the
    # default) leaves every corner quadrant bit-identical to the untreated
    # pre-correction result (mirrors the module's pre-existing documented
    # behavior); (c) a synthetic corner tonal step is reduced with corners
    # on; (d) the corner fill respects `clamp`.
    op_band_max = max(OFFSET_PROP_LF_ROWS_DEFAULT, OFFSET_PROP_HF_ROWS_DEFAULT)

    op_out_off, op_meta_off = apply_seam_offset_propagation(
        op_result, op_placed, op_rect, (op_W, op_H), fill_corners=False,
    )
    assert op_meta_off["corners"] == [], "fill_corners=False must leave corners untouched"
    for _cy, _cx in ((oy1, ox1), (oy1, ox0 - op_band_max), (oy0 - op_band_max, ox1), (oy0 - op_band_max, ox0 - op_band_max)):
        _region_off = op_out_off[_cy:_cy + op_band_max, _cx:_cx + op_band_max]
        _region_pre = op_result[_cy:_cy + op_band_max, _cx:_cx + op_band_max]
        assert np.array_equal(_region_off, _region_pre), \
            "(b) fill_corners=False must be bit-identical to the untreated corner quadrant (no regression pre-change)"

    op_out_on, op_meta_on = apply_seam_offset_propagation(
        op_result, op_placed, op_rect, (op_W, op_H), fill_corners=True,
    )
    assert np.array_equal(op_out_on[oy0:oy1, ox0:ox1], op_result[oy0:oy1, ox0:ox1]), \
        "(a) corner fill must never modify rect pixels"
    assert set(op_meta_on["corners"]) == {"bottom_right", "bottom_left", "top_right", "top_left"}

    # (c) diagonal corner jump reduced: op_result carries a UNIFORM bias
    # (-10) relative to `placed` everywhere, including the corner-diagonal
    # pixel immediately beyond the vertex (oy1, ox1) -- a pixel neither the
    # bottom strip (cols restricted to [x0, x1)) nor the right strip (rows
    # restricted to [y0, y1)) ever writes, so it stays at the untreated bias
    # unless the corner fill reaches it.
    op_corner_pre_jump = float(np.mean(np.abs(
        op_result[oy1, ox1, :].astype(np.float32) - op_placed[-1, -1, :].astype(np.float32)
    )))
    op_corner_off_jump = float(np.mean(np.abs(
        op_out_off[oy1, ox1, :].astype(np.float32) - op_placed[-1, -1, :].astype(np.float32)
    )))
    op_corner_on_jump = float(np.mean(np.abs(
        op_out_on[oy1, ox1, :].astype(np.float32) - op_placed[-1, -1, :].astype(np.float32)
    )))
    assert abs(op_corner_off_jump - op_corner_pre_jump) < 1e-3, \
        "fill_corners=False must leave the diagonal corner pixel unchanged"
    assert op_corner_on_jump < op_corner_off_jump, \
        f"(c) expected corner-diagonal jump reduced: off={op_corner_off_jump} on={op_corner_on_jump}"

    # (d) clamp respected: every corner-quadrant offset actually applied must
    # stay within +/- CLAMP_DEFAULT (the same bound the edge strips honor).
    op_corner_diff = np.abs(op_out_on[oy0 - op_band_max:oy1 + op_band_max, ox0 - op_band_max:ox1 + op_band_max].astype(np.int16)
                             - op_result[oy0 - op_band_max:oy1 + op_band_max, ox0 - op_band_max:ox1 + op_band_max].astype(np.int16))
    assert op_corner_diff.max() <= CLAMP_DEFAULT + 1, \
        f"(d) corner fill exceeded the clamp bound: max |delta|={op_corner_diff.max()}"

    op_out_on_zero, op_meta_on_zero = apply_seam_offset_propagation(
        op_result, op_placed, op_rect, (op_W, op_H), strength=0.0, fill_corners=True,
    )
    assert np.array_equal(op_out_on_zero, op_result), "strength=0 must be an exact no-op even with fill_corners=True"
    assert not op_meta_on_zero["applied"]

    print(f"[seam_offset_propagation corner-fill self-test] OK: rect byte-exact, corners-off matches "
          f"untreated ({op_corner_off_jump:.3f}), corner-diagonal jump {op_corner_pre_jump:.3f} -> "
          f"{op_corner_on_jump:.3f} with fill_corners=True, meta={op_meta_on}")
