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
