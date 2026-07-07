"""In-loop hard-flatten of a flat background region (pixel space).

This is the region-detect + hard-replace + feather math validated in the
step-flatten prototype (scratchpad ``proto2.py``, the ``hardflat`` mode). It is
distinct from ``color_flatten.py`` (an RGB-guided chroma-soft filter): here the
detected flat background region is REPLACED with its single dominant (median)
colour, the region mask is feathered, and only that region is altered - textured
backgrounds are protected by an explicit area gate that returns a no-op.

All functions here are pure (numpy / OpenCV / scipy), operate on an HWC float
image in ``[0, 1]``, and have no torch / GPU / model dependency, so they are
unit-testable on CPU. The caller (the SD1.5/SDXL denoise loop) is responsible
for the VAE decode -> ``hard_flatten`` -> VAE encode round-trip and the latent
injection.

Reference:
  Zhang et al. is unrelated; this is a heuristic image-space correction. The
  gradient threshold (6/255), min-area gate, erosion (3 iters) and gaussian
  feather (sigma 6) reproduce the validated prototype exactly.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# cv2 / scipy are hard dependencies of the inference stack (already used by
# color-space helpers and the prototype). Import at module load so a missing
# dependency fails loudly rather than silently disabling the feature.
import cv2
from scipy import ndimage

# Validated prototype constants (proto2.py hardflat mode).
_GRAD_THRESH = 6.0        # Sobel magnitude threshold, in 0-255 luma units.
_EROSION_ITERS = 3        # erode the hard mask before feathering.
_BLUR_SIGMA = 6.0         # gaussian feather sigma (px).


def detect_flat_region(
    img01: np.ndarray,
    grad_thresh: float = _GRAD_THRESH,
    min_region_frac: float = 0.02,
) -> Optional[np.ndarray]:
    """Detect the flat background region of an image.

    The region is the largest connected low-gradient (Sobel magnitude below
    ``grad_thresh``) component that TOUCHES the image border - backgrounds do.
    It must cover at least ``min_region_frac`` of the frame, otherwise there is
    no confident flat background and ``None`` is returned (the caller then makes
    the step a no-op, protecting textured-background intents).

    Unlike the prototype's ``detect_flat_region`` there is NO border-band
    fallback: for the in-loop correction a low-confidence detection must be a
    hard no-op, not a forced band replacement.

    Args:
        img01: HWC float image in [0, 1].
        grad_thresh: gradient magnitude threshold in 0-255 luma units.
        min_region_frac: minimum region area as a fraction of the frame.

    Returns:
        Boolean HxW mask of the background region, or ``None`` if no region
        qualifies.
    """
    if img01.ndim != 3 or img01.shape[2] < 3:
        raise ValueError(f"detect_flat_region expects HWC image, got shape {img01.shape}")
    g = (img01[..., :3].mean(2) * 255.0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    low = grad < grad_thresh
    lbl, _n = ndimage.label(low)
    Hh, Ww = g.shape
    border_ids = set(np.unique(np.concatenate([
        lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])).tolist()) - {0}
    best = None
    bsz = 0
    for i in border_ids:
        sz = int((lbl == i).sum())
        if sz > bsz:
            bsz = sz
            best = i
    if best is not None and bsz >= min_region_frac * Hh * Ww:
        return lbl == best
    return None


def hard_flatten(
    img01: np.ndarray,
    min_region_frac: float = 0.02,
    grad_thresh: float = _GRAD_THRESH,
    erosion_iters: int = _EROSION_ITERS,
    blur_sigma: float = _BLUR_SIGMA,
) -> Tuple[np.ndarray, bool]:
    """Replace the detected flat background region with its dominant colour.

    The region is detected with :func:`detect_flat_region`. When a confident
    region is found, its per-channel median colour is computed and blended in
    over a FEATHERED mask (erode the hard mask, then gaussian-blur it to a soft
    alpha) so the VAE re-encode never sees a hard seam. When no confident region
    is found the input is returned UNCHANGED (a true no-op).

    Args:
        img01: HWC float image in [0, 1].
        min_region_frac: area gate passed to :func:`detect_flat_region`.
        grad_thresh: gradient threshold passed to :func:`detect_flat_region`.
        erosion_iters: binary-erosion iterations applied to the hard mask.
        blur_sigma: gaussian sigma (px) for feathering the eroded mask.

    Returns:
        ``(out_img01, applied)`` - the (possibly) corrected HWC image in [0, 1]
        and a bool indicating whether a region was flattened.
    """
    mask = detect_flat_region(img01, grad_thresh=grad_thresh, min_region_frac=min_region_frac)
    if mask is None or not mask.any():
        return img01, False

    arr = img01.astype(np.float32, copy=True)
    dom = np.median(arr[mask], axis=0)              # (C,) dominant colour
    m = ndimage.binary_erosion(mask, iterations=max(0, int(erosion_iters)))
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), float(blur_sigma))[..., None]
    arr = arr * (1.0 - soft) + dom[None, None, :] * soft
    return np.clip(arr, 0.0, 1.0), True
