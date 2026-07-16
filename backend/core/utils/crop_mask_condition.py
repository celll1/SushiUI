"""Shared 4-channel outpaint ControlNet conditioning builder.

Used by BOTH sides of the outpaint-native ControlNet (PART B), so training and
inference build the byte-identical conditioning format (no train/infer skew):

  - TRAINING (self-supervised crop->full): the FULL target image is the teacher;
    a deterministic sub-rectangle (the OutpaintControlPlanner "crop") is the KNOWN
    region. The conditioning shows the model the known crop pixels at their true
    position, a neutral gray everywhere else, and a binary channel marking which
    is which -- so the net learns P(full | known-crop-at-position).

  - INFERENCE (crop_mask mode): the placed input occupies ``rect`` on the canvas;
    the SAME operation yields the identical 4-ch conditioning the model trained on.

Conditioning channels (H, W, 4), float32 in [0, 1]:
    0..2  known-region RGB pixels in [0,1] inside ``rect``; a constant ``gray``
          fill (default 0.5) everywhere outside. The fill is deliberately NOT a
          replicate/reflect of the crop -- the net must read "unknown" as a flat
          neutral, never as fabricated content.
    3     binary known-mask: 1.0 inside ``rect`` (given), 0.0 outside (to generate).
          This channel is what lets the very first ControlNet conv compute a
          distance-to-known and disambiguate "given" from "dark content".

Also returns the generate-side residual GATE (H, W) float32 = 1.0 outside ``rect``
(the region whose ControlNet residuals are kept) and 0.0 inside (the B1-pinned /
paste-preserved keep region, which must never be ControlNet-constrained). For a
TRAINED model no distance taper is needed (it learned where structure ends), so
the gate is a flat 1.0 over the whole generate region -- contrast PART A's
edge-extrapolation gate, which tapers with distance because its geometry is a guess.

CRITICAL: this module must import ONLY numpy / PIL / stdlib -- no ``api.*``, no
``core.inference`` / ``core.training``. It is imported by BOTH the training spine
(``base_trainer``) and the inference path (``outpaint_control``); a heavier import
here would create a training<->inference dependency cycle (see the Explore
hook-point map, section 7).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _to_float01(image: np.ndarray) -> np.ndarray:
    """Coerce an HxWx3 RGB array (uint8 or float) to float32 in [0, 1]."""
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"crop-mask condition expects HxWx3 RGB, got shape {a.shape}")
    a = a.astype(np.float32)
    if a.max() > 1.0 + 1e-6:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def build_crop_mask_condition(
    image: np.ndarray,
    rect: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    gray: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the shared 4-ch outpaint conditioning + generate-side residual gate.

    Args:
        image: HxWx3 RGB (uint8 or float). Its pixels inside ``rect`` are the KNOWN
            region. Must already be at the canvas/target resolution -- (H, W) must
            equal (canvas_h, canvas_w).
        rect: (x0, y0, x1, y1) the known-region rectangle in canvas pixels, half-open
            [x0, x1) x [y0, y1). Clamped to the canvas.
        canvas_size: (W, H) of the target canvas.
        gray: constant fill value for the unknown (to-generate) region, in [0, 1].

    Returns:
        (cond, gate):
            cond: (H, W, 4) float32 in [0, 1] -- channels 0..2 RGB (known pixels /
                  gray fill), channel 3 binary known-mask.
            gate: (H, W) float32 -- 1.0 over the generate region, 0.0 over the keep
                  rect (flat; a trained model needs no distance taper).
    """
    W, H = int(canvas_size[0]), int(canvas_size[1])
    img = _to_float01(image)
    if img.shape[0] != H or img.shape[1] != W:
        raise ValueError(
            f"image {img.shape[:2]} must match canvas (H,W)=({H},{W}); resize before calling"
        )

    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty/invalid rect {rect} on canvas {canvas_size}")

    known = np.zeros((H, W), dtype=np.float32)
    known[y0:y1, x0:x1] = 1.0
    m3 = known[:, :, None]

    rgb = img * m3 + float(gray) * (1.0 - m3)
    cond = np.concatenate([rgb.astype(np.float32), known[:, :, None]], axis=2)
    gate = (1.0 - known).astype(np.float32)
    return cond, gate
