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
    edge_feather_px: float = 0.0,
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
        edge_feather_px: half-width, in canvas pixels, of a soft ramp replacing the
            razor-sharp known/unknown perimeter transition. The ramp is drawn
            INWARD -- entirely inside the rect. Applied identically to BOTH channel
            3 (the known-mask goes from a binary 1.0/0.0 step to a linear ramp that
            is 1.0 in the eroded core >= edge_feather_px from the nearest rect edge,
            falls to 0.0 AT the boundary row, and is 0.0 everywhere outside the
            rect) and channels 0..2 (the RGB known-crop -> gray-fill step uses that
            same ramp as its blend weight), so the two channels never disagree about
            where the boundary is. 0.0 (default) reproduces the exact prior
            razor-sharp step byte-for-byte -- this parameter is purely additive and
            opt-in. Because the softening is inward, EVERYTHING OUTSIDE the rect is
            always exact flat ``gray`` + mask 0 regardless of feather: the cond
            outside the rect is fill-mode-independent (a replicate/reflect/other
            fill of the placed input never leaks into the ControlNet cond -- only
            the crop's own interior pixels, faded toward gray at their own edge, do).
            Only the eroded core stays pure known RGB; the inner perimeter band is
            what softens. ``feather`` is additionally clamped to 0.25 * the shorter
            rect side so even a tiny crop keeps a known core.

            Root-cause fix (D3-R1, ``scratchpad/outpaint_boundary_structure_fix.md``):
            a hard, axis-aligned known/unknown edge is the ONE thing this
            conditioning holds constant across every training sample (position,
            size, aspect and anchor mode are already randomized by
            ``OutpaintControlPlanner``), so a ControlNet trained on it learns to
            render the rect perimeter as scene structure (a "frame"). Per-sample
            RANDOMIZED softness (drawn in training, see
            ``OutpaintControlPlanner.feather_for``) removes that invariant. Must be
            applied identically in training and inference (the no-skew contract
            this module exists to uphold) -- softening at inference alone against a
            model trained on hard edges is a measured no-op (D2 of the same doc).

    Returns:
        (cond, gate):
            cond: (H, W, 4) float32 in [0, 1] -- channels 0..2 RGB (known pixels /
                  gray fill, softened at the perimeter iff edge_feather_px > 0),
                  channel 3 known-mask (binary iff edge_feather_px == 0, else a
                  ramp using the same softening).
            gate: (H, W) float32 -- 1.0 over the generate region, 0.0 over the keep
                  rect (flat; a trained model needs no distance taper). Always tied
                  to the HARD rect boundary regardless of edge_feather_px -- this
                  drives the self-supervised loss weight map (known vs generate
                  region), not the ControlNet's input signal, so it must reflect
                  the true crop, never a softened proxy of it.
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

    hard_known = np.zeros((H, W), dtype=np.float32)
    hard_known[y0:y1, x0:x1] = 1.0

    feather = float(edge_feather_px)
    if feather <= 0.0:
        # Default path: byte-identical to the pre-R1 razor-sharp implementation.
        known = hard_known
    else:
        # Inward rectangular distance: >=0 inside the rect (px to nearest rect
        # edge, per-axis min), <0 outside. x1-1/y1-1 is the last IN-rect index
        # (rect is half-open [x0,x1)).
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        dxi = np.minimum(xs - x0, (x1 - 1) - xs)      # per-axis inward dist
        dyi = np.minimum(ys - y0, (y1 - 1) - ys)
        din = np.minimum(dxi[None, :], dyi[:, None])  # (H,W); <0 outside rect
        feather = min(feather, 0.25 * min(x1 - x0, y1 - y0))  # guarantee a known core for small crops
        known = np.clip(din / max(feather, 1e-6), 0.0, 1.0).astype(np.float32)

    m3 = known[:, :, None]
    rgb = img * m3 + float(gray) * (1.0 - m3)
    cond = np.concatenate([rgb.astype(np.float32), known[:, :, None]], axis=2)
    # Gate always uses the HARD rect (see docstring) -- independent of feather.
    gate = (1.0 - hard_known).astype(np.float32)
    return cond, gate


def _self_test() -> None:
    """Lightweight correctness check for the edge_feather_px addition (R1).

    Not run automatically on import (see ``if __name__ == "__main__"`` below) --
    only exercised by an explicit ``python crop_mask_condition.py`` invocation or
    a dedicated test harness.
    """
    rng = np.random.default_rng(0)
    H, W = 64, 96
    image = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    rect = (20, 10, 70, 50)  # x0, y0, x1, y1
    canvas_size = (W, H)

    # 1) feather=0 (default) must equal the original razor-sharp construction.
    cond0, gate0 = build_crop_mask_condition(image, rect, canvas_size)
    x0, y0, x1, y1 = rect
    known_ref = np.zeros((H, W), dtype=np.float32)
    known_ref[y0:y1, x0:x1] = 1.0
    img01 = _to_float01(image)
    rgb_ref = img01 * known_ref[:, :, None] + 0.5 * (1.0 - known_ref[:, :, None])
    cond_ref = np.concatenate([rgb_ref.astype(np.float32), known_ref[:, :, None]], axis=2)
    gate_ref = (1.0 - known_ref).astype(np.float32)
    assert np.array_equal(cond0, cond_ref), "feather=0 must be byte-identical to the razor-sharp cond"
    assert np.array_equal(gate0, gate_ref), "feather=0 gate must be byte-identical"

    # 2) feather>0 (INWARD ramp): the softening lives entirely inside the rect.
    #    - The eroded core (>= feather px from the nearest rect edge) must still be
    #      exact known RGB + mask 1.
    #    - EVERYTHING outside the rect must be exact flat gray + mask 0 for ANY
    #      feather (the ramp never crosses the boundary; cond is fill-mode-
    #      independent outside the rect).
    #    - The inner perimeter band (0 <= din < feather) must contain strictly-
    #      intermediate mask values (evidence it actually softened something).
    #    - The gate must be unaffected by feather (still the hard rect).
    feather_px = 8.0
    cond_f, gate_f = build_crop_mask_condition(image, rect, canvas_size, edge_feather_px=feather_px)
    assert np.array_equal(gate_f, gate_ref), "gate must stay tied to the hard rect regardless of feather"

    # Inward distance field matching the implementation (feather is unclamped here
    # since feather_px=8 < 0.25*min(50,40)=10, so no clamp occurs).
    xs = np.arange(W)
    ys = np.arange(H)
    dxi = np.minimum(xs - x0, (x1 - 1) - xs)
    dyi = np.minimum(ys - y0, (y1 - 1) - ys)
    din = np.minimum(dxi[None, :], dyi[:, None])  # (H,W); <0 outside rect

    # Eroded core: known RGB + mask 1, exactly as the razor-sharp cond there.
    core = din >= feather_px
    assert core.any(), "test rect must be large enough to have an eroded known core"
    assert np.array_equal(cond_f[core], cond_ref[core]), "eroded core (>= feather px inside) must be exact known RGB + mask 1"

    # Outside the rect: exact flat gray + mask 0, for any feather (inward ramp).
    outside = din < 0
    assert np.all(cond_f[outside, 3] == 0.0), "outside-rect mask channel must stay exactly 0"
    assert np.allclose(cond_f[outside][:, :3], 0.5), "outside-rect RGB must stay exactly gray"

    # Inner perimeter band (inside the rect, within feather of an edge): soft ramp.
    band = (din >= 0) & (din < feather_px)
    assert band.any(), "test rect/canvas must produce a non-empty inner perimeter band"
    band_vals = cond_f[band, 3]
    assert np.any((band_vals > 0.0) & (band_vals < 1.0)), "inner perimeter band must contain a soft ramp, not a step"

    # 3) Determinism: same inputs -> exact same output (pure function).
    cond_f2, gate_f2 = build_crop_mask_condition(image, rect, canvas_size, edge_feather_px=feather_px)
    assert np.array_equal(cond_f, cond_f2) and np.array_equal(gate_f, gate_f2), "must be a pure function of its inputs"

    print("crop_mask_condition._self_test: OK")


if __name__ == "__main__":
    _self_test()
