"""Per-corner ControlNet residual gate for outpaint crop_mask conditioning.

Root cause (H1, CN-residual, vertex-feature-lock; see
scratchpad/outpaint_seam_diagnosis.md): the trained outpaint ControlNet locks
onto the 90-degree vertex feature of the rectangular known/unknown boundary
and re-projects it as a spurious seam LINE running from each placed-rect
corner into the generated region. The fix is NOT to gate the whole generate
region (that starves the CN of the edge continuation signal it needs, and is
already covered/rejected by the flat gateless design -- see
crop_mask_condition.py's module docstring) but to locally attenuate the CN
residual ONLY in small disks centered on the 4 rect vertices, leaving the
edges (away from corners) at full strength for cross-boundary continuation.

This module builds that field. It multiplies (not replaces) the existing
per-shape residual gate machinery in custom_sampling.py, which already knows
how to consume an (H, W) float32 gate and apply it as ``residual * gate``.

CRITICAL: same import constraints as crop_mask_condition.py -- numpy / PIL /
stdlib only, no ``api.*`` / ``core.inference`` / ``core.training``, so this
stays importable from both the inference path and (if ever needed) training.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def build_corner_gate(
    rect: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    radius_px: float,
    g_min: float,
) -> np.ndarray:
    """Build an (H, W) float32 field: 1.0 everywhere except local dips to
    ``g_min`` at the 4 vertices of ``rect``, cosine-tapered back to 1.0 at
    ``radius_px``.

    Args:
        rect: (x0, y0, x1, y1) half-open placed-input rectangle, canvas pixels.
        canvas_size: (W, H) of the target canvas (PIL ``Image.size`` order).
        radius_px: disk radius (canvas px) around each vertex within which the
            gate dips below 1.0. Clamped to 0.25 * the shorter rect side so a
            small rect can't have its two opposite-corner disks overlap and
            eat the whole rect. <= 0 is the caller's responsibility to treat
            as "disabled" (this function still returns an all-1.0 field for
            radius_px <= 0, matching the cosine formula's limit).
        g_min: gate value AT each vertex center, in [0, 1]. 1.0 = no dip
            (field is uniformly 1.0, i.e. disabled).

    Returns:
        gate: (H, W) float32 in [g_min, 1.0]. Field is built over the FULL
            disk around each vertex (both inside and outside the rect) --
            NOT restricted to the generate side, because a half-disk would
            introduce a new sharp semicircular boundary along the rect edge.
            The keep side of the final image is always byte-exact restored
            by the boundary paste regardless of what the gate does there, so
            a dip on the keep side is harmless.
    """
    W, H = int(canvas_size[0]), int(canvas_size[1])
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty/invalid rect {rect} on canvas {canvas_size}")

    g_min = float(np.clip(g_min, 0.0, 1.0))
    R = max(0.0, min(float(radius_px), 0.25 * min(x1 - x0, y1 - y0)))

    gate = np.ones((H, W), dtype=np.float32)
    if R <= 0.0 or g_min >= 1.0:
        return gate

    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)  # (H, W) each

    d_min = None
    for (cx, cy) in corners:
        d = np.sqrt((XX - float(cx)) ** 2 + (YY - float(cy)) ** 2)
        d_min = d if d_min is None else np.minimum(d_min, d)

    d_clip = np.clip(d_min / R, 0.0, 1.0)
    # Cosine taper: gate == g_min at d=0 (the vertex), == 1.0 at d>=R.
    taper = g_min + (1.0 - g_min) * 0.5 * (1.0 - np.cos(np.pi * d_clip))
    gate = taper.astype(np.float32)
    return gate


def _self_test() -> None:
    H, W = 80, 120
    rect = (20, 15, 90, 65)  # x0, y0, x1, y1
    canvas_size = (W, H)

    # 1) Disabled cases -> uniform 1.0 field.
    g0 = build_corner_gate(rect, canvas_size, radius_px=0.0, g_min=0.3)
    assert g0.shape == (H, W) and g0.dtype == np.float32
    assert np.array_equal(g0, np.ones((H, W), dtype=np.float32)), "radius=0 must be a no-op (all 1.0)"

    g1 = build_corner_gate(rect, canvas_size, radius_px=15.0, g_min=1.0)
    assert np.array_equal(g1, np.ones((H, W), dtype=np.float32)), "g_min=1.0 must be a no-op (all 1.0)"

    # 2) Enabled: far-from-corner pixels stay at 1.0.
    radius_px, g_min = 15.0, 0.2
    gate = build_corner_gate(rect, canvas_size, radius_px=radius_px, g_min=g_min)
    assert gate.shape == (H, W) and gate.dtype == np.float32
    x0, y0, x1, y1 = rect
    far_y, far_x = (y0 + y1) // 2, (x0 + x1) // 2  # rect center, far from all 4 corners
    assert abs(gate[far_y, far_x] - 1.0) < 1e-5, "gate far from any corner must be 1.0"

    # 3) At the exact vertex, gate == g_min.
    assert abs(gate[y0, x0] - g_min) < 1e-5, "gate at a vertex must equal g_min"
    assert abs(gate[y0, min(x1, W - 1)] - g_min) < 1e-4, "gate at the (x1,y0) vertex must equal g_min"

    # 4) Monotonic cosine return to 1.0 at exactly radius_px.
    edge_pt_y = y0
    edge_pt_x = x0 + int(radius_px)
    assert abs(gate[edge_pt_y, edge_pt_x] - 1.0) < 1e-3, "gate must return to ~1.0 at distance == radius_px"

    # 5) Radius clamp: a tiny rect (shorter side 8px) clamps radius to 2.0px.
    tiny_rect = (10, 10, 18, 40)  # width=8, height=30 -> 0.25*8=2.0 clamp
    gate_tiny = build_corner_gate(tiny_rect, canvas_size, radius_px=50.0, g_min=0.0)
    # At the clamped radius (2px) from a vertex along the short axis, gate should be ~1.0.
    tx0, ty0, tx1, ty1 = tiny_rect
    clamped_r = 0.25 * min(tx1 - tx0, ty1 - ty0)
    assert abs(clamped_r - 2.0) < 1e-6
    probe_x = min(tx0 + int(round(clamped_r)), W - 1)
    assert gate_tiny[ty0, probe_x] > 0.9, "radius must clamp to 0.25*shorter side, not the requested 50px"

    # 6) Determinism.
    gate2 = build_corner_gate(rect, canvas_size, radius_px=radius_px, g_min=g_min)
    assert np.array_equal(gate, gate2), "must be a pure function of its inputs"

    print("outpaint_corner_gate._self_test: OK")


if __name__ == "__main__":
    _self_test()
