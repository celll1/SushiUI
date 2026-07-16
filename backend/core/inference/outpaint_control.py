"""Outpaint ControlNet control-image builder (PART A -- edge extrapolation).

scratchpad/outpaint_controlnet_synthesis.md (fable + codex CONVERGED). Builds the
ControlNet control image that CONTINUES the structures crossing the outpaint
boundary into the generate region, for driving a general structure ControlNet
(e.g. the anytest CN) at inference. This is deliberately a SEPARATE module from
outpaint_utils.py (which enforces a numpy/PIL/stdlib-only import policy) because
it needs cv2 (edge detection) + scipy (geometry).

Mechanism (honest framing -- this ENFORCES a GUESSED geometry, so it decays
gracefully rather than pretending the guess is correct):
  1. detect edges on the PLACED (preserved) pixels ONLY -- never the synthetic
     canvas fill (its smeared streaks would become hallucinated control lines);
  2. find edge strands that cross a generate-adjacent boundary of the rect;
     fit their orientation (PCA/structure-tensor), gate on crossing angle,
     support, coherence, cap 8, drop-don't-guess on ambiguity;
  3. extrapolate each strand a SHORT range into the generate region with a
     DISTANCE-TAPERED confidence (cosine-squared) that hits exactly 0 at a finite
     depth -- NEVER rendering an endpoint/cap. The model decides where and how
     the structure terminates in the faded region.
Returns (control_img, gate): the canvas-size RGB control image, and a canvas-size
[H,W] float32 confidence field used TWICE -- as the rendered line intensity AND
(by the caller) as a spatial gate on the ControlNet residuals (a nonlinear CN
does not treat a half-bright line as half-strength conditioning). Returns None
when no eligible crossing exists (feature no-ops -> byte-identical).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# --- internal geometry containers -------------------------------------------
class _Crossing:
    __slots__ = ("py", "px", "ty", "tx", "width", "conf", "depth")

    def __init__(self, py, px, ty, tx, width, conf, depth):
        self.py = float(py); self.px = float(px)   # crossing point on the seam (canvas px)
        self.ty = float(ty); self.tx = float(tx)   # unit tangent, oriented OUTWARD into the gen region
        self.width = float(width)                  # strand half-width for rendering (px)
        self.conf = float(conf)                    # confidence s_k in [0,1]
        self.depth = float(depth)                  # extrapolation depth D_k (px)


def _run_detector(placed_np: np.ndarray, detector: str) -> np.ndarray:
    """Edge/lineart map (uint8 [h,w], 0..255, high = edge) over the placed pixels.
    canny = cv2 only (no model download). lineart/lineart_anime = controlnet_aux
    (opt-in). Falls back to canny if the aux detector is unavailable."""
    import cv2

    if detector in ("lineart", "lineart_anime"):
        try:
            from controlnet_aux import LineartDetector, LineartAnimeDetector
            cls = LineartAnimeDetector if detector == "lineart_anime" else LineartDetector
            det = cls.from_pretrained("lllyasviel/Annotators")
            out = np.asarray(det(Image.fromarray(placed_np)).convert("L"))
            # controlnet_aux lineart is black-on-white; invert to high=edge like Canny
            return (255 - out).astype(np.uint8)
        except Exception:
            pass  # model unavailable/offline -> fall through to canny (no download needed)
    gray = cv2.cvtColor(placed_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=5)
    return cv2.Canny(gray, 80, 160)


def _gen_adjacent_edges(rect: Tuple[int, int, int, int], canvas_size: Tuple[int, int]):
    """The rect edges that border the GENERATE region (not canvas-flush), each
    with its OUTWARD unit normal (pointing into the generate region)."""
    x0, y0, x1, y1 = rect
    W, H = canvas_size
    edges = []
    if x0 > 0:   edges.append(("left",   x0, (0.0, -1.0)))   # (side, seam-coord, normal (ty,tx))
    if x1 < W:   edges.append(("right",  x1 - 1, (0.0, 1.0)))
    if y0 > 0:   edges.append(("top",    y0, (-1.0, 0.0)))
    if y1 < H:   edges.append(("bottom", y1 - 1, (1.0, 0.0)))
    return edges


def _find_crossings(
    edge_map: np.ndarray,          # [rh, rw] over the placed rect
    rect: Tuple[int, int, int, int],
    side: str, seam: int, normal: Tuple[float, float],
    support_px: int, min_support_px: int, cross_min: float,
    depth_px: int, max_depth_fraction: float, canvas_size: Tuple[int, int],
    conf_thr: float,
) -> List[_Crossing]:
    """Detect edge strands crossing one generate-adjacent rect edge and return
    their extrapolation geometry. edge_map is in RECT-local coords; results are
    in CANVAS coords."""
    import cv2
    from scipy import ndimage

    x0, y0, x1, y1 = rect
    W, H = canvas_size
    ny, nx = normal
    rh, rw = edge_map.shape

    # Ribbon of the edge map just INSIDE the rect, within support_px of the seam.
    if side in ("left", "right"):
        seam_local = 0 if side == "left" else rw - 1
        lo = max(0, seam_local - support_px) if side == "right" else 0
        hi = rw if side == "right" else min(rw, support_px)
        ribbon = np.zeros_like(edge_map); ribbon[:, lo:hi] = edge_map[:, lo:hi]
    else:
        seam_local = 0 if side == "top" else rh - 1
        lo = max(0, seam_local - support_px) if side == "bottom" else 0
        hi = rh if side == "bottom" else min(rh, support_px)
        ribbon = np.zeros_like(edge_map); ribbon[lo:hi, :] = edge_map[lo:hi, :]

    # 8-connectivity (structure = full 3x3): a diagonal edge is a STAIRCASE whose
    # steps are only diagonally adjacent -- 4-connectivity would shatter it into
    # 1-3px fragments (none reaching the support threshold). A light dilation
    # first bridges Canny's small gaps along the strand.
    import cv2 as _cv2
    ribbon = _cv2.dilate((ribbon > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
    lbl, n_lbl = ndimage.label(ribbon > 0, structure=np.ones((3, 3), dtype=int))
    crossings: List[_Crossing] = []
    for lab in range(1, n_lbl + 1):
        ys, xs = np.where(lbl == lab)
        if ys.size < min_support_px:
            continue
        # touches the seam?
        if side == "left" and xs.min() > 2: continue
        if side == "right" and xs.max() < rw - 3: continue
        if side == "top" and ys.min() > 2: continue
        if side == "bottom" and ys.max() < rh - 3: continue

        # PCA tangent + coherence over the component pixels.
        pts = np.stack([ys.astype(np.float64), xs.astype(np.float64)], 1)
        c = pts.mean(0); d = pts - c
        cov = (d.T @ d) / max(len(d), 1)
        evals, evecs = np.linalg.eigh(cov)  # ascending
        lam2, lam1 = float(evals[0]), float(evals[1])
        coh = (lam1 - lam2) / (lam1 + lam2 + 1e-6)
        ty, tx = float(evecs[1, 0]), float(evecs[1, 1])  # principal dir (tangent)
        tnorm = (ty * ty + tx * tx) ** 0.5 + 1e-9
        ty, tx = ty / tnorm, tx / tnorm
        # orient the tangent OUTWARD (into the gen region): tau . normal > 0
        if ty * ny + tx * nx < 0:
            ty, tx = -ty, -tx
        xcross = abs(ty * ny + tx * nx)  # |tau . n|: reject seam-parallel edges

        if coh < 0.30 or xcross < cross_min:
            continue

        # crossing point on the seam (canvas coords): the component pixel nearest the seam
        if side in ("left", "right"):
            k = int(np.argmin(xs)) if side == "left" else int(np.argmax(xs))
            cy = float(ys[k]) + y0
            cx = float(seam)
        else:
            k = int(np.argmin(ys)) if side == "top" else int(np.argmax(ys))
            cy = float(seam)
            cx = float(xs[k]) + x0

        # strand half-width via the local edge distance transform (clamped)
        w = 1.0
        try:
            dt = cv2.distanceTransform((ribbon > 0).astype(np.uint8), cv2.DIST_L2, 3)
            w = float(np.clip(dt[ys, xs].mean() + 0.5, 0.75, 6.0))
        except Exception:
            pass

        # confidence from support + coherence + crossing directness
        conf = float(np.clip((min(ys.size, 200) / 200.0) * coh * xcross, 0.0, 1.0))
        if conf < conf_thr:
            continue

        # available ray length to the canvas edge along +tau
        ray_len = _ray_to_canvas_edge(cy, cx, ty, tx, W, H)
        D = min(float(depth_px), max_depth_fraction * ray_len)
        if D < 4.0:
            continue
        crossings.append(_Crossing(cy, cx, ty, tx, w, conf, D))

    # cap + conflict suppression: sort by confidence, keep top 8, drop near-duplicates
    crossings.sort(key=lambda c: -c.conf)
    kept: List[_Crossing] = []
    for c in crossings:
        if len(kept) >= 8:
            break
        dup = False
        for k in kept:
            if (c.py - k.py) ** 2 + (c.px - k.px) ** 2 < 100.0:  # <10px apart
                # keep both only if orientations agree (parallel rod strands); else drop weaker
                if abs(c.ty * k.ty + c.tx * k.tx) < 0.87:  # >30deg apart
                    dup = True
                    break
        if not dup:
            kept.append(c)
    return kept


def _ray_to_canvas_edge(y, x, ty, tx, W, H) -> float:
    """Distance from (y,x) along +(ty,tx) to the canvas boundary."""
    ts = []
    if tx > 1e-6:   ts.append((W - 1 - x) / tx)
    elif tx < -1e-6: ts.append((0 - x) / tx)
    if ty > 1e-6:   ts.append((H - 1 - y) / ty)
    elif ty < -1e-6: ts.append((0 - y) / ty)
    return float(min([t for t in ts if t > 0], default=0.0))


def build_outpaint_control_image(
    placed_img: Image.Image,
    rect: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    detector: str = "canny",
    depth_px: int = 160,
    taper_power: float = 2.0,
    support_px: int = 64,
    min_support_px: int = 32,
    cross_min: float = 0.25,
    max_depth_fraction: float = 0.35,
    conf_thr: float = 0.30,
    supersample: int = 2,
) -> Optional[Tuple[Image.Image, np.ndarray]]:
    """Build the outpaint ControlNet control image + confidence gate.

    Returns (control_img RGB [W,H], gate float32 [H,W] in [0,1]) or None when no
    eligible boundary-crossing structure is found (-> caller injects no CN entry,
    byte-identical). The gate = max_k C_k(d,u) over the generate region (0 in the
    keep rect and beyond every strand's finite taper depth). See module docstring.
    """
    import cv2

    W, H = canvas_size
    x0, y0, x1, y1 = rect
    placed_np = np.asarray(placed_img.convert("RGB"))
    edge_rect = _run_detector(placed_np, detector)  # [rh, rw]

    edges = _gen_adjacent_edges(rect, canvas_size)
    all_cross: List[_Crossing] = []
    for side, seam, normal in edges:
        all_cross += _find_crossings(
            edge_rect, rect, side, seam, normal,
            support_px, min_support_px, cross_min,
            depth_px, max_depth_fraction, canvas_size, conf_thr,
        )
    if not all_cross:
        return None

    # --- render at supersample resolution, then downsample once (anti-alias) ---
    ss = max(1, int(supersample))
    Hs, Ws = H * ss, W * ss
    ctrl = np.zeros((Hs, Ws), dtype=np.float32)   # known edges + extrapolated strands
    gate = np.zeros((Hs, Ws), dtype=np.float32)   # confidence field (gen side only)

    # known-region edges: real detector output, at native strength, inside the rect
    ctrl[y0 * ss:y1 * ss, x0 * ss:x1 * ss] = cv2.resize(
        edge_rect.astype(np.float32), ((x1 - x0) * ss, (y1 - y0) * ss),
        interpolation=cv2.INTER_NEAREST,
    ) / 255.0

    # generate-region mask (1 = generate) at supersample res
    genmask = np.ones((Hs, Ws), dtype=np.float32)
    genmask[y0 * ss:y1 * ss, x0 * ss:x1 * ss] = 0.0

    for c in all_cross:
        # march the tapered strand from the seam outward to depth D
        n_steps = int(c.depth * ss) + 1
        wpx = max(c.width * ss, 1.0)
        for si in range(n_steps):
            d = si / ss  # distance in canvas px
            frac = d / max(c.depth, 1e-6)
            if frac >= 1.0:
                break
            # cosine-squared taper: full at the seam, smooth -> 0 at depth D
            conf = c.conf * (0.5 * (1.0 + np.cos(np.pi * frac))) ** taper_power
            cy = (c.py + c.ty * d) * ss
            cx = (c.px + c.tx * d) * ss
            iy, ix = int(round(cy)), int(round(cx))
            r = int(np.ceil(wpx))
            y_lo, y_hi = max(0, iy - r), min(Hs, iy + r + 1)
            x_lo, x_hi = max(0, ix - r), min(Ws, ix + r + 1)
            if y_lo >= y_hi or x_lo >= x_hi:
                continue
            yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
            u2 = (yy - cy) ** 2 + (xx - cx) ** 2
            stamp = np.exp(-u2 / (2.0 * wpx * wpx)) * conf
            ctrl[y_lo:y_hi, x_lo:x_hi] = np.maximum(ctrl[y_lo:y_hi, x_lo:x_hi], stamp)
            gate[y_lo:y_hi, x_lo:x_hi] = np.maximum(gate[y_lo:y_hi, x_lo:x_hi], stamp)

    # confine control's extrapolated part + the gate to the generate region
    gate *= genmask
    # downsample once
    if ss > 1:
        ctrl = cv2.resize(ctrl, (W, H), interpolation=cv2.INTER_AREA)
        gate = cv2.resize(gate, (W, H), interpolation=cv2.INTER_AREA)
    ctrl = np.clip(ctrl, 0.0, 1.0)
    gate = np.clip(gate, 0.0, 1.0)
    gate[y0:y1, x0:x1] = 0.0  # hard-zero the keep rect (B1 owns keep)

    control_img = Image.fromarray((ctrl * 255.0).astype(np.uint8), mode="L").convert("RGB")
    return control_img, gate.astype(np.float32)
