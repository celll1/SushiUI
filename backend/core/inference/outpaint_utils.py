"""Dependency-light image-outpaint orchestration helpers.

Outpaint places a (optionally cropped/resized) input image inside a LARGER
canvas and generates everything outside the placed region, while PRESERVING
the placed input rectangle byte-exact. These helpers build the enlarged
canvas + an outward-only-blurred mask consumed by the existing all-architecture
``PipelineManager.generate_inpaint`` (see ``core.pipeline.generate_outpaint``),
then perform an UNCONDITIONAL final pixel paste of the placed rectangle. That
paste -- not any per-arch inpaint compositing -- is the strict-preservation
guarantee (per-arch inpaint pixel/latent compositing is either gated on
strength=1.0 (SD1.5/SDXL) or latent-only/approximate (every other
architecture); see ``scratchpad/outpaint_design.md`` section 1.2).

CRITICAL: this module must import ONLY ``numpy``/``PIL``/stdlib -- no
``api.*``/``core.pipeline``/other backend package (mirrors
``core.utils.tile_blend``'s decoupling policy) so it stays trivially
unit-testable and side-effect-free at import time.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Placed rectangles smaller than this (in either dimension) are rejected --
# there would be nothing meaningful to preserve/generate around.
_MIN_RECT_PX = 8


def validate_and_snap_placement(
    params: Dict[str, Any],
    input_size: Tuple[int, int],
    align: int = 8,
    snap: int = 8,
) -> Dict[str, int]:
    """Resolve and clamp the outpaint placement geometry.

    Reads ``canvas_width``/``canvas_height``, ``place_x``/``place_y``/
    ``place_width``/``place_height`` and ``input_crop_x``/``input_crop_y``/
    ``input_crop_w``/``input_crop_h`` from ``params`` (all optional; 0/absent
    means "auto" as documented per-key below) and returns the fully resolved,
    integer, in-bounds geometry.

    Resolution order:
      1. Canvas size is rounded to the nearest multiple of ``align`` (the
         loaded architecture's required latent-grid multiple), minimum
         ``align``.
      2. The input crop (trim) is clamped to the input image bounds;
         ``input_crop_w``/``input_crop_h`` <= 0 means "to the input edge".
         An empty resulting crop is rejected.
      3. The placed size defaults to the (cropped) input's native size when
         ``place_width``/``place_height`` <= 0, then is capped to the canvas
         size (a placed rect can never exceed the canvas).
      4. The placed rect's top-left is clamped so the whole rect lies inside
         the canvas.
      5. If ``snap`` > 0, the placed rect (position + size) is snapped to
         that pixel grid and re-clamped. This is a UI/latent-grid convenience
         only -- the strict-preservation contract (see
         ``paste_preserved_region``) does not depend on alignment.
      6. Degenerate geometry is rejected: a placed rect under
         ``_MIN_RECT_PX`` in either dimension, or one that fully covers the
         canvas (nothing left to generate).

    Returns:
        Dict with the resolved integer keys: ``canvas_width``,
        ``canvas_height``, ``place_x``, ``place_y``, ``place_width``,
        ``place_height``, ``input_crop_x``, ``input_crop_y``,
        ``input_crop_w``, ``input_crop_h``.

    Raises:
        ValueError: on degenerate/empty geometry (empty crop, placed rect
            too small, placed rect fully covering the canvas).
    """
    in_w, in_h = input_size
    if in_w <= 0 or in_h <= 0:
        raise ValueError(f"Invalid input image size: {input_size}")

    align = max(1, int(align))
    canvas_w = int(params.get("canvas_width") or 0)
    canvas_h = int(params.get("canvas_height") or 0)
    canvas_w = max(align, int(round(canvas_w / align)) * align)
    canvas_h = max(align, int(round(canvas_h / align)) * align)

    # --- Crop (trim) of the input image before placement ---
    crop_x = int(params.get("input_crop_x") or 0)
    crop_y = int(params.get("input_crop_y") or 0)
    crop_w = int(params.get("input_crop_w") or 0)
    crop_h = int(params.get("input_crop_h") or 0)

    # Clamp to [0, in_w]/[0, in_h] (NOT in_w-1/in_h-1): crop_x/crop_y == the
    # input's edge is a valid clamp target that correctly resolves to a
    # zero-width/height crop below (rejected as empty), rather than being
    # silently recovered into a spurious 1px sliver.
    crop_x = max(0, min(crop_x, in_w))
    crop_y = max(0, min(crop_y, in_h))
    if crop_w <= 0:
        crop_w = in_w - crop_x
    if crop_h <= 0:
        crop_h = in_h - crop_y
    crop_w = min(crop_w, in_w - crop_x)
    crop_h = min(crop_h, in_h - crop_y)
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(
            f"Empty input crop: crop=({crop_x},{crop_y},{crop_w},{crop_h}) "
            f"input_size={input_size}"
        )

    # --- Placed size: 0 = input native size (after crop) ---
    place_w = int(params.get("place_width") or 0)
    place_h = int(params.get("place_height") or 0)
    if place_w <= 0:
        place_w = crop_w
    if place_h <= 0:
        place_h = crop_h

    # A placed rect can never be larger than the canvas itself.
    place_w = min(place_w, canvas_w)
    place_h = min(place_h, canvas_h)

    place_x = int(params.get("place_x") or 0)
    place_y = int(params.get("place_y") or 0)
    place_x = max(0, min(place_x, canvas_w - place_w))
    place_y = max(0, min(place_y, canvas_h - place_h))

    # --- Optional snap-to-grid (UI convenience; correctness independent) ---
    if snap and snap > 0:
        def _snap(value: int) -> int:
            return int(round(value / snap) * snap)

        place_w = max(_MIN_RECT_PX, _snap(place_w))
        place_h = max(_MIN_RECT_PX, _snap(place_h))
        place_x = _snap(place_x)
        place_y = _snap(place_y)
        # Re-clamp -- snapping can push the rect outside the canvas.
        place_w = min(place_w, canvas_w)
        place_h = min(place_h, canvas_h)
        place_x = max(0, min(place_x, canvas_w - place_w))
        place_y = max(0, min(place_y, canvas_h - place_h))

    # --- Reject degenerate geometry ---
    if place_w < _MIN_RECT_PX or place_h < _MIN_RECT_PX:
        raise ValueError(
            f"Placed rect too small: {place_w}x{place_h} "
            f"(minimum {_MIN_RECT_PX}px per side)"
        )
    if place_w >= canvas_w and place_h >= canvas_h:
        raise ValueError(
            f"Placed rect ({place_w}x{place_h}) fully covers the canvas "
            f"({canvas_w}x{canvas_h}) -- nothing to generate"
        )

    return {
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "place_x": place_x,
        "place_y": place_y,
        "place_width": place_w,
        "place_height": place_h,
        "input_crop_x": crop_x,
        "input_crop_y": crop_y,
        "input_crop_w": crop_w,
        "input_crop_h": crop_h,
    }


def _pad_with_mode(
    arr: np.ndarray, top: int, bottom: int, left: int, right: int, mode: str
) -> np.ndarray:
    """``np.pad`` wrapper that supports pad widths LARGER than the source
    array (numpy's own ``reflect``/``symmetric`` modes cap the pad width at
    ``dim - 1`` per call), by applying the pad incrementally.
    """
    if mode == "edge":
        # np.pad(mode="edge") has no such size restriction.
        return np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="edge")

    result = arr
    t, b, l, r = top, bottom, left, right
    while t or b or l or r:
        h, w = result.shape[0], result.shape[1]
        step_t = min(t, max(0, h - 1))
        step_b = min(b, max(0, h - 1))
        step_l = min(l, max(0, w - 1))
        step_r = min(r, max(0, w - 1))
        if step_t == 0 and step_b == 0 and step_l == 0 and step_r == 0:
            # Degenerate 1px source along an axis -- reflect cannot make
            # progress; finish the remaining pad with edge-extension.
            result = np.pad(result, ((t, b), (l, r), (0, 0)), mode="edge")
            break
        result = np.pad(
            result, ((step_t, step_b), (step_l, step_r), (0, 0)), mode=mode
        )
        t -= step_t
        b -= step_b
        l -= step_l
        r -= step_r
    return result


def _fill_canvas(
    canvas_w: int,
    canvas_h: int,
    placed_img: Image.Image,
    rect: Tuple[int, int, int, int],
    fill_mode: str,
) -> Image.Image:
    """Build the full canvas-sized background fill for ``outpaint_fill_mode``.

    The area under ``rect`` need not be exact here -- ``build_outpaint_canvas``
    always pastes ``placed_img`` on top afterward.
    """
    x0, y0, x1, y1 = rect
    fill_mode = (fill_mode or "replicate").lower()

    if fill_mode in ("replicate", "reflect"):
        arr = np.array(placed_img.convert("RGB"))
        top, bottom, left, right = y0, canvas_h - y1, x0, canvas_w - x1
        np_mode = "edge" if fill_mode == "replicate" else "reflect"
        padded = _pad_with_mode(arr, top, bottom, left, right, np_mode)
        return Image.fromarray(padded.astype(np.uint8), mode="RGB")

    if fill_mode == "mean":
        arr = np.array(placed_img.convert("RGB")).reshape(-1, 3)
        mean_color = tuple(int(round(c)) for c in arr.mean(axis=0))
        return Image.new("RGB", (canvas_w, canvas_h), mean_color)

    if fill_mode == "noise":
        noise = np.random.randint(0, 256, size=(canvas_h, canvas_w, 3), dtype=np.uint8)
        return Image.fromarray(noise, mode="RGB")

    raise ValueError(f"Unknown outpaint_fill_mode: {fill_mode!r}")


def build_outpaint_canvas(
    input_img: Image.Image, params: Dict[str, Any], align: int = 16
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """Build the enlarged outpaint canvas.

    Pipeline: crop (trim) -> resize to (place_width, place_height) with
    LANCZOS ONCE in pixel space (the result, ``placed_img``, IS the preserved
    content) -> paste onto a canvas of (canvas_width, canvas_height)
    pre-filled per ``outpaint_fill_mode``.

    ``align`` defaults to 16 (not 8): 7 of the 9 image architectures re-round
    their own canvas to a 16px grid internally (FLUX.2/Anima floor down,
    Lens rounds to nearest-16, Ideogram4/MiniT2I round to 16, Krea2 rounds
    UP) -- a canvas that is only 8-aligned can come back from
    ``generate_inpaint`` at a DIFFERENT size than it was sent in at, which
    would silently misalign (or clip) the preserved rect. 16-alignment is a
    fixed point for every architecture's grid (SD's own /8 divides evenly
    into it too), so it is universal-safe. See ``reconcile_and_paste`` for
    the defensive belt-and-suspenders half of this fix.

    Returns:
        (canvas_img, placed_img, rect) where ``rect`` is the placed
        rectangle in canvas pixel coordinates as
        (x0, y0, x1, y1) (half-open, i.e. x1/y1 are exclusive).
    """
    resolved = validate_and_snap_placement(params, input_img.size, align=align)

    crop_box = (
        resolved["input_crop_x"],
        resolved["input_crop_y"],
        resolved["input_crop_x"] + resolved["input_crop_w"],
        resolved["input_crop_y"] + resolved["input_crop_h"],
    )
    cropped = input_img.convert("RGB").crop(crop_box)
    placed_img = cropped.resize(
        (resolved["place_width"], resolved["place_height"]), Image.Resampling.LANCZOS
    )

    canvas_w, canvas_h = resolved["canvas_width"], resolved["canvas_height"]
    rect = (
        resolved["place_x"],
        resolved["place_y"],
        resolved["place_x"] + resolved["place_width"],
        resolved["place_y"] + resolved["place_height"],
    )

    fill_mode = params.get("outpaint_fill_mode", "replicate")
    canvas_img = _fill_canvas(canvas_w, canvas_h, placed_img, rect, fill_mode)
    # Unconditional -- guarantees the placed rect is exactly `placed_img`
    # regardless of any imprecision in the fill-mode construction above.
    canvas_img.paste(placed_img, (rect[0], rect[1]))

    return canvas_img, placed_img, rect


def build_outpaint_mask(
    canvas_size: Tuple[int, int],
    rect: Tuple[int, int, int, int],
    mask_blur: int,
) -> Image.Image:
    """Build the outpaint inpainting mask (mode "L").

    White (255) = generate, black (0) = keep original, everywhere EXCEPT the
    softened transition band lies entirely OUTSIDE the preserved rect
    (outward-only blur) -- unlike stock inpaint's symmetric mask_blur, which
    blends a band on both sides of the mask edge. This is achieved by
    Gaussian-blurring the hard mask, then re-clamping the interior of `rect`
    back to 0.
    """
    canvas_w, canvas_h = canvas_size
    x0, y0, x1, y1 = rect

    mask = Image.new("L", (canvas_w, canvas_h), 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=0)

    if mask_blur and mask_blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))

    # Hard-clamp: force the entire preserved rect back to 0, regardless of
    # blur bleed-in, so the transition band lies entirely outside it.
    mask_arr = np.array(mask)
    mask_arr[y0:y1, x0:x1] = 0
    return Image.fromarray(mask_arr, mode="L")


def compose_outpaint_start(keep_start: Any, native_noise_start: Any, mask: Any) -> Any:
    """Shared pure init-compose helper for outpaint's noise-init mode.

    ``z_t0 = (1 - mask) * keep_start + mask * native_noise_start``.

    Mask convention is 1 = GENERATE, matching ``build_outpaint_mask``'s
    white=generate convention and every backend's ``mask_latent``/packed-mask
    convention already in use in their inpaint sampling loops. The KEEP
    region (``mask == 0``) gets the architecture's normal noised/blended init
    (``keep_start``, e.g. ``scheduler.add_noise(z0, eps, t0)`` for SD/SDXL);
    the GENERATE region (``mask == 1``) gets the architecture's own NATIVE
    txt2img start (``native_noise_start``, e.g. ``eps * init_noise_sigma`` for
    SD/SDXL, or plain ``eps`` at sigma=1 for flow-matching archs) --
    independent of the canvas fill. This is what removes the
    encode(canvas-fill) artifact from the generated region (see
    ``scratchpad/outpaint_noise_init_design.md``).

    Deliberately generic: ``keep_start``/``native_noise_start``/``mask`` may be
    numpy arrays or torch tensors (whatever the calling backend's sampling
    loop already has in scope for its latent/patchified representation) --
    only ``-``/``*``/``+`` operators are used, so no tensor-library import is
    needed here, keeping this module's numpy/PIL/stdlib-only import policy
    intact. Each backend calls this locally (mask/latent layout -- BCHW /
    packed / patchified -- differs per architecture).
    """
    return (1 - mask) * keep_start + mask * native_noise_start


def match_generated_exposure(
    result_img: Image.Image,
    placed_img: Image.Image,
    rect: Tuple[int, int, int, int],
    mask_blur: int,
    strip_px: int = 16,
    gain_min: float = 0.67,
    gain_max: float = 1.5,
) -> Image.Image:
    """Arch-independent multiplicative exposure/tone harmonizer.

    Noise-init (``compose_outpaint_start``) removes the encode(fill) artifact
    but does NOT by itself remove tone/exposure mismatch between the model's
    generated pixels and the exact preserved rectangle -- large denoise spans
    commonly drift exposure, producing a visible tonal step right at the rect
    boundary. This measures, independently for each rect edge that borders a
    GENERATED region (i.e. there is canvas beyond that edge), the ratio
    between:
      - an INNER strip (``strip_px`` wide/tall), sampled from ``placed_img``
        (the ground-truth preserved content) just inside the rect at that
        edge, and
      - an OUTER strip (``strip_px``), sampled from ``result_img`` (the
        generated result), starting ``mask_blur`` pixels past the rect edge
        (skipping the blended transition band baked into the outpaint mask).

    A per-channel gain ``clip(median(inner)/median(outer), gain_min, gain_max)``
    is applied multiplicatively to the GENERATED pixels bordering that edge
    (never inside the rect), weighted by an outward cosine taper from 1.0 at
    the rect edge to 0.0 at distance ``W`` (no correction beyond that). A
    reliability gate skips any edge whose inner/outer strip is empty or
    near-clipped (median near 0 or 255) -- these are inherently unreliable to
    measure, so no correction (gain=1) is safer than a bad one.

    ``rect`` pixels are never read from ``result_img`` for gain measurement
    (only ``placed_img`` is used as the inner-strip source) and are never
    written by this function -- the crop under ``rect`` is copied back
    unchanged as a final defensive guard, even though the row/column slicing
    below already excludes it, keeping the invariant enforced by code rather
    than by care alone. The caller still performs the real, unconditional
    ``paste_preserved_region`` afterward regardless.
    """
    x0, y0, x1, y1 = rect
    canvas_w, canvas_h = result_img.size

    result_arr = np.array(result_img.convert("RGB")).astype(np.float64)
    placed_arr = np.array(placed_img.convert("RGB")).astype(np.float64)
    corrected = result_arr.copy()

    strip = max(1, int(strip_px))
    skip = max(0, int(mask_blur))
    extent = max(canvas_w, canvas_h)
    taper_w = float(min(max(extent // 2, 64), 256))

    def _reliable(inner: np.ndarray, outer: np.ndarray) -> bool:
        if inner.size == 0 or outer.size == 0:
            return False
        inner_med = np.median(inner.reshape(-1, 3), axis=0)
        outer_med = np.median(outer.reshape(-1, 3), axis=0)
        # Near-clipped strips (crushed blacks / blown highlights) make the
        # ratio unstable/meaningless -- skip rather than guess.
        if np.any(inner_med < 2) or np.any(inner_med > 253):
            return False
        if np.any(outer_med < 2) or np.any(outer_med > 253):
            return False
        return True

    def _apply(region_slice: Tuple[slice, slice], dist: np.ndarray, gain: np.ndarray, axis: int) -> None:
        """Blend `region` toward `region * gain` via an outward cosine taper.

        ``dist`` is the per-row (axis=0) or per-column (axis=1) distance (in
        px) from the rect edge; ``axis`` selects which side broadcasts.
        """
        w = np.where(dist < taper_w, 0.5 * (1.0 + np.cos(np.pi * dist / taper_w)), 0.0)
        w = w[:, None, None] if axis == 0 else w[None, :, None]
        region = corrected[region_slice]
        corrected[region_slice] = region * (1.0 + w * (gain[None, None, :] - 1.0))

    # top: canvas above the rect is generated content.
    if y0 > 0:
        inner = placed_arr[0:min(strip, y1 - y0), 0:(x1 - x0), :]
        outer_end = max(0, y0 - skip)
        outer_start = max(0, outer_end - strip)
        outer = result_arr[outer_start:outer_end, x0:x1, :]
        if _reliable(inner, outer):
            gain = np.clip(
                np.median(inner.reshape(-1, 3), axis=0) / (np.median(outer.reshape(-1, 3), axis=0) + 1e-6),
                gain_min, gain_max,
            )
            ys = np.arange(0, y0)
            dist = (y0 - ys).astype(np.float64) - 1.0  # 0 at the row touching the rect
            _apply((slice(0, y0), slice(x0, x1)), dist, gain, axis=0)

    # bottom: canvas below the rect is generated content.
    if y1 < canvas_h:
        inner = placed_arr[max(0, (y1 - y0) - strip):(y1 - y0), 0:(x1 - x0), :]
        outer_start = min(canvas_h, y1 + skip)
        outer_end = min(canvas_h, outer_start + strip)
        outer = result_arr[outer_start:outer_end, x0:x1, :]
        if _reliable(inner, outer):
            gain = np.clip(
                np.median(inner.reshape(-1, 3), axis=0) / (np.median(outer.reshape(-1, 3), axis=0) + 1e-6),
                gain_min, gain_max,
            )
            ys = np.arange(y1, canvas_h)
            dist = (ys - y1).astype(np.float64)
            _apply((slice(y1, canvas_h), slice(x0, x1)), dist, gain, axis=0)

    # left: canvas left of the rect is generated content.
    if x0 > 0:
        inner = placed_arr[0:(y1 - y0), 0:min(strip, x1 - x0), :]
        outer_end = max(0, x0 - skip)
        outer_start = max(0, outer_end - strip)
        outer = result_arr[y0:y1, outer_start:outer_end, :]
        if _reliable(inner, outer):
            gain = np.clip(
                np.median(inner.reshape(-1, 3), axis=0) / (np.median(outer.reshape(-1, 3), axis=0) + 1e-6),
                gain_min, gain_max,
            )
            xs = np.arange(0, x0)
            dist = (x0 - xs).astype(np.float64) - 1.0
            _apply((slice(y0, y1), slice(0, x0)), dist, gain, axis=1)

    # right: canvas right of the rect is generated content.
    if x1 < canvas_w:
        inner = placed_arr[0:(y1 - y0), max(0, (x1 - x0) - strip):(x1 - x0), :]
        outer_start = min(canvas_w, x1 + skip)
        outer_end = min(canvas_w, outer_start + strip)
        outer = result_arr[y0:y1, outer_start:outer_end, :]
        if _reliable(inner, outer):
            gain = np.clip(
                np.median(inner.reshape(-1, 3), axis=0) / (np.median(outer.reshape(-1, 3), axis=0) + 1e-6),
                gain_min, gain_max,
            )
            xs = np.arange(x1, canvas_w)
            dist = (xs - x1).astype(np.float64)
            _apply((slice(y0, y1), slice(x1, canvas_w)), dist, gain, axis=1)

    # Round (not truncate) before quantizing back to uint8 -- floating-point
    # imprecision in the gain math (e.g. an exact-1.0 ratio computed as
    # 0.999999...) would otherwise silently shave 1 off an unmodified pixel.
    corrected = np.clip(np.round(corrected), 0, 255).astype(np.uint8)
    out = Image.fromarray(corrected, mode="RGB")
    # Defensive guard (belt-and-suspenders): restore the rect crop from the
    # pre-harmonizer result unconditionally, even though none of the region
    # slices above ever include it.
    out.paste(result_img.crop(rect), (x0, y0))
    return out


def paste_preserved_region(
    result_img: Image.Image,
    placed_img: Image.Image,
    rect: Tuple[int, int, int, int],
) -> Image.Image:
    """Unconditional final pixel paste of the preserved input rectangle.

    THE strict-preservation contract: regardless of architecture, denoising
    strength, VAE round-trip drift, or any per-arch inpaint compositing
    (or lack thereof), the returned image's ``rect`` pixels are byte-identical
    to ``placed_img``.
    """
    result = result_img.copy()
    result.paste(placed_img, (rect[0], rect[1]))
    return result


def reconcile_and_paste(
    result_img: Image.Image,
    placed_img: Image.Image,
    rect: Tuple[int, int, int, int],
    canvas_size: Tuple[int, int],
    mask_blur: int = 4,
    outpaint_seam_fix: bool = True,
) -> Image.Image:
    """Defensive belt-and-suspenders wrapper around ``paste_preserved_region``.

    ``rect`` is computed against ``canvas_size`` (the canvas ``generate_inpaint``
    was invoked with), but several architectures re-round their working
    resolution to their own internal grid (see ``build_outpaint_canvas``'s
    ``align`` docstring), so the DECODED image that comes back can differ in
    size from ``canvas_size`` even when the canvas was built 16-aligned. If
    that happens, pasting ``placed_img`` at ``rect`` directly would land
    offset (or silently clip up to the size delta at the right/bottom edge).

    If ``result_img.size != canvas_size``, the (generated) result is resized
    back to ``canvas_size`` FIRST -- this only touches the generated
    surroundings. Then, when ``outpaint_seam_fix`` is True, the arch-independent
    exposure harmonizer (``match_generated_exposure``) corrects the generated
    surroundings' tone against the preserved rect (using ``mask_blur`` to skip
    the mask's blended transition band). Finally ``placed_img`` is pasted at
    ``rect`` exactly as ``paste_preserved_region`` does -- this paste is always
    the LAST mutation, re-establishing byte-exactness of the preserved rect
    regardless of any architecture-side re-rounding or the harmonizer above.
    """
    if result_img.size != canvas_size:
        result_img = result_img.resize(canvas_size, Image.Resampling.LANCZOS)
    if outpaint_seam_fix:
        result_img = match_generated_exposure(result_img, placed_img, rect, mask_blur)
    return paste_preserved_region(result_img, placed_img, rect)
