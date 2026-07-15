"""Dependency-light tile geometry + feather-blend helpers shared by any
tiled-decode/upscale backend (``core.upscaler``'s diffusion tile upscale,
``PidVaeWrapper``'s tiled PiD decode, and any future tiled backend).

CRITICAL: this module must import ONLY ``torch``/``numpy``/stdlib — no
``api.*`` or other backend package. ``PidVaeWrapper`` is constructed on the
hot generation path and must not transitively pull in ``api.error_handlers``
(which fires ``TaglistCache``/settings side effects at import time); moving
these two helpers out of ``core.upscaler`` (which DOES import
``api.error_handlers``) into this standalone module is what keeps that path
side-effect-free.
"""
from __future__ import annotations

from typing import List, Tuple

from PIL import Image


def compute_tile_boxes(
    width: int, height: int, tile_size: int, tile_overlap: int
) -> List[Tuple[int, int, int, int]]:
    """Compute tile crop boxes covering (width, height) in a given coordinate
    space (output pixels, or latent cells — the caller decides the unit).

    Tile dims are snapped to a multiple of 8. Edge tiles are pulled inward
    (rather than padded) so every tile is a real crop of the image; edge
    tiles may overlap their neighbor more than ``tile_overlap``.
    """
    if tile_size <= 0:
        return [(0, 0, width, height)]

    tile_size = max(8, int(round(tile_size / 8)) * 8)
    tile_w = min(tile_size, (width // 8) * 8 or width)
    tile_h = min(tile_size, (height // 8) * 8 or height)
    tile_w = max(8, tile_w)
    tile_h = max(8, tile_h)

    step_x = max(1, tile_w - tile_overlap)
    step_y = max(1, tile_h - tile_overlap)

    xs = list(range(0, max(1, width - tile_w) + 1, step_x))
    if not xs or xs[-1] != width - tile_w:
        xs.append(max(0, width - tile_w))
    ys = list(range(0, max(1, height - tile_h) + 1, step_y))
    if not ys or ys[-1] != height - tile_h:
        ys.append(max(0, height - tile_h))
    xs = sorted(set(max(0, min(x, width - tile_w)) for x in xs))
    ys = sorted(set(max(0, min(y, height - tile_h)) for y in ys))

    boxes = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, x + tile_w, y + tile_h))
    return boxes


def feather_blend_tiles(
    width: int,
    height: int,
    boxes: List[Tuple[int, int, int, int]],
    tile_images: List[Image.Image],
    tile_overlap: int,
) -> Image.Image:
    """Blend ``tile_images`` (aligned with ``boxes``) back into a
    ``(width, height)`` canvas using a linear feather ramp proportional to
    ``tile_overlap``."""
    import numpy as np

    canvas = np.zeros((height, width, 3), dtype="float32")
    weight = np.zeros((height, width, 1), dtype="float32")

    for (x1, y1, x2, y2), tile_img in zip(boxes, tile_images):
        tw, th = x2 - x1, y2 - y1
        arr = np.asarray(tile_img.convert("RGB"), dtype="float32")
        if arr.shape[0] != th or arr.shape[1] != tw:
            tile_img = tile_img.resize((tw, th), resample=Image.Resampling.LANCZOS)
            arr = np.asarray(tile_img.convert("RGB"), dtype="float32")

        feather_px = max(0, int(tile_overlap))
        ramp_y = np.ones(th, dtype="float32")
        ramp_x = np.ones(tw, dtype="float32")
        n = min(feather_px, th // 2)
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype="float32")
            ramp_y[:n] = ramp
            ramp_y[-n:] = ramp[::-1]
        n = min(feather_px, tw // 2)
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype="float32")
            ramp_x[:n] = ramp
            ramp_x[-n:] = ramp[::-1]
        w_mask = (ramp_y[:, None] * ramp_x[None, :])[:, :, None]

        canvas[y1:y2, x1:x2, :] += arr * w_mask
        weight[y1:y2, x1:x2, :] += w_mask

    weight = np.clip(weight, 1e-6, None)
    result = canvas / weight
    result = (result + 0.5).clip(0, 255).astype("uint8")
    return Image.fromarray(result, mode="RGB")
