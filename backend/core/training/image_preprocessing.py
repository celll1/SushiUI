"""Shared image-mode normalization and crop geometry for training inputs."""

from __future__ import annotations

from typing import Tuple

from PIL import Image


TRANSPARENT_WEBP_PREPROCESSING_VERSION = "alpha-white-v1"


def flatten_to_rgb(
    image: Image.Image,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Return RGB pixels with any transparency composited over ``background``."""
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if not has_alpha:
        return image if image.mode == "RGB" else image.convert("RGB")

    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*background, 255))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


def crop_window_in_original(
    orig_w: int,
    orig_h: int,
    resized_w: int,
    resized_h: int,
    left: int,
    top: int,
    win_w: int,
    win_h: int,
) -> Tuple[int, int, int, int]:
    """Map a crop window taken in RESIZED space back to original-image pixels.

    Returns a PIL box ``(x0, y0, x1, y1)``, clamped inside the original and never
    empty. Negative offsets are clamped rather than reproduced: PIL pads a
    negative box with black, which is not something the source image contains.
    """
    sx = orig_w / max(1, resized_w)
    sy = orig_h / max(1, resized_h)
    x0 = min(max(0, int(round(left * sx))), max(0, orig_w - 1))
    y0 = min(max(0, int(round(top * sy))), max(0, orig_h - 1))
    x1 = min(orig_w, max(x0 + 1, int(round((left + win_w) * sx))))
    y1 = min(orig_h, max(y0 + 1, int(round((top + win_h) * sy))))
    return (x0, y0, x1, y1)


def source_region_for_strategy(
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
    strategy: str,
) -> Tuple[int, int, int, int]:
    """The original-pixel box ``BaseTrainer.encode_image`` keeps for ``strategy``.

    Mirrors that method's own arithmetic, for callers that must reconstruct the
    region of a latent encoded earlier (a disk cache or swap-buffer hit). Only
    the deterministic strategies have an answer: ``random_crop`` draws its offset
    at encode time and is not a function of the sizes, so it raises.
    """
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"invalid source size {orig_w}x{orig_h}")
    if strategy == "resize":
        return (0, 0, orig_w, orig_h)
    if strategy == "crop":
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        return crop_window_in_original(
            orig_w, orig_h, new_w, new_h,
            (new_w - target_w) // 2, (new_h - target_h) // 2, target_w, target_h,
        )
    raise ValueError(
        f"bucket_strategy={strategy!r} has no deterministic source region "
        f"(only 'resize' and 'crop' do)"
    )
