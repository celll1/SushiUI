"""Shared image-mode normalization for training inputs."""

from __future__ import annotations

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
