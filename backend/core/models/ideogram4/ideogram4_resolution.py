"""Resolution helpers for Ideogram 4.

Ideogram 4 supports arbitrary resolutions whose height and width are multiples of
``vae_scale_factor * patch_size`` (= 8 * 2 = 16). The noise schedule auto-adjusts
per resolution (see ``ideogram4_pipeline_ops._resolution_aware_mu``).
"""

from __future__ import annotations

# VAE downscale (2 ** (len(block_out_channels) - 1) = 8) times the 2x2 patchify.
VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
GRID_ALIGN = VAE_SCALE_FACTOR * PATCH_SIZE  # 16

# Native supported pixel range (per side).
MIN_SIDE = 256
MAX_SIDE = 2048


def align_to_grid(value: int, align: int = GRID_ALIGN) -> int:
    """Round ``value`` to the nearest positive multiple of ``align``."""
    if value <= 0:
        return align
    rounded = round(value / align) * align
    return max(align, rounded)


def normalize_resolution(width: int, height: int) -> tuple[int, int]:
    """Clamp to the native range and snap both sides to the 16-pixel grid.

    Returns ``(width, height)`` ready for latent-grid computation.
    """
    width = min(max(int(width), MIN_SIDE), MAX_SIDE)
    height = min(max(int(height), MIN_SIDE), MAX_SIDE)
    return align_to_grid(width), align_to_grid(height)


def latent_grid(width: int, height: int) -> tuple[int, int]:
    """Return the ``(grid_h, grid_w)`` token grid for a (already aligned) resolution."""
    grid_w = width // GRID_ALIGN
    grid_h = height // GRID_ALIGN
    return grid_h, grid_w
