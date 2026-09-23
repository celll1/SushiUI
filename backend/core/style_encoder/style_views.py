"""Masked-view block masks on the NaFlex patch grid (design §7)."""

from __future__ import annotations

import math

import numpy as np
import torch

# Block policy: area ~ U{1..patches still to mask}, aspect log-uniform in [1/3, 3], clipped to the grid;
# a draw that would overshoot the target is retried, then replaced by one 1x1 block, so the count is exact.
BLOCK_ASPECT_MAX = 3.0
BLOCK_ATTEMPTS = 10
# Absorbs float error in r * V so that e.g. 0.3 * 10 = 3.0000000000000004 still needs 3 patches.
_RATIO_EPS = 1e-9


def artist_mask_ratios(confound: np.ndarray, ratio_min: float, ratio_max: float) -> np.ndarray:
    """r_a = r_min + (r_max - r_min) * s_a."""
    if not 0.0 <= ratio_min <= ratio_max < 1.0:
        raise ValueError(f"need 0 <= ratio_min <= ratio_max < 1, got {ratio_min!r}, {ratio_max!r}")
    s = np.asarray(confound, dtype=np.float64)
    if s.size and not (np.all(s >= 0.0) and np.all(s <= 1.0)):
        raise ValueError("confound scores must lie in [0, 1]")
    return ratio_min + (ratio_max - ratio_min) * s


def masked_patch_count(valid: int, ratio: float) -> int:
    """Smallest count reaching `ratio` of `valid` patches, keeping at least one patch visible."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"mask ratio must be in [0, 1], got {ratio!r}")
    if valid < 1:
        raise ValueError(f"an image needs at least one valid patch, got {valid}")
    return min(max(math.ceil(ratio * valid - _RATIO_EPS), 0), valid - 1)


def block_mask(height: int, width: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Row-major bool [height * width], True = masked."""
    target = masked_patch_count(height * width, ratio)
    grid = np.zeros((height, width), dtype=bool)
    count = 0
    log_aspect = math.log(BLOCK_ASPECT_MAX)
    while count < target:
        remaining = target - count
        for _ in range(BLOCK_ATTEMPTS):
            area = int(rng.integers(1, remaining + 1))
            aspect = math.exp(rng.uniform(-log_aspect, log_aspect))
            h = min(max(round(math.sqrt(area * aspect)), 1), height)
            w = min(max(round(math.sqrt(area / aspect)), 1), width)
            top = int(rng.integers(0, height - h + 1))
            left = int(rng.integers(0, width - w + 1))
            block = grid[top:top + h, left:left + w]
            delta = int(block.size - block.sum())
            if 0 < delta <= remaining:
                block[...] = True
                count += delta
                break
        else:
            free = np.flatnonzero(~grid.ravel())
            grid.ravel()[free[int(rng.integers(0, free.size))]] = True
            count += 1
    return grid.ravel()


def apply_block_masks(pixel_attention_mask: torch.Tensor, spatial_shapes: torch.Tensor, ratios: np.ndarray,
                      rng: np.random.Generator) -> torch.Tensor:
    """Masked-view attention mask; row b must be the NaFlex layout (h*w ones, then padding zeros)."""
    mask = pixel_attention_mask.detach().cpu().numpy()
    shapes = spatial_shapes.detach().cpu().numpy()
    ratios = np.asarray(ratios, dtype=np.float64)
    if mask.ndim != 2 or shapes.shape != (mask.shape[0], 2) or ratios.shape != (mask.shape[0],):
        raise ValueError(f"shape mismatch: mask {mask.shape}, spatial_shapes {shapes.shape}, ratios {ratios.shape}")
    out = mask.copy()
    for b in range(mask.shape[0]):
        h, w = int(shapes[b, 0]), int(shapes[b, 1])
        valid = h * w
        if not (0 < valid <= mask.shape[1] and np.all(mask[b, :valid] != 0) and np.all(mask[b, valid:] == 0)):
            raise ValueError(f"row {b}: pixel_attention_mask does not match spatial shape ({h}, {w})")
        out[b, :valid][block_mask(h, w, float(ratios[b]), rng)] = 0
    if np.any((out != 0).sum(axis=1) == 0):
        raise AssertionError("a masked view lost every valid patch")
    return torch.from_numpy(out).to(device=pixel_attention_mask.device)
