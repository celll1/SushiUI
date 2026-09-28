"""Masked RGB reconstruction components for FoveaTag pretraining."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchReconstructionHead(nn.Module):
    """Decode each stride-aligned feature cell into one RGB pixel patch."""

    def __init__(self, feature_dim: int, patch_size: int = 16) -> None:
        super().__init__()
        if feature_dim <= 0 or patch_size <= 0:
            raise ValueError("feature_dim and patch_size must be positive")
        self.patch_size = patch_size
        self.projection = nn.Conv2d(feature_dim, 3 * patch_size * patch_size, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [batch, channels, height, width]")
        return F.pixel_shuffle(self.projection(feature_map), self.patch_size)


def random_patch_mask(
    batch_size: int,
    height: int,
    width: int,
    mask_ratio: float,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose the same exact number of masked cells per sample."""
    if min(batch_size, height, width) <= 0:
        raise ValueError("mask dimensions must be positive")
    if not 0 <= mask_ratio <= 1:
        raise ValueError("mask_ratio must be in [0, 1]")
    cells = height * width
    masked_cells = round(cells * mask_ratio)
    noise = torch.rand((batch_size, cells), device=device, generator=generator)
    order = noise.argsort(dim=1)
    mask = torch.zeros((batch_size, cells), dtype=torch.bool, device=device)
    if masked_cells:
        mask.scatter_(1, order[:, :masked_cells], True)
    return mask.reshape(batch_size, height, width)


def mask_image_patches(
    images: torch.Tensor,
    patch_mask: torch.Tensor,
    *,
    patch_size: int = 16,
    fill: float = 0.0,
) -> torch.Tensor:
    """Replace selected non-overlapping image patches before encoder input."""
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    expected = (images.shape[0], images.shape[2] // patch_size, images.shape[3] // patch_size)
    if images.shape[2] % patch_size or images.shape[3] % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    if patch_mask.shape != expected:
        raise ValueError(f"patch_mask must have shape {expected}")
    pixel_mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return images.masked_fill(pixel_mask[:, None].to(images.device), fill)


def expand_patch_mask(patch_mask: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """Expand a token mask to the pixel mask expected by reconstruction losses."""
    if patch_mask.ndim != 3 or patch_size <= 0:
        raise ValueError("patch_mask must have shape [batch, height, width]")
    return patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
