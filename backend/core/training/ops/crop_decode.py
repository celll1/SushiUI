"""Autograd-enabled context crop decode (Phase 2-1).

Decodes a latent crop with surrounding context cells and discards the margin,
preserving the autograd computation graph from the decoded output back to the
input latent tensor.

The geometry is reused directly from core.inference.context_tiled_decode.
VAE parameters are assumed frozen (eval mode), but input latents are NOT detached.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from core.inference.context_tiled_decode import (
    DEFAULT_MARGIN_CELLS,
    TileRect,
    spatial_compression_of,
)


def make_crop_rect(
    lat_h: int,
    lat_w: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    margin_cells: int = DEFAULT_MARGIN_CELLS,
) -> TileRect:
    """Construct a TileRect defining inner crop (y0, y1, x0, x1) and its padded window.

    Margin cells are clamped at canvas boundaries (where real zero padding matches
    a full-image decode).
    """
    y0 = max(0, min(lat_h, y0))
    y1 = max(y0, min(lat_h, y1))
    x0 = max(0, min(lat_w, x0))
    x1 = max(x0, min(lat_w, x1))
    py0 = max(0, y0 - margin_cells)
    py1 = min(lat_h, y1 + margin_cells)
    px0 = max(0, x0 - margin_cells)
    px1 = min(lat_w, x1 + margin_cells)
    return TileRect(y0, y1, x0, x1, py0, py1, px0, px1)


def decode_crop_with_context(
    vae,
    latent: torch.Tensor,
    rect: TileRect,
    scale: Optional[int] = None,
) -> torch.Tensor:
    """Decode padded latent window and discard outer margin, keeping autograd graph.

    Args:
        vae: Autoencoder instance (AutoencoderKL, etc.)
        latent: Latent tensor [B, C, H, W] or [B, C, 1, H, W] (requires_grad may be True).
        rect: TileRect defining output footprint and padded window.
        scale: Spatial compression ratio (pixels per cell). Derived if None.

    Returns:
        Decoded RGB tensor of shape [B, 3, (y1-y0)*scale, (x1-x0)*scale]
        corresponding strictly to the inner footprint rect(y0, y1, x0, x1).
    """
    if scale is None:
        scale = spatial_compression_of(vae)

    # Slice padded window: autograd graph is preserved
    tile = latent[..., rect.py0:rect.py1, rect.px0:rect.px1]

    # Invoke VAE decoder (differentiable w.r.t tile input)
    if hasattr(vae, "decode"):
        dec_out = vae.decode(tile, return_dict=False)
        dec = dec_out[0] if isinstance(dec_out, (tuple, list)) else getattr(dec_out, "sample", dec_out)
    elif hasattr(vae, "decoder"):
        dec = vae.decoder(tile)
    else:
        raise AttributeError(f"VAE object {type(vae)} has neither decode nor decoder method")

    # Margin offsets in pixel space
    ty0 = (rect.y0 - rect.py0) * scale
    tx0 = (rect.x0 - rect.px0) * scale
    th = (rect.y1 - rect.y0) * scale
    tw = (rect.x1 - rect.x0) * scale

    # Slice interior region (differentiable)
    interior = dec[..., ty0:ty0 + th, tx0:tx0 + tw]
    return interior
