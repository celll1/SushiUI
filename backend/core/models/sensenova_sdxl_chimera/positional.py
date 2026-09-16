"""SenseNova-layout three-axis RoPE for the Chimera U-Net.

The implementation intentionally owns no Parameters.  A donor-equal SDXL
U-Net therefore keeps exactly the donor's trainable tensor census while its
attention processor can use SenseNova's t:h:w = 2:1:1 rotary layout.
"""

from __future__ import annotations

from typing import Tuple

import torch

POSITION_LAYOUT_VERSION = 1
SPATIAL_UNIT_PIXELS = 32.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _axis_cos_sin(
    positions: torch.Tensor,
    dim: int,
    *,
    base: float,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dim <= 0 or dim % 2:
        raise ValueError(f"rotary axis width must be a positive even number, got {dim}")
    inv_freq = 1.0 / (
        float(base)
        ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim)
    )
    with torch.autocast(device_type=positions.device.type, enabled=False):
        freq = positions.float().unsqueeze(-1) * inv_freq
        embedding = torch.cat((freq, freq), dim=-1)
        cos, sin = embedding.cos(), embedding.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_sensenova_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_theta: float = 1_000_000.0,
    rope_theta_hw: float = 10_000.0,
) -> torch.Tensor:
    """Apply SenseNova's 3-D rotary layout to a BHSD tensor.

    ``positions`` is ``[B,S,3]`` in ``(t,h,w)`` order.  A leading batch of one
    may broadcast over ``tensor``.  Float positions are supported so all U-Net
    resolutions can share one physical coordinate system.
    """
    if tensor.ndim != 4:
        raise ValueError(f"expected tensor [B,H,S,D], got {tuple(tensor.shape)}")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"expected positions [B,S,3], got {tuple(positions.shape)}")
    batch, _heads, sequence, width = tensor.shape
    if width % 4:
        raise ValueError(f"SenseNova rotary head width must be divisible by 4, got {width}")
    if positions.shape[0] not in (1, batch) or positions.shape[1] != sequence:
        raise ValueError(
            f"position shape {tuple(positions.shape)} cannot broadcast to tensor {tuple(tensor.shape)}"
        )

    t_width = width // 2
    hw_width = width // 4
    t_part, h_part, w_part = torch.split(tensor, (t_width, hw_width, hw_width), dim=-1)
    rotated = []
    for part, axis, base in (
        (t_part, 0, rope_theta),
        (h_part, 1, rope_theta_hw),
        (w_part, 2, rope_theta_hw),
    ):
        cos, sin = _axis_cos_sin(
            positions[..., axis], part.shape[-1], base=base, dtype=tensor.dtype
        )
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        rotated.append(part * cos + _rotate_half(part) * sin)
    return torch.cat(rotated, dim=-1)


def apply_sensenova_rope_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    rope_theta: float = 1_000_000.0,
    rope_theta_hw: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate query and key sequences that may have different lengths."""
    if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1] or query.shape[3] != key.shape[3]:
        raise ValueError(
            "query/key must share batch, head count, and head width; "
            f"got {tuple(query.shape)} and {tuple(key.shape)}"
        )
    kwargs = {"rope_theta": rope_theta, "rope_theta_hw": rope_theta_hw}
    return (
        apply_sensenova_rope(query, query_positions, **kwargs),
        apply_sensenova_rope(key, key_positions, **kwargs),
    )


def spatial_query_positions(
    height: int,
    width: int,
    *,
    target_height: int,
    target_width: int,
    crop_top: int = 0,
    crop_left: int = 0,
    prefix_terminal_t: float = 0.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return flattened ``[1,H*W,3]`` physical query coordinates.

    Coordinates use one unit per 32 output pixels.  Pixel centers, not cell
    corners, make the same physical point agree across U-Net resolutions.
    """
    values = (height, width, target_height, target_width)
    if any(int(value) <= 0 for value in values):
        raise ValueError(f"spatial dimensions must be positive, got {values}")
    rows = (
        float(crop_top) / SPATIAL_UNIT_PIXELS
        + (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
        * float(target_height)
        / (SPATIAL_UNIT_PIXELS * height)
        - 0.5
    )
    cols = (
        float(crop_left) / SPATIAL_UNIT_PIXELS
        + (torch.arange(width, device=device, dtype=torch.float32) + 0.5)
        * float(target_width)
        / (SPATIAL_UNIT_PIXELS * width)
        - 0.5
    )
    grid_h, grid_w = torch.meshgrid(rows, cols, indexing="ij")
    temporal = torch.full_like(grid_h, float(prefix_terminal_t))
    return torch.stack((temporal, grid_h, grid_w), dim=-1).reshape(1, height * width, 3)
