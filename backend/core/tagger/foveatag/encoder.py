"""Resolution-flexible from-scratch visual encoder for FoveaTag."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FoveaEncoderConfig:
    widths: Tuple[int, int, int, int] = (96, 192, 384, 512)
    depths: Tuple[int, int, int, int] = (2, 3, 8, 6)
    window_size: int = 8
    attention_heads: Tuple[int, int] = (8, 8)
    mlp_ratio: float = 3.0
    geometry_dim: int = 10
    scale_dropout: float = 0.2

    def __post_init__(self) -> None:
        if len(self.widths) != 4 or len(self.depths) != 4:
            raise ValueError("widths and depths must contain four stages")
        if any(value <= 0 for value in (*self.widths, *self.depths)):
            raise ValueError("widths and depths must be positive")
        if self.window_size <= 0 or self.geometry_dim < 10:
            raise ValueError("window_size must be positive and geometry_dim at least 10")
        if not 0 <= self.scale_dropout < 1:
            raise ValueError("scale_dropout must be in [0, 1)")
        for width, heads in zip(self.widths[2:], self.attention_heads):
            if width % heads or (width // heads) % 4:
                raise ValueError("attention head dimensions must be divisible by four")


@dataclass(frozen=True)
class FoveaEncoderOutput:
    local_map: torch.Tensor
    local_coordinates: torch.Tensor
    global_tokens: torch.Tensor
    global_coordinates: torch.Tensor
    global_summary: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized * self.weight


class BlurPool2d(nn.Module):
    """Fixed low-pass filter before stride-two subsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel = torch.tensor([1.0, 2.0, 1.0])
        kernel = (kernel[:, None] * kernel[None, :]) / 16.0
        self.channels = channels
        self.register_buffer("kernel", kernel[None, None], persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.expand(self.channels, 1, 3, 3)
        return F.conv2d(value, kernel, stride=2, padding=1, groups=self.channels)


class AntiAliasedDownsample(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.blur = BlurPool2d(input_dim)
        self.projection = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(self.blur(value))


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(dim, hidden_dim * 2)
        self.output = nn.Linear(hidden_dim, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        content, gate = self.input(value).chunk(2, dim=-1)
        return self.output(content * F.silu(gate))


class DepthwiseGatedBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = RMSNorm(dim)
        self.mlp = GatedMLP(dim, round(dim * mlp_ratio))
        self.layer_scale = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(value).permute(0, 2, 3, 1)
        residual = self.mlp(self.norm(residual)) * self.layer_scale
        return value + residual.permute(0, 3, 1, 2)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.reshape(*value.shape[:-1], -1, 2).unbind(-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def apply_2d_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    coordinates: torch.Tensor,
    theta: float = 10_000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply continuous x/y rotary coordinates to [B, heads, tokens, dim]."""
    head_dim = query.shape[-1]
    if head_dim % 4:
        raise ValueError("2-D RoPE requires head dimension divisible by four")
    axis_dim = head_dim // 2
    frequency = torch.arange(0, axis_dim, 2, device=query.device, dtype=torch.float32)
    frequency = theta ** (-frequency / axis_dim)

    def rotate_axis(value: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        angle = position.to(torch.float32)[..., None] * frequency
        cosine = torch.repeat_interleave(angle.cos(), 2, dim=-1).unsqueeze(1).to(value.dtype)
        sine = torch.repeat_interleave(angle.sin(), 2, dim=-1).unsqueeze(1).to(value.dtype)
        return value * cosine + _rotate_half(value) * sine

    query_x, query_y = query.split(axis_dim, dim=-1)
    key_x, key_y = key.split(axis_dim, dim=-1)
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    return (
        torch.cat((rotate_axis(query_x, x), rotate_axis(query_y, y)), dim=-1),
        torch.cat((rotate_axis(key_x, x), rotate_axis(key_y, y)), dim=-1),
    )


class SpatialAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)

    def forward(
        self,
        value: torch.Tensor,
        coordinates: torch.Tensor,
        allowed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, _ = value.shape
        qkv = self.qkv(value).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, content = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = apply_2d_rope(query, key, coordinates)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.head_dim ** -0.5
        if allowed is not None:
            scores = scores.masked_fill(~allowed[:, None], -torch.inf)
        weights = scores.softmax(dim=-1)
        attended = torch.matmul(weights, content).transpose(1, 2).reshape(batch, tokens, self.dim)
        return self.output(attended)


def _window_partition(value: torch.Tensor, window: int) -> torch.Tensor:
    batch, height, width, channels = value.shape
    return (
        value.reshape(batch, height // window, window, width // window, window, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window * window, channels)
    )


def _window_reverse(
    windows: torch.Tensor,
    window: int,
    batch: int,
    height: int,
    width: int,
) -> torch.Tensor:
    channels = windows.shape[-1]
    return (
        windows.reshape(batch, height // window, width // window, window, window, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, channels)
    )


class WindowAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window_size: int, shifted: bool, mlp_ratio: float) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift = window_size // 2 if shifted else 0
        self.norm1 = RMSNorm(dim)
        self.attention = SpatialAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = GatedMLP(dim, round(dim * mlp_ratio))
        self.attention_scale = nn.Parameter(torch.full((dim,), 1e-6))
        self.mlp_scale = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, value: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        window = self.window_size
        padded_height = math.ceil(height / window) * window
        padded_width = math.ceil(width / window) * window
        x = value.permute(0, 2, 3, 1)
        x = F.pad(x, (0, 0, 0, padded_width - width, 0, padded_height - height))
        coords = F.pad(coordinates, (0, 0, 0, padded_width - width, 0, padded_height - height))
        valid = torch.zeros((batch, padded_height, padded_width), dtype=torch.bool, device=value.device)
        valid[:, :height, :width] = True
        regions = _shift_regions(batch, padded_height, padded_width, window, self.shift, value.device)
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
            coords = torch.roll(coords, shifts=(-self.shift, -self.shift), dims=(1, 2))
            valid = torch.roll(valid, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = _window_partition(self.norm1(x), window)
        coordinate_windows = _window_partition(coords, window)
        valid_windows = _window_partition(valid[..., None], window).squeeze(-1)
        region_windows = _window_partition(regions[..., None], window).squeeze(-1)
        allowed = valid_windows[:, :, None] & valid_windows[:, None, :]
        allowed &= region_windows[:, :, None] == region_windows[:, None, :]
        identity = torch.eye(window * window, dtype=torch.bool, device=value.device)
        allowed |= (~valid_windows)[:, :, None] & identity[None]
        attended = self.attention(windows, coordinate_windows, allowed)
        attended = _window_reverse(attended, window, batch, padded_height, padded_width)
        x = x + attended * self.attention_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        if self.shift:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        return x[:, :height, :width].permute(0, 3, 1, 2)


def _shift_regions(
    batch: int,
    height: int,
    width: int,
    window: int,
    shift: int,
    device: torch.device,
) -> torch.Tensor:
    if not shift:
        return torch.zeros((batch, height, width), dtype=torch.long, device=device)
    labels = torch.zeros((1, height, width), dtype=torch.long, device=device)
    height_slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    width_slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    index = 0
    for height_slice in height_slices:
        for width_slice in width_slices:
            labels[:, height_slice, width_slice] = index
            index += 1
    return labels.expand(batch, -1, -1)


class GlobalAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attention = SpatialAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = GatedMLP(dim, round(dim * mlp_ratio))

    def forward(self, value: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(self.norm1(value), coordinates)
        return value + self.mlp(self.norm2(value))


class FoveaEncoder(nn.Module):
    """Shared encoder for overview and local canvases, trained from scratch."""

    def __init__(self, config: FoveaEncoderConfig = FoveaEncoderConfig()) -> None:
        super().__init__()
        self.config = config
        widths = config.widths
        self.stem_conv = nn.Conv2d(3, widths[0], kernel_size=5, stride=1, padding=2)
        self.stem_blur = BlurPool2d(widths[0])
        self.stem_downsample = AntiAliasedDownsample(widths[0], widths[0])
        self.early_stage = nn.Sequential(*[
            DepthwiseGatedBlock(widths[0], config.mlp_ratio) for _ in range(config.depths[0])
        ])
        self.down_to_texture = AntiAliasedDownsample(widths[0], widths[1])
        self.texture_stage = nn.Sequential(*[
            DepthwiseGatedBlock(widths[1], config.mlp_ratio) for _ in range(config.depths[1])
        ])
        self.down_to_local = AntiAliasedDownsample(widths[1], widths[2])
        self.local_geometry = nn.Linear(config.geometry_dim, widths[2])
        self.window_stage = nn.ModuleList([
            WindowAttentionBlock(
                widths[2],
                config.attention_heads[0],
                config.window_size,
                shifted=bool(index % 2),
                mlp_ratio=config.mlp_ratio,
            )
            for index in range(config.depths[2])
        ])
        self.global_merge = AntiAliasedDownsample(widths[2], widths[3])
        self.global_geometry = nn.Linear(config.geometry_dim, widths[3])
        self.global_stage = nn.ModuleList([
            GlobalAttentionBlock(widths[3], config.attention_heads[1], config.mlp_ratio)
            for _ in range(config.depths[3])
        ])
        self.global_norm = RMSNorm(widths[3])

    def forward(self, images: torch.Tensor, geometry: torch.Tensor) -> FoveaEncoderOutput:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if geometry.shape != (images.shape[0], self.config.geometry_dim):
            raise ValueError(
                f"geometry must have shape [batch, {self.config.geometry_dim}]"
            )
        geometry = self._drop_scale(geometry)
        value = self.stem_downsample(self.stem_blur(self.stem_conv(images)))
        value = self.early_stage(value)
        value = self.texture_stage(self.down_to_texture(value))
        value = self.down_to_local(value)
        local_coordinates = source_coordinates(
            geometry,
            value.shape[-2],
            value.shape[-1],
            device=value.device,
            dtype=value.dtype,
        )
        value = value + self.local_geometry(geometry.to(value.dtype)).unsqueeze(-1).unsqueeze(-1)
        for block in self.window_stage:
            value = block(value, local_coordinates)
        local_map = value

        value = self.global_merge(value)
        global_coordinates = source_coordinates(
            geometry,
            value.shape[-2],
            value.shape[-1],
            device=value.device,
            dtype=value.dtype,
        )
        tokens = value.flatten(2).transpose(1, 2)
        tokens = tokens + self.global_geometry(geometry.to(tokens.dtype)).unsqueeze(1)
        flat_coordinates = global_coordinates.flatten(1, 2)
        for block in self.global_stage:
            tokens = block(tokens, flat_coordinates)
        tokens = self.global_norm(tokens)
        return FoveaEncoderOutput(
            local_map=local_map,
            local_coordinates=local_coordinates,
            global_tokens=tokens,
            global_coordinates=global_coordinates,
            global_summary=tokens.mean(dim=1),
        )

    def _drop_scale(self, geometry: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.scale_dropout == 0:
            return geometry
        keep = torch.rand((geometry.shape[0], 1), device=geometry.device) >= self.config.scale_dropout
        result = geometry.clone()
        result[:, 4:6] *= keep.to(result.dtype)
        return result

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def source_coordinates(
    geometry: torch.Tensor,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map feature-cell centers to continuous normalized source coordinates."""
    if height <= 0 or width <= 0:
        raise ValueError("feature dimensions must be positive")
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height - 0.5
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width - 0.5
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    canvas_x = grid_x + 0.5
    canvas_y = grid_y + 0.5
    content_left = geometry[:, 6, None, None]
    content_top = geometry[:, 7, None, None]
    content_width = geometry[:, 8, None, None].clamp_min(1e-8)
    content_height = geometry[:, 9, None, None].clamp_min(1e-8)
    local_x = (canvas_x[None] - content_left) / content_width - 0.5
    local_y = (canvas_y[None] - content_top) / content_height - 0.5
    center_x = geometry[:, 0, None, None]
    center_y = geometry[:, 1, None, None]
    field_width = geometry[:, 2, None, None]
    field_height = geometry[:, 3, None, None]
    source_x = center_x + local_x * field_width
    source_y = center_y + local_y * field_height
    return torch.stack((source_x, source_y), dim=-1)
