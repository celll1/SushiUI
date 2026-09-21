"""Complete-coverage target partitioning for Qwen-Image 2.1 training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PartitionBox:
    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def tokens(self) -> int:
        return self.height * self.width

    def expand(self, halo: int, *, full_height: int, full_width: int) -> "PartitionBox":
        return PartitionBox(
            max(0, self.top - halo),
            max(0, self.left - halo),
            min(full_height, self.bottom + halo),
            min(full_width, self.right + halo),
        )


@dataclass(frozen=True)
class PartitionRegion:
    core: PartitionBox
    input: PartitionBox

    @property
    def core_in_input(self) -> PartitionBox:
        return PartitionBox(
            self.core.top - self.input.top,
            self.core.left - self.input.left,
            self.core.bottom - self.input.top,
            self.core.right - self.input.left,
        )


@dataclass(frozen=True)
class PartitionPlan:
    full_height: int
    full_width: int
    regions: tuple[PartitionRegion, ...]
    reason: str

    @property
    def full_tokens(self) -> int:
        return self.full_height * self.full_width

    @property
    def largest_input_tokens(self) -> int:
        return max(region.input.tokens for region in self.regions)


def _draw_u01(*parts: object) -> float:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value / float(1 << 64)


def _even_cut(length: int, ratio: float, *, minimum: int = 2) -> int:
    if length < minimum * 2:
        raise ValueError(f"Cannot split latent side {length} with minimum side {minimum}")
    cut = int(round(length * ratio / 2.0)) * 2
    return min(length - minimum, max(minimum, cut))


def _ratio(seed: int, epoch: int, occurrence: int, axis: str, lo: float, hi: float) -> float:
    return lo + (hi - lo) * _draw_u01(seed, epoch, occurrence, axis)


def _validate_regions(height: int, width: int, regions: Iterable[PartitionRegion]) -> None:
    owner = torch.zeros((height, width), dtype=torch.int16)
    count = 0
    for region in regions:
        count += 1
        core, input_box = region.core, region.input
        for box, label in ((core, "core"), (input_box, "input")):
            if not (0 <= box.top < box.bottom <= height and 0 <= box.left < box.right <= width):
                raise ValueError(f"Qwen partition {label} is outside the latent canvas: {box}")
        if input_box.tokens % 4:
            raise ValueError(
                "Qwen partition transformer input token count must be divisible by four, "
                f"got {input_box.height}x{input_box.width}={input_box.tokens}"
            )
        if not (
            input_box.top <= core.top < core.bottom <= input_box.bottom
            and input_box.left <= core.left < core.right <= input_box.right
        ):
            raise ValueError(f"Qwen partition core {core} is not contained in input {input_box}")
        owner[core.top : core.bottom, core.left : core.right] += 1
    if count == 0 or not torch.all(owner == 1):
        missing = int((owner == 0).sum())
        overlapping = int((owner > 1).sum())
        raise ValueError(
            "Qwen partition loss cores must cover the canvas exactly once; "
            f"missing={missing}, overlapping={overlapping}"
        )


def build_fixed_partition_plan(
    height: int,
    width: int,
    *,
    count: int,
    halo: int = 0,
    seed: int = 0,
    epoch: int = 0,
    occurrence: int = 0,
    split_ratio_min: float = 0.35,
    split_ratio_max: float = 0.65,
) -> PartitionPlan:
    """Build a deterministic, epoch-varying two- or four-region plan."""
    height, width, count, halo = int(height), int(width), int(count), int(halo)
    if count not in (2, 4):
        raise ValueError(f"qwen_partition_fixed_count must be 2 or 4, got {count}")
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError(
            "Fixed Qwen partition prototype requires positive even latent sides, "
            f"got {height}x{width}"
        )
    if halo < 0 or halo % 2:
        raise ValueError(f"qwen_partition_halo_tokens must be a non-negative even number, got {halo}")
    lo, hi = float(split_ratio_min), float(split_ratio_max)
    if not 0.0 < lo <= 0.5 <= hi < 1.0:
        raise ValueError(f"Invalid Qwen partition split ratio range [{lo}, {hi}]")

    if count == 2:
        if height > width:
            axis = "h"
        elif width > height:
            axis = "w"
        else:
            axis = "h" if _draw_u01(seed, epoch, occurrence, "axis") < 0.5 else "w"
        if axis == "h":
            cut = _even_cut(height, _ratio(seed, epoch, occurrence, "h", lo, hi))
            cores = (
                PartitionBox(0, 0, cut, width),
                PartitionBox(cut, 0, height, width),
            )
        else:
            cut = _even_cut(width, _ratio(seed, epoch, occurrence, "w", lo, hi))
            cores = (
                PartitionBox(0, 0, height, cut),
                PartitionBox(0, cut, height, width),
            )
    else:
        cut_h = _even_cut(height, _ratio(seed, epoch, occurrence, "h", lo, hi))
        cut_w = _even_cut(width, _ratio(seed, epoch, occurrence, "w", lo, hi))
        cores = (
            PartitionBox(0, 0, cut_h, cut_w),
            PartitionBox(0, cut_w, cut_h, width),
            PartitionBox(cut_h, 0, height, cut_w),
            PartitionBox(cut_h, cut_w, height, width),
        )

    regions = tuple(
        PartitionRegion(
            core=core,
            input=core.expand(halo, full_height=height, full_width=width),
        )
        for core in cores
    )
    _validate_regions(height, width, regions)
    return PartitionPlan(height, width, regions, reason="fixed")


def full_canvas_position_ids(
    full_height: int,
    full_width: int,
    box: PartitionBox,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return row-major `(height, width)` RoPE indices from the full canvas."""
    full_y = torch.arange(
        -(full_height - full_height // 2), full_height // 2, device=device, dtype=torch.long
    )
    full_x = torch.arange(
        -(full_width - full_width // 2), full_width // 2, device=device, dtype=torch.long
    )
    yy = full_y[box.top : box.bottom, None].expand(box.height, box.width)
    xx = full_x[box.left : box.right][None, :].expand(box.height, box.width)
    return torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)


def flatten_region(grid: torch.Tensor, box: PartitionBox) -> torch.Tensor:
    """Slice `[B,H,W,C]` and flatten its rectangular region row-major."""
    return grid[:, box.top : box.bottom, box.left : box.right].reshape(
        grid.shape[0], box.tokens, grid.shape[-1]
    )


class QwenPartitionGlobalAdapter(nn.Module):
    """Small trainable full-canvas communication path for partitioned training."""

    ADAPTER_ALGORITHM = "qwen_partition_global"
    WEIGHT_DECOMPOSE = False

    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int,
        *,
        rank: int = 64,
        summary_tokens: int = 16,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if rank <= 0 or summary_tokens <= 0:
            raise ValueError("Qwen partition global rank and summary tokens must be positive")
        self.rank = int(rank)
        self.summary_tokens = int(summary_tokens)
        input_dim = int(latent_channels) + 2
        self.key = nn.Linear(input_dim, self.rank, bias=False, dtype=dtype)
        self.value = nn.Linear(input_dim, self.rank, bias=False, dtype=dtype)
        self.query = nn.Linear(input_dim, self.rank, bias=False, dtype=dtype)
        self.summary_queries = nn.Parameter(
            torch.randn(self.summary_tokens, self.rank, dtype=dtype) / self.rank**0.5
        )
        self.out = nn.Linear(self.rank, int(hidden_dim), bias=False, dtype=dtype)
        nn.init.zeros_(self.out.weight)

    @staticmethod
    def _features(grid: torch.Tensor, box: PartitionBox | None = None) -> torch.Tensor:
        batch, height, width, _ = grid.shape
        y = torch.linspace(-1, 1, height, device=grid.device, dtype=grid.dtype)
        x = torch.linspace(-1, 1, width, device=grid.device, dtype=grid.dtype)
        yy = y[:, None].expand(height, width)
        xx = x[None, :].expand(height, width)
        coords = torch.stack((yy, xx), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        features = torch.cat((grid, coords), dim=-1)
        if box is not None:
            features = features[:, box.top : box.bottom, box.left : box.right]
        return features.reshape(batch, -1, features.shape[-1])

    def forward(self, full_grid: torch.Tensor, box: PartitionBox) -> torch.Tensor:
        full = self._features(full_grid)
        local = self._features(full_grid, box)
        keys = self.key(full)
        values = self.value(full)
        scale = self.rank**-0.5
        summary_weights = torch.softmax(
            torch.einsum("sr,bnr->bsn", self.summary_queries, keys) * scale,
            dim=-1,
            dtype=torch.float32,
        ).to(values.dtype)
        summaries = torch.einsum("bsn,bnr->bsr", summary_weights, values)
        queries = self.query(local)
        local_weights = torch.softmax(
            torch.einsum("bnr,bsr->bns", queries, summaries) * scale,
            dim=-1,
            dtype=torch.float32,
        ).to(summaries.dtype)
        context = torch.einsum("bns,bsr->bnr", local_weights, summaries)
        return self.out(context)

    def branch_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "key.weight": self.key.weight,
            "value.weight": self.value.weight,
            "query.weight": self.query.weight,
            "summary_queries": self.summary_queries,
            "out.weight": self.out.weight,
        }

    def trainable_parameters(self):
        return iter(self.parameters())

    def export_tensors(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.branch_tensors().items()}

    def load_tensors(self, tensors: Mapping[str, torch.Tensor]) -> None:
        for name, parameter in self.branch_tensors().items():
            value = tensors.get(name)
            if value is not None:
                parameter.data.copy_(value)

    def spec_constants(self) -> tuple[str, ...]:
        return ()
