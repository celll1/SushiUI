"""Topology-neutral planning and coverage bookkeeping for spatial DiT partitions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import torch


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


def validate_partition_regions(
    height: int,
    width: int,
    regions: Iterable[PartitionRegion],
    *,
    input_token_multiple: int = 1,
) -> None:
    """Require loss cores to cover the canvas exactly once."""
    owner = torch.zeros((height, width), dtype=torch.int16)
    count = 0
    for region in regions:
        count += 1
        core, input_box = region.core, region.input
        for box, label in ((core, "core"), (input_box, "input")):
            if not (0 <= box.top < box.bottom <= height and 0 <= box.left < box.right <= width):
                raise ValueError(f"DiT partition {label} is outside the latent canvas: {box}")
        if input_box.tokens % input_token_multiple:
            raise ValueError(
                "DiT partition transformer input token count must be divisible by "
                f"{input_token_multiple}, got {input_box.height}x{input_box.width}="
                f"{input_box.tokens}"
            )
        if not (
            input_box.top <= core.top < core.bottom <= input_box.bottom
            and input_box.left <= core.left < core.right <= input_box.right
        ):
            raise ValueError(f"DiT partition core {core} is not contained in input {input_box}")
        owner[core.top : core.bottom, core.left : core.right] += 1
    if count == 0 or not torch.all(owner == 1):
        missing = int((owner == 0).sum())
        overlapping = int((owner > 1).sum())
        raise ValueError(
            "DiT partition loss cores must cover the canvas exactly once; "
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
    input_token_multiple: int = 1,
) -> PartitionPlan:
    """Build a deterministic, epoch-varying two- or four-region plan."""
    height, width, count, halo = int(height), int(width), int(count), int(halo)
    if count not in (2, 4):
        raise ValueError(f"dit_partition_fixed_count must be 2 or 4, got {count}")
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError(
            "Fixed DiT partition prototype requires positive even latent sides, "
            f"got {height}x{width}"
        )
    if halo < 0 or halo % 2:
        raise ValueError(f"dit_partition_halo_tokens must be non-negative and even, got {halo}")
    lo, hi = float(split_ratio_min), float(split_ratio_max)
    if not 0.0 < lo <= 0.5 <= hi < 1.0:
        raise ValueError(f"Invalid DiT partition split ratio range [{lo}, {hi}]")

    if count == 2:
        if height > width:
            axis = "h"
        elif width > height:
            axis = "w"
        else:
            axis = "h" if _draw_u01(seed, epoch, occurrence, "axis") < 0.5 else "w"
        if axis == "h":
            cut = _even_cut(height, _ratio(seed, epoch, occurrence, "h", lo, hi))
            cores = (PartitionBox(0, 0, cut, width), PartitionBox(cut, 0, height, width))
        else:
            cut = _even_cut(width, _ratio(seed, epoch, occurrence, "w", lo, hi))
            cores = (PartitionBox(0, 0, height, cut), PartitionBox(0, cut, height, width))
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
        PartitionRegion(core=core, input=core.expand(
            halo, full_height=height, full_width=width))
        for core in cores
    )
    validate_partition_regions(
        height, width, regions, input_token_multiple=input_token_multiple
    )
    return PartitionPlan(height, width, regions, reason="fixed")


def flatten_region(grid: torch.Tensor, box: PartitionBox) -> torch.Tensor:
    """Slice ``[B,H,W,C]`` and flatten its rectangular region row-major."""
    return grid[:, box.top : box.bottom, box.left : box.right].reshape(
        grid.shape[0], box.tokens, grid.shape[-1]
    )
