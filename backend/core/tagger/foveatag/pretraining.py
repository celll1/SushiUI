"""Teacher-free FoveaTag pretraining data and model contracts."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from .encoder import FoveaEncoder, FoveaEncoderConfig
from .objectives import FoveaSelfSupervisedLoss, SelfSupervisedLossOutput
from .reconstruction import PatchReconstructionHead, mask_image_patches, random_patch_mask
from .tiled_pyramid import resolve_cached_tiled_pyramid
from .views import (
    RegionImageSource,
    ViewConfig,
    ViewSpec,
    local_spec,
    overview_spec,
    pil_to_tensor,
    render_source_view,
)


@dataclass(frozen=True)
class FoveaPretrainingSample:
    first_image: torch.Tensor
    first_geometry: torch.Tensor
    second_image: torch.Tensor
    second_geometry: torch.Tensor


@dataclass(frozen=True)
class FoveaPretrainingBatch:
    first_images: torch.Tensor
    first_geometry: torch.Tensor
    second_images: torch.Tensor
    second_geometry: torch.Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> "FoveaPretrainingBatch":
        image_options = {"device": device, "non_blocking": non_blocking}
        if dtype is not None:
            image_options["dtype"] = dtype
        return FoveaPretrainingBatch(
            first_images=self.first_images.to(**image_options),
            first_geometry=self.first_geometry.to(device=device, non_blocking=non_blocking),
            second_images=self.second_images.to(**image_options),
            second_geometry=self.second_geometry.to(device=device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class FoveaPretrainingOutput:
    losses: SelfSupervisedLossOutput
    reconstruction_mask: torch.Tensor
    first_features: torch.Tensor
    second_features: torch.Tensor


def sample_overlapping_specs(
    image_width: int,
    image_height: int,
    config: ViewConfig,
    *,
    seed: int,
) -> tuple[ViewSpec, ViewSpec]:
    """Sample two local views with a shared source region at different scales."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    rng = random.Random(seed)
    first_density = rng.choice(config.field_levels)
    low, high = config.scale_jitter
    first_density *= math.exp(rng.uniform(math.log(low), math.log(high)))
    second_density = first_density * math.exp(rng.uniform(math.log(0.75), math.log(1.25)))
    maximum_density = max(0.5, min(image_width, image_height) / config.local_size)
    first_density = min(maximum_density, max(0.5, first_density))
    second_density = min(maximum_density, max(0.5, second_density))

    maximum_side = max(config.local_size * first_density, config.local_size * second_density)
    center_x = _bounded_center(image_width, maximum_side, rng)
    center_y = _bounded_center(image_height, maximum_side, rng)
    shared_side = min(config.local_size * first_density, config.local_size * second_density)
    jitter = shared_side * 0.2
    second_side = config.local_size * second_density
    second_x = _clamp_center(center_x + rng.uniform(-jitter, jitter), image_width, second_side)
    second_y = _clamp_center(center_y + rng.uniform(-jitter, jitter), image_height, second_side)
    return (
        local_spec(
            center_x=center_x,
            center_y=center_y,
            canvas_size=config.local_size,
            pixels_per_canvas_pixel=first_density,
            role="pretrain_a",
        ),
        local_spec(
            center_x=second_x,
            center_y=second_y,
            canvas_size=config.local_size,
            pixels_per_canvas_pixel=second_density,
            role="pretrain_b",
        ),
    )


def _bounded_center(length: int, source_side: float, rng: random.Random) -> float:
    half = source_side / 2.0
    if source_side >= length:
        return length / 2.0
    return rng.uniform(half, length - half)


def _clamp_center(center: float, length: int, source_side: float) -> float:
    if source_side >= length:
        return length / 2.0
    half = source_side / 2.0
    return min(length - half, max(half, center))


def build_pretraining_sample(
    image: Image.Image | RegionImageSource,
    config: ViewConfig,
    *,
    seed: int,
) -> FoveaPretrainingSample:
    first_spec, _ = sample_overlapping_specs(
        image.width,
        image.height,
        config,
        seed=seed,
    )
    second_spec = overview_spec(image.width, image.height, config.overview_size)
    first = render_source_view(image, first_spec, config.pad_color)
    second = render_source_view(image, second_spec, config.pad_color)
    return FoveaPretrainingSample(
        first_image=pil_to_tensor(first),
        first_geometry=torch.tensor(
            first_spec.normalized_geometry(image.width, image.height), dtype=torch.float32
        ),
        second_image=pil_to_tensor(second),
        second_geometry=torch.tensor(
            second_spec.normalized_geometry(image.width, image.height), dtype=torch.float32
        ),
    )


class FoveaImagePathDataset(Dataset):
    """Decode source images and make deterministic overlapping view pairs."""

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        *,
        view_config: ViewConfig = ViewConfig(local_views=2),
        seed: int = 0,
        max_source_pixels: Optional[int] = 40_000_000,
        pyramid_cache_root: Optional[str | Path] = None,
    ) -> None:
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        self.image_paths = tuple(str(path) for path in image_paths)
        self.view_config = view_config
        self.seed = int(seed)
        if max_source_pixels is not None and max_source_pixels <= 0:
            raise ValueError("max_source_pixels must be positive when provided")
        self.max_source_pixels = max_source_pixels
        self.pyramid_cache_root = (
            None if pyramid_cache_root is None else str(pyramid_cache_root)
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Optional[FoveaPretrainingSample]:
        sample_seed = self.seed + self.epoch * len(self.image_paths) + index
        return load_pretraining_sample(
            self.image_paths[index],
            self.view_config,
            seed=sample_seed,
            max_source_pixels=self.max_source_pixels,
            pyramid_cache_root=self.pyramid_cache_root,
        )


def load_pretraining_sample(
    image_path: str | Path,
    view_config: ViewConfig,
    *,
    seed: int,
    max_source_pixels: Optional[int] = 40_000_000,
    pyramid_cache_root: Optional[str | Path] = None,
) -> Optional[FoveaPretrainingSample]:
    """Load one bounded source and produce its deterministic local/overview pair."""
    if max_source_pixels is not None and max_source_pixels <= 0:
        raise ValueError("max_source_pixels must be positive when provided")
    try:
        with Image.open(image_path) as source:
            if (
                max_source_pixels is not None
                and source.width * source.height > max_source_pixels
            ):
                if pyramid_cache_root is None:
                    return None
                cached = resolve_cached_tiled_pyramid(image_path, pyramid_cache_root)
                if cached is None:
                    return None
                return build_pretraining_sample(cached, view_config, seed=seed)
            image = source.convert("RGB")
            image.load()
    except (OSError, ValueError):
        return None
    return build_pretraining_sample(image, view_config, seed=seed)


def collate_pretraining_samples(
    samples: Sequence[Optional[FoveaPretrainingSample]],
) -> Optional[FoveaPretrainingBatch]:
    valid = [sample for sample in samples if sample is not None]
    if not valid:
        return None
    return FoveaPretrainingBatch(
        first_images=torch.stack([sample.first_image for sample in valid]),
        first_geometry=torch.stack([sample.first_geometry for sample in valid]),
        second_images=torch.stack([sample.second_image for sample in valid]),
        second_geometry=torch.stack([sample.second_geometry for sample in valid]),
    )


class FoveaPretrainer(nn.Module):
    """Shared encoder, reconstruction decoder, and teacher-free objectives."""

    def __init__(
        self,
        encoder_config: FoveaEncoderConfig = FoveaEncoderConfig(),
        *,
        mask_ratio: float = 0.4,
        objective: Optional[FoveaSelfSupervisedLoss] = None,
    ) -> None:
        super().__init__()
        if not 0 <= mask_ratio <= 1:
            raise ValueError("mask_ratio must be in [0, 1]")
        self.encoder = FoveaEncoder(encoder_config)
        self.reconstruction_head = PatchReconstructionHead(
            encoder_config.widths[2], patch_size=16
        )
        self.objective = objective or FoveaSelfSupervisedLoss()
        self.mask_ratio = mask_ratio

    def forward(
        self,
        batch: FoveaPretrainingBatch,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> FoveaPretrainingOutput:
        height, width = batch.first_images.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("pretraining canvas dimensions must be divisible by 16")
        patch_mask = random_patch_mask(
            batch.first_images.shape[0],
            height // 16,
            width // 16,
            self.mask_ratio,
            device=batch.first_images.device,
            generator=generator,
        )
        masked_first = mask_image_patches(batch.first_images, patch_mask, patch_size=16)
        first = self.encoder(masked_first, batch.first_geometry)
        second = self.encoder(batch.second_images, batch.second_geometry)
        reconstructed = self.reconstruction_head(first.local_map)
        pixel_mask = patch_mask.repeat_interleave(16, 1).repeat_interleave(16, 2)
        losses = self.objective(
            first.local_map,
            first.local_coordinates,
            second.local_map,
            second.local_coordinates,
            reconstructed,
            batch.first_images,
            pixel_mask,
        )
        return FoveaPretrainingOutput(
            losses=losses,
            reconstruction_mask=patch_mask,
            first_features=first.local_map,
            second_features=second.local_map,
        )
