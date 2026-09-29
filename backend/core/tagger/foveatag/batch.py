"""Tensor batching for coordinate-aware FoveaTag views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .views import ViewBatch, pil_to_tensor


@dataclass(frozen=True)
class FoveaViewTensorBatch:
    overview_images: torch.Tensor
    overview_geometry: torch.Tensor
    local_images: torch.Tensor
    local_geometry: torch.Tensor
    local_view_mask: torch.Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "FoveaViewTensorBatch":
        image_options = {"device": device, "non_blocking": non_blocking}
        if dtype is not None:
            image_options["dtype"] = dtype
        return FoveaViewTensorBatch(
            overview_images=self.overview_images.to(**image_options),
            overview_geometry=self.overview_geometry.to(device=device, non_blocking=non_blocking),
            local_images=self.local_images.to(**image_options),
            local_geometry=self.local_geometry.to(device=device, non_blocking=non_blocking),
            local_view_mask=self.local_view_mask.to(device=device, non_blocking=non_blocking),
        )


def collate_view_batches(batches: Sequence[ViewBatch]) -> FoveaViewTensorBatch:
    """Pad local-view count while retaining exact per-view geometry and validity."""
    if not batches:
        raise ValueError("at least one ViewBatch is required")
    overview_shape = batches[0].overview.size
    local_size = batches[0].local_canvas_size
    if any(batch.overview.size != overview_shape for batch in batches):
        raise ValueError("overview canvases in one batch must have equal dimensions")
    if any(batch.local_canvas_size != local_size for batch in batches):
        raise ValueError("local canvases in one batch must have equal dimensions")
    maximum_views = max(len(batch.local_views) for batch in batches)
    batch_size = len(batches)
    overview_images = torch.stack([pil_to_tensor(batch.overview) for batch in batches])
    overview_geometry = torch.tensor([
        batch.overview_spec.normalized_geometry(batch.overview_spec.source_box.width, batch.overview_spec.source_box.height)
        for batch in batches
    ], dtype=torch.float32)
    local_images = torch.zeros(
        (batch_size, maximum_views, 3, local_size, local_size),
        dtype=torch.float32,
    )
    local_geometry = torch.zeros((batch_size, maximum_views, 10), dtype=torch.float32)
    local_geometry[..., 8:10] = 1.0
    local_view_mask = torch.zeros((batch_size, maximum_views), dtype=torch.bool)
    for batch_index, batch in enumerate(batches):
        image_width = batch.overview_spec.source_box.width
        image_height = batch.overview_spec.source_box.height
        for view_index, (image, spec) in enumerate(zip(batch.local_views, batch.local_specs)):
            local_images[batch_index, view_index] = pil_to_tensor(image)
            local_geometry[batch_index, view_index] = torch.tensor(
                spec.normalized_geometry(image_width, image_height),
                dtype=torch.float32,
            )
            local_view_mask[batch_index, view_index] = True
    return FoveaViewTensorBatch(
        overview_images=overview_images,
        overview_geometry=overview_geometry,
        local_images=local_images,
        local_geometry=local_geometry,
        local_view_mask=local_view_mask,
    )
