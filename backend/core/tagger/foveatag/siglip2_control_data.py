"""Processor bridge for the Phase-1 SigLIP2 encoder-swap control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from .siglip2_control import SigLIP2ViewInputs
from .supervised_data import FoveaSupervisedSample


@dataclass(frozen=True)
class SigLIP2ControlBatch:
    overview: SigLIP2ViewInputs
    local: SigLIP2ViewInputs
    local_view_mask: torch.Tensor
    labels: torch.Tensor
    loss_masks: torch.Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "SigLIP2ControlBatch":
        return SigLIP2ControlBatch(
            overview=_inputs_to(self.overview, device, non_blocking),
            local=_inputs_to(self.local, device, non_blocking),
            local_view_mask=self.local_view_mask.to(device=device, non_blocking=non_blocking),
            labels=self.labels.to(device=device, non_blocking=non_blocking),
            loss_masks=self.loss_masks.to(device=device, non_blocking=non_blocking),
        )


def _inputs_to(
    inputs: SigLIP2ViewInputs,
    device: torch.device | str,
    non_blocking: bool,
) -> SigLIP2ViewInputs:
    return SigLIP2ViewInputs(
        pixel_values=inputs.pixel_values.to(device=device, non_blocking=non_blocking),
        pixel_attention_mask=(
            inputs.pixel_attention_mask.to(device=device, non_blocking=non_blocking)
            if inputs.pixel_attention_mask is not None else None
        ),
        spatial_shapes=(
            inputs.spatial_shapes.to(device=device, non_blocking=non_blocking)
            if inputs.spatial_shapes is not None else None
        ),
    )


def collate_siglip2_control_samples(
    samples: Sequence[Optional[FoveaSupervisedSample]],
    *,
    processor: Any,
    is_naflex: bool,
    overview_patch_budget: int = 784,
    local_patch_budget: int = 256,
) -> Optional[SigLIP2ControlBatch]:
    """Batch PIL views through the existing encoder's matching processor."""
    valid = [sample for sample in samples if sample is not None]
    if not valid:
        return None
    view_counts = {len(sample.views.local_views) for sample in valid}
    if len(view_counts) != 1:
        raise ValueError("SigLIP2 control samples must have equal local-view counts")
    view_count = view_counts.pop()

    overview_kwargs = {
        "images": [sample.views.overview for sample in valid],
        "return_tensors": "pt",
    }
    if is_naflex:
        overview_kwargs["max_num_patches"] = overview_patch_budget
    overview_raw = processor(**overview_kwargs)
    batch_size = len(valid)

    overview = _view_inputs(overview_raw, is_naflex=is_naflex)
    if view_count:
        local_kwargs = {
            "images": [image for sample in valid for image in sample.views.local_views],
            "return_tensors": "pt",
        }
        if is_naflex:
            local_kwargs["max_num_patches"] = local_patch_budget
        local_raw = processor(**local_kwargs)
        flat_local = _view_inputs(local_raw, is_naflex=is_naflex)
        local = SigLIP2ViewInputs(
            pixel_values=flat_local.pixel_values.reshape(
                batch_size, view_count, *flat_local.pixel_values.shape[1:]
            ),
            pixel_attention_mask=_reshape_optional(
                flat_local.pixel_attention_mask, batch_size, view_count
            ),
            spatial_shapes=_reshape_optional(
                flat_local.spatial_shapes, batch_size, view_count
            ),
        )
    else:
        if is_naflex:
            pixel_shape = (
                batch_size,
                0,
                local_patch_budget,
                overview.pixel_values.shape[-1],
            )
            local = SigLIP2ViewInputs(
                pixel_values=overview.pixel_values.new_empty(pixel_shape),
                pixel_attention_mask=torch.empty(
                    (batch_size, 0, local_patch_budget), dtype=torch.int32
                ),
                spatial_shapes=torch.empty((batch_size, 0, 2), dtype=torch.long),
            )
        else:
            local = SigLIP2ViewInputs(
                pixel_values=overview.pixel_values.new_empty(
                    (batch_size, 0, *overview.pixel_values.shape[1:])
                )
            )
    return SigLIP2ControlBatch(
        overview=overview,
        local=local,
        local_view_mask=torch.ones((batch_size, view_count), dtype=torch.bool),
        labels=torch.stack([sample.label for sample in valid]),
        loss_masks=torch.stack([sample.loss_mask for sample in valid]),
    )


def _view_inputs(raw: Any, *, is_naflex: bool) -> SigLIP2ViewInputs:
    return SigLIP2ViewInputs(
        pixel_values=raw["pixel_values"],
        pixel_attention_mask=raw.get("pixel_attention_mask") if is_naflex else None,
        spatial_shapes=raw.get("spatial_shapes") if is_naflex else None,
    )


def _reshape_optional(
    tensor: Optional[torch.Tensor], batch_size: int, view_count: int
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.reshape(batch_size, view_count, *tensor.shape[1:])
