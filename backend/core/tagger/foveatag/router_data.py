"""Source-native dense-candidate data bridge for generic router training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Protocol, Sequence

import torch
from torch.utils.data import Dataset

from ..tagger_dataset import RawTaggerSample
from .batch import FoveaViewTensorBatch, collate_view_batches
from .supervised_data import SupervisionPolicy
from .views import ViewBatch, ViewConfig, ViewSpec, build_views


class RawRouterDataset(Protocol):
    def __len__(self) -> int: ...

    def get_raw_item(self, index: int) -> Optional[RawTaggerSample]: ...


@dataclass(frozen=True)
class FoveaRouterSample:
    views: ViewBatch
    candidate_boxes: torch.Tensor
    label: torch.Tensor
    loss_mask: torch.Tensor
    diagnostic_source: Optional[dict] = None


@dataclass(frozen=True)
class FoveaRouterBatch:
    views: FoveaViewTensorBatch
    candidate_boxes: torch.Tensor
    labels: torch.Tensor
    loss_masks: torch.Tensor
    diagnostic_sources: tuple[Optional[dict], ...] = ()

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> "FoveaRouterBatch":
        return FoveaRouterBatch(
            views=self.views.to(device, dtype=dtype, non_blocking=non_blocking),
            candidate_boxes=self.candidate_boxes.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
            labels=self.labels.to(device=device, non_blocking=non_blocking),
            loss_masks=self.loss_masks.to(device=device, non_blocking=non_blocking),
            diagnostic_sources=self.diagnostic_sources,
        )

    def stage_for_routing(
        self,
        device: torch.device | str,
        *,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> "FoveaRouterBatch":
        """Move overview and routing metadata while retaining local pixels on CPU."""
        image_options = {"device": device, "non_blocking": non_blocking}
        if dtype is not None:
            image_options["dtype"] = dtype
        views = FoveaViewTensorBatch(
            overview_images=self.views.overview_images.to(**image_options),
            overview_geometry=self.views.overview_geometry.to(
                device=device, non_blocking=non_blocking
            ),
            local_images=self.views.local_images,
            local_geometry=self.views.local_geometry,
            local_view_mask=self.views.local_view_mask.to(
                device=device, non_blocking=non_blocking
            ),
        )
        return FoveaRouterBatch(
            views=views,
            candidate_boxes=self.candidate_boxes.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
            labels=self.labels.to(device=device, non_blocking=non_blocking),
            loss_masks=self.loss_masks.to(device=device, non_blocking=non_blocking),
            diagnostic_sources=self.diagnostic_sources,
        )

    def pin_memory(self) -> "FoveaRouterBatch":
        return FoveaRouterBatch(
            views=self.views.pin_memory(),
            candidate_boxes=self.candidate_boxes.pin_memory(),
            labels=self.labels.pin_memory(),
            loss_masks=self.loss_masks.pin_memory(),
            diagnostic_sources=self.diagnostic_sources,
        )


class FoveaRouterDataset(Dataset):
    """Render deterministic spatial/scale candidates without semantic priors."""

    def __init__(
        self,
        source: RawRouterDataset,
        *,
        view_config: ViewConfig = ViewConfig(),
        candidate_count: int,
        seed: int = 0,
        training: bool = True,
        supervision_policies: Optional[Mapping[int, SupervisionPolicy]] = None,
    ) -> None:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        self.source = source
        self.view_config = replace(view_config, local_views=candidate_count)
        self.candidate_count = int(candidate_count)
        self.seed = int(seed)
        self.training = bool(training)
        self.supervision_policies = dict(supervision_policies or {})
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Optional[FoveaRouterSample]:
        return self.sample(index)

    def sample(
        self, index: int, *, sample_seed: Optional[int] = None
    ) -> Optional[FoveaRouterSample]:
        raw = self.source.get_raw_item(index)
        if raw is None:
            return None
        if sample_seed is None:
            sample_seed = self.seed + self.epoch * len(self) + index
        loss_mask = raw.loss_mask
        if raw.dataset_id is not None:
            policy = self.supervision_policies.get(raw.dataset_id)
            if policy is not None:
                loss_mask = policy.apply(raw.label, loss_mask)
        views = build_views(
            raw.image,
            self.view_config,
            seed=sample_seed,
            training=self.training,
        )
        boxes = candidate_boxes_from_specs(
            views.local_specs,
            image_width=raw.image.width,
            image_height=raw.image.height,
        )
        return FoveaRouterSample(
            views=views,
            candidate_boxes=boxes,
            label=raw.label,
            loss_mask=loss_mask,
            diagnostic_source={"source_path": raw.source_path, "source_index": index,
                "sample_seed": sample_seed, "dataset_id": raw.dataset_id},
        )


def candidate_boxes_from_specs(
    specs: Sequence[ViewSpec], *, image_width: int, image_height: int
) -> torch.Tensor:
    """Convert source-pixel view boxes to normalized center/size proposals."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    rows = []
    for spec in specs:
        center_x, center_y = spec.source_box.center
        rows.append((
            center_x / image_width,
            center_y / image_height,
            spec.source_box.width / image_width,
            spec.source_box.height / image_height,
        ))
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)


def collate_router_samples(
    samples: Sequence[Optional[FoveaRouterSample]],
) -> Optional[FoveaRouterBatch]:
    valid = [sample for sample in samples if sample is not None]
    if not valid:
        return None
    candidate_counts = {sample.candidate_boxes.shape[0] for sample in valid}
    if len(candidate_counts) != 1:
        raise ValueError("router samples in one batch must have equal candidate counts")
    return FoveaRouterBatch(
        views=collate_view_batches([sample.views for sample in valid]),
        candidate_boxes=torch.stack([sample.candidate_boxes for sample in valid]),
        labels=torch.stack([sample.label for sample in valid]),
        loss_masks=torch.stack([sample.loss_mask for sample in valid]),
        diagnostic_sources=tuple(sample.diagnostic_source for sample in valid),
    )
