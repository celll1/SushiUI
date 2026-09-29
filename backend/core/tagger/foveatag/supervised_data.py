"""Bridge the existing tag dataset lifecycle to FoveaTag multi-view batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import torch
from torch.utils.data import Dataset

from ..tagger_dataset import RawTaggerSample
from .batch import FoveaViewTensorBatch, collate_view_batches
from .views import ViewBatch, ViewConfig, build_views


class RawTaggerDataset(Protocol):
    def __len__(self) -> int: ...

    def get_raw_item(self, index: int) -> Optional[RawTaggerSample]: ...


@dataclass(frozen=True)
class FoveaSupervisedSample:
    views: ViewBatch
    label: torch.Tensor
    loss_mask: torch.Tensor


@dataclass(frozen=True)
class FoveaSupervisedBatch:
    views: FoveaViewTensorBatch
    labels: torch.Tensor
    loss_masks: torch.Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> "FoveaSupervisedBatch":
        return FoveaSupervisedBatch(
            views=self.views.to(device, dtype=dtype, non_blocking=non_blocking),
            labels=self.labels.to(device=device, non_blocking=non_blocking),
            loss_masks=self.loss_masks.to(device=device, non_blocking=non_blocking),
        )


class FoveaSupervisedDataset(Dataset):
    """Apply deterministic epoch-aware FoveaTag views to raw tagged samples."""

    def __init__(
        self,
        source: RawTaggerDataset,
        *,
        view_config: ViewConfig = ViewConfig(),
        seed: int = 0,
        training: bool = True,
    ) -> None:
        self.source = source
        self.view_config = view_config
        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Optional[FoveaSupervisedSample]:
        raw = self.source.get_raw_item(index)
        if raw is None:
            return None
        sample_seed = self.seed + self.epoch * len(self) + index
        return FoveaSupervisedSample(
            views=build_views(
                raw.image,
                self.view_config,
                seed=sample_seed,
                training=self.training,
            ),
            label=raw.label,
            loss_mask=raw.loss_mask,
        )


def collate_supervised_samples(
    samples: Sequence[Optional[FoveaSupervisedSample]],
) -> Optional[FoveaSupervisedBatch]:
    valid = [sample for sample in samples if sample is not None]
    if not valid:
        return None
    return FoveaSupervisedBatch(
        views=collate_view_batches([sample.views for sample in valid]),
        labels=torch.stack([sample.label for sample in valid]),
        loss_masks=torch.stack([sample.loss_mask for sample in valid]),
    )
