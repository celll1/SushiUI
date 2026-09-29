"""Bridge the existing tag dataset lifecycle to FoveaTag multi-view batches."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal, Mapping, Optional, Protocol, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..tagger_dataset import RawTaggerSample
from .batch import FoveaViewTensorBatch, collate_view_batches
from .views import (
    ViewBatch,
    ViewConfig,
    build_views,
    preserve_source_overview,
    render_overview_placebo_locals,
)


FoveaInputMode = Literal["overview_only", "overview_placebo", "source_native"]
PHASE3_INPUT_MODES: tuple[FoveaInputMode, ...] = (
    "overview_only",
    "overview_placebo",
    "source_native",
)
_UINT64_MASK = (1 << 64) - 1


class RawTaggerDataset(Protocol):
    def __len__(self) -> int: ...

    def get_raw_item(self, index: int) -> Optional[RawTaggerSample]: ...


@dataclass(frozen=True)
class FoveaSupervisedSample:
    views: ViewBatch
    label: torch.Tensor
    loss_mask: torch.Tensor


@dataclass(frozen=True)
class SupervisionPolicy:
    """Per-dataset confidence policy for mixed-provenance labels."""

    loss_weight: float = 1.0
    positive_only_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.loss_weight) or self.loss_weight < 0:
            raise ValueError("loss_weight must be finite and non-negative")
        if any(index < 0 for index in self.positive_only_indices):
            raise ValueError("positive_only_indices must be non-negative")

    def apply(self, label: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        adjusted = loss_mask.clone().mul_(self.loss_weight)
        if self.positive_only_indices:
            indices = torch.tensor(
                self.positive_only_indices,
                dtype=torch.long,
                device=label.device,
            )
            if int(indices.max()) >= label.numel():
                raise ValueError("positive_only_indices exceed the label dimension")
            adjusted[indices] *= label[indices]
        return adjusted


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
        supervision_policies: Optional[Mapping[int, SupervisionPolicy]] = None,
        baseline_overview: bool = False,
        input_mode: FoveaInputMode = "source_native",
    ) -> None:
        self.source = source
        self.view_config = view_config
        self.seed = int(seed)
        self.training = bool(training)
        self.supervision_policies = dict(supervision_policies or {})
        self.baseline_overview = bool(baseline_overview)
        if input_mode not in PHASE3_INPUT_MODES:
            raise ValueError(f"unsupported FoveaTag input mode: {input_mode}")
        if self.baseline_overview and input_mode != "source_native":
            raise ValueError("baseline_overview is only valid with source_native input")
        self.input_mode = input_mode
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Optional[FoveaSupervisedSample]:
        return self.sample(index)

    def sample(
        self,
        index: int,
        *,
        sample_seed: Optional[int] = None,
    ) -> Optional[FoveaSupervisedSample]:
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
        views = build_supervised_views(
            raw.image,
            self.view_config,
            seed=sample_seed,
            training=self.training,
            input_mode=self.input_mode,
        )
        if self.baseline_overview:
            views = preserve_source_overview(raw.image, views)
        return FoveaSupervisedSample(
            views=views,
            label=raw.label,
            loss_mask=loss_mask,
        )


class DomainBalancedSupervisedDataset(Dataset):
    """Stateless domain-balanced sampling with distinct view seeds per draw."""

    def __init__(
        self,
        source: FoveaSupervisedDataset,
        domain_indices: Mapping[int, Sequence[int]],
        domain_weights: Mapping[int, float],
        *,
        epoch_samples: int,
        seed: int,
    ) -> None:
        if epoch_samples <= 0:
            raise ValueError("epoch_samples must be positive")
        if set(domain_indices) != set(domain_weights) or not domain_indices:
            raise ValueError("domain indices and weights must contain the same domains")
        if any(not indices for indices in domain_indices.values()):
            raise ValueError("every supervised domain must contain samples")
        if any(not math.isfinite(float(weight)) or weight <= 0 for weight in domain_weights.values()):
            raise ValueError("domain weights must be finite and positive")
        self.source = source
        self.domain_ids = tuple(sorted(domain_indices))
        self.domain_indices = {
            domain: tuple(int(index) for index in domain_indices[domain])
            for domain in self.domain_ids
        }
        total = sum(float(domain_weights[domain]) for domain in self.domain_ids)
        cumulative = 0.0
        thresholds = []
        for domain in self.domain_ids:
            cumulative += float(domain_weights[domain]) / total
            thresholds.append(cumulative)
        thresholds[-1] = 1.0
        self.thresholds = tuple(thresholds)
        self.epoch_samples = int(epoch_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.source.set_epoch(epoch)

    def __len__(self) -> int:
        return self.epoch_samples

    def selection(self, index: int) -> tuple[int, int, int]:
        if index < 0 or index >= self.epoch_samples:
            raise IndexError(index)
        key = _splitmix64(self.seed + self.epoch * self.epoch_samples + index)
        fraction = key / float(1 << 64)
        position = next(
            position for position, threshold in enumerate(self.thresholds) if fraction < threshold
        )
        domain = self.domain_ids[position]
        item_key = _splitmix64(key)
        indices = self.domain_indices[domain]
        source_index = indices[item_key % len(indices)]
        sample_seed = _splitmix64(item_key) & 0x7FFFFFFF
        return domain, source_index, sample_seed

    def __getitem__(self, index: int) -> Optional[FoveaSupervisedSample]:
        _, source_index, sample_seed = self.selection(index)
        return self.source.sample(source_index, sample_seed=sample_seed)


def build_supervised_views(
    image: Image.Image,
    view_config: ViewConfig,
    *,
    seed: int,
    training: bool,
    input_mode: FoveaInputMode,
) -> ViewBatch:
    """Build exactly the A/B/C input contract shared by training and evaluation."""
    if input_mode not in PHASE3_INPUT_MODES:
        raise ValueError(f"unsupported FoveaTag input mode: {input_mode}")
    effective_view_config = (
        replace(view_config, local_views=0)
        if input_mode == "overview_only"
        else view_config
    )
    views = build_views(
        image,
        effective_view_config,
        seed=seed,
        training=training,
    )
    if input_mode == "overview_placebo":
        return render_overview_placebo_locals(
            views,
            pad_color=view_config.pad_color,
        )
    return views


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


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK
