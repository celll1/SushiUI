"""CPU tests for the existing-dataset to FoveaTag bridge."""

import pytest
import torch
from PIL import Image

from core.tagger.foveatag.supervised_data import (
    FoveaSupervisedDataset,
    SupervisionPolicy,
    collate_supervised_samples,
)
from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.model import FoveaTagger
from core.tagger.foveatag.views import ViewConfig
from core.tagger.tagger_dataset import RawTaggerSample


class _RawDataset:
    def __init__(self):
        self.sample = RawTaggerSample(
            image=Image.new("RGB", (320, 240), (40, 80, 120)),
            label=torch.tensor([1.0, 0.0, 1.0]),
            loss_mask=torch.tensor([1.0, 0.0, 1.0]),
            dataset_id=16,
        )

    def __len__(self):
        return 2

    def get_raw_item(self, index):
        return self.sample if index == 0 else None


def test_supervised_adapter_preserves_labels_and_epoch_changes_views():
    dataset = FoveaSupervisedDataset(
        _RawDataset(),
        view_config=ViewConfig(overview_size=64, local_size=64, local_views=2),
        seed=10,
    )
    first = dataset[0]
    repeat = dataset[0]
    assert first is not None and repeat is not None
    assert first.views.local_specs == repeat.views.local_specs
    assert torch.equal(first.label, torch.tensor([1.0, 0.0, 1.0]))
    assert dataset[1] is None

    dataset.set_epoch(1)
    changed = dataset[0]
    assert changed is not None
    assert first.views.local_specs != changed.views.local_specs


def test_supervised_collate_filters_failures_and_retains_masks():
    dataset = FoveaSupervisedDataset(
        _RawDataset(),
        view_config=ViewConfig(overview_size=64, local_size=64, local_views=2),
    )
    batch = collate_supervised_samples([dataset[0], dataset[1]])
    assert batch is not None
    assert batch.views.overview_images.shape == (1, 3, 64, 64)
    assert batch.views.local_images.shape == (1, 2, 3, 64, 64)
    assert torch.equal(batch.labels, torch.tensor([[1.0, 0.0, 1.0]]))
    assert torch.equal(batch.loss_masks, torch.tensor([[1.0, 0.0, 1.0]]))
    assert collate_supervised_samples([None]) is None


def test_supervised_adapter_applies_mixed_provenance_policy():
    dataset = FoveaSupervisedDataset(
        _RawDataset(),
        view_config=ViewConfig(overview_size=64, local_size=64, local_views=0),
        supervision_policies={
            16: SupervisionPolicy(loss_weight=0.25, positive_only_indices=(0, 1))
        },
    )
    sample = dataset[0]
    assert sample is not None
    assert torch.equal(sample.loss_mask, torch.tensor([0.25, 0.0, 0.25]))


@pytest.mark.parametrize("loss_weight", [-0.1, float("nan"), float("inf")])
def test_supervision_policy_rejects_invalid_loss_weight(loss_weight):
    with pytest.raises(ValueError, match="finite and non-negative"):
        SupervisionPolicy(loss_weight=loss_weight)


def test_supervised_batch_runs_end_to_end_backward():
    dataset = FoveaSupervisedDataset(
        _RawDataset(),
        view_config=ViewConfig(overview_size=64, local_size=64, local_views=2),
    )
    batch = collate_supervised_samples([dataset[0]])
    assert batch is not None
    model = FoveaTagger(
        3,
        encoder_config=FoveaEncoderConfig(
            widths=(8, 12, 16, 32),
            depths=(1, 1, 1, 1),
            attention_heads=(4, 4),
            window_size=4,
        ),
        feature_dim=16,
        evidence_slots_per_view=1,
        pool_heads=4,
    )
    logits = model(
        batch.views.overview_images,
        batch.views.overview_geometry,
        batch.views.local_images,
        batch.views.local_geometry,
        local_view_mask=batch.views.local_view_mask,
    )
    element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.labels, reduction="none"
    )
    loss = (element_loss * batch.loss_masks).sum() / batch.loss_masks.sum()
    loss.backward()
    assert logits.shape == (1, 3)
    assert model.head.weight.grad is not None
