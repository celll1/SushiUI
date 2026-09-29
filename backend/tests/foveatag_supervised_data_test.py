"""CPU tests for the existing-dataset to FoveaTag bridge."""

import pytest
import torch
from PIL import Image

from core.tagger.foveatag.supervised_data import (
    DomainBalancedSupervisedDataset,
    FoveaSupervisedDataset,
    SupervisionPolicy,
    build_supervised_views,
    collate_supervised_samples,
    supervision_mask_statistics,
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

    def label_and_mask_for_index(self, index):
        return self.sample.label.clone(), self.sample.loss_mask.clone()


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


def test_supervision_statistics_report_global_mean_mask_dilution():
    statistics = supervision_mask_statistics(
        _RawDataset(),
        {16: (0, 1)},
        {16: SupervisionPolicy(loss_weight=0.25, positive_only_indices=(0, 1))},
        max_samples_per_domain=1,
    )

    domain = statistics["domains"]["16"]
    assert statistics["reduction"] == "multiply_mask_then_mean_all_batch_tag_elements"
    assert statistics["num_tags"] == 3
    assert domain["active_label_fraction"] == pytest.approx(2 / 3)
    assert domain["mean_mask_weight"] == pytest.approx(1 / 6)
    assert domain["mean_active_weight"] == pytest.approx(0.25)
    assert domain["active_positives_per_sample"] == 2


@pytest.mark.parametrize(
    ("input_mode", "expected_views"),
    [("overview_only", 0), ("overview_placebo", 2), ("source_native", 2)],
)
def test_supervised_adapter_selects_explicit_input_branch(input_mode, expected_views):
    dataset = FoveaSupervisedDataset(
        _RawDataset(),
        view_config=ViewConfig(overview_size=64, local_size=64, local_views=2),
        input_mode=input_mode,
    )

    sample = dataset[0]

    assert sample is not None
    assert len(sample.views.local_views) == expected_views
    assert len(sample.views.local_specs) == expected_views


def test_supervised_adapter_rejects_unknown_input_mode():
    with pytest.raises(ValueError, match="unsupported FoveaTag input mode"):
        FoveaSupervisedDataset(_RawDataset(), input_mode="unsupported")


def test_shared_view_builder_matches_dataset_branch_contract():
    source = _RawDataset()
    config = ViewConfig(overview_size=64, local_size=64, local_views=2)
    dataset = FoveaSupervisedDataset(
        source,
        view_config=config,
        seed=10,
        training=False,
        input_mode="overview_placebo",
    )

    sample = dataset[0]
    direct = build_supervised_views(
        source.sample.image,
        config,
        seed=10,
        training=False,
        input_mode="overview_placebo",
    )

    assert sample is not None
    assert sample.views.local_specs == direct.local_specs
    assert tuple(view.tobytes() for view in sample.views.local_views) == tuple(
        view.tobytes() for view in direct.local_views
    )


def test_control_adapter_preserves_original_overview_for_baseline_processor():
    dataset = FoveaSupervisedDataset(
        _RawDataset(),
        view_config=ViewConfig(overview_size=64, local_size=64, local_views=2),
        baseline_overview=True,
    )

    sample = dataset[0]

    assert sample is not None
    assert sample.views.overview.size == (320, 240)
    assert sample.views.overview_spec.role == "baseline-overview"
    assert all(view.size == (64, 64) for view in sample.views.local_views)


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


def test_domain_balanced_sampling_is_stateless_and_changes_repeated_view_seed():
    source = FoveaSupervisedDataset(
        _RawDataset(),
        view_config=ViewConfig(overview_size=64, local_size=64, local_views=2),
        seed=10,
    )
    balanced = DomainBalancedSupervisedDataset(
        source,
        domain_indices={16: (0,), 25: (0,)},
        domain_weights={16: 0.25, 25: 0.75},
        epoch_samples=1000,
        seed=7,
    )

    first = [balanced.selection(index) for index in range(1000)]
    repeat = [balanced.selection(index) for index in range(1000)]

    assert first == repeat
    assert 200 < sum(domain == 16 for domain, _, _ in first) < 300
    assert len({sample_seed for _, _, sample_seed in first}) == 1000
    first_sample = balanced[0]
    assert first_sample is not None
    balanced.set_epoch(1)
    assert balanced.selection(0) != first[0]
