"""CPU tests for source-native generic router candidates."""

import torch
from PIL import Image

from core.tagger.foveatag.router_data import (
    FoveaRouterDataset,
    collate_router_samples,
)
from core.tagger.foveatag.supervised_data import SupervisionPolicy
from core.tagger.foveatag.views import ViewConfig
from core.tagger.tagger_dataset import RawTaggerSample


class _RawDataset:
    def __init__(self):
        self.samples = [
            RawTaggerSample(
                image=Image.new("RGB", (80, 48), (10, 20, 30)),
                label=torch.tensor([1.0, 0.0]),
                loss_mask=torch.ones(2),
                source_path="sample.png",
                dataset_id=7,
            )
        ]

    def __len__(self):
        return len(self.samples)

    def get_raw_item(self, index):
        return self.samples[index]


def test_router_dataset_builds_generic_multiscale_source_candidates():
    dataset = FoveaRouterDataset(
        _RawDataset(),
        view_config=ViewConfig(
            overview_size=32,
            local_size=16,
            local_views=1,
            field_levels=(0.5, 1.0, 2.0),
            scale_jitter=(1.0, 1.0),
        ),
        candidate_count=6,
        seed=4,
        training=False,
    )

    sample = dataset[0]
    assert sample is not None
    assert len(sample.views.local_views) == 6
    assert sample.candidate_boxes.shape == (6, 4)
    assert len(set(sample.candidate_boxes[:, 2].tolist())) == 3
    assert torch.all((sample.candidate_boxes[:, :2] >= 0) & (sample.candidate_boxes[:, :2] <= 1))


def test_router_collation_preserves_boxes_geometry_and_unknown_label_mask():
    dataset = FoveaRouterDataset(
        _RawDataset(),
        view_config=ViewConfig(
            overview_size=32,
            local_size=16,
            local_views=1,
            field_levels=(0.5, 1.0),
            scale_jitter=(1.0, 1.0),
        ),
        candidate_count=4,
        supervision_policies={7: SupervisionPolicy(
            loss_weight=0.25,
            positive_only_indices=(0, 1),
        )},
        training=False,
    )
    sample = dataset[0]
    batch = collate_router_samples([sample])

    assert batch is not None
    torch.testing.assert_close(batch.candidate_boxes, batch.views.local_geometry[..., :4])
    torch.testing.assert_close(batch.loss_masks, torch.tensor([[0.25, 0.0]]))
    assert batch.views.local_view_mask.all()


def test_router_candidate_sampling_is_epoch_deterministic():
    dataset = FoveaRouterDataset(
        _RawDataset(),
        view_config=ViewConfig(
            overview_size=32,
            local_size=16,
            local_views=1,
            field_levels=(0.5, 1.0),
        ),
        candidate_count=4,
        seed=9,
        training=True,
    )
    first = dataset[0].candidate_boxes
    second = dataset[0].candidate_boxes
    dataset.set_epoch(1)
    next_epoch = dataset[0].candidate_boxes

    torch.testing.assert_close(first, second)
    assert not torch.equal(first, next_epoch)


def test_staged_router_transfer_keeps_dense_local_pixels_on_host():
    dataset = FoveaRouterDataset(
        _RawDataset(),
        view_config=ViewConfig(
            overview_size=16,
            local_size=8,
            local_views=4,
            field_levels=(0.5, 1.0),
            scale_jitter=(1.0, 1.0),
        ),
        candidate_count=4,
        training=False,
    )
    batch = collate_router_samples([dataset[0]])

    staged = batch.stage_for_routing("meta")

    assert staged.views.overview_images.device.type == "meta"
    assert staged.candidate_boxes.device.type == "meta"
    assert staged.labels.device.type == "meta"
    assert staged.views.local_view_mask.device.type == "meta"
    assert staged.views.local_images.device.type == "cpu"
    assert staged.views.local_geometry.device.type == "cpu"
