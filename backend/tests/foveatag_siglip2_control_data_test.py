"""CPU tests for the SigLIP2 control processor bridge."""

import torch
from PIL import Image

from core.tagger.foveatag.siglip2_control_data import collate_siglip2_control_samples
from core.tagger.foveatag.supervised_data import FoveaSupervisedSample
from core.tagger.foveatag.views import ViewConfig, build_views


class _Processor:
    def __init__(self, *, naflex: bool):
        self.naflex = naflex
        self.budgets = []

    def __call__(self, *, images, return_tensors, **kwargs):
        assert return_tensors == "pt"
        self.budgets.append(kwargs.get("max_num_patches"))
        count = len(images)
        if self.naflex:
            patches = kwargs["max_num_patches"]
            return {
                "pixel_values": torch.zeros(count, patches, 6),
                "pixel_attention_mask": torch.ones(count, patches, dtype=torch.int32),
                "spatial_shapes": torch.tensor([[1, patches]]).repeat(count, 1),
            }
        return {"pixel_values": torch.zeros(count, 3, 32, 32)}


def _sample(seed: int) -> FoveaSupervisedSample:
    views = build_views(
        Image.new("RGB", (160, 120)),
        ViewConfig(overview_size=64, local_size=32, local_views=2),
        seed=seed,
        training=True,
    )
    return FoveaSupervisedSample(
        views=views,
        label=torch.tensor([1.0, 0.0]),
        loss_mask=torch.tensor([1.0, 1.0]),
    )


def test_naflex_control_collate_keeps_view_axes_and_patch_budgets():
    processor = _Processor(naflex=True)
    batch = collate_siglip2_control_samples(
        [_sample(1), _sample(2), None],
        processor=processor,
        is_naflex=True,
        overview_patch_budget=12,
        local_patch_budget=7,
    )
    assert batch is not None
    assert batch.overview.pixel_values.shape == (2, 12, 6)
    assert batch.local.pixel_values.shape == (2, 2, 7, 6)
    assert batch.local.pixel_attention_mask.shape == (2, 2, 7)
    assert batch.local.spatial_shapes.shape == (2, 2, 2)
    assert processor.budgets == [12, 7]


def test_fixed_control_collate_omits_naflex_auxiliary_inputs():
    processor = _Processor(naflex=False)
    batch = collate_siglip2_control_samples(
        [_sample(3)], processor=processor, is_naflex=False
    )
    assert batch is not None
    assert batch.overview.pixel_values.shape == (1, 3, 32, 32)
    assert batch.local.pixel_values.shape == (1, 2, 3, 32, 32)
    assert batch.local.pixel_attention_mask is None
    assert processor.budgets == [None, None]
