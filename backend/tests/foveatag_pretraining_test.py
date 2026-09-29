"""CPU tests for the FoveaTag teacher-free pretraining path."""

import math

import torch
from PIL import Image

from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.pretraining import (
    FoveaImagePathDataset,
    FoveaPretrainer,
    build_pretraining_sample,
    collate_pretraining_samples,
    sample_overlapping_specs,
)
from core.tagger.foveatag.views import ViewConfig


def _tiny_encoder() -> FoveaEncoderConfig:
    return FoveaEncoderConfig(
        widths=(8, 12, 16, 32),
        depths=(1, 1, 1, 1),
        attention_heads=(4, 4),
        window_size=4,
    )


def test_overlapping_specs_share_source_area_and_vary_by_seed():
    config = ViewConfig(overview_size=64, local_size=64, local_views=2)
    first, second = sample_overlapping_specs(800, 600, config, seed=7)
    other, _ = sample_overlapping_specs(800, 600, config, seed=8)

    overlap_width = min(first.source_box.x + first.source_box.width, second.source_box.x + second.source_box.width) - max(first.source_box.x, second.source_box.x)
    overlap_height = min(first.source_box.y + first.source_box.height, second.source_box.y + second.source_box.height) - max(first.source_box.y, second.source_box.y)
    assert overlap_width > 0
    assert overlap_height > 0
    for spec in (first, second):
        assert spec.source_box.x >= 0
        assert spec.source_box.y >= 0
        assert spec.source_box.x + spec.source_box.width <= 800
        assert spec.source_box.y + spec.source_box.height <= 600
    assert first != other


def test_pretrainer_runs_backward_on_collated_views():
    config = ViewConfig(overview_size=96, local_size=64, local_views=2)
    image = Image.new("RGB", (320, 240), (60, 100, 180))
    batch = collate_pretraining_samples([
        build_pretraining_sample(image, config, seed=1),
        build_pretraining_sample(image, config, seed=2),
    ])
    assert batch is not None
    model = FoveaPretrainer(_tiny_encoder(), mask_ratio=0.5)
    output = model(batch)
    output.losses.loss.backward()

    assert batch.first_images.shape == (2, 3, 64, 64)
    assert batch.second_images.shape == (2, 3, 96, 96)
    torch.testing.assert_close(
        batch.second_geometry[0],
        torch.tensor([
            0.5,
            0.5,
            1.0,
            1.0,
            0.0,
            math.log2(320 / 96),
            0.0,
            0.125,
            1.0,
            0.75,
        ]),
    )
    assert output.reconstruction_mask.shape == (2, 4, 4)
    assert output.reconstruction_mask.sum().item() == 16
    assert output.first_features.shape[:2] == (2, 16)
    assert output.second_features.shape[:2] == (2, 16)
    assert torch.isfinite(output.losses.loss)
    assert model.encoder.stem_conv.weight.grad is not None


def test_image_path_dataset_is_epoch_deterministic_and_skips_corrupt(tmp_path):
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    Image.new("RGB", (192, 160), (20, 30, 40)).save(good)
    bad.write_text("not an image", encoding="utf-8")
    dataset = FoveaImagePathDataset(
        [good, bad],
        view_config=ViewConfig(overview_size=64, local_size=64, local_views=2),
        seed=4,
    )

    first = dataset[0]
    repeat = dataset[0]
    assert first is not None and repeat is not None
    assert torch.equal(first.first_geometry, repeat.first_geometry)
    assert dataset[1] is None
    dataset.set_epoch(1)
    changed = dataset[0]
    assert changed is not None
    assert not torch.equal(first.first_geometry, changed.first_geometry)
    assert collate_pretraining_samples([None]) is None


def test_image_path_dataset_skips_oversized_source_before_decode(tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (128, 128), (20, 30, 40)).save(path)
    dataset = FoveaImagePathDataset(
        [path],
        view_config=ViewConfig(overview_size=64, local_size=64, local_views=2),
        max_source_pixels=10_000,
    )

    assert dataset[0] is None
