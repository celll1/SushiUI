"""Native data/model contracts for shared tagger training."""

import json
import pytest
import torch
from PIL import Image

from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.processor import FoveaProcessor, concatenate_view_tensors
from core.tagger.foveatag.trainer_model import FoveaTrainerModel
from core.tagger.foveatag.views import ViewConfig
from core.tagger.tagger_dataset import tagger_collate_fn


def processor(training=False):
    return FoveaProcessor(ViewConfig(overview_size=32, local_size=32, local_views=2),
                          seed=13, training=training)


def model(method):
    return FoveaTrainerModel(
        5, encoder_config=FoveaEncoderConfig(
            widths=(8, 16, 32, 32), depths=(1, 1, 1, 1),
            attention_heads=(2, 2), scale_dropout=0,
        ), feature_dim=16, evidence_slots=2, pool_heads=2, temperature=.1,
        training_method=method, lora_rank=2, lora_alpha=2,
    )


def inputs():
    return processor()(images=[Image.new("RGB", (96, 48), (16, 64, 128))],
                       return_tensors="pt")["fovea_views"]


def test_native_collate_retains_labels_masks_and_coordinates():
    sample = inputs()
    labels = torch.tensor([1., 0., 0., 1., 0.])
    masks = torch.tensor([1., 0., 1., .25, 1.])
    item = (sample, torch.zeros(0, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int64), labels, masks)
    batch = tagger_collate_fn([item, None, item])
    assert batch[0].overview_images.shape == (2, 3, 32, 32)
    assert batch[0].local_images.shape == (2, 2, 3, 32, 32)
    torch.testing.assert_close(batch[0].local_geometry[0], sample.local_geometry[0])
    torch.testing.assert_close(batch[3], labels.repeat(2, 1))
    torch.testing.assert_close(batch[4], masks.repeat(2, 1))
    assert batch[1].shape == (2, 0)
    assert batch[2].shape == (2, 0)


def test_validation_views_do_not_consume_rng():
    state = torch.get_rng_state()
    first, second = inputs(), inputs()
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(first.local_images, second.local_images)
    torch.testing.assert_close(first.local_geometry, second.local_geometry)


def test_training_views_use_seeded_worker_rng():
    image = Image.new("RGB", (256, 128))
    torch.manual_seed(71)
    first = processor(True)(images=[image], return_tensors="pt")["fovea_views"]
    torch.manual_seed(71)
    second = processor(True)(images=[image], return_tensors="pt")["fovea_views"]
    torch.testing.assert_close(first.local_geometry, second.local_geometry)


def test_shape_mismatch_is_not_silently_padded():
    different = FoveaProcessor(ViewConfig(overview_size=64, local_size=32, local_views=2),
                               seed=13, training=False)(
        images=[Image.new("RGB", (96, 48))], return_tensors="pt")["fovea_views"]
    with pytest.raises(ValueError, match="shapes differ"):
        concatenate_view_tensors([inputs(), different])


@pytest.mark.parametrize("method", ["full", "lora"])
def test_model_training_and_exact_weight_roundtrip(method, tmp_path):
    torch.manual_seed(18)
    original = model(method)
    original.vision_encoder.set_sdpa(True)
    batch = inputs()
    original.train()
    optimizer = torch.optim.AdamW(original.trainable_parameters(), lr=.01)
    original(batch).square().mean().backward()
    assert original.head.weight.grad is not None
    if method == "lora":
        assert original._lora_modules
        for adapter in original._lora_modules.values():
            assert adapter.base.weight.grad is None
            assert adapter.lora_B.grad is not None
    else:
        assert any(parameter.grad is not None for parameter in original.vision_encoder.parameters())
    optimizer.step()
    path = original.save_checkpoint(tmp_path, "latest")
    metadata = json.loads((tmp_path / "latest_metadata.json").read_text())
    assert metadata["feature_dim"] == 16
    assert metadata["pool_heads"] == 2
    assert metadata["contains_frozen_base"] is True
    restored = model(method)
    restored.vision_encoder.set_sdpa(True)
    restored.load_weights_inplace(path)
    original.eval()
    restored.eval()
    torch.testing.assert_close(original(batch), restored(batch), rtol=0, atol=0)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("method", ["full", "lora"])
def test_head_growth_preserves_existing_rows(method):
    instance = model(method)
    weight = instance.head.weight.detach().clone()
    instance.expand_head(8)
    assert instance.head.out_features == 8
    torch.testing.assert_close(instance.head.weight[:5], weight)


def test_native_model_rejects_siglip_input():
    with pytest.raises(TypeError, match="native"):
        model("full")(torch.zeros(1, 3, 32, 32))


@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_native_bfloat16_backward(method, checkpointing):
    instance = model(method).train()
    instance.vision_encoder.set_sdpa(True)
    if checkpointing:
        instance.vision_encoder.gradient_checkpointing_enable()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = instance(inputs())
        loss = logits.float().square().mean()
    loss.backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(instance.head.weight.grad).all()
    if method == "lora":
        assert all(adapter.lora_B.grad is not None for adapter in instance._lora_modules.values())
