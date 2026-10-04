"""Scratch lineage and native reconstruction are checked before training."""

from dataclasses import asdict
import json

import pytest
import torch

from api.param_defaults import TAGGER_TRAINING_DEFAULTS
from core.tagger.foveatag.encoder import FoveaEncoder, FoveaEncoderConfig
from core.tagger.foveatag.pretraining_checkpoint import save_pretrained_encoder
from core.tagger.foveatag.shared_factory import build_model, build_processor


@pytest.fixture
def native_config(tmp_path):
    cfg = FoveaEncoderConfig(widths=(8, 16, 32, 32), depths=(1, 1, 1, 1),
                             attention_heads=(2, 2), scale_dropout=0)
    encoder = FoveaEncoder(cfg)
    path = save_pretrained_encoder(tmp_path / "encoder.safetensors", encoder,
        run_contract={"encoder": asdict(cfg)}, epoch=1, global_step=1, validation={})
    config = dict(TAGGER_TRAINING_DEFAULTS)
    config.update(architecture="foveatag", vision_encoder_path=str(path), training_method="full",
                  fovea_feature_dim=16, fovea_overview_size=32, fovea_local_size=32,
                  lora_rank=2, lora_alpha=2)
    return config, encoder


@pytest.mark.parametrize("method", ["full", "lora"])
def test_loads_scratch_encoder_before_adapter_insertion(native_config, method):
    config, encoder = native_config
    config["training_method"] = method
    instance = build_model(config, 5, vocabulary={}, lineage={})
    for key, value in encoder.state_dict().items():
        if method == "lora":
            for name in instance._lora_modules:
                if key in (f"{name}.weight", f"{name}.bias"):
                    key = key.rsplit(".", 1)[0] + ".base." + key.rsplit(".", 1)[1]
                    break
        torch.testing.assert_close(value, instance.vision_encoder.state_dict()[key], rtol=0, atol=0)


def test_rejects_external_weight_lineage(native_config):
    config, _ = native_config
    path = config["vision_encoder_path"].replace(".safetensors", "_metadata.json")
    metadata = json.loads(open(path, encoding="utf-8").read())
    metadata["external_weights"] = ["external"]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    with pytest.raises(ValueError, match="scratch lineage"):
        build_model(config, 5, vocabulary={}, lineage={})


def test_resume_reconstructs_without_original_base(native_config, tmp_path):
    config, _ = native_config
    instance = build_model(config, 5, vocabulary={}, lineage={})
    path = instance.save_checkpoint(tmp_path, "latest")
    config["vision_encoder_path"] = str(tmp_path / "unavailable.safetensors")
    restored = build_model(config, 5, vocabulary={}, lineage={}, resume_path=path)
    restored.load_weights_inplace(path)
    for key, value in instance.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    config["fovea_feature_dim"] = 32
    with pytest.raises(ValueError, match="feature_dim"):
        build_model(config, 5, vocabulary={}, lineage={}, resume_path=path)


def test_random_frozen_backbone_is_refused(native_config):
    config, _ = native_config
    config.update(training_method="lora", vision_encoder_path="")
    with pytest.raises(ValueError, match="random frozen"):
        build_model(config, 5, vocabulary={}, lineage={})


def test_siglip_pooling_dimensions_are_not_silently_ignored(native_config):
    config, _ = native_config
    config["cls_dim"] = 32
    with pytest.raises(ValueError, match="pooling dimensions"):
        build_model(config, 5, vocabulary={}, lineage={})


def test_processor_training_and_validation_are_separate(native_config):
    config, _ = native_config
    assert build_processor(config, training=True).training
    assert not build_processor(config, training=False).training


@pytest.mark.parametrize("enabled", [False, True])
def test_factory_applies_native_flash_permission(native_config, enabled):
    config, _ = native_config
    config["use_flash_attention"] = enabled
    instance = build_model(config, 5, vocabulary={}, lineage={})
    attention = [module for module in instance.modules() if hasattr(module, "use_sdpa")]
    assert attention
    assert all(module.use_sdpa and module.allow_flash == enabled
               and module.use_external_flash == enabled for module in attention)


def test_lora_resume_is_self_contained_without_initial_backbone_path(native_config, tmp_path):
    config, _ = native_config
    config["training_method"] = "lora"
    instance = build_model(config, 5, vocabulary={}, lineage={})
    path = instance.save_checkpoint(tmp_path, "latest")
    config["vision_encoder_path"] = ""
    restored = build_model(config, 5, vocabulary={}, lineage={}, resume_path=path)
    restored.load_weights_inplace(path)
    for key, value in instance.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
