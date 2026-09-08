"""Regression tests for latent-cache options emitted by training configs."""

from __future__ import annotations

import sys
from unittest.mock import patch
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.training_config import TrainingConfigGenerator
from core.training.train_runner import _apply_reference_training_contract


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "generator",
    [
        TrainingConfigGenerator.generate_lora_config,
        TrainingConfigGenerator.generate_full_finetune_config,
        TrainingConfigGenerator.generate_controlnet_config,
    ],
)
def test_force_recache_reaches_the_dataset_section(generator):
    """The removed run keys are dropped from the dataset section (invariant)."""
    text = generator(
        {
            "total_steps": 1,
            "base_resolutions": [2048, 4096],
            "cache_latents_to_disk": True,
            "force_recache": True,
        },
        run_name="cache-wiring",
        base_model_path="model.safetensors",
        output_dir="output",
        dataset_path="dataset",
    )

    process = yaml.safe_load(text)["config"]["process"][0]
    dataset = process["datasets"][0]
    assert "cache_latents_to_disk" not in dataset
    assert "force_recache" not in dataset
    assert process["train"]["base_resolutions"] == [2048, 4096]


def test_visible_frontend_controls_are_sent_and_restored():
    """Each of these controls is visible in the form, so a run started from it
    must carry the value and an edit of that run must show it again.

    `getRequestData` no longer names the fields: it spreads
    `passThroughParams(params)`, which copies every PARAM_KEYS entry that
    `getRequestData` does not build itself, and `applyParamsToState` restores
    through the same list. So the question "is this control sent and restored"
    is now "is it in PARAM_KEYS, and if it is also a computed key, does the
    request build it" -- which is what this asks, through the same readers
    training_edit_restore_coverage_test.py uses.
    """
    from training_edit_restore_coverage_test import (
        _param_keys, _request_data, _request_keys, _source,
    )

    source = _source()
    param_keys = set(_param_keys(source))
    request_keys = _request_keys(source)
    request = _request_data(source)

    keys = (
        "anima_lora_scope",
        "train_llm_adapter",
        "anima_attn_mlp_lr_factor",
        "anima_mod_lr_factor",
        "anima_llm_adapter_lr_factor",
        "lens_lora_scope",
        "lens_img_lr_factor",
        "lens_txt_lr_factor",
        "ideogram4_lora_scope",
        "ideogram4_train_uncond",
        "ideogram4_uncond_loss_weight",
        "ideogram4_lr_factor",
        "minit2i_lora_scope",
        "minit2i_te_lora_scope",
        "text_encoding_prefetch_depth",
        "cpu_offload_checkpointing",
        "async_cpu_offload_checkpointing",
        "fp8_base_dtype",
    )
    for key in keys:
        # In PARAM_KEYS is what makes it restorable: applyParamsToState's loop
        # is over that list, so a key outside it silently reverts to the default
        # when a run is edited.
        assert key in param_keys, f"{key} is not in PARAM_KEYS, so an edit loses it"
        assert key in request_keys, f"{key} never reaches the request"

    panel = (ROOT / "frontend/src/components/training/TrainingConfig.tsx").read_text(
        encoding="utf-8"
    )
    assert "[Math.max(...params.base_resolutions!)]" in request
    assert 'type={enableBucketing ? "checkbox" : "radio"}' in panel
    assert 'updateParam("base_resolutions", [res])' in source
    assert "Cache latents to disk (reduces VRAM usage)" not in source


def test_siglip2_selection_arms_reference_conditioning_in_generated_config():
    text = TrainingConfigGenerator.generate_lora_config(
        {
            "total_steps": 1,
            "use_reference_images": False,
            "vision_encoder_path": "siglip2.safetensors",
        },
        run_name="sdxl-ve-wiring",
        base_model_path="sdxl.safetensors",
        output_dir="output",
        dataset_path="dataset",
    )

    train = yaml.safe_load(text)["config"]["process"][0]["train"]
    assert train["use_reference_images"] is True
    assert train["vision_encoder_path"] == "siglip2.safetensors"


def test_frontend_uses_arch_specific_reference_controls():
    source = (ROOT / "frontend/src/components/training/TrainingConfig.tsx").read_text(
        encoding="utf-8"
    )

    assert "Enable reference image conditioning (FLUX.2 only)" not in source
    assert 'updateParam("use_reference_images", !!path)' in source
    assert "isFlux2Model(baseModelPath) || isSenseNovaModel(baseModelPath)" in source
    assert "referenceConditioningEnabled" in source
    assert "Block text-encoder gradients on reference batches" in source


def test_sdxl_reference_contract_derives_flag_from_vision_encoder():
    train = {
        "use_reference_images": False,
        "vision_encoder_path": "siglip2.safetensors",
    }
    with patch("core.model_loader.ModelLoader.detect_model_type", return_value="sdxl"):
        _apply_reference_training_contract("model", train)

    assert train["use_reference_images"] is True


@pytest.mark.parametrize(
    ("model_type", "train", "message"),
    [
        ("sdxl", {"use_reference_images": True}, "requires vision_encoder_path"),
        (
            "zimage",
            {"vision_encoder_path": "siglip2.safetensors"},
            "only for SD1.5/SDXL",
        ),
        (
            "zimage",
            {"use_reference_images": True},
            "supported only for FLUX.2, SenseNova",
        ),
    ],
)
def test_reference_contract_rejects_unwired_combinations(model_type, train, message):
    with patch("core.model_loader.ModelLoader.detect_model_type", return_value=model_type):
        with pytest.raises(ValueError, match=message):
            _apply_reference_training_contract("model", train)
