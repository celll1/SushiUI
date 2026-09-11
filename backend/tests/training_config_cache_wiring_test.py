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
