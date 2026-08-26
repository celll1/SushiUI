"""Regression tests for latent-cache options emitted by training configs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.training_config import TrainingConfigGenerator


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
    assert dataset["cache_latents_to_disk"] is True
    assert dataset["force_recache"] is True
    assert process["train"]["base_resolutions"] == [2048, 4096]


def test_visible_frontend_controls_are_sent_and_restored():
    source = (ROOT / "frontend/src/components/training/TrainingConfig.tsx").read_text(
        encoding="utf-8"
    )
    request = source[source.index("const getRequestData"):source.index("const applyParamsToState")]
    restore = source[source.index("const applyParamsToState"):]

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
        assert f"{key}: params.{key}" in request, key
        assert f'"{key}"' in restore, key

    assert "base_resolutions: params.base_resolutions" in request
    assert "Cache latents to disk (reduces VRAM usage)" not in source

