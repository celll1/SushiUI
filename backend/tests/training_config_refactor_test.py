"""Equivalence gates for shared training-configuration plumbing."""

from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.training_config import TrainingConfigGenerator  # noqa: E402


COMMON = {
    "run_name": "refactor-gate",
    "base_model_path": "model.safetensors",
    "output_dir": "output",
    "dataset_path": "fallback",
}


def _process(text: str) -> dict:
    return yaml.safe_load(text)["config"]["process"][0]


@pytest.mark.parametrize(
    "generator",
    [
        TrainingConfigGenerator.generate_lora_config,
        TrainingConfigGenerator.generate_full_finetune_config,
        TrainingConfigGenerator.generate_controlnet_config,
    ],
)
def test_diffusion_generators_keep_duration_validation(generator):
    with pytest.raises(ValueError, match="Either total_steps or epochs"):
        generator({}, **COMMON)
    with pytest.raises(ValueError, match="Cannot specify both"):
        generator({"total_steps": 1, "epochs": 1}, **COMMON)


@pytest.mark.parametrize(
    "generator",
    [
        TrainingConfigGenerator.generate_lora_config,
        TrainingConfigGenerator.generate_full_finetune_config,
        TrainingConfigGenerator.generate_controlnet_config,
        TrainingConfigGenerator.generate_vae_config,
    ],
)
def test_generators_keep_dataset_shape_and_legacy_override(generator):
    params = {
        "total_steps": 2,
        "learning_rate": 1e-4,
        "timestep_sampling": "uniform",
    }
    dataset_configs = [
        {"path": "first", "dataset_id": 0},
        {"path": "second", "dataset_id": 7, "caption_types": ["tags"]},
    ]
    text = generator(
        params,
        dataset_configs=dataset_configs,
        learning_rate=2e-4,
        timestep_sampling_config="logit_normal",
        **COMMON,
    )

    process = _process(text)
    first, second = process["datasets"]
    assert first["folder_path"] == "first"
    assert "dataset_id" not in first
    assert second["dataset_id"] == 7
    assert second["caption_types"] == ["tags"]
    if generator is TrainingConfigGenerator.generate_lora_config:
        assert first["resolution"] == [512, 768, 1024]
    else:
        assert "resolution" not in first
    if generator is not TrainingConfigGenerator.generate_vae_config:
        assert process["train"]["timestep_sampling"] == "uniform"
    assert process["train"]["lr"] == 2e-4
    assert params == {
        "total_steps": 2,
        "learning_rate": 1e-4,
        "timestep_sampling": "uniform",
    }
