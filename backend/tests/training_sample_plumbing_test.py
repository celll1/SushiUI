"""Regression coverage for common in-training sample defaults and plumbing."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.param_defaults import (
    SENSENOVA_GENERATION_DEFAULTS,
    TRAINING_DEFAULTS,
    TRAINING_SAMPLE_DEFAULTS_BY_ARCH,
)
from api.routes import TrainingRunCreateRequest, get_training_defaults
from core.training.arch.base_arch import SampleContext
from core.training.arch.sd15 import SD15ArchHandler
from core.training.arch.sdxl import SDXLArchHandler
from core.training.arch.sensenova import SenseNovaArchHandler
from core.training.base_trainer import BaseTrainer
from core.training.train_runner import _resolve_training_sample_config
from core.training.training_config import TrainingConfigGenerator


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_KEYS = (
    "sample_every",
    "sample_prompts",
    "sample_width",
    "sample_height",
    "sample_steps",
    "sample_cfg_scale",
    "sample_sampler",
    "sample_schedule_type",
    "sample_seed",
    "sensenova_sample_timestep_shift",
    "sensenova_sample_img_cfg_scale",
    "sensenova_sample_cfg_norm",
)


class _ConcreteTrainer(BaseTrainer):
    def setup_trainable_parameters(self):
        return []

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError


def test_training_request_sample_defaults_match_ssot():
    request = TrainingRunCreateRequest(training_method="lora", base_model_path="model")
    for key in SAMPLE_KEYS:
        assert getattr(request, key) == TRAINING_DEFAULTS[key], key

    request.sample_prompts[0]["positive"] = "mutated"
    fresh = TrainingRunCreateRequest(training_method="lora", base_model_path="model")
    assert fresh.sample_prompts == TRAINING_DEFAULTS["sample_prompts"]


@pytest.mark.parametrize(
    "generator",
    [
        TrainingConfigGenerator.generate_lora_config,
        TrainingConfigGenerator.generate_relora_config,
        TrainingConfigGenerator.generate_full_finetune_config,
        TrainingConfigGenerator.generate_controlnet_config,
    ],
)
def test_generated_sample_sections_use_ssot_defaults(generator):
    text = generator(
        {"total_steps": 1},
        run_name="sample-defaults",
        base_model_path="model.safetensors",
        output_dir="output",
        dataset_path="dataset",
    )
    sample = yaml.safe_load(text)["config"]["process"][0]["sample"]
    expected = {
        "sample_every": TRAINING_DEFAULTS["sample_every"],
        "prompts": TRAINING_DEFAULTS["sample_prompts"],
        "width": TRAINING_DEFAULTS["sample_width"],
        "height": TRAINING_DEFAULTS["sample_height"],
        "sample_steps": TRAINING_DEFAULTS["sample_steps"],
        "guidance_scale": TRAINING_DEFAULTS["sample_cfg_scale"],
        "sampler": TRAINING_DEFAULTS["sample_sampler"],
        "schedule_type": TRAINING_DEFAULTS["sample_schedule_type"],
        "seed": TRAINING_DEFAULTS["sample_seed"],
        "sensenova_timestep_shift": TRAINING_DEFAULTS["sensenova_sample_timestep_shift"],
        "sensenova_img_cfg_scale": TRAINING_DEFAULTS["sensenova_sample_img_cfg_scale"],
        "sensenova_cfg_norm": TRAINING_DEFAULTS["sensenova_sample_cfg_norm"],
    }
    for key, value in expected.items():
        assert sample[key] == value, key


def test_runner_resolves_and_preserves_sampler_and_schedule():
    resolved = _resolve_training_sample_config(
        {"sample": {"sampler": "dpmpp_2m", "schedule_type": "karras"}}
    )
    assert resolved["sampler"] == "dpmpp_2m"
    assert resolved["schedule_type"] == "karras"
    assert resolved["seed"] == TRAINING_DEFAULTS["sample_seed"]
    assert resolved["prompts"] == TRAINING_DEFAULTS["sample_prompts"]
    assert resolved["sensenova_timestep_shift"] == TRAINING_DEFAULTS["sensenova_sample_timestep_shift"]
    assert resolved["sensenova_img_cfg_scale"] == TRAINING_DEFAULTS["sensenova_sample_img_cfg_scale"]
    assert resolved["sensenova_cfg_norm"] == TRAINING_DEFAULTS["sensenova_sample_cfg_norm"]


def test_runner_resolves_legacy_sensenova_yaml_with_arch_defaults():
    resolved = _resolve_training_sample_config({"sample": {}}, "sensenova")
    assert resolved["sample_steps"] == SENSENOVA_GENERATION_DEFAULTS["steps"]
    assert resolved["guidance_scale"] == SENSENOVA_GENERATION_DEFAULTS["cfg_scale"]

    explicit = _resolve_training_sample_config(
        {"sample": {"sample_steps": 12, "guidance_scale": 2.0}}, "sensenova"
    )
    assert explicit["sample_steps"] == 12
    assert explicit["guidance_scale"] == 2.0


def test_all_training_method_calls_forward_sampler_and_schedule():
    source = (ROOT / "backend/core/training/train_runner.py").read_text(encoding="utf-8")
    assert source.count("sample_sampler=sample_sampler") == 4
    assert source.count("sample_schedule_type=sample_schedule_type") == 4
    assert source.count("sensenova_sample_timestep_shift=sensenova_sample_timestep_shift") == 4
    assert source.count("sensenova_sample_img_cfg_scale=sensenova_sample_img_cfg_scale") == 4
    assert source.count("sensenova_sample_cfg_norm=sensenova_sample_cfg_norm") == 4


def test_base_trainer_sample_defaults_match_ssot():
    parameters = inspect.signature(BaseTrainer.train).parameters
    assert parameters["sample_every_n_steps"].default == TRAINING_DEFAULTS["sample_every"]
    assert parameters["sample_guidance_scale"].default == TRAINING_DEFAULTS["sample_cfg_scale"]
    assert parameters["sample_steps"].default == TRAINING_DEFAULTS["sample_steps"]
    assert parameters["sample_width"].default == TRAINING_DEFAULTS["sample_width"]
    assert parameters["sample_height"].default == TRAINING_DEFAULTS["sample_height"]
    assert parameters["sample_seed"].default == TRAINING_DEFAULTS["sample_seed"]
    assert parameters["sample_sampler"].default == TRAINING_DEFAULTS["sample_sampler"]
    assert parameters["sample_schedule_type"].default == TRAINING_DEFAULTS["sample_schedule_type"]
    assert parameters["sensenova_sample_timestep_shift"].default == TRAINING_DEFAULTS["sensenova_sample_timestep_shift"]
    assert parameters["sensenova_sample_img_cfg_scale"].default == TRAINING_DEFAULTS["sensenova_sample_img_cfg_scale"]
    assert parameters["sensenova_sample_cfg_norm"].default == TRAINING_DEFAULTS["sensenova_sample_cfg_norm"]


def test_sensenova_omitted_sample_defaults_use_generation_defaults():
    text = TrainingConfigGenerator.generate_lora_config(
        {"total_steps": 1, "_explicit_fields": ["total_steps"]},
        run_name="sensenova-sample-defaults",
        base_model_path="SenseNova-U1.5.safetensors",
        output_dir="output",
        dataset_path="dataset",
    )
    sample = yaml.safe_load(text)["config"]["process"][0]["sample"]
    assert sample["sample_steps"] == SENSENOVA_GENERATION_DEFAULTS["steps"]
    assert sample["guidance_scale"] == SENSENOVA_GENERATION_DEFAULTS["cfg_scale"]
    assert sample["sensenova_timestep_shift"] == SENSENOVA_GENERATION_DEFAULTS["timestep_shift"]
    assert sample["sensenova_img_cfg_scale"] == SENSENOVA_GENERATION_DEFAULTS["img_cfg_scale"]
    assert sample["sensenova_cfg_norm"] == SENSENOVA_GENERATION_DEFAULTS["cfg_norm"]


def test_sensenova_explicit_sample_steps_and_cfg_are_preserved():
    text = TrainingConfigGenerator.generate_full_finetune_config(
        {
            "total_steps": 1,
            "sample_steps": 9,
            "sample_cfg_scale": 2.5,
            "_explicit_fields": ["total_steps", "sample_steps", "sample_cfg_scale"],
        },
        run_name="sensenova-explicit-sample",
        base_model_path="SenseNova-U1.5.safetensors",
        output_dir="output",
        dataset_path="dataset",
    )
    sample = yaml.safe_load(text)["config"]["process"][0]["sample"]
    assert sample["sample_steps"] == 9
    assert sample["guidance_scale"] == 2.5


def test_training_defaults_schema_serves_sample_overlay():
    import asyncio

    payload = asyncio.run(get_training_defaults())
    assert payload["_sample_defaults_by_arch"] == TRAINING_SAMPLE_DEFAULTS_BY_ARCH


def test_dispatch_and_sd_handlers_forward_sampler_and_schedule():
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.arch = SimpleNamespace(sample=Mock(return_value="sample"))
    assert trainer._dispatch_sample(
        "prompt",
        width=64,
        height=64,
        num_inference_steps=2,
        guidance_scale=4.0,
        seed=1,
        sampler="dpmpp_2m",
        schedule_type="karras",
    ) == "sample"
    context = trainer.arch.sample.call_args.args[1]
    assert context.sampler == "dpmpp_2m"
    assert context.schedule_type == "karras"

    context = SampleContext(
        prompt="prompt",
        width=64,
        height=64,
        num_inference_steps=2,
        guidance_scale=4.0,
        seed=1,
        sampler="dpmpp_2m",
        schedule_type="karras",
    )
    for handler_type in (SD15ArchHandler, SDXLArchHandler):
        target = SimpleNamespace(generate_sample=Mock(return_value="sample"))
        assert handler_type().sample(target, context) == "sample"
        assert target.generate_sample.call_args.kwargs["sampler"] == "dpmpp_2m"
        assert target.generate_sample.call_args.kwargs["schedule_type"] == "karras"


def test_sensenova_handler_forwards_preview_specific_controls():
    context = SampleContext(
        prompt="prompt",
        width=64,
        height=64,
        num_inference_steps=2,
        guidance_scale=4.0,
        seed=1,
        sensenova_timestep_shift=6.0,
        sensenova_img_cfg_scale=1.75,
        sensenova_cfg_norm="none",
    )
    trainer = SimpleNamespace()
    with pytest.MonkeyPatch.context() as monkeypatch:
        from core.training.ops import sensenova_ops

        generate = Mock(return_value="sample")
        monkeypatch.setattr(sensenova_ops, "generate_sample", generate)
        assert SenseNovaArchHandler().sample(trainer, context) == "sample"
    assert generate.call_args.kwargs["timestep_shift"] == 6.0
    assert generate.call_args.kwargs["img_cfg_scale"] == 1.75
    assert generate.call_args.kwargs["cfg_norm"] == "none"


def test_sd_and_controlnet_scheduler_creation_uses_selected_sampler():
    sd_source = (ROOT / "backend/core/training/ops/sd_sdxl_ops.py").read_text(
        encoding="utf-8"
    )
    controlnet_source = (
        ROOT / "backend/core/training/controlnet_trainer.py"
    ).read_text(encoding="utf-8")
    assert sd_source.count("sampler=sampler") == 2
    assert controlnet_source.count("sampler=sampler") >= 6
    assert 'sampler="euler"' not in sd_source
    assert 'sampler="euler"' not in controlnet_source


def test_frontend_sample_fallbacks_do_not_diverge_from_default_object():
    source = (
        ROOT / "frontend/src/components/training/TrainingConfig.tsx"
    ).read_text(encoding="utf-8")
    assert 'params.sample_schedule_type ?? "uniform"' not in source
    assert 'updateParam("sample_seed", 42)' not in source
    for key in ("sample_every", "sample_width", "sample_height", "sample_seed"):
        assert f'updateParam("{key}", DEFAULT_PARAMS.{key})' in source
    assert 'updateParam("sample_steps", sampleStepsDefault)' in source
    assert 'updateParam("sample_cfg_scale", sampleCfgScaleDefault)' in source
