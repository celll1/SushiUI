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

from api.param_defaults import TRAINING_DEFAULTS
from api.routes import TrainingRunCreateRequest
from core.training.arch.base_arch import SampleContext
from core.training.arch.sd15 import SD15ArchHandler
from core.training.arch.sdxl import SDXLArchHandler
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


def test_all_training_method_calls_forward_sampler_and_schedule():
    source = (ROOT / "backend/core/training/train_runner.py").read_text(encoding="utf-8")
    assert source.count("sample_sampler=sample_sampler") == 4
    assert source.count("sample_schedule_type=sample_schedule_type") == 4


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
    for key in ("sample_every", "sample_width", "sample_height", "sample_steps", "sample_cfg_scale", "sample_seed"):
        assert f'updateParam("{key}", DEFAULT_PARAMS.{key})' in source
