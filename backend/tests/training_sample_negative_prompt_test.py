"""Training sample negative prompts reach each image architecture's sampler."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.arch.base_arch import SampleContext
from core.training.arch.flux2 import Flux2ArchHandler
from core.training.arch.sd15 import SD15ArchHandler
from core.training.arch.sdxl import SDXLArchHandler
from core.training.arch.zimage import ZImageArchHandler
from core.training.base_trainer import BaseTrainer
from core.training.controlnet_trainer import ControlNetTrainer


class _ConcreteTrainer(BaseTrainer):
    def setup_trainable_parameters(self):
        return []

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError


def _sample_context() -> SampleContext:
    return SampleContext(
        prompt="a cat",
        width=64,
        height=64,
        num_inference_steps=4,
        guidance_scale=4.0,
        seed=1,
        negative_prompt="blurry",
    )


@pytest.mark.parametrize("handler_type", [SD15ArchHandler, SDXLArchHandler])
def test_sd_handlers_forward_negative_prompt(handler_type):
    trainer = SimpleNamespace(generate_sample=Mock(return_value="sample"))

    assert handler_type().sample(trainer, _sample_context()) == "sample"
    assert trainer.generate_sample.call_args.kwargs["negative_prompt"] == "blurry"


@pytest.mark.parametrize(
    ("handler_type", "ops_target"),
    [
        (Flux2ArchHandler, "core.training.ops.flux2_ops.generate_sample"),
        (ZImageArchHandler, "core.training.ops.zimage_ops.generate_sample"),
    ],
)
def test_dit_handlers_forward_negative_prompt(handler_type, ops_target):
    with patch(ops_target, return_value="sample") as generate:
        assert handler_type().sample(object(), _sample_context()) == "sample"

    assert generate.call_args.kwargs["negative_prompt"] == "blurry"


def test_sd_base_delegator_forwards_negative_prompt_to_sampler_ops():
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    with patch(
        "core.training.ops.sd_sdxl_ops.generate_sample", return_value="sample"
    ) as generate:
        assert trainer.generate_sample("a cat", negative_prompt="blurry") == "sample"

    assert generate.call_args.kwargs["negative_prompt"] == "blurry"


@pytest.mark.parametrize(
    ("controlnet_type", "method_name"),
    [("standard", "_generate_sample_standard"), ("lllite", "_generate_sample_lllite")],
)
def test_controlnet_sample_dispatch_keeps_negative_prompt(controlnet_type, method_name):
    trainer = ControlNetTrainer.__new__(ControlNetTrainer)
    trainer.controlnet_type = controlnet_type
    trainer._load_sample_condition_image = Mock(return_value=object())
    setattr(trainer, method_name, Mock(return_value="sample"))

    assert trainer.generate_sample("a cat", negative_prompt="blurry") == "sample"
    assert getattr(trainer, method_name).call_args.kwargs["negative_prompt"] == "blurry"


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        (
            "core/training/ops/sd_sdxl_ops.py",
            "trainer.encode_prompt(negative_prompt, requires_grad=False)",
        ),
        (
            "core/training/ops/flux2_ops.py",
            "_flux2_encode_prompt_for_sample(trainer, negative_prompt)",
        ),
        ("core/training/ops/zimage_ops.py", 'trainer.encode_prompt_zimage(negative_prompt)'),
    ],
)
def test_sampler_ops_use_the_forwarded_negative_prompt(relative_path, expected):
    source = (Path(__file__).resolve().parents[1] / relative_path).read_text(encoding="utf-8")
    assert expected in source
