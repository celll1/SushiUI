"""Empty-caption CFG null routing and checkpoint-family guards."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.base_trainer import BaseTrainer
from core.pipeline_backends.qwen_image_21 import QwenImage21Mixin


class _CaptionTrainer:
    encode_caption = BaseTrainer.encode_caption

    def __init__(self, arch):
        self.arch = arch
        self.is_qwen_image_21 = arch.name == "qwen_image_21"
        self.is_sdxl = arch.name == "sdxl"

    def __getattr__(self, name):
        if name.startswith("is_"):
            return False
        raise AttributeError(name)

    def encode_prompt(self, prompt, requires_grad=False):
        self.seen = prompt
        return torch.ones(1, 2, 3), torch.ones(1, 3)


def test_caption_stage_uses_empty_prompt_and_preserves_qwen_mask():
    class Arch:
        name = "qwen_image_21"
        cfg_null_stage = "caption"

        def encode_prompt(self, trainer, prompt):
            trainer.seen = prompt
            return torch.ones(1, 2, 3), torch.ones(2, dtype=torch.bool)

    trainer = _CaptionTrainer(Arch())
    embeds, mask = trainer.encode_caption("character", cfg_null=True)
    assert trainer.seen == ""
    assert embeds.shape == (1, 2, 3)
    assert mask.shape == (2,)


def test_caption_stage_preserves_sdxl_pooled_embedding():
    trainer = _CaptionTrainer(SimpleNamespace(name="sdxl", cfg_null_stage="caption"))
    embeds, pooled = trainer.encode_caption("character", cfg_null=True)
    assert trainer.seen == ""
    assert embeds.shape == (1, 2, 3)
    assert pooled.shape == (1, 3)


@pytest.mark.parametrize("arch,attribute", [
    ("flux2", "is_distilled"),
    ("krea2", "krea2_is_distilled"),
])
def test_distilled_variant_refuses_null_drop(arch, attribute):
    trainer = SimpleNamespace(
        arch=SimpleNamespace(name=arch, cfg_null_stage="caption"),
        config={"cfg_uncond_drop_rate": 0.2},
        log_prefix="[test]",
    )
    setattr(trainer, attribute, True)
    with pytest.raises(ValueError, match="non-distilled"):
        BaseTrainer.cfg_null_drop_rate(trainer)


def test_zimage_turbo_refuses_null_drop():
    trainer = SimpleNamespace(
        arch=SimpleNamespace(name="zimage", cfg_null_stage="caption"),
        config={"cfg_uncond_drop_rate": 0.2},
        model_path="Z-Image-Turbo.safetensors",
        log_prefix="[test]",
    )
    with pytest.raises(ValueError, match="non-distilled"):
        BaseTrainer.cfg_null_drop_rate(trainer)


def test_qwen_cfg_with_blank_negative_is_not_disabled():
    class Pipe:
        def __call__(self, **kwargs):
            self.call = kwargs
            return SimpleNamespace(images=[object()])

    pipe = Pipe()
    backend = SimpleNamespace(
        device=torch.device("cpu"),
        cancel_requested=False,
        _qwen21_lora_session=SimpleNamespace(set_step=lambda *_: None),
        _qwen_image_21_pipe=lambda: pipe,
        _load_lora_qwen21=lambda *_: None,
        _unload_lora_qwen21=lambda: None,
    )
    QwenImage21Mixin._qwen_image_21_run(
        backend, {"prompt": "character", "negative_prompt": "", "cfg_scale": 7})
    assert pipe.call["negative_prompt"] == ""
    assert pipe.call["true_cfg_scale"] == 7.0
