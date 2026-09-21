"""Qwen training-time sample must support FP32 LoRA weights with BF16 inputs."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.ops.qwen_image_21_ops import generate_sample


def test_sample_uses_training_autocast_for_lora(monkeypatch):
    transformer = torch.nn.Linear(2, 2, bias=False)
    transformer.train()

    class Pipe:
        def enable_model_cpu_offload(self, *, device):
            assert device.type == "cpu"

        def __call__(self, **kwargs):
            assert torch.is_autocast_enabled("cpu")
            output = transformer(torch.ones(1, 2, dtype=torch.bfloat16))
            assert output.dtype == torch.bfloat16
            return SimpleNamespace(images=["sample"])

    trainer = SimpleNamespace(
        qwen_image_21_pipeline=Pipe(),
        transformer=transformer,
        text_encoder=torch.nn.Linear(1, 1),
        vae=torch.nn.Linear(1, 1),
        device=torch.device("cpu"),
        training_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = generate_sample(
        trainer, prompt="test", height=64, width=64,
        num_inference_steps=1, guidance_scale=1.0, seed=1,
    )

    assert result == "sample"
    assert transformer.training
