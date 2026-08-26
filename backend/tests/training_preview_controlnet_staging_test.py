"""Training-preview ControlNet staging across repeated previews.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_preview_controlnet_staging_test.py -q

The preview path's finally offloads the process-global ControlNet cache to CPU
after every preview, but `load_controlnet` only stages on the FIRST load — a
cache hit returns the module exactly as it was left. Nothing between the two
re-stages it, so preview #2 onward died on a device mismatch for the life of the
run. (The generation path is safe for a different reason: `_apply_controlnets`
ends with `cn_pipeline.to(self.device)`.)
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.extensions import controlnet_manager as cn_module
from core.training.training_inference import TrainingPreviewGenerator


class _FakeControlNet:
    def __init__(self):
        self.device = "cpu"
        self.dtype = None
        self.moves = []

    def to(self, device=None, dtype=None):
        self.moves.append((str(device), dtype))
        self.device = str(device)
        self.dtype = dtype
        return self


class _FakeTempPipeline:
    is_sdxl = False
    vae = object()
    text_encoder = object()
    tokenizer = object()
    unet = object()
    scheduler = object()
    vae_scale_factor = 8


def _generator(device, dtype):
    gen = TrainingPreviewGenerator.__new__(TrainingPreviewGenerator)
    gen.trainer = types.SimpleNamespace(device=device, training_dtype=dtype)
    return gen


def test_cached_controlnet_is_restaged_on_every_preview(monkeypatch):
    """MUTANT: dropping the re-stage after load_controlnet. Preview #1 works;
    the preview finally moves the cached module to CPU; preview #2 raises
    'Expected all tensors to be on the same device' forever after."""
    import torch

    cn = _FakeControlNet()
    monkeypatch.setattr(cn_module.controlnet_manager, "load_controlnet",
                        lambda **kwargs: cn)
    monkeypatch.setattr("diffusers.StableDiffusionControlNetPipeline",
                        lambda **kwargs: types.SimpleNamespace(**kwargs))

    gen = _generator("cuda", torch.float16)
    configs = [{"model": "cn.safetensors"}]

    gen._maybe_build_controlnet_pipeline(_FakeTempPipeline(), configs, 512, 512)
    assert cn.device == "cuda"

    # What the preview's own finally does after every request.
    cn.to("cpu")
    assert cn.device == "cpu"

    gen._maybe_build_controlnet_pipeline(_FakeTempPipeline(), configs, 512, 512)
    assert cn.device == "cuda"
    assert cn.dtype is torch.float16


def test_no_controlnets_returns_none(monkeypatch):
    import torch
    gen = _generator("cuda", torch.float16)
    assert gen._maybe_build_controlnet_pipeline(_FakeTempPipeline(), [], 512, 512) is None
