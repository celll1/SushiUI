"""Wrong diffusers pipeline classes can succeed with missing XL conditioners."""
from types import SimpleNamespace

import pytest
import torch

from core.model_loader import ModelLoader
from core.training.ops import sd_sdxl_ops as ops


@pytest.mark.parametrize("architecture", ["sd15", "sdxl"])
def test_header_selection_does_not_probe_the_other_pipeline(monkeypatch, architecture):
    calls = []
    pipelines = {}
    for name in ("sd15", "sdxl"):
        def load(path, *, name=name, **kwargs):
            calls.append((name, path, kwargs))
            return SimpleNamespace(tokenizer=object(), tokenizer_2=None)
        pipelines[name] = SimpleNamespace(from_single_file=load)
    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda path: architecture)
    monkeypatch.setattr(ops, "StableDiffusionPipeline", pipelines["sd15"])
    monkeypatch.setattr(ops, "StableDiffusionXLPipeline", pipelines["sdxl"])
    replacement = object()
    result, is_sdxl = ops._load_native_single_file(
        "native.safetensors", dtype=torch.bfloat16, overrides={"vae": replacement})
    assert result.tokenizer is not None
    assert is_sdxl == (architecture == "sdxl")
    assert calls == [(architecture, "native.safetensors", {
        "torch_dtype": torch.bfloat16, "use_safetensors": True, "vae": replacement})]


def test_corrupt_xl_error_is_not_reinterpreted_as_sd15(monkeypatch):
    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda path: "sdxl")
    def fail(*args, **kwargs):
        raise RuntimeError("corrupt XL weights")
    monkeypatch.setattr(ops, "StableDiffusionXLPipeline", SimpleNamespace(from_single_file=fail))
    monkeypatch.setattr(ops, "StableDiffusionPipeline", SimpleNamespace(
        from_single_file=lambda *a, **k: pytest.fail("wrong fallback")))
    with pytest.raises(RuntimeError, match="corrupt XL"):
        ops._load_native_single_file("bad.safetensors", dtype=torch.bfloat16, overrides={})


def test_non_sd_architecture_is_refused_before_weight_loading(monkeypatch):
    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda path: "qwen_image_21")
    with pytest.raises(ValueError, match="qwen_image_21"):
        ops._load_native_single_file("qwen.safetensors", dtype=torch.bfloat16, overrides={})
