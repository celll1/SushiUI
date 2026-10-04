"""Native precision and nonfinite-output regression gates."""
import pytest
import torch
from ming_image_01_pipeline_test import manager, params
from ming_image_01_conditioning_test import write_tiny_package
from ming_image_01_sampling_test import make_sampler, condition, options


def test_native_bf16_components_keep_fp32_vae(tmp_path):
    write_tiny_package(tmp_path)
    from core.models.ming_image_01.loader import load_components
    loaded = load_components(tmp_path, dtype=torch.bfloat16, conditioner_attention="eager")
    assert loaded.transformer.dtype == torch.bfloat16
    assert next(loaded.conditioner.parameters()).dtype == torch.bfloat16
    assert loaded.vae.dtype == torch.float32


def test_nonfinite_conditioning_refused_before_denoising(monkeypatch):
    value, components = manager("generation_edit")
    from core.models.ming_image_01.conditioning import MingConditioning
    monkeypatch.setattr(components.conditioner, "forward", lambda **kwargs:
        MingConditioning(torch.full((1, 4, 16), float("nan")), torch.zeros(1, 4, 32)))
    with pytest.raises(ValueError, match="nonfinite conditioning"):
        value.generate_txt2img(params())


def test_nonfinite_denoising_refused_before_image_save(monkeypatch):
    sampler = make_sampler("generation_edit")
    original = sampler.components.transformer.forward
    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        return ([torch.full_like(item, float("nan")) for item in result[0]],)
    monkeypatch.setattr(sampler.components.transformer, "forward", broken)
    with pytest.raises(ValueError, match="nonfinite latents at step 1"):
        sampler.run(task="text-to-image", conditioning=condition(), **options())


def test_nonfinite_vae_refused_before_png_conversion(monkeypatch):
    sampler = make_sampler("generation_edit")
    monkeypatch.setattr(sampler.components.vae, "decode", lambda *args, **kwargs:
        (torch.full((1, 4, 32, 32), float("nan")),))
    with pytest.raises(ValueError, match="nonfinite decoded pixels"):
        sampler.run(task="text-to-image", conditioning=condition(), **options())
