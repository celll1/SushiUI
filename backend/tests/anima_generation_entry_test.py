"""Generation entry points dispatch NegPip and restore its wrappers."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from core.pipeline_backends.anima import AnimaMixin
from core.models.anima import anima_pipeline_ops as ops


@pytest.mark.parametrize("mode", ["txt2img", "img2img", "inpaint"])
@pytest.mark.parametrize("signed", [False, True])
def test_entry_dispatches_signed_prompt_without_foreign_local(monkeypatch, mode, signed):
    owner = AnimaMixin()
    model = SimpleNamespace(in_channels=16, patch_spatial=2)
    owner.device = "cpu"
    owner.anima_components = dict(transformer=model, text_encoder=None,
        tokenizer=None, t5_tokenizer=None, vae=None, scheduler=None)
    owner._anima_set_attention_backend = Mock()
    owner._anima_move = Mock()
    owner._anima_stage_transformer = Mock(return_value=model)
    owner._anima_unstage_transformer = Mock()
    owner._load_lora_anima = Mock(return_value=0)
    owner._unload_lora_anima = Mock(return_value=0)
    owner._anima_rewiring_metadata = Mock()
    owner._anima_encode_conditioning = Mock(return_value=(({}, {}, None), False))
    owner._anima_build_nag_wrapper = Mock(return_value=None)
    cond_wrapper = SimpleNamespace(restore=Mock())
    uncond_wrapper = SimpleNamespace(restore=Mock())
    owner._anima_build_negpip = Mock(return_value=(cond_wrapper, uncond_wrapper))
    owner._apply_vae_tiling = Mock()
    sampler = Mock(return_value=torch.zeros(1, 16, 1, 4, 4))
    monkeypatch.setattr(ops, "sample_" + mode, sampler)
    monkeypatch.setattr(ops, "vae_encode_image", lambda *a, **k: torch.zeros(1,16,1,4,4))
    monkeypatch.setattr(ops, "make_mask_latents", lambda *a, **k: torch.ones(1,1,1,4,4))
    image = Image.new("RGB", (32,32))
    monkeypatch.setattr(ops, "vae_decode_latents", lambda *a, **k: [image])
    params = dict(seed=63, ancestral_seed=64, width=32, height=32,
        prompt="(fox:-1)" if signed else "fox")
    args = [] if mode == "txt2img" else [image]
    if mode == "inpaint":
        args.append(Image.new("L", (32,32), 255))
    result = getattr(owner, "_generate_" + mode + "_anima")(params, *args)
    assert result[0] is image
    assert owner._anima_build_negpip.call_count == int(signed)
    kwargs = sampler.call_args.kwargs
    assert kwargs["nag_transformer"] is (cond_wrapper if signed else None)
    assert kwargs["negpip_uncond_transformer"] is (uncond_wrapper if signed else None)
    assert cond_wrapper.restore.call_count == int(signed)
    assert uncond_wrapper.restore.call_count == int(signed)
