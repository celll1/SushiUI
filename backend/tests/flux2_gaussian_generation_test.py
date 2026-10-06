"""Exercise production Flux2 loops with synthetic codecs and a real small DiT."""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from backend.tests.flux2_gaussian_boundaries_test import aligned, ids
from core.pipeline_backends import flux2 as backend
from core.models.flux2.vae_alignment import _contexts
from core.inference.reference_style import StyleTransferConfig


class SyntheticVAE(torch.nn.Linear):
    dtype = torch.float32
    spatial_scale_factor = 16
    config = SimpleNamespace(latent_channels=5)

    def encode(self, pixels):
        latent = torch.full((1, 5, pixels.shape[-2] // 16, pixels.shape[-1] // 16), .25)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, latent, return_dict=False):
        return (torch.nn.functional.interpolate(latent[:, :3], scale_factor=16),)


class Owner(backend.Flux2Mixin):
    device = torch.device("cpu")
    cancel_requested = False

    def __init__(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.flux2_components = dict(transformer=aligned(), vae=SyntheticVAE(1, 1),
            text_encoder=torch.nn.Linear(12, 12), tokenizer=None,
            scheduler=FlowMatchEulerDiscreteScheduler(), config={})

    def _load_lora_flux2(self, *args): pass
    def _flux2_rewiring_metadata(self, *args): pass
    def _flux2_te_quantization_with_lora(self, value): return value
    def _flux2_runtime_int8(self, params, transformer, **kwargs): return transformer
    def _flux2_negpip_eligible(self, *args): return False
    def _flux2_build_fbcache(self, *args, **kwargs): return None
    def _flux2_cleanup(self, *args, **kwargs): pass
    def _apply_vae_tiling(self, *args): pass

    def _flux2_encode_conditioning(self, *args):
        text = torch.randn(1, 3, 12)
        return (text, ids(1, 3), text.clone(), ids(1, 3), None, None), False, False, ""

    def encode_flux2_image_refs(self, *args, **kwargs):
        return torch.full((1, 4, 20), .4), ids(2, 2, 10)


@pytest.mark.parametrize("route", ["txt2img", "img2img", "inpaint"])
@pytest.mark.parametrize("cfg", [1., 2.])
@pytest.mark.parametrize("reference", [False, True])
def test_production_loop_keeps_target_noise_and_cleans_context(route, cfg, reference):
    owner = Owner()
    received = []
    real_forward = backend._aligned_forward
    def record(driver, noise, **kwargs):
        received.append((noise.clone(), kwargs["hidden_states"].clone()))
        return real_forward(driver, noise, **kwargs)
    params = dict(prompt="synthetic", height=64, width=96, steps=3,
        cfg_scale=cfg, seed=63, ancestral_seed=63, strength=.8, attention_type="native")
    if reference: params["ref_images"] = [Image.new("RGB", (64, 64), "blue")]
    with ExitStack() as stack:
        stack.enter_context(patch.object(backend, "_aligned_forward", side_effect=record))
        stack.enter_context(patch.object(backend, "set_flux2_attention_backend"))
        stack.enter_context(patch.object(backend, "_vae_normalize", side_effect=lambda x, vae: x))
        stack.enter_context(patch.object(backend, "_vae_denormalize", side_effect=lambda x, vae: x))
        stack.enter_context(patch("core.vram_optimization.move_flux2_transformer_to_gpu", side_effect=lambda model, *a, **kw: model))
        image = Image.new("RGB", (96, 64), "blue")
        if route == "txt2img": result = owner._generate_txt2img_flux2(params)
        elif route == "img2img": result = owner._generate_img2img_flux2(params, image)
        else: result = owner._generate_inpaint_flux2(params, image, Image.new("L", image.size, 0))
    assert result[0].size == (96, 64)
    assert received and received[0][0].shape == (1, 6, 20)
    for noise, state in received:
        torch.testing.assert_close(noise, received[0][0])
        assert state.shape[1:] == (10 if reference else 6, 20)
        if reference: torch.testing.assert_close(state[:, 6:], torch.full_like(state[:, 6:], .4))
    if route == "inpaint":
        import numpy as np
        assert np.unique(np.asarray(result[0])).tolist() == [159]
    assert _contexts.get() == ()


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("cfg", [False, True])
def test_style_capture_uses_each_reference_noise(multi, cfg):
    owner = Owner()
    value = owner.flux2_components["transformer"]
    noise = torch.randn(1, 6, 20)
    refs = [(StyleTransferConfig(block_range=(0, 2), style_guidance_scale=1.5),
        torch.randn_like(noise), torch.randn_like(noise)) for _ in range(2)]
    received = []
    real_forward = backend._aligned_forward
    def record(driver, eps, **kwargs):
        received.append(eps.clone())
        return real_forward(driver, eps, **kwargs)
    common = dict(transformer_wrapper=value, style_processors=[], step_idx=1, total_steps=3,
        t=torch.tensor(500.), sigma_now=.5, latents=torch.randn_like(noise),
        prompt_embeds=torch.randn(1, 3, 12), text_ids=ids(1, 3),
        negative_prompt_embeds=torch.randn(1, 3, 12), negative_text_ids=ids(1, 3),
        latent_ids=ids(2, 3), do_classifier_free_guidance=cfg, guidance_scale=2.,
        guidance_vec=None if cfg else torch.ones(1), transformer_input_dtype=torch.float32,
        exposure_noise=noise)
    with patch.object(backend, "_aligned_forward", side_effect=record):
        if multi:
            result = owner._flux2_style_step_multi(**common, style_refs=refs, style_combine_mode="stack")
        else:
            result = owner._flux2_style_step(**common, style_cfg=refs[0][0], style_ref_x0=refs[0][1], style_eps_ref=refs[0][2])
    count = 2 if multi else 1
    for index in range(count): torch.testing.assert_close(received[index], refs[index][2])
    for eps in received[count:]: torch.testing.assert_close(eps, noise)
    assert result.shape == noise.shape and torch.isfinite(result).all()
    assert _contexts.get() == ()
