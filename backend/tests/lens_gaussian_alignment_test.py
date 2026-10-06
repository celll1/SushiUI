"""Actual Lens noise, timestep and conditioning boundaries on synthetic tensors."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from backend.tests.gaussian_alignment_test import pair
from backend.tests.lens_spatial_swap_test import model, no_cuda
from core.models.components.spatial_swap import map_packed_image


@pytest.mark.parametrize("rewired", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_and_unet_forward_use_actual_sigma_and_noise(rewired, dtype):
    value = model(5, rewired).to(dtype)
    value.add_module("vae_latent_bridge", pair(dtype))
    bridge = value.vae_latent_bridge
    clean = torch.randn(2, 96, 20, dtype=dtype)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([.2, .8], dtype=dtype)
    states = (1-sigma[:, None, None])*clean + sigma[:, None, None]*noise
    features = [torch.randn(2, 5, 12, dtype=dtype) for _ in range(2)]
    mask = torch.ones(2, 5, dtype=torch.bool)
    actual = value(states, features, mask, sigma, [(1, 8, 12)], alignment_noise=noise)
    native = value._forward_native(map_packed_image(bridge, states, (8, 12), sigma=sigma, noise=noise),
        features, mask, sigma, [(1, 8, 12)])
    torch.testing.assert_close(actual, map_packed_image(bridge, native, (8, 12), output=True, noise=noise))
    actual.float().square().mean().backward()
    assert bridge.input.projection.weight.grad.abs().sum() > 0
    assert bridge.output.projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="noise"):
        value(states, features, mask, sigma, [(1, 8, 12)])


class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module("vae_latent_bridge", pair())
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append((hidden_states.clone(), kwargs))
        return hidden_states * .1


@pytest.mark.parametrize("mode", ["txt", "img", "inpaint"])
def test_generation_loops_keep_real_noise_and_cfg_batch(mode):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from core.models.lens import lens_pipeline_ops as ops
    value = Recorder()
    initial = torch.randn(1, 6, 20)
    features = [torch.randn(2, 3, 12)]
    kwargs = dict(encoder_features=features, encoder_mask=torch.ones(2, 3, dtype=torch.bool),
        guidance_scale=2, num_inference_steps=3, latent_h=2, latent_w=3)
    scheduler = FlowMatchEulerDiscreteScheduler()
    if mode == "txt":
        result = ops.denoise_loop(value, scheduler, initial, **kwargs)
        expected = initial.repeat(2, 1, 1)
    else:
        generator = torch.Generator().manual_seed(63)
        expected = torch.randn(initial.shape, generator=generator).repeat(2, 1, 1)
        if mode == "img":
            result = ops.denoise_loop_img2img(value, scheduler, initial, .8, seed=63, **kwargs)
        else:
            result = ops.denoise_loop_inpaint(value, scheduler, initial, torch.full((1, 6, 1), .4),
                .8, seed=63, **kwargs)
    assert torch.isfinite(result).all() and value.calls
    for _, call in value.calls:
        torch.testing.assert_close(call["alignment_noise"], expected)
        assert call["timestep"].max() <= 1


def test_nag_expands_noise_with_the_cond_image():
    from core.inference.nag_lens import LensNAGWrapper
    value = Recorder()
    wrapper = LensNAGWrapper(value)
    images = torch.randn(2, 6, 20)
    noise = torch.randn_like(images)
    result = wrapper(images, [torch.randn(3, 4, 12)], torch.ones(3, 4, dtype=torch.bool),
        torch.tensor([.4, .4]), [(1, 2, 3)], alignment_noise=noise)
    assert result.shape == images.shape
    received, call = value.calls[0]
    torch.testing.assert_close(received, torch.cat([images, images[:1]]))
    torch.testing.assert_close(call["alignment_noise"], torch.cat([noise, noise[:1]]))


def test_inpaint_kept_region_matches_each_forward_sigma_and_finishes_clean():
    from diffusers import FlowMatchEulerDiscreteScheduler
    from core.models.lens.lens_pipeline_ops import denoise_loop_inpaint
    value = Recorder()
    initial = torch.randn(1, 6, 20)
    result = denoise_loop_inpaint(value, FlowMatchEulerDiscreteScheduler(), initial, torch.zeros(1, 6, 1), .9,
        [torch.randn(2, 3, 12)], torch.ones(2, 3, dtype=torch.bool), 2, 4, 2, 3, seed=63)
    for received, call in value.calls:
        sigma = call["timestep"][:, None, None]
        expected = (1-sigma)*initial.repeat(2, 1, 1)+sigma*call["alignment_noise"]
        torch.testing.assert_close(received, expected)
    torch.testing.assert_close(result, initial)


@pytest.mark.parametrize("multi,guidance", [(False,None), (False,1.5), (True,None)])
def test_style_capture_uses_own_noise_and_prediction_uses_target_noise(multi,guidance):
    from core.models.lens import lens_pipeline_ops as ops
    from core.inference.reference_style import StyleTransferConfig
    value = Recorder()
    noise = torch.randn(1, 6, 20)
    refs = [(StyleTransferConfig(style_guidance_scale=guidance), torch.randn_like(noise), torch.randn_like(noise)) for _ in range(2)]
    args = (0, 3, torch.tensor(500.), .5, torch.randn_like(noise), [torch.randn(2, 3, 12)],
        torch.ones(2, 3, dtype=torch.bool), 2., [(1, 2, 3)], None)
    if multi:
        ops._lens_style_step_multi(value, refs, "stack", *args, exposure_noise=noise)
        reference_noises = [ref[2] for ref in refs]
    else:
        ops._lens_style_step(value, *refs[0], *args, exposure_noise=noise)
        reference_noises = [refs[0][2]]
    for (_, call), expected in zip(value.calls, reference_noises):
        torch.testing.assert_close(call["alignment_noise"], expected)
    for _, call in value.calls[len(reference_noises):]:
        torch.testing.assert_close(call["alignment_noise"], noise)


def test_stochastic_scheduler_refused_only_for_gaussian():
    from core.models.lens.vae_alignment import validate_scheduler, noise_kwargs
    schedule = SimpleNamespace(config=SimpleNamespace(stochastic_sampling=True))
    with pytest.raises(ValueError, match="deterministic"):
        validate_scheduler(Recorder(), schedule)
    validate_scheduler(torch.nn.Linear(2, 2), schedule)
    assert noise_kwargs(torch.nn.Linear(2, 2), None) == {}


@pytest.mark.parametrize("channels,scale,gaussian", [(8,16,True),(4,16,True),(8,8,False),(2,16,False)])
def test_new_policy_requires_stronger_compression_and_no_channel_loss(channels,scale,gaussian):
    from core.models.lens.vae_alignment import apply_swap
    from core.models.components.wiring import LENS_WIRING
    from core.models.components.gaussian_alignment import POLICY
    value = model(4)
    trainer = SimpleNamespace(transformer=value, config={"vae_swap_new_channel_init":"paired_spatial"})
    handler = SimpleNamespace(resolve_wiring=lambda t: LENS_WIRING.replace(latent_channels=4))
    with patch("core.models.lens.vae_alignment.initialize_image_swap") as initialize:
        apply_swap(trainer, SimpleNamespace(latent_channels=channels, scale_factor=scale), handler=handler)
    assert initialize.call_args.kwargs["noise_policy"] == (POLICY if gaussian else None)


@pytest.mark.parametrize("mode", ["txt", "img", "inpaint"])
def test_cancel_releases_wrappers_and_style_context(mode):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from core.models.lens import lens_pipeline_ops as ops
    value = Recorder()
    value._style_ctx = object()
    wrapper = SimpleNamespace(restore=lambda: restored.append(True))
    restored = []
    initial = torch.randn(1, 6, 20)
    kwargs = dict(encoder_features=[torch.randn(2, 3, 12)], encoder_mask=torch.ones(2, 3, dtype=torch.bool),
        guidance_scale=2, num_inference_steps=3, latent_h=2, latent_w=3)
    with patch.object(ops, "_maybe_setup_nag", side_effect=lambda m, f, mask, p: (m, f, mask, wrapper)), \
         patch.object(ops, "raise_if_cancelled", side_effect=RuntimeError("cancelled")), \
         pytest.raises(RuntimeError, match="cancelled"):
        scheduler = FlowMatchEulerDiscreteScheduler()
        if mode == "txt":
            ops.denoise_loop(value, scheduler, initial, **kwargs)
        elif mode == "img":
            ops.denoise_loop_img2img(value, scheduler, initial, .8, **kwargs)
        else:
            ops.denoise_loop_inpaint(value, scheduler, initial, torch.ones(1, 6, 1), .8, **kwargs)
    assert restored == [True] and value._style_ctx is None


@pytest.mark.parametrize("autocast", [False, True])
def test_training_uses_sigma_and_individual_exposure_noise(autocast):
    from core.training.ops.lens_ops import train_step
    value = Recorder()
    trainer = SimpleNamespace(device=torch.device("cpu"), training_dtype=torch.bfloat16 if autocast else torch.float32,
        mixed_precision=autocast, timestep_sampler=None, transformer=value,
        stash_cfg_null_per_sample_loss=lambda *a: None)
    clean = torch.randn(2, 6, 20)
    noise = torch.randn_like(clean).to(trainer.training_dtype)
    sigma = torch.tensor([.2, .8])
    with patch("core.training.mnt.training_noise_like", return_value=noise), \
         patch("core.training.ops.crop_decode_loss.crop_decode_or_x0_capture_needed", return_value=False):
        train_step(trainer, clean, torch.randn(2, 1, 3, 12), torch.ones(2, 3, dtype=torch.bool),
            sigma, latent_h=2, latent_w=3)
    received, call = value.calls[0]
    torch.testing.assert_close(call["timestep"], sigma.to(trainer.training_dtype))
    torch.testing.assert_close(call["alignment_noise"], noise)
    s = sigma.to(trainer.training_dtype)[:, None, None]
    torch.testing.assert_close(received, (1-s)*clean.to(trainer.training_dtype)+s*noise)
