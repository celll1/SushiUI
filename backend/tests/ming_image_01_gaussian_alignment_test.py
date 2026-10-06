"""Real small Ming DiTs preserve exposure noise across independent RGBA frames."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file

from core.models.ming_image_01 import vae_alignment as alignment
from core.models.ming_image_01.conditioning import MingConditioning
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.sampling import MingSampler, MingGenerationCancelled
from core.models.ming_image_01.training import flow_loss
from core.models.ming_image_01.reference_guide import PreparedGuide, GuideConfig, blend_guide_noise
from core.models.ming_image_01.style_generation import PreparedStyle
from core.inference.reference_style import StyleTransferConfig
from ming_image_01_rewiring_test import config, spec_for, restore_topology
from ming_image_01_vae_alignment_test import declaration, initialize
from ming_image_01_sampling_test import condition
from ming_image_01_partition_generation_test import plan


def model(profile, dtype=torch.float32, rewired=False):
    value = create_transformer(config("learned" if profile == "layer_decompose" else "zero_masked"), device="cpu", dtype=dtype)
    if profile == "layer_decompose":
        torch.nn.init.constant_(value.x_pad_token, .02)
        torch.nn.init.constant_(value.cap_pad_token, .03)
    alignment.resize_backbone(value, 24)
    declared = dict(declaration(24, ndim=4), noise_policy="gaussian_row_orthogonal_v1")
    pair = initialize(alignment.MingVAEAlignment(declared))
    pair.noise_alignment.capture(pair.input, pair.output)
    value.add_module("vae_latent_bridge", pair.to(dtype=dtype))
    if rewired: restore_topology(value, spec_for(value))
    return value


class Codec(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1)
        from diffusers.configuration_utils import FrozenDict
        self.config = FrozenDict(scaling_factor=1., shift_factor=0.)

    def encode(self, pixels):
        value = torch.full((pixels.shape[0], 24, pixels.shape[-2] // 16, pixels.shape[-1] // 16), .2)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value))

    def decode(self, latent, return_dict=False):
        return (torch.nn.functional.interpolate(latent[:, :4], scale_factor=16),)


def sampler(profile, rewired=False):
    value = model(profile, rewired=rewired).eval()
    vae = alignment.NormalizedFrameVAE(Codec(), value.vae_latent_bridge.declaration)
    artifact = SimpleNamespace(profile=profile, configs={"transformer": dict(value.config), "vae": {"input_channels": 4}})
    return MingSampler(SimpleNamespace(artifact=artifact, transformer=value, vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=6.)))


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rewired", [False, True])
def test_training_projection_oracle_per_item_sigma_and_independent_frames(profile, dtype, rewired):
    value = model(profile, dtype, rewired)
    frames = 3 if profile == "layer_decompose" else 1
    clean = torch.randn(2, 24, frames, 8, 8, dtype=dtype)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([.2, .8])
    refs = [torch.randn(24, 2, 8, 8, dtype=dtype), None]
    cond = MingConditioning(torch.randn(2, 3, 8, dtype=dtype), torch.randn(2, 2, 24, dtype=dtype))
    native = value._native_forward
    observed = []
    def record(x, t, caps, **kwargs):
        result = native(x, t, caps, **kwargs)
        observed.append((x, kwargs["ref_x"], result[0]))
        return result
    with patch.object(value, "_native_forward", side_effect=record):
        result = flow_loss(value, clean=clean, noise=noise, sigma=sigma, conditioning=cond,
            frame_weights=torch.ones(2, frames), reference_latents=refs, cfg_drop_mask=torch.tensor([False, True]))
    pair = value.vae_latent_bridge
    noisy = (clean * (1-sigma[:, None, None, None, None]) + noise * sigma[:, None, None, None, None]).to(dtype)
    expected = pair.flow_state(noisy, sigma, noise=noise)
    torch.testing.assert_close(torch.stack(observed[0][0]), expected)
    torch.testing.assert_close(observed[0][1][0], pair.input(refs[0][None])[0])
    raw = torch.stack([item[:, :frames] for item in observed[0][2]])
    torch.testing.assert_close(result.prediction, pair.flow_velocity(raw, convention="x0_minus_eps", noise=noise))
    result.loss.backward()
    assert pair.input.projection.weight.grad.abs().sum() > 0
    assert pair.output.projection.weight.grad.abs().sum() > 0
    assert alignment._noise_contexts.get() == ()


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
@pytest.mark.parametrize("mode", ["plain", "style", "style_multi", "partition", "guide"])
def test_actual_sampler_cfg_references_and_noise_ownership(profile, mode):
    value = sampler(profile, rewired=True)
    batch = 2
    frames = 3 if profile == "layer_decompose" else 1
    kwargs = dict(task="layer-decompose" if frames > 1 else "image-edit", conditioning=condition(batch),
        width=128, height=128, steps=2, cfg=2., seed=53, requested_layers=2 if frames > 1 else 1,
        output_type="latent", reference_pixels=torch.randn(batch, 4, 128, 128))
    style_noise = torch.randn(batch, 24, 8, 8)
    if mode in {"style", "style_multi"}:
        kwargs.update(styles=[PreparedStyle(torch.randn_like(style_noise), style_noise,
            StyleTransferConfig(block_range=(0, 5), ref_k_strength=.4, adain_strength=.3))], style_combine_mode="stack")
        if mode == "style_multi":
            kwargs["styles"].append(PreparedStyle(torch.randn_like(style_noise), torch.randn_like(style_noise),
                StyleTransferConfig(block_range=(0, 5), ref_k_strength=.3, adain_strength=.2)))
            kwargs["style_combine_mode"] = "common_concept"
    elif mode == "partition": kwargs["partition_plan"] = plan(4, halo=4)
    elif mode == "guide":
        kwargs["reference_guides"] = [PreparedGuide(torch.full_like(style_noise, .3), style_noise,
            GuideConfig(None, .4, 0, 1000))]
    initial = torch.randn((batch*frames, 24, 8, 8), generator=torch.Generator().manual_seed(53))
    initial = torch.stack(initial.chunk(frames, dim=0), dim=2)
    received = []
    original = alignment.forward_with_noise
    def record(owner, noise, *args, **kw):
        received.append(noise.clone())
        return original(owner, noise, *args, **kw)
    with patch("core.models.ming_image_01.sampling.forward_with_noise", side_effect=record), \
         patch("core.models.ming_image_01.style_generation.forward_with_noise", side_effect=record), \
         patch("core.models.ming_image_01.partition_generation.forward_with_noise", side_effect=record):
        result = value.run(**kwargs)
    assert len(result) == batch and all(len(group.values) == frames for group in result)
    assert all(output.shape == (24, 8, 8) for group in result for output in group.values)
    if mode in {"style", "style_multi"}:
        torch.testing.assert_close(received[0], style_noise.unsqueeze(2).expand(-1, -1, frames, -1, -1))
        count = len(kwargs["styles"])
        if count == 2:
            torch.testing.assert_close(received[1], kwargs["styles"][1].noise.unsqueeze(2).expand(-1, -1, frames, -1, -1))
        torch.testing.assert_close(received[count], initial)
        torch.testing.assert_close(received[count+1], initial)
    else:
        torch.testing.assert_close(received[0], initial.repeat(2, 1, 1, 1, 1) if mode == "partition" else initial)
        if mode == "guide":
            exposed = initial.squeeze(2) if frames == 1 else initial
            expected = blend_guide_noise(exposed, kwargs["reference_guides"], index=0, total=2, profile=profile)
            torch.testing.assert_close(received[1], expected.unsqueeze(2) if frames == 1 else expected)
    assert value.pipe.transformer.alignment_noise is None
    assert "prepare_latents" not in value.pipe.__dict__
    assert alignment._noise_contexts.get() == ()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fixed_noise_coefficients_restore_exactly(tmp_path, dtype):
    source = model("generation_edit")
    pair = source.vae_latent_bridge
    save_file(pair.state_dict(), str(tmp_path / alignment.WEIGHTS))
    target = create_transformer(config("zero_masked"), device="cpu", dtype=dtype)
    artifact = SimpleNamespace(root=tmp_path, configs={"transformer": {"vae_alignment": pair.declaration}})
    alignment.restore(target, artifact, dtype)
    for key, value in pair.noise_alignment.state_dict().items():
        restored = target.vae_latent_bridge.noise_alignment.state_dict()[key]
        assert torch.equal(value, restored)
        if key in ("incoming", "outgoing", "white"): assert restored.dtype == torch.float32


def test_refuse_untracked_stochastic_sampling_and_cleanup_after_cancel():
    value = sampler("generation_edit")
    args = dict(task="text-to-image", conditioning=condition(), width=128, height=128,
        steps=2, cfg=1., seed=53, requested_layers=1, output_type="latent")
    value.pipe.scheduler.register_to_config(stochastic_sampling=True)
    with pytest.raises(ValueError, match="deterministic"): value.run(**args)
    value.pipe.scheduler.register_to_config(stochastic_sampling=False)
    stopped = []
    with pytest.raises(MingGenerationCancelled):
        value.run(**args, cancelled=lambda: bool(stopped), step_callback=lambda *a: stopped.append(True))
    assert value.pipe.transformer.alignment_noise is None
    assert "prepare_latents" not in value.pipe.__dict__
    assert alignment._noise_contexts.get() == ()
    assert value.run(**args)[0].values[0].shape == (24, 8, 8)


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
@pytest.mark.parametrize("guided", [False, True])
def test_partition_backward_crops_each_exposure_and_maps_global_context(profile, guided):
    from core.models.ming_image_01.partition_training import backward_regions, training_plan
    from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter
    value = model(profile, rewired=True)
    adapter = MingPartitionGlobalAdapter(24, 24, rank=4, summary_tokens=2)
    torch.nn.init.normal_(adapter.out.weight, std=.02)
    value.add_module("ming_partition_global_adapter", adapter)
    frames = 3 if profile == "layer_decompose" else 1
    clean = torch.randn(2, 24, frames, 16, 16)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([.2, .7])
    settings = dict(dit_partition_training_enabled=True, dit_partition_mode="fixed",
        dit_partition_fixed_count=2, dit_partition_halo_tokens=0,
        dit_partition_split_ratio_min=.5, dit_partition_split_ratio_max=.5,
        qwen_guidance_loss_weight=.4 if guided else 0., qwen_guidance_loss_mix_mode="blend",
        qwen_guidance_loss_scale=4., qwen_guidance_loss_schedule="sigma",
        qwen_guidance_loss_extrapolation_max_rms=.25)
    trainer = SimpleNamespace(transformer=value, device="cpu", training_dtype=torch.float32,
        mixed_precision=False, config=settings, reconstruction_loss_weight=.3,
        _collect_concept_sample_loss=True, log_extra_metric=lambda *a: None)
    batch = SimpleNamespace(clean=clean, frame_weights=torch.ones(2, frames), conditioning=condition(2),
        reference_latents=[torch.randn(24, 1, 16, 16), torch.randn(24, 2, 16, 16)])
    layout = training_plan(trainer, clean)
    observed = []
    handle = adapter.register_forward_pre_hook(lambda m, args: observed.append(args[0].detach().clone()))
    from core.models.ming_image_01 import partition_training
    real = partition_training.forward_with_noise
    exposures = []
    def record(owner, eps, *args, **kwargs):
        exposures.append(eps.clone())
        return real(owner, eps, *args, **kwargs)
    with patch.object(partition_training, "forward_with_noise", side_effect=record):
        losses = backward_regions(trainer, batch, sigma=sigma, noise=noise, plan=layout, backward_scale=1.,
            drop_mask=torch.tensor([False, True]) if guided else None)
    handle.remove()
    assert all(torch.isfinite(torch.as_tensor(loss)) for loss in losses)
    noisy = clean * (1-sigma[:, None, None, None, None]) + noise * sigma[:, None, None, None, None]
    expected = value.vae_latent_bridge.flow_state(noisy, sigma, noise=noise)
    for grid in observed: torch.testing.assert_close(grid, expected[:grid.shape[0]])
    for index, eps in enumerate(exposures):
        box = layout.regions[index % len(layout.regions)].input
        torch.testing.assert_close(eps, noise[:eps.shape[0], ..., box.top*2:box.bottom*2, box.left*2:box.right*2])
    assert adapter.out.weight.grad.abs().sum() > 0
    assert value.vae_latent_bridge.input.projection.weight.grad.abs().sum() > 0
    assert value.vae_latent_bridge.output.projection.weight.grad.abs().sum() > 0
    assert alignment._noise_contexts.get() == ()


@pytest.mark.parametrize("new_channels", [8, 16, 24])
@pytest.mark.parametrize("scale", [8, 16])
def test_fresh_policy_geometry_and_real_codec_calibration(tmp_path, new_channels, scale):
    from diffusers import AutoencoderKL
    from core.training.arch.ming_image_01 import MingImage01ArchHandler
    from PIL import Image
    import numpy as np
    def codec(channels, compression):
        levels = 4 if compression == 8 else 5
        return AutoencoderKL(in_channels=4, out_channels=4, latent_channels=channels,
            block_out_channels=(8,)*levels, down_block_types=("DownEncoderBlock2D",)*levels,
            up_block_types=("UpDecoderBlock2D",)*levels, norm_num_groups=4, layers_per_block=1,
            scaling_factor=.7, shift_factor=.2)
    target = codec(new_channels, scale)
    directory = tmp_path / "target"
    target.save_pretrained(directory)
    value = create_transformer(config("zero_masked"), device="cpu", dtype=torch.float32)
    source_decl = dict(declaration(16, ndim=4), new_scale=8)
    source = alignment.NormalizedFrameVAE(codec(16, 8), source_decl)
    trainer = SimpleNamespace(transformer=value, vae=source,
        config=dict(training_method="full_finetune", vae_swap_source=f"file:{directory}",
            vae_swap_calibration_samples=2, vae_swap_calibration_resolution=128),
        weight_dtype=torch.float32, vae_dtype=torch.float32, device="cpu", _check_stop_requested=lambda: None,
        ming_image_01_components=SimpleNamespace(transformer=value, vae=source, conditioner=torch.nn.Linear(2, 2)))
    trainer.arch = MingImage01ArchHandler(trainer)
    alignment.configure_swap(trainer)
    pair = value.vae_latent_bridge
    assert alignment.uses_gaussian(value) == (scale > 8 and new_channels >= 16)
    if new_channels == 24 and scale == 16:
        paths = []
        for index in range(2):
            path = tmp_path / f"synthetic{index}.png"
            Image.fromarray(np.random.default_rng(index).integers(0, 256, (128, 128, 4), dtype=np.uint8)).save(path)
            paths.append(dict(image_path=str(path)))
        alignment.calibrate(trainer, [SimpleNamespace(items=paths)])
        assert pair.ready and not hasattr(trainer, "_ming_vae_calibration_source")
        white = pair.noise_alignment.white.flatten(1)
        torch.testing.assert_close(white @ white.T, torch.eye(16), atol=2e-6, rtol=0)
