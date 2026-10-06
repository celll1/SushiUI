"""Independent mixed-coordinate updates and the public trainer's drift loss."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from core.models.sensenova_sdxl_chimera.flow import (
    FLOW_V4_PREDICTION, mixed_coordinate_polar_exp_euler_step,
    polar_state, polar_tangent_projection, destruction_coordinate_recover_clean,
)


@pytest.mark.parametrize("start,end", [(0., .2), (.3, .6), (1., .8), (.8, .2), (.4, .4)])
@pytest.mark.parametrize("limit", [None, .07])
def test_mixed_drift_step_matches_spherical_displacement_oracle(start, end, limit):
    torch.manual_seed(711)
    sample = torch.randn(2, 3, 6, 8, dtype=torch.float64)
    mean = torch.tensor([.2, -.3, .1], dtype=torch.float64)[None, :, None, None]
    centered = sample-start**2*(2-start)*mean
    radius = centered.square().flatten(1).mean(1).sqrt()[:, None, None, None]
    direction = centered/radius
    angular = polar_tangent_projection(torch.randn_like(sample), direction)
    drift = polar_tangent_projection(torch.randn_like(sample), direction)
    radial = torch.tensor([-.3, .2], dtype=torch.float64)
    d = lambda t: 4*t**3-3*t**4
    displacement = ((end-start)*drift+(d(end)-d(start))*angular)/radius
    angle = displacement.square().flatten(1).mean(1).sqrt()[:, None, None, None]
    scale = torch.ones_like(angle) if limit is None else (limit/angle.clamp_min(1e-8)).clamp(max=1.)
    updated = torch.cos(angle*scale)*direction + torch.sin(angle*scale)*displacement/angle.clamp_min(1e-8)
    updated = torch.where((angle > 1e-8).expand_as(updated), updated, direction)
    updated /= updated.square().flatten(1).mean(1).sqrt()[:, None, None, None]
    new_radius = radius*torch.exp((end-start)*radial[:, None, None, None]/radius)
    expected = end**2*(2-end)*mean + new_radius*updated
    result = mixed_coordinate_polar_exp_euler_step(sample, radial, angular, start, end,
        latent_mean=mean.flatten().tolist(), tangent_drift=drift, angular_step_limit=limit)
    torch.testing.assert_close(result.sample, expected, atol=1e-13, rtol=1e-13)
    assert torch.isfinite(result.sample).all()


def test_drift_clean_estimate_keeps_endpoints_and_matches_interior_effective_tangent():
    torch.manual_seed(712)
    sample = torch.randn(2, 3, 6, 8, dtype=torch.float64)
    radial = torch.tensor([.2, -.4], dtype=torch.float64)
    tangent, drift = torch.randn_like(sample), torch.randn_like(sample)
    for t in (0., .3, 1.):
        kwargs = dict(latent_mean=[.1, -.2, .3], latent_centered_second_moment=.7)
        actual = destruction_coordinate_recover_clean(sample, radial, tangent, t,
            tangent_drift=drift, **kwargs)
        assert torch.isfinite(actual).all()
        if t == 1.:
            torch.testing.assert_close(actual, sample)
        else:
            coordinate = 4*t**3-3*t**4
            effective = tangent + (1-t)/(1-coordinate)*drift
            expected = destruction_coordinate_recover_clean(sample, radial, effective, t, **kwargs)
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mode", ["none", "batched", "sequential"])
@pytest.mark.parametrize("normalization", ["none", "global"])
def test_cfg_guides_both_tangent_fields_and_anchors_radial_to_positive(mode, normalization):
    from core.models.sensenova_sdxl_chimera.pipeline_ops import ChimeraConditioning, _polar_cfg_step
    from core.models.sensenova_sdxl_chimera.polar_vae_alignment import PolarInferencePrediction
    torch.manual_seed(713)
    sample = torch.randn(1, 2, 4, 6)
    t, following = torch.tensor(.3), torch.tensor(.6)
    contract = dict(type=FLOW_V4_PREDICTION, latent_mean=[.1, -.2], radius_floor=1e-8,
        latent_centered_second_moment=.8)
    _, radius, direction = polar_state(sample, t, latent_mean=contract["latent_mean"])
    angular = [polar_tangent_projection(torch.randn_like(sample), direction) for _ in range(2)]
    drift = [polar_tangent_projection(torch.randn_like(sample), direction) for _ in range(2)]
    radial = [torch.tensor([-.2]), torch.tensor([.4])]
    def conditioning(name):
        return ChimeraConditioning(encoder_hidden_states=torch.ones(1, 3, 6),
            pooled_text_embeds=torch.ones(1, 5), context_positions=torch.zeros(1, 3, 3),
            attention_mask=torch.ones(1, 3, dtype=torch.bool), fingerprint=name, key_lengths=(3,))
    def predict(unet, x, time, cond, time_ids, **kwargs):
        indices = [0, 1] if cond.fingerprint == "negative:positive" else [int(cond.fingerprint == "positive")]
        return PolarInferencePrediction(torch.cat([radial[i] for i in indices]),
            torch.cat([angular[i] for i in indices]), direction.repeat(len(indices), 1, 1, 1),
            radius.repeat(len(indices)), torch.cat([drift[i] for i in indices]))
    needs_cfg = mode != "none"
    guided_angular = angular[0]+3*(angular[1]-angular[0]) if needs_cfg else angular[1]
    guided_drift = drift[0]+3*(drift[1]-drift[0]) if needs_cfg else drift[1]
    if needs_cfg and normalization == "global":
        positive_norm = torch.cat((angular[1], drift[1]), dim=1).square().sum().sqrt()
        guided_norm = torch.cat((guided_angular, guided_drift), dim=1).square().sum().sqrt()
        factor = (positive_norm/guided_norm.clamp_min(1e-8)).clamp(max=1.)
        guided_angular, guided_drift = guided_angular*factor, guided_drift*factor
    expected = mixed_coordinate_polar_exp_euler_step(sample, radial[1], guided_angular,
        t, following, latent_mean=contract["latent_mean"], tangent_drift=guided_drift).sample
    with patch("core.models.sensenova_sdxl_chimera.pipeline_ops._unet_polar", side_effect=predict):
        result, probe = _polar_cfg_step(SimpleNamespace(vae_latent_bridge=object()),
            conditioning("positive"), conditioning("negative"), sample=sample, timestep=t,
            next_timestep=following, time_ids=torch.zeros(1, 6), cache_metadata=(32, 48, 0, 0, 1.),
            prediction=contract, needs_cfg=needs_cfg, cfg_mode=mode, cfg_scale=3., cfg_norm=normalization,
            collect_probe=True)
    torch.testing.assert_close(result, expected)
    prime = 12*t.square()*(1-t)
    mean_prime = (4*t-3*t.square())*torch.tensor(contract["latent_mean"])[None, :, None, None]
    full = mean_prime + radial[1][:, None, None, None]*direction + prime*guided_angular+guided_drift
    torch.testing.assert_close(probe["guided_post"], full)
    assert all(torch.isfinite(x).all() for x in probe["x0_estimates"])


@pytest.mark.parametrize("channels", [2, 6])
def test_actual_train_step_supervises_coordinate_tangent_radial_and_boundary_drift(channels):
    from diffusers import UNet2DConditionModel
    from chimera_polar_spatial_boundaries_test import _pair
    from core.models.sensenova_sdxl_chimera.artifact import prediction_contract
    from core.models.sensenova_sdxl_chimera.unet import install_polar_radial_head
    from core.models.sensenova_sdxl_chimera.polar_vae_alignment import forward_prediction
    from core.models.sensenova_sdxl_chimera.flow import destruction_coordinate_polar_flow_target, gate_conditioning_tensor
    from core.training.ops.sensenova_sdxl_chimera_ops import train_step
    torch.manual_seed(714)
    width = max(4, channels)
    model = UNet2DConditionModel(in_channels=width, out_channels=width, block_out_channels=(8,),
        down_block_types=("CrossAttnDownBlock2D",), up_block_types=("CrossAttnUpBlock2D",),
        cross_attention_dim=6, layers_per_block=1, norm_num_groups=4, attention_head_dim=1)
    install_polar_radial_head(model)
    model.vae_latent_bridge = _pair(channels, torch.float32)
    model._chimera_source_prediction = prediction_contract(FLOW_V4_PREDICTION,
        latent_channels=width, latent_mean=[.1]*width, latent_centered_second_moment=.8)
    contract = prediction_contract(FLOW_V4_PREDICTION, latent_channels=channels,
        latent_mean=[-.2]*channels, latent_centered_second_moment=.7)
    logs = {}
    trainer = SimpleNamespace(config=dict(chimera_training_stage="unet"), device="cpu",
        training_dtype=torch.float32, unet=model, chimera_prediction=contract,
        repa_enable=False, log_extra_metric=lambda key, value: logs.update({key: value}))
    latents = torch.randn(2, channels, 8, 12)
    noise = torch.randn_like(latents)
    ctx = SimpleNamespace(latents=latents, text_embeddings=torch.randn(2, 3, 6),
        timesteps=torch.tensor([.2, .8]), time_ids=None,
        attention_mask=dict(pooled_text_embeds=torch.randn(2, 5), context_positions=torch.zeros(2, 3, 3),
            context_attention_mask=torch.ones(2, 3, dtype=torch.bool)))
    target = destruction_coordinate_polar_flow_target(latents, noise, ctx.timesteps,
        latent_mean=contract["latent_mean"])
    predicted = forward_prediction(model, target.sample, ctx.timesteps, prediction=contract,
        encoder_hidden_states=gate_conditioning_tensor(ctx.text_embeddings, ctx.timesteps, latents),
        encoder_attention_mask=ctx.attention_mask["context_attention_mask"],
        added_cond_kwargs=dict(text_embeds=gate_conditioning_tensor(ctx.attention_mask["pooled_text_embeds"],
            ctx.timesteps, latents), time_ids=torch.tensor([[128., 192., 0., 0., 128., 192.]]).repeat(2, 1)))
    expected = (predicted.tangent_per_coordinate-target.tangent_velocity).square().mean()
    expected += (predicted.radial_velocity-target.radial_velocity).square().mean()
    expected += predicted.drift_tangent_per_time.square().mean()
    with patch("core.training.mnt.training_noise_like", return_value=noise):
        loss, value, recon = train_step(trainer, ctx)
    torch.testing.assert_close(loss, expected)
    assert value > 0 and recon >= 0 and logs["chimera_v4_boundary_drift_loss"] > 0
    loss.backward()
    for bridge in (model.vae_latent_bridge.input, model.vae_latent_bridge.output):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bridge.parameters())
