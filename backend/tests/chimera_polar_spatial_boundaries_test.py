"""Polar transfers preserve Cartesian derivatives, including v4 endpoints."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from core.models.components.latent_adapter import SpatialAlignmentPair
from core.models.sensenova_sdxl_chimera.flow import (
    FLOW_V3_PREDICTION, FLOW_V4_PREDICTION, polar_state,
)
from core.models.sensenova_sdxl_chimera.polar_vae_alignment import transfer_prediction


def _pair(channels, dtype=torch.float64):
    pair = SpatialAlignmentPair(dict(
        version=1, method="paired_spatial", token_policy="target_native",
        old_channels=4, new_channels=channels, old_scale=8, new_scale=16,
        source="synthetic", ndim=4, norm="identity", norm_pack=1, input_channels=3,
    )).to(dtype=dtype)
    rng = torch.Generator().manual_seed(315)
    for bridge in (pair.input, pair.output):
        bridge.initialize(torch.randn(bridge.source_channels*9, bridge.target_channels,
            generator=rng, dtype=dtype)/10,
            torch.randn(bridge.target_channels, generator=rng, dtype=dtype)/10)
    return pair


@pytest.mark.parametrize("channels", [2, 6])
@pytest.mark.parametrize("kind", [FLOW_V3_PREDICTION, FLOW_V4_PREDICTION])
@pytest.mark.parametrize("scalar", [False, True])
def test_transfer_matches_independent_cartesian_oracle(channels, kind, scalar):
    torch.manual_seed(31)
    pair = _pair(channels)
    owner = SimpleNamespace(vae_latent_bridge=pair)
    target = torch.randn(2, channels, 8, 12, dtype=torch.float64)
    time = torch.tensor(.6 if scalar else [0., 1.], dtype=torch.float64)
    t = time.expand(2).reshape(2, 1, 1, 1)
    alpha, derivative = t.square()*(2-t), 4*t-3*t.square()
    incoming = pair.input.projection
    source = F.conv2d(target, incoming.weight, padding=1) + alpha*incoming.bias[None, :, None, None]
    source_mean = torch.linspace(-.2, .3, source.shape[1], dtype=torch.float64)
    target_mean = torch.linspace(.4, -.1, channels, dtype=torch.float64)
    source_contract = dict(type=kind, latent_mean=source_mean.tolist(), radius_floor=1e-8)
    target_contract = dict(type=kind, latent_mean=target_mean.tolist(), radius_floor=1e-8)
    direction = polar_state(source, time, latent_mean=source_mean)[2]
    raw = torch.randn_like(source, requires_grad=True)
    radial = torch.randn(2, dtype=torch.float64, requires_grad=True)
    projected = raw-direction*(raw*direction).flatten(1).mean(1)[:, None, None, None]
    coordinate_derivative = 12*t.square()*(1-t) if kind == FLOW_V4_PREDICTION else torch.ones_like(t)
    source_velocity = derivative*source_mean[None, :, None, None] + radial[:, None, None, None]*direction + coordinate_derivative*projected
    outgoing = pair.output.projection
    expected = F.conv2d(source_velocity, outgoing.weight, padding=1) + derivative*outgoing.bias[None, :, None, None]
    result = transfer_prediction(owner, source, radial, raw, target, time,
        source_prediction=source_contract, target_prediction=target_contract)
    torch.testing.assert_close(result.full_velocity, expected)
    reconstructed = derivative*target_mean[None, :, None, None] + result.radial_velocity[:, None, None, None]*result.direction + result.tangent_per_time
    torch.testing.assert_close(reconstructed, expected)
    torch.testing.assert_close(result.tangent_per_time,
        result.drift_tangent_per_time + coordinate_derivative*result.tangent_per_coordinate)
    torch.testing.assert_close((result.tangent_per_time*result.direction).flatten(1).mean(1),
        torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
    assert result.full_velocity.shape == target.shape
    if kind == FLOW_V4_PREDICTION and not scalar:
        assert result.drift_tangent_per_time.abs().sum() > 0
        assert torch.isfinite(result.tangent_per_coordinate).all()
    result.full_velocity.square().mean().backward()
    for parameter in (radial, pair.input.projection.weight, pair.output.projection.weight,
                      pair.output.projection.bias):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    if scalar or kind == FLOW_V3_PREDICTION:
        assert raw.grad.abs().sum() > 0


@pytest.mark.parametrize("channels", [2, 6])
@pytest.mark.parametrize("kind", [FLOW_V3_PREDICTION, FLOW_V4_PREDICTION])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_radial_head_and_both_bridges_train_on_target_grid(channels, kind, dtype):
    from diffusers import UNet2DConditionModel
    from core.models.sensenova_sdxl_chimera.unet import install_polar_radial_head
    from core.models.sensenova_sdxl_chimera.polar_vae_alignment import forward_prediction
    torch.manual_seed(32)
    pair = _pair(channels, torch.float32)
    width = max(4, channels)
    unet = UNet2DConditionModel(in_channels=width, out_channels=width,
        block_out_channels=(8,), down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",), layers_per_block=1,
        cross_attention_dim=6, attention_head_dim=1, norm_num_groups=4)
    install_polar_radial_head(unet)
    unet.vae_latent_bridge = pair
    unet.to(dtype=dtype)
    unet._chimera_source_prediction = dict(type=kind, latent_mean=[.1]*width, radius_floor=1e-8)
    target = torch.randn(2, channels, 8, 12, dtype=dtype)
    prediction = dict(type=kind, latent_mean=[-.2]*channels, radius_floor=1e-8)
    observed = []
    hook = unet.conv_in.register_forward_pre_hook(lambda module, args: observed.append(args[0].shape))
    try:
        result = forward_prediction(unet, target, torch.tensor([.2, .8]), prediction=prediction,
            encoder_hidden_states=torch.randn(2, 3, 6, dtype=dtype))
    finally:
        hook.remove()
    assert observed == [torch.Size([2, width, 8, 12])]
    assert result.full_velocity.shape == target.shape
    result.full_velocity.square().mean().backward()
    for bridge in (pair.input, pair.output):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bridge.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in unet.chimera_polar_radial_head.parameters())
