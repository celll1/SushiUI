"""Gaussian codec boundaries against independent time-dependent paths."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from core.models.components.image_noise_context import noise_scope, exposure_noise
from core.models.components.latent_adapter import SpatialAlignmentPair
from core.models.sensenova_sdxl_chimera.vae_alignment import map_state, map_velocity


def _pair():
    torch.manual_seed(193)
    pair = SpatialAlignmentPair(dict(
        version=1, method="paired_spatial", token_policy="target_native",
        old_channels=4, new_channels=6, old_scale=8, new_scale=16,
        source="synthetic", ndim=4, norm="identity", norm_pack=1, input_channels=3,
        noise_policy="gaussian_row_orthogonal_v1",
    ))
    for bridge in (pair.input, pair.output):
        bridge.initialize(torch.randn(bridge.source_channels*9, bridge.target_channels)/10,
                          torch.randn(bridge.target_channels)/10)
    pair.noise_alignment.capture(pair.input, pair.output)
    return pair


def _coefficients(t, kind):
    t = t[:, None, None, None]
    if kind == "flow_velocity":
        return t, 1-t, torch.ones_like(t), -torch.ones_like(t)
    return t.square()*(2-t), (1-t).square()*(1+t), 4*t-3*t.square(), -1-2*t+3*t.square()


@pytest.mark.parametrize("kind", ["flow_velocity", "endpoint_observable_velocity", "endpoint_observable_residual"])
@pytest.mark.parametrize("cfg_repeat", [1, 2])
def test_state_and_velocity_follow_noise_corrected_path(kind, cfg_repeat):
    pair = _pair()
    owner = SimpleNamespace(vae_latent_bridge=pair)
    noise = torch.randn(2, 6, 8, 12)
    n = noise.repeat(cfg_repeat, 1, 1, 1)
    clean = torch.randn_like(n)
    t = torch.tensor([.2, .8]).repeat(cfg_repeat)
    a, s, ap, sp = _coefficients(t, kind)
    white = F.conv2d(n, pair.noise_alignment.white)
    incoming = F.conv2d(n, pair.noise_alignment.incoming, padding=1)
    expected = F.conv2d(a*clean+s*n, pair.input.projection.weight, padding=1)
    expected += a*pair.input.projection.bias[None, :, None, None]
    expected[:, :4] += s*(white-incoming)
    source_clean = torch.randn_like(n)
    source_noise = torch.zeros_like(n)
    source_noise[:, :4] = white
    native_velocity = ap*source_clean+sp*source_noise
    residual = n-F.conv2d(white, pair.noise_alignment.outgoing, padding=1)

    def output_path(time):
        alpha, sigma, _, _ = _coefficients(time, kind)
        return (F.conv2d(alpha*source_clean+sigma*source_noise,
                        pair.output.projection.weight, padding=1)
                + alpha*pair.output.projection.bias[None, :, None, None]+sigma*residual)

    with noise_scope(noise):
        actual = map_state(owner, a*clean+s*n, t, prediction={"type": kind})
        velocity = map_velocity(owner, native_velocity, t, prediction={"type": kind})
    torch.testing.assert_close(actual, expected)
    numerical = (output_path(t+.001)-output_path(t-.001))/.002
    torch.testing.assert_close(velocity, numerical, atol=3e-4, rtol=3e-4)
    velocity.square().mean().backward()
    assert pair.output.projection.weight.grad.abs().sum() > 0
    assert pair.output.projection.bias.grad.abs().sum() > 0


def test_missing_exposure_refuses_and_nested_error_restores_context():
    pair = _pair()
    owner = SimpleNamespace(vae_latent_bridge=pair)
    value = torch.randn(2, 6, 8, 12)
    with pytest.raises(ValueError, match="exposure"):
        map_state(owner, value, .5, prediction={"type": "flow_velocity"})
    noise = torch.randn_like(value)
    with noise_scope(noise):
        with pytest.raises(RuntimeError):
            with noise_scope(torch.zeros_like(noise)):
                raise RuntimeError("cancel")
        torch.testing.assert_close(exposure_noise(value), noise)
    with pytest.raises(ValueError, match="exposure"):
        exposure_noise(value)


def test_frozen_noise_restores_and_does_not_follow_bridge_updates():
    pair = _pair()
    saved = {key: value.clone() for key, value in pair.state_dict().items()}
    restored = SpatialAlignmentPair(pair.declaration)
    restored.load_state_dict(saved)
    assert restored.ready
    noise = torch.randn(2, 6, 8, 12)
    expected = restored.correct_diffusion_velocity(noise, noise_derivative=torch.tensor([-1., 0.]), noise=noise)
    with torch.no_grad():
        restored.output.projection.weight.add_(.1)
    torch.testing.assert_close(restored.correct_diffusion_velocity(
        noise, noise_derivative=torch.tensor([-1., 0.]), noise=noise), expected, rtol=0, atol=0)
    for key, value in restored.noise_alignment.state_dict().items():
        torch.testing.assert_close(value, saved['noise_alignment.'+key], rtol=0, atol=0)
