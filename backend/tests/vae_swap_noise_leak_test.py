"""Wide VAE swaps must not feed the exposure noise to the backbone input (VAE_SWAP_NOISE_LEAK.md)."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.models.components.gaussian_alignment import POLICY
from core.models.components.latent_adapter import (
    NOISE_ESTIMATE, SpatialAlignmentPair, backbone_width, leaks_exposure_noise, validate_declaration)
from core.models.components.noise_prior import RadialNoisePrior


def _declaration(old, new, *, noise, passthrough=None, estimate=False):
    value = dict(version=1, method="paired_spatial", token_policy="target_native", old_channels=old,
                 new_channels=new, old_scale=8, new_scale=16, input_channels=3, norm_pack=1, ndim=4,
                 source="test", norm="per_channel")
    if noise:
        value["noise_policy"] = POLICY
    if estimate:
        value["noise_estimate"] = NOISE_ESTIMATE
    if passthrough:
        value["passthrough"] = passthrough
    return dict(validate_declaration(value))


@pytest.mark.parametrize("old,new,passthrough,leaks", [
    (16, 64, "full", True), (16, 64, None, True), (16, 16, None, False)])
def test_exposure_noise_leaks_only_beside_a_second_view_of_x_t(old, new, passthrough, leaks):
    declaration = _declaration(old, new, noise=True, passthrough=passthrough)
    assert backbone_width(declaration) > old or not leaks
    assert leaks_exposure_noise(declaration) is leaks
    assert not leaks_exposure_noise(_declaration(old, new, noise=False, passthrough=passthrough))
    if new > old:
        assert not leaks_exposure_noise(_declaration(old, new, noise=True, passthrough=passthrough, estimate=True))


def test_noise_estimate_requires_the_gaussian_policy():
    with pytest.raises(ValueError, match="noise estimate"):
        _declaration(16, 64, noise=False, passthrough="full", estimate=True)


def _smooth_latents(count, channels=4, side=32):
    """Latents with a falling spectrum, like a VAE's, and a per-channel offset."""
    raw = torch.randn(count, channels, side, side)
    smooth = torch.nn.functional.avg_pool2d(raw, 5, stride=1, padding=2) * 3
    return smooth + torch.arange(channels).view(1, -1, 1, 1) * .5


def test_wiener_prior_beats_the_flat_prior_and_sees_only_x_t():
    torch.manual_seed(0)
    prior = RadialNoisePrior(4)
    prior.capture([_smooth_latents(64)])
    x0 = _smooth_latents(32)
    for s in (.3, .5, .7):
        eps = torch.randn_like(x0)
        x_t = (1 - s) * x0 + s * eps
        sigma = torch.full((32,), s)
        estimate = prior.estimate(x_t, sigma)
        flat = s / ((1 - s) ** 2 + s ** 2) * x_t
        assert (estimate - eps).square().mean() < .6 * (flat - eps).square().mean()
    assert torch.equal(prior.estimate(x_t, sigma), estimate)


def test_noise_prior_survives_a_state_dict_round_trip_and_needs_calibration():
    with pytest.raises(RuntimeError):
        RadialNoisePrior(4).estimate(torch.zeros(1, 4, 8, 8), torch.tensor([.5]))
    prior = RadialNoisePrior(4)
    prior.capture([_smooth_latents(8, side=16), _smooth_latents(8, side=24)])
    restored = RadialNoisePrior(4)
    restored.load_state_dict(prior.state_dict())
    x_t = torch.randn(2, 4, 1, 20, 20)
    torch.testing.assert_close(restored.estimate(x_t, torch.tensor([.4, .8])), prior.estimate(x_t, torch.tensor([.4, .8])))


def test_a_wide_pair_with_an_estimate_owns_a_noise_prior():
    pair = SpatialAlignmentPair(_declaration(16, 64, noise=True, passthrough="full", estimate=True))
    assert isinstance(pair.noise_prior, RadialNoisePrior)
