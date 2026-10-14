"""Wide VAE swaps must not feed the exposure noise to the backbone input (VAE_SWAP_NOISE_LEAK.md)."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.models.components.gaussian_alignment import POLICY
from core.models.components.latent_adapter import (
    NOISE_ESTIMATE, backbone_width, estimated_noise, leaks_exposure_noise, validate_declaration)


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


def test_estimated_noise_is_the_standard_normal_posterior_mean():
    torch.manual_seed(0)
    x0, eps = torch.randn(20000), torch.randn(20000)
    for s in (.3, .7, .95):
        x_t = (1 - s) * x0 + s * eps
        estimate = estimated_noise(x_t[None], torch.tensor([s]))[0]
        slope = (eps * x_t).mean() / x_t.square().mean()
        assert float(estimate[0] / x_t[0]) == pytest.approx(float(slope), rel=.05)
