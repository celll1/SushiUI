import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from core.models.components.patch_noise_alignment import POLICY, PatchNoiseAlignment


def fixture(hidden=7, channels=4):
    torch.manual_seed(71)
    declaration = dict(policy=POLICY, channels=channels, bridge_channels=5,
                       hidden_channels=hidden, patch_size=2)
    module = PatchNoiseAlignment(declaration)
    bridge = nn.Conv2d(channels, 5, 3, padding=1)
    projection = nn.Conv2d(5, hidden, 2, stride=2)
    native = torch.randn(hidden, 3, 4, 4)
    module.capture(native, bridge, projection)
    return module, native, bridge, projection


def test_native_patch_covariance_is_preserved():
    module, native, _, _ = fixture()
    matrix = native.flatten(1)
    white = module.white.flatten(1)
    torch.testing.assert_close(white @ white.T, matrix @ matrix.T, atol=1e-5, rtol=1e-6)
    assert int(module.rank) == 7
    assert float(module.covariance_error) < 1e-10


def test_noisy_projection_matches_native_covariance_map():
    module, _, _, _ = fixture()
    noise = torch.randn(2, 4, 8, 12) * 2
    current = F.conv2d(F.conv2d(noise, module.incoming, padding=1), module.projection, stride=2)
    actual = module.correct(current, noise, torch.ones(2))
    desired = F.conv2d(noise, module.white, stride=2)
    torch.testing.assert_close(actual, desired)


def test_clean_boundary_and_gradient_are_unchanged():
    module, _, _, _ = fixture()
    output = torch.randn(2, 7, 4, 6, requires_grad=True)
    noise = torch.randn(2, 4, 8, 12, requires_grad=True)
    clean = module.correct(output, noise, torch.zeros(2))
    assert torch.equal(clean, output)
    clean.sum().backward()
    assert torch.equal(output.grad, torch.ones_like(output))
    assert noise.grad is None
    assert list(module.parameters()) == []


def test_restoration_keeps_coefficients_fp32_and_prediction_exact():
    module, _, _, _ = fixture()
    restored = PatchNoiseAlignment(module.declaration)
    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    restored.to(dtype=torch.bfloat16)
    assert restored.white.dtype == restored.incoming.dtype == restored.projection.dtype == torch.float32
    noise = torch.randn(2, 4, 8, 12)
    output = torch.randn(2, 7, 4, 6).bfloat16()
    sigma = torch.tensor([0.2, 0.8])
    assert torch.equal(module.correct(output, noise, sigma), restored.correct(output, noise, sigma))


def test_rank_limited_target_reports_covariance_loss():
    module, _, _, _ = fixture(hidden=7, channels=1)
    assert int(module.rank) == 4
    assert float(module.covariance_error) > 0


def test_capture_does_not_consume_global_rng():
    module, native, bridge, projection = fixture()
    state = torch.get_rng_state().clone()
    module.capture(native, bridge, projection)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("corruption", ["identity", "nonfinite"])
def test_corrupt_checkpoint_is_rejected(corruption):
    module, _, _, _ = fixture()
    state = {key: value.clone() for key, value in module.state_dict().items()}
    if corruption == "identity":
        state["identity"][0] ^= 1
    else:
        state["white"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        PatchNoiseAlignment(module.declaration).load_state_dict(state)


def test_mismatched_exposure_grid_is_rejected():
    module, _, _, _ = fixture()
    with pytest.raises(ValueError, match="divisible BCHW"):
        module.correct(torch.zeros(2, 7, 4, 6), torch.zeros(2, 4, 9, 12), torch.ones(2))
    with pytest.raises(ValueError, match="per image"):
        module.correct(torch.zeros(2, 7, 4, 6), torch.zeros(2, 4, 8, 12), torch.ones(1))
