"""Spatial bridge semantics; CPU only, no dataset image reads."""

import io

import pytest
import torch

from core.experiments.vae_alignment import SpatialMap, fit_spatial
from core.models.components.vae_alignment import SpatialLatentBridge, SpatialBridgeCalibrator


def bridge(source=3, target=5, source_pack=1, target_pack=1, dtype=torch.float32):
    g = torch.Generator().manual_seed(142)
    weight = torch.randn(source * source_pack**2 * 9, target * target_pack**2,
                         generator=g, dtype=torch.float64) / 9
    bias = torch.randn(target * target_pack**2, generator=g, dtype=torch.float64)
    module = SpatialLatentBridge(source, target, source_pack=source_pack,
                                 target_pack=target_pack).to(dtype=dtype)
    return module.initialize(weight, bias), SpatialMap(weight, bias)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packs", [(1, 1), (2, 1), (1, 2)])
def test_matches_unfold_oracle_and_has_gradients(dtype, packs):
    module, oracle = bridge(source_pack=packs[0], target_pack=packs[1], dtype=dtype)
    latent = torch.randn(2, 3, 8, 12, dtype=dtype, requires_grad=True)
    expected = torch.nn.functional.pixel_shuffle(
        oracle(torch.nn.functional.pixel_unshuffle(latent.detach().float(), packs[0])), packs[1])
    actual = module(latent)
    torch.testing.assert_close(actual.float(), expected, atol=.045 if dtype == torch.bfloat16 else 1e-6,
                               rtol=.045 if dtype == torch.bfloat16 else 1e-5)
    actual.float().square().mean().backward()
    assert torch.isfinite(latent.grad).all()
    assert module.projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("convention,sign", [("eps_minus_x0", 1), ("x0_minus_eps", -1)])
def test_velocity_derivative_and_noise_endpoint(convention, sign):
    module, _ = bridge(source_pack=2)
    clean, noise = torch.randn(2, 3, 8, 12), torch.randn(2, 3, 8, 12)
    sigma = torch.tensor([.2, .7])
    expand = sigma[:, None, None, None]
    state = (1-expand)*clean + expand*noise
    mapped = module.flow_state(state, sigma)
    expected = (1-expand)*module(clean) + expand*module.epsilon(noise)
    torch.testing.assert_close(mapped, expected)
    velocity = sign*(noise-clean)
    torch.testing.assert_close(module.flow_velocity(velocity, convention=convention),
                               sign*(module.epsilon(noise)-module(clean)))
    torch.testing.assert_close(module.flow_state(noise, 1), module.epsilon(noise))


def test_frame_independence_and_state_round_trip():
    module, _ = bridge(3, 4, 2, 1)
    latent = torch.randn(2, 3, 3, 8, 12)
    result = module(latent)
    for i in range(3):
        torch.testing.assert_close(result[:, :, i], module(latent[:, :, i]))
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    buffer.seek(0)
    restored = SpatialLatentBridge(**module.geometry())
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(latent), result)
    # These projections have equal tensor widths but different pixel geometry.
    wrong = SpatialLatentBridge(12, 4)
    with pytest.raises(ValueError, match="geometry/version"):
        wrong.load_state_dict(module.state_dict())


def test_uncalibrated_and_incompatible_geometry_fail():
    module = SpatialLatentBridge(3, 4, source_pack=2)
    with pytest.raises(RuntimeError, match="no paired calibration"):
        module(torch.randn(1, 3, 8, 8))
    with pytest.raises(ValueError, match="shape"):
        module.initialize(torch.zeros(3, 4), torch.zeros(4))
    module.initialize(torch.zeros(108, 4), torch.zeros(4))
    with pytest.raises(ValueError, match="divisible"):
        module(torch.randn(1, 3, 7, 8))
    with pytest.raises(ValueError, match="Sigma"):
        module.flow_state(torch.randn(1, 3, 8, 8), torch.ones(2))


def test_streaming_calibration_matches_whole_dataset_fit():
    source, target = torch.randn(4, 3, 8, 12), torch.randn(4, 5, 8, 12)
    module = SpatialLatentBridge(3, 5)
    calibrator = SpatialBridgeCalibrator(module, ridge=1e-2)
    for i in range(4):
        calibrator.add(source[i:i+1], target[i:i+1])
    calibrator.finish()
    oracle = fit_spatial(source, target)
    unseen = torch.randn(2, 3, 6, 10)
    torch.testing.assert_close(module(unseen), oracle(unseen), atol=1e-6, rtol=1e-5)
    assert not any(isinstance(v, list) for v in vars(calibrator).values())


def test_streaming_packed_frame_calibration():
    module = SpatialLatentBridge(2, 3, source_pack=2)
    calibrator = SpatialBridgeCalibrator(module, ridge=1e-2)
    source, target = torch.randn(2, 2, 3, 8, 12), torch.randn(2, 3, 3, 4, 6)
    calibrator.add(source, target)
    assert calibrator.finish()(source).shape == target.shape
    with pytest.raises(ValueError, match="grids"):
        calibrator.add(source, target[..., :3, :])
    with pytest.raises(ValueError, match="nonfinite"):
        calibrator.add(source * float('nan'), target)


def test_empty_calibration_does_not_initialize():
    module = SpatialLatentBridge(2, 3)
    with pytest.raises(ValueError, match="insufficient"):
        SpatialBridgeCalibrator(module, ridge=1e-2).finish()
    assert not module._ready
