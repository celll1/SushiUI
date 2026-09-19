"""Pure geometric contracts for Chimera v3."""

import math

import pytest
import torch

from core.models.sensenova_sdxl_chimera.artifact import (
    FORMAT_VERSION,
    V3_FORMAT_VERSION,
    ChimeraArtifactError,
    prediction_contract,
    validated_prediction_contract,
)
from core.models.sensenova_sdxl_chimera.flow import (
    FLOW_V3_PREDICTION,
    polar_flow_target,
    polar_tangent_projection,
    terminal_flat_angular_schedule,
)


def _rms_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).flatten(1).mean(dim=1)


def _rms_norm(value: torch.Tensor) -> torch.Tensor:
    return _rms_inner(value, value).sqrt()


def test_terminal_flat_family_has_declared_endpoint_derivatives():
    probe = torch.empty(2, 4, 1, 1, dtype=torch.float64)
    endpoints = torch.tensor([0.0, 1.0], dtype=torch.float64)
    for slope in (0.0, 0.75, 2.0):
        gamma, derivative = terminal_flat_angular_schedule(
            endpoints, probe, endpoint_slope=slope
        )
        assert gamma.flatten().tolist() == pytest.approx([0.0, 1.0])
        assert derivative.flatten().tolist() == pytest.approx([slope, 0.0])

    with pytest.raises(ValueError, match="\[0, 2\]"):
        terminal_flat_angular_schedule(endpoints, probe, endpoint_slope=2.1)


def test_polar_path_is_endpoint_exact_and_tangent():
    generator = torch.Generator().manual_seed(13)
    mean = [0.2, -0.1, 0.05, 0.3]
    clean = torch.randn(2, 4, 7, 9, generator=generator, dtype=torch.float64)
    clean = clean + torch.tensor(mean, dtype=clean.dtype).reshape(1, 4, 1, 1)
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    target = polar_flow_target(
        clean,
        noise,
        torch.tensor([0.0, 1.0], dtype=clean.dtype),
        latent_mean=mean,
    )

    assert torch.allclose(target.sample[0], noise[0], atol=1e-12, rtol=1e-12)
    assert torch.allclose(target.sample[1], clean[1], atol=1e-12, rtol=1e-12)
    expected_noise_tangent = target.full_velocity[0] + noise[0]
    assert torch.allclose(
        target.tangent_velocity[0], expected_noise_tangent, atol=1e-12, rtol=1e-12
    )
    assert torch.allclose(
        target.full_velocity[1], clean[1], atol=1e-12, rtol=1e-12
    )
    assert target.radial_velocity.tolist() == pytest.approx(
        [-target.radius[0].item(), target.radius[1].item()]
    )
    assert torch.allclose(_rms_norm(target.direction), torch.ones(2, dtype=clean.dtype))
    assert torch.allclose(
        _rms_inner(target.direction, target.tangent_velocity),
        torch.zeros(2, dtype=clean.dtype),
        atol=1e-12,
    )


def test_noise_end_target_energy_matches_orthogonal_reference():
    width = 64
    noise = torch.ones(1, 1, 1, width, dtype=torch.float64)
    clean = torch.cat(
        (torch.ones(width // 2), -torch.ones(width // 2))
    ).reshape(1, 1, 1, width).to(dtype=torch.float64)
    target = polar_flow_target(clean, noise, 0.0, latent_mean=[0.0])

    assert _rms_inner(noise, clean).item() == pytest.approx(0.0)
    assert _rms_norm(target.tangent_velocity).item() == pytest.approx(math.pi)
    assert _rms_inner(target.tangent_velocity, target.tangent_velocity).item() == pytest.approx(
        math.pi**2
    )


def test_projection_and_cfg_are_pointwise_tangent_and_loss_decomposes():
    generator = torch.Generator().manual_seed(21)
    direction = torch.randn(3, 4, 5, 6, generator=generator, dtype=torch.float64)
    direction = direction / _rms_norm(direction).reshape(3, 1, 1, 1)
    conditional = torch.randn(direction.shape, generator=generator, dtype=direction.dtype)
    unconditional = torch.randn(direction.shape, generator=generator, dtype=direction.dtype)
    tau_c = polar_tangent_projection(conditional, direction)
    tau_u = polar_tangent_projection(unconditional, direction)

    for scale in (-4.0, 0.0, 1.0, 30.0):
        guided = tau_u + scale * (tau_c - tau_u)
        assert torch.allclose(
            _rms_inner(direction, guided),
            torch.zeros(3, dtype=direction.dtype),
            atol=2e-15,
        )

    predicted_radial = torch.tensor([0.2, -0.3, 1.1], dtype=direction.dtype)
    target_radial = torch.tensor([-0.1, 0.4, 0.7], dtype=direction.dtype)
    target_tangent = polar_tangent_projection(
        torch.randn(direction.shape, generator=generator, dtype=direction.dtype),
        direction,
    )
    direct_error = (
        (predicted_radial - target_radial).reshape(3, 1, 1, 1) * direction
        + tau_c
        - target_tangent
    )
    direct_mse = direct_error.square().flatten(1).mean(dim=1)
    split_mse = (predicted_radial - target_radial).square()
    split_mse = split_mse + (tau_c - target_tangent).square().flatten(1).mean(dim=1)
    assert torch.allclose(direct_mse, split_mse, atol=2e-15, rtol=2e-15)


def test_coincident_and_antipodal_policies_are_finite_and_deterministic():
    noise = torch.tensor(
        [[[[1.0, -1.0, 1.0, -1.0]]], [[[1.0, -1.0, 1.0, -1.0]]]],
        dtype=torch.float64,
    )
    clean = torch.stack((noise[0], -noise[1]))
    first = polar_flow_target(clean, noise, 0.5, latent_mean=[0.0])
    second = polar_flow_target(clean, noise, 0.5, latent_mean=[0.0])

    assert first.small_angle_mask.tolist() == [True, False]
    assert first.antipodal_mask.tolist() == [False, True]
    assert torch.isfinite(first.sample).all()
    assert torch.isfinite(first.tangent_velocity).all()
    assert torch.equal(first.sample, second.sample)
    assert torch.equal(first.tangent_velocity, second.tangent_velocity)

    with pytest.raises(ValueError, match="radius"):
        polar_flow_target(torch.zeros_like(clean), noise, 0.5, latent_mean=[0.0])


def test_v3_prediction_contract_is_strictly_format_four():
    contract = prediction_contract(
        FLOW_V3_PREDICTION,
        latent_mean=[0.1, -0.2, 0.3, -0.4],
        latent_centered_second_moment=1.25,
        angular_endpoint_slope=0.75,
        angular_step_limit=None,
    )
    manifest = {"format_version": V3_FORMAT_VERSION, "prediction": contract}
    assert validated_prediction_contract(manifest) == contract
    assert contract["cfg_mode"] == "tangent_only_v1"
    assert contract["integrator"] == "polar_exp_euler_v1"
    assert contract["spatial_input"] == "unit_centered_direction_v1"
    assert contract["angular_endpoint_slope"] == pytest.approx(0.75)

    with pytest.raises(ChimeraArtifactError, match="requires Chimera format v4"):
        validated_prediction_contract(
            {"format_version": FORMAT_VERSION, "prediction": contract}
        )
    with pytest.raises(ChimeraArtifactError, match="requires polar_tangent_flow"):
        validated_prediction_contract(
            {
                "format_version": V3_FORMAT_VERSION,
                "prediction": {"type": "flow_velocity"},
            }
        )
    with pytest.raises(ChimeraArtifactError, match="angular_endpoint_slope"):
        prediction_contract(
            FLOW_V3_PREDICTION,
            latent_mean=[0.0] * 4,
            latent_centered_second_moment=1.0,
            angular_endpoint_slope=3.0,
        )
    invalid = dict(contract)
    invalid["integrator"] = "cartesian_euler"
    with pytest.raises(ChimeraArtifactError, match="prediction contract fields"):
        validated_prediction_contract(
            {"format_version": V3_FORMAT_VERSION, "prediction": invalid}
        )
