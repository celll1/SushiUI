"""Pure algebra and artifact contracts for Chimera v2."""

import pytest
import torch

from core.models.sensenova_sdxl_chimera.artifact import (
    FORMAT_VERSION,
    LEGACY_FORMAT_VERSION,
    ChimeraArtifactError,
    migrate_manifest_prediction,
    prediction_contract,
    validated_prediction_contract,
)
from core.models.sensenova_sdxl_chimera.flow import (
    FLOW_V1_PREDICTION,
    FLOW_V2_PATH,
    FLOW_V2_PREDICTION,
    endpoint_observable_coefficients,
    endpoint_observable_noising,
    endpoint_observable_preconditioning,
    endpoint_observable_reconstruct_velocity,
    endpoint_observable_residual_target,
    endpoint_observable_velocity_target,
)


def test_endpoint_observable_path_is_symmetric_and_endpoint_exact():
    clean = torch.randn(2, 4, 3, 5, dtype=torch.float64)
    noise = torch.randn_like(clean)
    endpoints = torch.tensor([0.0, 1.0], dtype=torch.float64)
    sample = endpoint_observable_noising(clean, noise, endpoints)
    velocity = endpoint_observable_velocity_target(clean, noise, endpoints)

    assert torch.equal(sample[0], noise[0])
    assert torch.equal(sample[1], clean[1])
    assert torch.equal(velocity[0], -noise[0])
    assert torch.equal(velocity[1], clean[1])

    grid = torch.linspace(0, 1, 101, dtype=torch.float64)
    probe = torch.empty(101, 4, 1, 1, dtype=torch.float64)
    alpha, sigma, alpha_prime, sigma_prime = endpoint_observable_coefficients(grid, probe)
    reflected = torch.flip(sigma, dims=(0,))
    assert torch.allclose(alpha, reflected, atol=1e-14, rtol=1e-14)
    assert torch.allclose(alpha_prime, -torch.flip(sigma_prime, dims=(0,)), atol=1e-14)


def test_centered_residual_vanishes_at_endpoints_and_reconstructs_velocity():
    clean = torch.randn(7, 4, 2, 3, dtype=torch.float64)
    noise = torch.randn_like(clean)
    times = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], dtype=torch.float64)
    mean = [0.25, -0.5, 0.125, 0.75]
    q = 1.7
    sample, residual, target = endpoint_observable_residual_target(
        clean,
        noise,
        times,
        latent_mean=mean,
        latent_centered_second_moment=q,
    )
    reconstructed = endpoint_observable_reconstruct_velocity(
        sample,
        residual,
        times,
        latent_mean=mean,
        latent_centered_second_moment=q,
    )

    assert torch.equal(residual[0], torch.zeros_like(residual[0]))
    assert torch.allclose(residual[-1], torch.zeros_like(residual[-1]), atol=5e-16)
    assert torch.allclose(reconstructed, target, atol=1e-12, rtol=1e-12)

    analytic, c_skip = endpoint_observable_preconditioning(
        sample,
        times,
        latent_mean=mean,
        latent_centered_second_moment=q,
    )
    assert torch.isfinite(analytic).all()
    assert torch.isfinite(c_skip).all()
    assert c_skip[0].item() == pytest.approx(-1.0)
    assert c_skip[-1].item() == pytest.approx(1.0)


def test_prediction_contract_keeps_v1_legacy_and_validates_v2_stats():
    legacy = {
        "format_version": LEGACY_FORMAT_VERSION,
        "prediction": {"type": FLOW_V1_PREDICTION},
    }
    assert validated_prediction_contract(legacy)["type"] == FLOW_V1_PREDICTION

    migrated = migrate_manifest_prediction(
        legacy,
        latent_mean=[0.1, 0.2, 0.3, 0.4],
        latent_centered_second_moment=1.25,
    )
    assert migrated["format_version"] == FORMAT_VERSION
    assert migrated["prediction"] == prediction_contract(
        FLOW_V2_PREDICTION,
        latent_mean=[0.1, 0.2, 0.3, 0.4],
        latent_centered_second_moment=1.25,
    )
    assert migrated["prediction"]["path"] == FLOW_V2_PATH

    with pytest.raises(ChimeraArtifactError, match="four finite"):
        prediction_contract(
            FLOW_V2_PREDICTION,
            latent_mean=[0.0, 0.0],
            latent_centered_second_moment=1.0,
        )
    with pytest.raises(ChimeraArtifactError, match="finite and > 0"):
        prediction_contract(
            FLOW_V2_PREDICTION,
            latent_mean=[0.0] * 4,
            latent_centered_second_moment=0.0,
        )
