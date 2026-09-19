"""Flow-matching algebra for SenseNova SDXL Chimera.

Time follows SenseNova rather than diffusers' usual descending-timestep
notation: t=0 is noise and t=1 is clean.  The U-Net predicts velocity directly;
there is no x0 reconstruction followed by division near t=1.
"""

from __future__ import annotations

import math

import torch


FLOW_V1_PREDICTION = "flow_velocity"
FLOW_V2_PREDICTION = "endpoint_observable_residual"
FLOW_V2_PATH = "symmetric_cubic_observable_v1"


def _batch_scalar(value: torch.Tensor | float, sample: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=sample.device, dtype=sample.dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(sample.shape[0])
    if tensor.ndim != 1 or tensor.shape[0] != sample.shape[0]:
        raise ValueError(
            f"expected a scalar or [B] value for batch {sample.shape[0]}, got {tuple(tensor.shape)}"
        )
    return tensor.reshape(sample.shape[0], *((1,) * (sample.ndim - 1)))


def _latent_mean(value: torch.Tensor | list[float] | tuple[float, ...], sample: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(value, device=sample.device, dtype=sample.dtype)
    if mean.ndim != 1 or mean.shape[0] != sample.shape[1]:
        raise ValueError(
            f"latent_mean must have one value per channel ({sample.shape[1]}), "
            f"got {tuple(mean.shape)}"
        )
    return mean.reshape(1, sample.shape[1], *((1,) * (sample.ndim - 2)))


def endpoint_observable_coefficients(
    timestep: torch.Tensor | float,
    sample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``alpha, sigma, alpha', sigma'`` for Chimera v2 clean time."""
    t = _batch_scalar(timestep, sample)
    alpha = t.square() * (2.0 - t)
    sigma = (1.0 - t).square() * (1.0 + t)
    alpha_prime = 4.0 * t - 3.0 * t.square()
    sigma_prime = -1.0 - 2.0 * t + 3.0 * t.square()
    return alpha, sigma, alpha_prime, sigma_prime


def endpoint_observable_noising(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor | float,
) -> torch.Tensor:
    """Return ``alpha(s)*x0 + sigma(s)*epsilon`` for the v2 path."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shape mismatch: {tuple(clean.shape)} vs {tuple(noise.shape)}")
    alpha, sigma, _alpha_prime, _sigma_prime = endpoint_observable_coefficients(
        timestep, clean
    )
    return alpha * clean + sigma * noise


def endpoint_observable_velocity_target(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor | float,
) -> torch.Tensor:
    """Return the endpoint-observable path velocity ``alpha'*x0 + sigma'*epsilon``."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shape mismatch: {tuple(clean.shape)} vs {tuple(noise.shape)}")
    _alpha, _sigma, alpha_prime, sigma_prime = endpoint_observable_coefficients(
        timestep, clean
    )
    return alpha_prime * clean + sigma_prime * noise


def endpoint_observable_preconditioning(
    sample: torch.Tensor,
    timestep: torch.Tensor | float,
    *,
    latent_mean: torch.Tensor | list[float] | tuple[float, ...],
    latent_centered_second_moment: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(analytic_state_drift, c_skip)`` for the v2 residual head."""
    q = float(latent_centered_second_moment)
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("latent_centered_second_moment must be finite and > 0")
    alpha, sigma, alpha_prime, sigma_prime = endpoint_observable_coefficients(
        timestep, sample
    )
    mean = _latent_mean(latent_mean, sample)
    centered = sample - alpha * mean
    denominator = q * alpha.square() + sigma.square()
    c_skip = (q * alpha * alpha_prime + sigma * sigma_prime) / denominator
    analytic = alpha_prime * mean + c_skip * centered
    return analytic, c_skip


def endpoint_observable_residual_target(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor | float,
    *,
    latent_mean: torch.Tensor | list[float] | tuple[float, ...],
    latent_centered_second_moment: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(z, residual_target, velocity_target)`` for Chimera v2 training."""
    sample = endpoint_observable_noising(clean, noise, timestep)
    velocity = endpoint_observable_velocity_target(clean, noise, timestep)
    analytic, _c_skip = endpoint_observable_preconditioning(
        sample,
        timestep,
        latent_mean=latent_mean,
        latent_centered_second_moment=latent_centered_second_moment,
    )
    return sample, velocity - analytic, velocity


def endpoint_observable_reconstruct_velocity(
    sample: torch.Tensor,
    residual: torch.Tensor,
    timestep: torch.Tensor | float,
    *,
    latent_mean: torch.Tensor | list[float] | tuple[float, ...],
    latent_centered_second_moment: float,
) -> torch.Tensor:
    """Add the analytic state drift to a learned v2 residual."""
    if sample.shape != residual.shape:
        raise ValueError(
            f"sample/residual shape mismatch: {tuple(sample.shape)} vs {tuple(residual.shape)}"
        )
    analytic, _c_skip = endpoint_observable_preconditioning(
        sample,
        timestep,
        latent_mean=latent_mean,
        latent_centered_second_moment=latent_centered_second_moment,
    )
    return analytic + residual


def flow_noising(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor | float,
    *,
    noise_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Return ``t*x0 + (1-t)*sigma*epsilon`` with batch-safe scalars."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shape mismatch: {tuple(clean.shape)} vs {tuple(noise.shape)}")
    t = _batch_scalar(timestep, clean)
    sigma = _batch_scalar(noise_scale, clean)
    return t * clean + (1.0 - t) * sigma * noise


def flow_velocity_target(
    clean: torch.Tensor,
    noise: torch.Tensor,
    *,
    noise_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """The constant velocity of Chimera's straight noising path."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shape mismatch: {tuple(clean.shape)} vs {tuple(noise.shape)}")
    sigma = _batch_scalar(noise_scale, clean)
    return clean - sigma * noise


def flow_euler_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor | float,
    next_timestep: torch.Tensor | float,
) -> torch.Tensor:
    """Advance along increasing clean-time by one explicit Euler step."""
    if sample.shape != velocity.shape:
        raise ValueError(
            f"sample/velocity shape mismatch: {tuple(sample.shape)} vs {tuple(velocity.shape)}"
        )
    t = _batch_scalar(timestep, sample)
    t_next = _batch_scalar(next_timestep, sample)
    return sample + (t_next - t) * velocity
