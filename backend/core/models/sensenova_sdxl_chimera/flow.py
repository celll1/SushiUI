"""Flow-matching algebra for SenseNova SDXL Chimera.

Time follows SenseNova rather than diffusers' usual descending-timestep
notation: t=0 is noise and t=1 is clean.  The U-Net predicts velocity directly;
there is no x0 reconstruction followed by division near t=1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


FLOW_V1_PREDICTION = "flow_velocity"
FLOW_V2_PREDICTION = "endpoint_observable_residual"
FLOW_V2_VELOCITY_PREDICTION = "endpoint_observable_velocity"
FLOW_V2_PATH = "symmetric_cubic_observable_v1"
FLOW_V3_PREDICTION = "polar_tangent_flow"
FLOW_V3_PATH = "observable_polar_geodesic_v1"
FLOW_V3_RADIAL_SCHEDULE = "cubic_quadrature_v1"
FLOW_V3_ANGULAR_SCHEDULE = "terminal_flat_cubic_v1"


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


def _geometry_dtype(sample: torch.Tensor) -> torch.dtype:
    if not sample.is_floating_point():
        raise TypeError("Chimera flow tensors must use a floating dtype")
    return torch.float64 if sample.dtype == torch.float64 else torch.float32


def _rms_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(
            f"inner-product shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    return (left * right).flatten(1).mean(dim=1).reshape(
        left.shape[0], *((1,) * (left.ndim - 1))
    )


def _rms_norm(value: torch.Tensor) -> torch.Tensor:
    return _rms_inner(value, value).clamp_min(0.0).sqrt()


def terminal_flat_angular_schedule(
    timestep: torch.Tensor | float,
    sample: torch.Tensor,
    *,
    endpoint_slope: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a terminal-flat angular schedule and its clean-time derivative."""
    slope = float(endpoint_slope)
    if not math.isfinite(slope) or not 0.0 <= slope <= 2.0:
        raise ValueError("endpoint_slope must be finite and in [0, 2]")
    t = _batch_scalar(timestep, sample)
    gamma = slope * t + (3.0 - 2.0 * slope) * t.square()
    gamma = gamma + (slope - 2.0) * t.pow(3)
    gamma_prime = slope + 2.0 * (3.0 - 2.0 * slope) * t
    gamma_prime = gamma_prime + 3.0 * (slope - 2.0) * t.square()
    return gamma, gamma_prime


def polar_tangent_projection(
    field: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Project one latent field onto the per-sample RMS tangent space."""
    if field.shape != direction.shape:
        raise ValueError(
            f"field/direction shape mismatch: {tuple(field.shape)} vs {tuple(direction.shape)}"
        )
    geometry_dtype = _geometry_dtype(field)
    field_geometry = field.to(dtype=geometry_dtype)
    direction_geometry = direction.to(device=field.device, dtype=geometry_dtype)
    projected = field_geometry - direction_geometry * _rms_inner(
        direction_geometry, field_geometry
    )
    return projected.to(dtype=field.dtype)


def _deterministic_orthogonal(direction: torch.Tensor) -> torch.Tensor:
    """Choose a deterministic RMS-unit tangent for an antipodal pair."""
    flat = direction.flatten(1)
    width = flat.shape[1]
    indices = flat.abs().argmin(dim=1, keepdim=True)
    basis = torch.zeros_like(flat)
    basis.scatter_(1, indices, math.sqrt(width))
    basis = basis.reshape_as(direction)
    tangent = basis - direction * _rms_inner(direction, basis)
    return tangent / _rms_norm(tangent).clamp_min(torch.finfo(direction.dtype).eps)


@dataclass(frozen=True)
class PolarFlowTarget:
    sample: torch.Tensor
    radial_velocity: torch.Tensor
    tangent_velocity: torch.Tensor
    full_velocity: torch.Tensor
    direction: torch.Tensor
    radius: torch.Tensor
    small_angle_mask: torch.Tensor
    antipodal_mask: torch.Tensor


@dataclass(frozen=True)
class PolarStepResult:
    sample: torch.Tensor
    radius: torch.Tensor
    direction: torch.Tensor
    angular_displacement: torch.Tensor
    angular_cap_scale: torch.Tensor


def polar_state(
    sample: torch.Tensor,
    timestep: torch.Tensor | float,
    *,
    latent_mean: torch.Tensor | list[float] | tuple[float, ...],
    radius_floor: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return centered state, RMS radius, and RMS-unit direction in fp32 geometry."""
    floor = float(radius_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("radius_floor must be finite and > 0")
    geometry_dtype = _geometry_dtype(sample)
    value = sample.to(dtype=geometry_dtype)
    alpha, _sigma, _alpha_prime, _sigma_prime = endpoint_observable_coefficients(
        timestep, value
    )
    centered = value - alpha * _latent_mean(latent_mean, value)
    radius = _rms_norm(centered)
    if bool((radius <= floor).any().item()):
        raise ValueError("polar state radius is at or below radius_floor")
    return centered, radius.flatten(1)[:, 0], centered / radius


def polar_compose_velocity(
    sample: torch.Tensor,
    radial_velocity: torch.Tensor,
    tangent_velocity: torch.Tensor,
    timestep: torch.Tensor | float,
    *,
    latent_mean: torch.Tensor | list[float] | tuple[float, ...],
    radius_floor: float = 1e-8,
) -> torch.Tensor:
    """Compose the full Cartesian velocity from v3 polar predictions."""
    if sample.shape != tangent_velocity.shape:
        raise ValueError("sample/tangent_velocity shape mismatch")
    _centered, _radius, direction = polar_state(
        sample, timestep, latent_mean=latent_mean, radius_floor=radius_floor
    )
    geometry_dtype = direction.dtype
    value = sample.to(dtype=geometry_dtype)
    _alpha, _sigma, alpha_prime, _sigma_prime = endpoint_observable_coefficients(
        timestep, value
    )
    mean = _latent_mean(latent_mean, value)
    radial = _batch_scalar(radial_velocity, value).to(dtype=geometry_dtype)
    tangent = tangent_velocity.to(dtype=geometry_dtype)
    return (alpha_prime * mean + radial * direction + tangent).to(sample.dtype)


def polar_recover_clean(
    sample: torch.Tensor,
    radial_velocity: torch.Tensor,
    tangent_velocity: torch.Tensor,
    timestep: torch.Tensor | float,
    *,
    latent_mean: torch.Tensor | list[float] | tuple[float, ...],
    latent_centered_second_moment: float,
    angular_endpoint_slope: float = 2.0,
    radius_floor: float = 1e-8,
    determinant_floor: float = 1e-8,
) -> torch.Tensor:
    """Recover a local clean-latent estimate for v3 diagnostics."""
    q = float(latent_centered_second_moment)
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("latent_centered_second_moment must be finite and > 0")
    centered, radius_vector, direction = polar_state(
        sample, timestep, latent_mean=latent_mean, radius_floor=radius_floor
    )
    radius = radius_vector.reshape(sample.shape[0], *((1,) * (sample.ndim - 1)))
    radial = _batch_scalar(radial_velocity, centered).to(direction)
    tangent = polar_tangent_projection(
        tangent_velocity.to(direction), direction
    ).to(direction)
    alpha, sigma, alpha_prime, sigma_prime = endpoint_observable_coefficients(
        timestep, centered
    )
    gamma, gamma_prime = terminal_flat_angular_schedule(
        timestep, centered, endpoint_slope=angular_endpoint_slope
    )

    determinant = sigma.square() * alpha * alpha_prime
    determinant = determinant - alpha.square() * sigma * sigma_prime
    clean_radius_square = (
        sigma.square() * radius * radial
        - sigma * sigma_prime * radius.square()
    )
    valid_radius = determinant.abs() >= float(determinant_floor)
    safe_determinant = torch.where(
        valid_radius, determinant, torch.ones_like(determinant)
    )
    clean_radius_square = clean_radius_square / safe_determinant
    fallback_radius_square = torch.full_like(clean_radius_square, q)
    clean_radius = torch.where(
        valid_radius,
        clean_radius_square.clamp_min(float(radius_floor) ** 2),
        fallback_radius_square,
    ).sqrt()

    tangent_norm = _rms_norm(tangent)
    safe_angular_rate = (radius * gamma_prime.abs()).clamp_min(float(radius_floor))
    pair_angle = tangent_norm / safe_angular_rate
    tangent_direction = tangent / tangent_norm.clamp_min(float(radius_floor))
    remaining_angle = (1.0 - gamma) * pair_angle
    clean_direction = (
        remaining_angle.cos() * direction
        + remaining_angle.sin() * tangent_direction
    )
    valid_tangent = (tangent_norm > float(radius_floor)) & (
        gamma_prime.abs() > float(radius_floor)
    )
    clean_direction = torch.where(
        valid_tangent.expand_as(clean_direction), clean_direction, direction
    )
    mean = _latent_mean(latent_mean, centered)
    recovered = mean + clean_radius * clean_direction
    clean_endpoint = alpha >= 1.0 - float(determinant_floor)
    recovered = torch.where(clean_endpoint.expand_as(recovered), sample.to(recovered), recovered)
    return recovered.to(sample.dtype)


def polar_exp_euler_step(
    sample: torch.Tensor,
    radial_velocity: torch.Tensor,
    tangent_velocity: torch.Tensor,
    timestep: torch.Tensor | float,
    next_timestep: torch.Tensor | float,
    *,
    latent_mean: torch.Tensor | list[float] | tuple[float, ...],
    radius_floor: float = 1e-8,
    angular_step_limit: float | None = None,
) -> PolarStepResult:
    """Advance one v3 step with exponential radial and spherical updates."""
    centered, radius_vector, direction = polar_state(
        sample, timestep, latent_mean=latent_mean, radius_floor=radius_floor
    )
    tangent = polar_tangent_projection(
        tangent_velocity.to(direction), direction
    ).to(direction)
    radius = radius_vector.reshape(sample.shape[0], *((1,) * (sample.ndim - 1)))
    radial = _batch_scalar(radial_velocity, centered).to(direction)
    current = _batch_scalar(timestep, centered).to(direction)
    following = _batch_scalar(next_timestep, centered).to(direction)
    delta = following - current

    log_radius_rate = radial / radius
    next_radius = radius * (delta * log_radius_rate).exp()
    if not bool(torch.isfinite(next_radius).all().item()):
        raise FloatingPointError("polar radial update produced a non-finite radius")

    omega = tangent / radius
    omega_norm = _rms_norm(omega)
    angular_displacement = delta * omega_norm
    cap_scale = torch.ones_like(angular_displacement)
    if angular_step_limit is not None:
        limit = float(angular_step_limit)
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("angular_step_limit must be null or finite and > 0")
        cap_scale = (
            torch.full_like(angular_displacement, limit)
            / angular_displacement.abs().clamp_min(float(radius_floor))
        ).clamp(max=1.0)
    effective_angle = angular_displacement * cap_scale
    omega_direction = omega / omega_norm.clamp_min(float(radius_floor))
    next_direction = (
        effective_angle.cos() * direction
        + effective_angle.sin() * omega_direction
    )
    has_angle = omega_norm > float(radius_floor)
    next_direction = torch.where(
        has_angle.expand_as(next_direction), next_direction, direction
    )
    next_direction = next_direction / _rms_norm(next_direction).clamp_min(
        float(radius_floor)
    )
    next_alpha, _next_sigma, _next_alpha_prime, _next_sigma_prime = (
        endpoint_observable_coefficients(next_timestep, centered)
    )
    mean = _latent_mean(latent_mean, centered)
    next_sample = next_alpha * mean + next_radius * next_direction
    return PolarStepResult(
        sample=next_sample.to(sample.dtype),
        radius=next_radius.flatten(1)[:, 0],
        direction=next_direction,
        angular_displacement=effective_angle.flatten(1)[:, 0],
        angular_cap_scale=cap_scale.flatten(1)[:, 0],
    )


def polar_flow_target(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor | float,
    *,
    latent_mean: torch.Tensor | list[float] | tuple[float, ...],
    angular_endpoint_slope: float = 2.0,
    radius_floor: float = 1e-8,
    angular_singularity_threshold: float = 1e-6,
) -> PolarFlowTarget:
    """Build the Chimera v3 polar path and its radial/tangent targets."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shape mismatch: {tuple(clean.shape)} vs {tuple(noise.shape)}")
    if clean.ndim < 2 or clean.shape[0] == 0:
        raise ValueError("clean/noise must have shape [B, ...] with a non-empty batch")
    floor = float(radius_floor)
    threshold = float(angular_singularity_threshold)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("radius_floor must be finite and > 0")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("angular_singularity_threshold must be finite and > 0")

    geometry_dtype = _geometry_dtype(clean)
    clean_geometry = clean.to(dtype=geometry_dtype)
    noise_geometry = noise.to(device=clean.device, dtype=geometry_dtype)
    mean = _latent_mean(latent_mean, clean_geometry)
    centered_clean = clean_geometry - mean
    rho0 = _rms_norm(noise_geometry)
    rho1 = _rms_norm(centered_clean)
    if bool((rho0 <= floor).any().item()) or bool((rho1 <= floor).any().item()):
        raise ValueError("polar path endpoint radius is at or below radius_floor")

    n0 = noise_geometry / rho0
    n1 = centered_clean / rho1
    dot = _rms_inner(n0, n1).clamp(-1.0, 1.0)
    raw_tangent = n1 - dot * n0
    tangent_norm = _rms_norm(raw_tangent)
    theta = torch.atan2(tangent_norm, dot)
    singular = tangent_norm <= threshold
    same_direction = singular & (dot >= 0.0)
    antipodal = singular & (dot < 0.0)

    safe_norm = tangent_norm.clamp_min(threshold)
    great_circle_direction = raw_tangent / safe_norm
    great_circle_direction = torch.where(
        antipodal.expand_as(great_circle_direction),
        _deterministic_orthogonal(n0),
        great_circle_direction,
    )

    alpha, sigma, alpha_prime, sigma_prime = endpoint_observable_coefficients(
        timestep, clean_geometry
    )
    gamma, gamma_prime = terminal_flat_angular_schedule(
        timestep,
        clean_geometry,
        endpoint_slope=angular_endpoint_slope,
    )
    phase = gamma * theta
    direction = phase.cos() * n0 + phase.sin() * great_circle_direction
    direction_prime = gamma_prime * theta * (
        -phase.sin() * n0 + phase.cos() * great_circle_direction
    )

    linear_direction = (1.0 - gamma) * n0 + gamma * n1
    linear_radius = _rms_norm(linear_direction).clamp_min(floor)
    linear_direction = linear_direction / linear_radius
    linear_prime_raw = gamma_prime * (n1 - n0)
    linear_direction_prime = polar_tangent_projection(
        linear_prime_raw, linear_direction
    ).to(dtype=geometry_dtype) / linear_radius
    direction = torch.where(same_direction.expand_as(direction), linear_direction, direction)
    direction_prime = torch.where(
        same_direction.expand_as(direction_prime),
        linear_direction_prime,
        direction_prime,
    )

    radius_square = sigma.square() * rho0.square() + alpha.square() * rho1.square()
    radius = radius_square.clamp_min(floor * floor).sqrt()
    radial_velocity = (
        sigma * sigma_prime * rho0.square()
        + alpha * alpha_prime * rho1.square()
    ) / radius
    tangent_velocity = radius * direction_prime
    sample_geometry = alpha * mean + radius * direction
    full_velocity_geometry = (
        alpha_prime * mean + radial_velocity * direction + tangent_velocity
    )

    output_dtype = clean.dtype
    return PolarFlowTarget(
        sample=sample_geometry.to(dtype=output_dtype),
        radial_velocity=radial_velocity.flatten(1)[:, 0],
        tangent_velocity=tangent_velocity.to(dtype=output_dtype),
        full_velocity=full_velocity_geometry.to(dtype=output_dtype),
        direction=direction,
        radius=radius.flatten(1)[:, 0],
        small_angle_mask=same_direction.flatten(1)[:, 0],
        antipodal_mask=antipodal.flatten(1)[:, 0],
    )


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


def endpoint_observable_recover_clean(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor | float,
    *,
    latent_mean: torch.Tensor | list[float] | tuple[float, ...],
    determinant_floor: float = 1e-8,
) -> torch.Tensor:
    """Recover x0 from an interior v2 state/velocity for diagnostics."""
    if sample.shape != velocity.shape:
        raise ValueError(
            f"sample/velocity shape mismatch: {tuple(sample.shape)} vs {tuple(velocity.shape)}"
        )
    alpha, sigma, alpha_prime, sigma_prime = endpoint_observable_coefficients(
        timestep, sample
    )
    determinant = alpha * sigma_prime - sigma * alpha_prime
    valid = determinant.abs() >= float(determinant_floor)
    safe = torch.where(valid, determinant, torch.ones_like(determinant))
    recovered = (sigma_prime * sample - sigma * velocity) / safe
    mean = _latent_mean(latent_mean, sample).expand_as(sample)
    # At clean time z is x0; at noise time only the calibrated mean is defined.
    fallback = torch.where(alpha >= sigma, sample, mean)
    return torch.where(valid, recovered, fallback)


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
