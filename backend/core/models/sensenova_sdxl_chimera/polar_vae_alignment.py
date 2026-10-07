"""Polar codec boundaries retain time drift separately from coordinate tangent."""
from dataclasses import dataclass

import torch

from .flow import (
    FLOW_V3_PREDICTION, FLOW_V4_PREDICTION, _latent_mean, _batch_scalar,
    endpoint_observable_coefficients, destruction_coordinate,
    polar_state, polar_tangent_projection,
)


@dataclass(frozen=True)
class PolarBoundaryPrediction:
    radial_velocity: torch.Tensor
    tangent_per_time: torch.Tensor
    tangent_per_coordinate: torch.Tensor
    drift_tangent_per_time: torch.Tensor
    direction: torch.Tensor
    radius: torch.Tensor
    full_velocity: torch.Tensor


@dataclass(frozen=True)
class PolarInferencePrediction:
    radial_velocity: torch.Tensor
    tangent_per_coordinate: torch.Tensor
    direction: torch.Tensor
    radius: torch.Tensor
    drift_tangent_per_time: torch.Tensor

    def __iter__(self):
        return iter((self.radial_velocity, self.tangent_per_coordinate,
                     self.direction, self.radius))


def transfer_prediction(unet, source_sample, source_radial, source_tangent,
                        target_sample, timestep, *, source_prediction, target_prediction):
    kind = source_prediction["type"]
    if kind not in (FLOW_V3_PREDICTION, FLOW_V4_PREDICTION) or kind != target_prediction["type"]:
        raise ValueError("Polar VAE transfer requires matching source/target coordinate contracts")
    _, _, source_direction = polar_state(source_sample, timestep,
        latent_mean=source_prediction["latent_mean"], radius_floor=source_prediction["radius_floor"])
    _, target_radius, target_direction = polar_state(target_sample, timestep,
        latent_mean=target_prediction["latent_mean"], radius_floor=target_prediction["radius_floor"])
    _, _, derivative, _ = endpoint_observable_coefficients(timestep, source_direction)
    source_mean = _latent_mean(source_prediction["latent_mean"], source_direction)
    target_mean = _latent_mean(target_prediction["latent_mean"], target_direction)
    source_tangent = polar_tangent_projection(source_tangent, source_direction)
    radial = _batch_scalar(source_radial, source_direction)
    output = unet.vae_latent_bridge.output
    weight = output.projection.weight
    # Affine mean motion and radial motion live in t, even when d'(t)=0.
    drift = output.diffusion_velocity(
        (derivative*source_mean + radial*source_direction).to(weight),
        clean_derivative=derivative.reshape(-1).expand(source_sample.shape[0]),
    ).to(target_direction) - derivative*target_mean
    from .vae_alignment import correct_velocity_noise
    drift = correct_velocity_noise(unet, drift, timestep, prediction=target_prediction)
    angular = output.epsilon(source_tangent.to(weight)).to(target_direction)
    coordinate_derivative = (destruction_coordinate(timestep, target_direction)[1]
                             if kind == FLOW_V4_PREDICTION else torch.ones_like(derivative))
    centered_velocity = drift + coordinate_derivative*angular
    radial_target = (centered_velocity*target_direction).flatten(1).mean(1)
    tangent = polar_tangent_projection(centered_velocity, target_direction)
    return PolarBoundaryPrediction(
        radial_target, tangent, polar_tangent_projection(angular, target_direction),
        polar_tangent_projection(drift, target_direction), target_direction, target_radius,
        centered_velocity + derivative*target_mean,
    )


def forward_prediction(unet, sample, timestep, *, prediction, **kwargs):
    from .unet import polar_unet_forward
    source = getattr(unet, "_chimera_source_prediction", None)
    pair = getattr(unet, "vae_latent_bridge", None)
    if pair is None or source is None:
        raise ValueError("Polar aligned forward requires its saved spatial bridge and source contract")
    if source["type"] != prediction["type"]:
        raise ValueError("Polar VAE forward cannot reinterpret its source coordinate contract")
    alpha, sigma, _, _ = endpoint_observable_coefficients(timestep, sample)
    if getattr(pair, "noise_alignment", None) is None:
        internal = pair.input.diffusion_state(sample.to(pair.input.projection.weight),
            alpha=alpha.reshape(-1).expand(sample.shape[0]))
    else:
        from core.models.components.image_noise_context import exposure_noise
        internal = pair.diffusion_state(sample.to(pair.input.projection.weight),
            alpha=alpha.reshape(-1).expand(sample.shape[0]), sigma=sigma.reshape(-1),
            noise=exposure_noise(sample))
    _, radius, direction = polar_state(internal, timestep,
        latent_mean=source["latent_mean"], radius_floor=source["radius_floor"])
    kwargs["return_dict"] = False
    tangent, radial = polar_unet_forward(unet, direction.to(internal), timestep,
        log_radius=radius.clamp_min(source["radius_floor"]).log(), **kwargs)
    return transfer_prediction(unet, internal, radial, tangent, sample, timestep,
        source_prediction=source, target_prediction=prediction)


def polar_forward(unet, sample, timestep, *, prediction, **kwargs):
    """Physical-state entry point shared by training and CFG inference."""
    if getattr(unet, "vae_latent_bridge", None) is not None:
        result = forward_prediction(unet, sample, timestep, prediction=prediction, **kwargs)
        if prediction["type"] == FLOW_V4_PREDICTION:
            return PolarInferencePrediction(result.radial_velocity, result.tangent_per_coordinate,
                result.direction, result.radius, result.drift_tangent_per_time)
        return result.radial_velocity, result.tangent_per_time, result.direction, result.radius
    from .unet import polar_unet_forward
    _, radius, direction = polar_state(sample, timestep,
        latent_mean=prediction["latent_mean"], radius_floor=prediction["radius_floor"])
    kwargs["return_dict"] = False
    tangent, radial = polar_unet_forward(unet, direction.to(sample), timestep,
        log_radius=radius.clamp_min(prediction["radius_floor"]).log(), **kwargs)
    return radial, polar_tangent_projection(tangent, direction), direction, radius
