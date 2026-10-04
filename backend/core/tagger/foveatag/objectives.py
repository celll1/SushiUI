"""Self-supervised objectives for training FoveaTag without external weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def overlap_correspondence_loss(
    first_features: torch.Tensor,
    first_coordinates: torch.Tensor,
    second_features: torch.Tensor,
    second_coordinates: torch.Tensor,
    *,
    max_coordinate_distance: float,
) -> torch.Tensor:
    """Match nearest feature cells only where two source-coordinate maps overlap."""
    _validate_feature_map(first_features, first_coordinates, "first")
    _validate_feature_map(second_features, second_coordinates, "second")
    if first_features.shape[0] != second_features.shape[0]:
        raise ValueError("feature-map batch sizes must match")
    if first_features.shape[1] != second_features.shape[1]:
        raise ValueError("feature channel dimensions must match")
    if max_coordinate_distance <= 0:
        raise ValueError("max_coordinate_distance must be positive")

    first = first_features.flatten(2).transpose(1, 2)
    second = second_features.flatten(2).transpose(1, 2)
    first_xy = first_coordinates.flatten(1, 2).to(torch.float32)
    second_xy = second_coordinates.flatten(1, 2).to(torch.float32)
    distances = torch.cdist(first_xy, second_xy)
    first_distance, first_index = distances.min(dim=2)
    second_distance, second_index = distances.min(dim=1)
    losses = []
    losses.extend(_matched_cosine(first, second, first_index, first_distance, max_coordinate_distance))
    losses.extend(_matched_cosine(second, first, second_index, second_distance, max_coordinate_distance))
    if not losses:
        return (first_features.sum() + second_features.sum()) * 0.0
    return torch.cat(losses).mean()


def _matched_cosine(
    source: torch.Tensor,
    target: torch.Tensor,
    target_index: torch.Tensor,
    distance: torch.Tensor,
    threshold: float,
):
    matched = torch.gather(
        target,
        dim=1,
        index=target_index.unsqueeze(-1).expand(-1, -1, target.shape[-1]),
    )
    loss = 1.0 - F.cosine_similarity(source, matched, dim=-1)
    valid = distance <= threshold
    return [loss[valid]] if valid.any() else []


def masked_patch_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier reconstruction over selected spatial cells."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have equal [batch, channels, h, w] shapes")
    selected = _spatial_mask(mask, prediction)
    if not selected.any():
        return prediction.sum() * 0.0
    error = torch.sqrt((prediction - target).pow(2) + epsilon**2).mean(dim=1)
    return error[selected].mean()


def rgb_to_chroma(rgb: torch.Tensor) -> torch.Tensor:
    """Return two opponent-color channels without discarding small RGB differences."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [batch, 3, height, width]")
    red, green, blue = rgb.unbind(dim=1)
    red_green = (red - green) / 2.0
    yellow_blue = (red + green - 2.0 * blue) / 4.0
    return torch.stack((red_green, yellow_blue), dim=1)


def chroma_reconstruction_loss(
    prediction_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction_rgb.shape != target_rgb.shape:
        raise ValueError("prediction_rgb and target_rgb shapes must match")
    prediction = rgb_to_chroma(prediction_rgb)
    target = rgb_to_chroma(target_rgb)
    selected = _spatial_mask(mask, prediction)
    if not selected.any():
        return prediction_rgb.sum() * 0.0
    error = (prediction - target).abs().mean(dim=1)
    return error[selected].mean()


def feature_distribution_regularization(
    first_features: torch.Tensor,
    second_features: torch.Tensor,
    *,
    target_standard_deviation: float = 1.0,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return VICReg-style variance/covariance plus explicit feature centering."""
    if first_features.ndim != 4 or second_features.ndim != 4:
        raise ValueError("features must have shape [batch, channels, height, width]")
    if first_features.shape[1] != second_features.shape[1]:
        raise ValueError("feature channel dimensions must match")
    if target_standard_deviation <= 0 or epsilon <= 0:
        raise ValueError("distribution targets must be positive")
    terms = [
        _feature_distribution_terms(
            features,
            target_standard_deviation=target_standard_deviation,
            epsilon=epsilon,
        )
        for features in (first_features, second_features)
    ]
    return tuple(torch.stack(values).mean() for values in zip(*terms))


def _feature_distribution_terms(
    features: torch.Tensor,
    *,
    target_standard_deviation: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).float()
    if tokens.shape[0] <= 1:
        zero = tokens.sum() * 0.0
        return zero, zero, zero
    mean = tokens.mean(dim=0)
    centered = tokens - mean
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=True) + epsilon)
    variance = F.relu(target_standard_deviation - standard_deviation).mean()
    covariance_matrix = centered.T @ centered / (tokens.shape[0] - 1)
    covariance = (
        covariance_matrix.square().sum() - covariance_matrix.diagonal().square().sum()
    ) / covariance_matrix.shape[0]
    centering = mean.square().mean()
    return variance, covariance, centering


@dataclass(frozen=True)
class SelfSupervisedLossOutput:
    loss: torch.Tensor
    correspondence: torch.Tensor
    reconstruction: torch.Tensor
    chroma: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    centering: torch.Tensor


class FoveaSelfSupervisedLoss(nn.Module):
    def __init__(
        self,
        *,
        correspondence_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        chroma_weight: float = 0.5,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.01,
        centering_weight: float = 0.1,
        target_standard_deviation: float = 1.0,
        distribution_epsilon: float = 1e-4,
        max_coordinate_distance: float = 0.02,
    ) -> None:
        super().__init__()
        if min(
            correspondence_weight,
            reconstruction_weight,
            chroma_weight,
            variance_weight,
            covariance_weight,
            centering_weight,
        ) < 0:
            raise ValueError("loss weights must be non-negative")
        if target_standard_deviation <= 0 or distribution_epsilon <= 0:
            raise ValueError("distribution targets must be positive")
        self.correspondence_weight = correspondence_weight
        self.reconstruction_weight = reconstruction_weight
        self.chroma_weight = chroma_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.centering_weight = centering_weight
        self.target_standard_deviation = target_standard_deviation
        self.distribution_epsilon = distribution_epsilon
        self.max_coordinate_distance = max_coordinate_distance

    def forward(
        self,
        first_features: torch.Tensor,
        first_coordinates: torch.Tensor,
        second_features: torch.Tensor,
        second_coordinates: torch.Tensor,
        reconstructed_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> SelfSupervisedLossOutput:
        correspondence = overlap_correspondence_loss(
            first_features,
            first_coordinates,
            second_features,
            second_coordinates,
            max_coordinate_distance=self.max_coordinate_distance,
        )
        reconstruction = masked_patch_reconstruction_loss(
            reconstructed_rgb,
            target_rgb,
            reconstruction_mask,
        )
        chroma = chroma_reconstruction_loss(
            reconstructed_rgb,
            target_rgb,
            reconstruction_mask,
        )
        variance, covariance, centering = feature_distribution_regularization(
            first_features,
            second_features,
            target_standard_deviation=self.target_standard_deviation,
            epsilon=self.distribution_epsilon,
        )
        loss = (
            correspondence * self.correspondence_weight
            + reconstruction * self.reconstruction_weight
            + chroma * self.chroma_weight
            + variance * self.variance_weight
            + covariance * self.covariance_weight
            + centering * self.centering_weight
        )
        return SelfSupervisedLossOutput(
            loss,
            correspondence,
            reconstruction,
            chroma,
            variance,
            covariance,
            centering,
        )


def _validate_feature_map(features: torch.Tensor, coordinates: torch.Tensor, name: str) -> None:
    if features.ndim != 4:
        raise ValueError(f"{name}_features must have shape [batch, channels, h, w]")
    expected = (features.shape[0], features.shape[2], features.shape[3], 2)
    if coordinates.shape != expected:
        raise ValueError(f"{name}_coordinates must have shape {expected}")


def _spatial_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    expected = (reference.shape[0], reference.shape[2], reference.shape[3])
    if mask.shape != expected:
        raise ValueError(f"mask must have shape {expected} or [batch, 1, h, w]")
    return mask.to(device=reference.device, dtype=torch.bool)
