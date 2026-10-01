"""Native Ming flow conventions shared by inference and training."""

import math

import torch


def model_time_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    return 1.0 - sigma.float()


def flow_training_pair(clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor):
    if clean.shape != noise.shape:
        raise ValueError("Ming clean/noise shapes must match")
    if sigma.ndim != 1 or sigma.shape[0] != clean.shape[0]:
        raise ValueError("Ming sigma must contain one value per image")
    weight = sigma.float().reshape(-1, *([1] * (clean.ndim - 1)))
    noisy = clean * (1.0 - weight) + noise * weight
    return noisy, clean - noise, model_time_from_sigma(sigma)


def native_guidance(positive: torch.Tensor, zero: torch.Tensor, scale: float) -> torch.Tensor:
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Ming guidance must be finite and nonnegative")
    if positive.shape != zero.shape:
        raise ValueError("Ming guidance branch shapes differ")
    return positive if scale <= 1 else positive + scale * (positive - zero)


def scheduler_prediction(raw: torch.Tensor) -> torch.Tensor:
    return -raw.float()
