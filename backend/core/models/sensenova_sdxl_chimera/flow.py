"""Flow-matching algebra for SenseNova SDXL Chimera.

Time follows SenseNova rather than diffusers' usual descending-timestep
notation: t=0 is noise and t=1 is clean.  The U-Net predicts velocity directly;
there is no x0 reconstruction followed by division near t=1.
"""

from __future__ import annotations

import torch


def _batch_scalar(value: torch.Tensor | float, sample: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=sample.device, dtype=sample.dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(sample.shape[0])
    if tensor.ndim != 1 or tensor.shape[0] != sample.shape[0]:
        raise ValueError(
            f"expected a scalar or [B] value for batch {sample.shape[0]}, got {tuple(tensor.shape)}"
        )
    return tensor.reshape(sample.shape[0], *((1,) * (sample.ndim - 1)))


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
