"""Flow-matching (Rectified Flow) scheduler for Anima.

Anima uses a discrete-time flow matching scheduler where:
  - sigmas: linearly spaced from 1.0 (pure noise) to 0.0 (clean) across N steps
  - x_t = (1 - sigma) * x_0 + sigma * noise
  - model predicts velocity v = noise - x_0
  - Euler update: x_{t-dt} = x_t - dt * v

The "shift" parameter modulates the sigma schedule to favor higher-noise or
lower-noise regions (timestep_shift in flux-style schedulers). shift=1.0
means uniform; values >1.0 push more steps toward high noise (recommended
for high-resolution inference).
"""

from typing import List, Optional, Tuple

import torch
import numpy as np


class AnimaFlowMatchScheduler:
    """Minimal flow-matching Euler scheduler for Anima."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

    @staticmethod
    def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
        # Used by flux-style schedulers. Reduces to identity when mu == 0.
        return mu * t / (1 + (mu - 1) * t) if mu != 1.0 else t

    def set_timesteps(self, num_inference_steps: int, device: torch.device,
                       shift: Optional[float] = None) -> None:
        if shift is None:
            shift = self.shift
        # sigmas linear from 1.0 -> 0.0 across N+1 boundaries, drop final 0.
        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, device=device, dtype=torch.float32)
        if shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        # Append a final 0.0 so we can compute dt = sigmas[i] - sigmas[i+1]
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=torch.float32)])
        self.sigmas = sigmas
        # Anima expects the timestep to be the sigma value directly (range [0, 1]),
        # NOT sigma * num_train_timesteps — see sd-scripts anima_train_utils.sample().
        self.timesteps = sigmas[:-1].clone()
        self.num_inference_steps = num_inference_steps

    def scale_noise(self, sample: torch.Tensor, step_index: int, noise: torch.Tensor) -> torch.Tensor:
        """For img2img/inpaint init: x_t = (1 - sigma) * x_0 + sigma * noise."""
        sigma = self.sigmas[step_index].to(sample.dtype).to(sample.device)
        return (1.0 - sigma) * sample + sigma * noise

    def step(self, model_output: torch.Tensor, step_index: int, sample: torch.Tensor) -> torch.Tensor:
        """One Euler step: x_{t-1} = x_t - dt * v(x_t, t)."""
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        dt = (sigma_next - sigma).to(sample.dtype).to(sample.device)
        return sample + dt * model_output.to(sample.dtype)

    def get_timestep(self, step_index: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.timesteps[step_index].to(device=device, dtype=dtype)


def calculate_shift_anima(latent_seq_len: int, base_seq_len: int = 256,
                          max_seq_len: int = 4096,
                          base_shift: float = 0.5, max_shift: float = 1.15) -> float:
    """Compute resolution-dependent shift (same idea as FLUX/Z-Image)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = m * latent_seq_len + b
    # Convert linear mu to shift factor (exp); flux uses mu directly for time_shift.
    return float(np.exp(mu))
