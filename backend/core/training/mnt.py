"""Noise coupling policies for multi-noise-timestep (MNT) windows."""

from __future__ import annotations

import math
from typing import Optional

import torch


MNT_NOISE_MODES = frozenset({"independent", "shared", "trajectory", "antithetic"})
_MODE_ALIASES = {"trajectory_blend": "trajectory"}


def normalize_mnt_noise_mode(mode: str) -> str:
    normalized = str(mode or "independent").strip().lower()
    normalized = _MODE_ALIASES.get(normalized, normalized)
    if normalized not in MNT_NOISE_MODES:
        choices = ", ".join(sorted(MNT_NOISE_MODES))
        raise ValueError(f"multi_noise_mode must be one of: {choices} (got {mode!r})")
    return normalized


class MNTNoiseWindow:
    """Generate one marginally-standard-normal noise tensor per MNT iteration.

    The state is deliberately held on the clean latent's device (normally CPU),
    so shared/correlated modes do not pin another full latent on the GPU between
    backwards. The architecture train step performs the usual device/dtype move.
    """

    def __init__(self, mode: str, iterations: int, trajectory_alpha: float = 0.7):
        self.mode = normalize_mnt_noise_mode(mode)
        self.iterations = max(1, int(iterations))
        self.trajectory_alpha = float(trajectory_alpha)
        if not 0.0 <= self.trajectory_alpha <= 1.0:
            raise ValueError("trajectory_blend_alpha must be between 0 and 1")
        self._shared: Optional[torch.Tensor] = None
        self._pair: Optional[torch.Tensor] = None
        self._pair_index = -1

    @staticmethod
    def _draw(reference: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(reference, requires_grad=False)

    def noise_for(self, iteration: int, reference: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the noise for this iteration; None preserves the legacy draw."""
        if self.mode == "independent" or self.iterations <= 1:
            return None
        if not 0 <= int(iteration) < self.iterations:
            raise IndexError(f"MNT iteration {iteration} is outside [0, {self.iterations})")

        ref = reference.detach()
        if self.mode == "shared":
            if self._shared is None:
                self._shared = self._draw(ref)
            return self._shared

        if self.mode == "trajectory":
            alpha = self.trajectory_alpha
            if alpha == 0.0:
                return self._draw(ref)
            if self._shared is None:
                self._shared = self._draw(ref)
            if alpha == 1.0:
                return self._shared
            innovation = self._draw(ref)
            # alpha is the Pearson correlation with the shared trajectory anchor.
            # The sqrt term keeps Var(epsilon) exactly one for every iteration.
            return self._shared.mul(alpha).add(
                innovation, alpha=math.sqrt(1.0 - alpha * alpha)
            )

        pair_index = int(iteration) // 2
        if self._pair is None or pair_index != self._pair_index:
            self._pair = self._draw(ref)
            self._pair_index = pair_index
        return self._pair if int(iteration) % 2 == 0 else -self._pair


def training_noise_like(trainer, reference: torch.Tensor) -> torch.Tensor:
    """Return this MNT iteration's noise, or the legacy independent draw."""
    noise = getattr(trainer, "_active_mnt_noise", None)
    if noise is None:
        return torch.randn_like(reference)
    if tuple(noise.shape) != tuple(reference.shape):
        raise ValueError(
            f"MNT noise shape {tuple(noise.shape)} does not match clean input "
            f"shape {tuple(reference.shape)}"
        )
    return noise.to(device=reference.device, dtype=reference.dtype, non_blocking=True)
