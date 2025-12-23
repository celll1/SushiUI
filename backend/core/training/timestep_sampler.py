"""
Timestep Sampling Strategies for Diffusion Training.

This module provides an extensible framework for sampling timesteps during training.
Currently implements uniform sampling, with support for future distributions.

Future extensions:
- NormalTimestepSampler: Sample from normal distribution
- LogNormalTimestepSampler: Sample from log-normal distribution
- BetaTimestepSampler: Sample from beta distribution
- CustomTimestepSampler: Sample from custom weighted distribution
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any


class TimestepSampler(ABC):
    """
    Abstract base class for timestep sampling strategies.

    Timestep samplers control how timesteps are sampled during training,
    allowing for different distributions and weighting schemes.

    Example:
        >>> sampler = UniformTimestepSampler(min_timestep=0.0, max_timestep=1.0)
        >>> timesteps = sampler.sample(batch_size=4, device=torch.device("cuda"))
        >>> print(timesteps)  # Tensor([0.234, 0.876, 0.512, 0.099])
    """

    def __init__(self, min_timestep: float = 0.0, max_timestep: float = 1.0):
        """
        Initialize timestep sampler.

        Args:
            min_timestep: Minimum timestep value (0.0-1.0 for Flow Matching)
            max_timestep: Maximum timestep value (0.0-1.0 for Flow Matching)

        Raises:
            ValueError: If timestep range is invalid
        """
        if not (0.0 <= min_timestep <= 1.0 and 0.0 <= max_timestep <= 1.0):
            raise ValueError(
                f"min/max_timestep must be in [0, 1], got [{min_timestep}, {max_timestep}]"
            )
        if min_timestep >= max_timestep:
            raise ValueError(
                f"min_timestep ({min_timestep}) must be < max_timestep ({max_timestep})"
            )

        self.min_timestep = min_timestep
        self.max_timestep = max_timestep

    @abstractmethod
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps for a batch.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on

        Returns:
            Tensor of shape [batch_size] with timesteps in [min_timestep, max_timestep]
        """
        pass

    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'TimestepSampler':
        """
        Create timestep sampler from configuration dictionary.

        Args:
            config: Timestep sampling configuration dict with keys:
                - distribution: str (default: "uniform")
                - min_timestep: float (default: 0.0)
                - max_timestep: float (default: 1.0)
                Future keys:
                - mean, std: for normal/lognormal
                - alpha, beta: for beta distribution
                - custom_weights: for custom distribution

        Returns:
            TimestepSampler instance

        Raises:
            ValueError: If distribution type is unknown

        Example:
            >>> config = {"distribution": "uniform", "min_timestep": 0.2, "max_timestep": 0.8}
            >>> sampler = TimestepSampler.from_config(config)
        """
        distribution = config.get("distribution", "uniform").lower()
        min_timestep = config.get("min_timestep", 0.0)
        max_timestep = config.get("max_timestep", 1.0)

        if distribution == "uniform":
            return UniformTimestepSampler(min_timestep, max_timestep)
        elif distribution == "normal":
            mean = config.get("mean", 0.5)
            std = config.get("std", 0.2)
            return NormalTimestepSampler(min_timestep, max_timestep, mean, std)
        elif distribution == "lognormal":
            mean = config.get("mean", 0.0)
            std = config.get("std", 1.0)
            return LogNormalTimestepSampler(min_timestep, max_timestep, mean, std)
        elif distribution == "beta":
            alpha = config.get("alpha", 2.0)
            beta = config.get("beta", 2.0)
            return BetaTimestepSampler(min_timestep, max_timestep, alpha, beta)
        elif distribution == "custom":
            weights = config.get("custom_weights", [])
            return CustomTimestepSampler(min_timestep, max_timestep, weights)
        else:
            raise ValueError(
                f"Unknown timestep distribution: '{distribution}'. "
                f"Supported: 'uniform', 'normal', 'lognormal', 'beta', 'custom'"
            )


class UniformTimestepSampler(TimestepSampler):
    """
    Uniform timestep sampling (current default implementation).

    Samples timesteps uniformly from [min_timestep, max_timestep].
    This is the standard approach for Flow Matching training.

    Example:
        >>> sampler = UniformTimestepSampler(min_timestep=0.1, max_timestep=0.9)
        >>> timesteps = sampler.sample(batch_size=2, device=torch.device("cpu"))
        >>> print(timesteps)  # e.g., Tensor([0.345, 0.782])
        >>> assert torch.all((timesteps >= 0.1) & (timesteps <= 0.9))
    """

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps uniformly from [min_timestep, max_timestep].

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on

        Returns:
            Tensor of shape [batch_size] with uniformly distributed timesteps
        """
        # Sample from [0, 1] uniformly
        timesteps = torch.rand(batch_size, device=device)

        # Scale to [min_timestep, max_timestep]
        timesteps = timesteps * (self.max_timestep - self.min_timestep) + self.min_timestep

        return timesteps


# ============================================================
# Future Implementations (not implemented yet)
# ============================================================

class NormalTimestepSampler(TimestepSampler):
    """
    Sample timesteps from normal (Gaussian) distribution.

    Useful for focusing training on specific timestep ranges while still
    covering the full range with lower probability.
    """

    def __init__(
        self,
        min_timestep: float = 0.0,
        max_timestep: float = 1.0,
        mean: float = 0.5,
        std: float = 0.2
    ):
        super().__init__(min_timestep, max_timestep)
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from normal distribution, clamped to [min, max]."""
        timesteps = torch.randn(batch_size, device=device) * self.std + self.mean
        timesteps = torch.clamp(timesteps, self.min_timestep, self.max_timestep)
        return timesteps


class LogNormalTimestepSampler(TimestepSampler):
    """
    Sample timesteps from log-normal distribution.

    Useful for emphasizing early or late timesteps depending on parameters.
    """

    def __init__(
        self,
        min_timestep: float = 0.0,
        max_timestep: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0
    ):
        super().__init__(min_timestep, max_timestep)
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from log-normal distribution, scaled to [min, max]."""
        timesteps = torch.exp(torch.randn(batch_size, device=device) * self.std + self.mean)
        # Normalize to [0, 1] then scale to [min, max]
        timesteps = (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min())
        timesteps = timesteps * (self.max_timestep - self.min_timestep) + self.min_timestep
        return timesteps


class BetaTimestepSampler(TimestepSampler):
    """
    Sample timesteps from beta distribution.

    Beta distribution allows flexible control over timestep distribution shape.
    - alpha=beta=1: Uniform
    - alpha>1, beta>1: Bell-shaped (concentrated in middle)
    - alpha<1, beta<1: U-shaped (concentrated at edges)
    """

    def __init__(
        self,
        min_timestep: float = 0.0,
        max_timestep: float = 1.0,
        alpha: float = 2.0,
        beta: float = 2.0
    ):
        super().__init__(min_timestep, max_timestep)
        self.alpha = alpha
        self.beta = beta

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from beta distribution."""
        from torch.distributions import Beta
        beta_dist = Beta(self.alpha, self.beta)
        timesteps = beta_dist.sample((batch_size,)).to(device)
        # Scale to [min, max]
        timesteps = timesteps * (self.max_timestep - self.min_timestep) + self.min_timestep
        return timesteps


class CustomTimestepSampler(TimestepSampler):
    """
    Sample timesteps from custom weighted distribution.

    Allows arbitrary weighting of timestep ranges for targeted training.
    """

    def __init__(
        self,
        min_timestep: float = 0.0,
        max_timestep: float = 1.0,
        weights: list = None
    ):
        super().__init__(min_timestep, max_timestep)
        if weights is None or len(weights) == 0:
            raise ValueError("CustomTimestepSampler requires non-empty weights list")
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = self.weights / self.weights.sum()  # Normalize

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from custom distribution using provided weights."""
        # Create bins based on weights
        num_bins = len(self.weights)
        bins = torch.linspace(self.min_timestep, self.max_timestep, num_bins + 1)

        # Sample bin indices according to weights
        bin_indices = torch.multinomial(
            self.weights.to(device), batch_size, replacement=True
        )

        # Sample uniformly within selected bins
        timesteps = bins[bin_indices] + torch.rand(batch_size, device=device) * (
            bins[bin_indices + 1] - bins[bin_indices]
        )

        return timesteps
