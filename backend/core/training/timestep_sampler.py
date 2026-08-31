"""
Timestep Sampling Strategies for Diffusion Training.

This module provides an extensible framework for sampling timesteps during training.

Supported distributions:
- UniformTimestepSampler: Standard uniform distribution [min, max]
- NormalTimestepSampler: Gaussian distribution, clamped to [min, max]
- LogitNormalTimestepSampler: Logit-normal (sigmoid of normal), used in FLUX/SD3
- BetaTimestepSampler: Beta distribution for flexible shape control
- CustomTimestepSampler: Arbitrary weighted distribution

NOTE on terminology:
- "logit_normal" / "lognormal" in this codebase refers to sigmoid(normal(mean, std))
- This matches sd-scripts, ai-toolkit, and diffusers implementations
- It is NOT the mathematical log-normal distribution (exp of normal)

Timestep interpretation is ARCHITECTURE-DEPENDENT, not a property of this
module. This sampler only draws a value in [min_timestep, max_timestep] -- it
does not decide which end is "clean" and which is "noise". Each architecture's
``train_step`` (``core/training/ops/*.py``) fixes that mapping, and the
authoritative declaration per architecture is
``core.training.arch.base_arch.ArchHandler.timestep_convention`` /
``resolve_timestep_convention()``:

- "t0" (most architectures: SD3/FLUX/Z-Image/FLUX.2/Krea 2/Ideogram 4/Lens/
  Anima/LTX-2.3/MiniMax-H3/ACE-Step, and SD1.5/SDXL when noise_process="flow"):
  t=0 is the clean image, t=1 is pure noise.
- "t1" (SenseNova, MiniT2I, and SD1.5/SDXL when noise_process="ddpm"): t=1 is
  the clean image, t=0 is pure noise -- the INVERSE of the above.

Do not assume "t0" when configuring ``mean``/``std`` for a "t1" architecture:
the sign that biases toward "clean" or "noisy" flips.
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

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        """Quantile function: map u in [0,1) to a timestep, same law as ``sample``.

        Only implemented where the quantile is available in closed form. A
        subclass without one raises, and ``sample_stratified`` falls back to
        independent draws rather than silently sampling a different law.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form quantile function"
        )

    def sample_stratified(
        self, n_strata: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """``n_strata`` draws per batch element, one from each equal-probability stratum.

        Returns ``[n_strata, batch_size]``. Row i is the draw from stratum i, so a
        caller running an MNT window indexes row ``mnt_idx``.

        Why: a multi-noise-timestep window is a Monte-Carlo estimate of a
        one-dimensional integral over t, and at batch size 1 the t draw is very
        nearly the ONLY source of within-window variance. Drawing the strata
        u_i = (i + v_i)/n with v_i ~ U(0,1) iid is proportional stratified
        sampling: the marginal law of each row is unchanged (so the estimator
        stays unbiased for the SAME objective, and the architecture's configured
        density is preserved exactly), while the between-strata component of the
        variance is removed. Stratified sampling never increases the variance of
        the mean -- that is a theorem, not a tuning choice, so this needs no
        hyperparameter and carries no quality risk.

        Reference for the construction: Kingma, Salimans, Poole & Ho,
        "Variational Diffusion Models", NeurIPS 2021, appendix I.1, which uses
        the low-discrepancy variant u_i = mod(u_0 + i/n, 1) (one shared jitter).
        The independent-jitter form used here is the one with the general
        variance theorem attached; the shared-jitter lattice has lower variance
        for smooth integrands but no guarantee off that assumption.

        The row ORDER is permuted before returning. Callers give mnt_idx 0 a
        special meaning (SenseNova reuses the batch's already-built prefix on
        iteration 0 and recomputes it after, and debug_latents saves there), so
        a monotone stratum order would permanently bind those behaviours to the
        noisiest stratum.
        """
        if n_strata < 1:
            raise ValueError(f"n_strata must be >= 1, got {n_strata}")
        edges = torch.arange(n_strata, device=device, dtype=torch.float32).unsqueeze(1)
        u = (edges + torch.rand(n_strata, batch_size, device=device)) / float(n_strata)
        t = self.icdf(u)
        return t[torch.randperm(n_strata, device=device)]

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
        elif distribution in ("lognormal", "logit_normal", "logit-normal", "logitnormal"):
            mean = config.get("mean", 0.0)
            std = config.get("std", 1.0)
            return LogitNormalTimestepSampler(min_timestep, max_timestep, mean, std)
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
                f"Supported: 'uniform', 'normal', 'logit_normal' (or 'lognormal'), 'beta', 'custom'"
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

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        return u * (self.max_timestep - self.min_timestep) + self.min_timestep


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

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        # Clamping is monotone, so the quantile of the clamped variable is the
        # clamped quantile -- the same law ``sample`` produces, atoms included.
        z = torch.special.ndtri(u.clamp(1e-7, 1 - 1e-7))
        return torch.clamp(z * self.std + self.mean, self.min_timestep, self.max_timestep)


class LogitNormalTimestepSampler(TimestepSampler):
    """
    Sample timesteps from logit-normal distribution (sd-scripts/ai-toolkit/diffusers style).

    This is the standard "logit_normal" distribution used in FLUX/SD3 training.
    It applies sigmoid to a normal distribution to get values in [0, 1].

    Formula: timestep = sigmoid(normal(mean, std))

    Parameter effects (in terms of the RAW [0,1] value this sampler emits,
    independent of which end an architecture calls "clean" -- see the module
    docstring's "t0"/"t1" convention before reading "high"/"low" as
    "noisy"/"clean"):
    - mean=0, std=1: Centered around 0.5, smooth bell curve
    - mean=-1, std=1: Biased toward LOW output values (~0.27 mean)
    - mean=1, std=1: Biased toward HIGH output values (~0.73 mean)
    - mean=0, std=0.5: Very concentrated around 0.5
    - mean=0, std=2: Spread out but still [0,1] bounded

    Whether "low" means clean or noisy depends on the consuming architecture's
    convention ("t0": low=clean/high=noisy; "t1": low=noisy/high=clean -- see
    the module docstring). The SAME mean sign biases toward opposite ends of
    the noise schedule depending on that convention.

    Example:
        >>> # Bias toward high output values
        >>> sampler = LogitNormalTimestepSampler(mean=1.0, std=1.0)
        >>> timesteps = sampler.sample(1000, torch.device("cpu"))
        >>> print(f"Mean: {timesteps.mean():.3f}")  # ~0.73

        >>> # Bias toward low output values
        >>> sampler = LogitNormalTimestepSampler(mean=-1.0, std=1.0)
        >>> timesteps = sampler.sample(1000, torch.device("cpu"))
        >>> print(f"Mean: {timesteps.mean():.3f}")  # ~0.27
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
        """
        Sample from logit-normal distribution.

        Process:
        1. Sample from normal distribution N(mean, std)
        2. Apply sigmoid to get [0, 1]
        3. Scale to [min_timestep, max_timestep]

        Returns:
            Timesteps in [min_timestep, max_timestep]
        """
        # Sample from normal distribution
        u = torch.randn(batch_size, device=device) * self.std + self.mean

        # Apply sigmoid to get [0, 1] - this is the "logit-normal" transformation
        timesteps = torch.sigmoid(u)

        # Scale to [min_timestep, max_timestep]
        timesteps = timesteps * (self.max_timestep - self.min_timestep) + self.min_timestep

        return timesteps

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        # sigmoid and the affine rescale are both monotone, so composing them
        # with the normal quantile gives this sampler's quantile exactly.
        z = torch.special.ndtri(u.clamp(1e-7, 1 - 1e-7))
        t = torch.sigmoid(z * self.std + self.mean)
        return t * (self.max_timestep - self.min_timestep) + self.min_timestep


# Alias for backward compatibility (config may use "lognormal")
LogNormalTimestepSampler = LogitNormalTimestepSampler


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

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear inverse CDF of the binned density.

        ``sample`` draws a bin by weight and then uniformly inside it, so the CDF
        is linear across each bin with the bin's weight as its rise -- inverting
        it is a bucketize plus one linear interpolation.
        """
        device = u.device
        n = len(self.weights)
        w = self.weights.to(device=device, dtype=torch.float32)
        edges = torch.linspace(self.min_timestep, self.max_timestep, n + 1, device=device)
        cum = torch.cat([torch.zeros(1, device=device), torch.cumsum(w, 0)])
        cum[-1] = 1.0  # absorb float error so u just below 1 still lands in bin n-1
        idx = (torch.bucketize(u.contiguous(), cum, right=True) - 1).clamp(0, n - 1)
        span = (cum[idx + 1] - cum[idx]).clamp_min(1e-12)
        frac = ((u - cum[idx]) / span).clamp(0.0, 1.0)
        return edges[idx] + frac * (edges[idx + 1] - edges[idx])
