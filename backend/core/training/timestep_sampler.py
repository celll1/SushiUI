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
import math
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


# ============================================================
# Canonicalisation, serialisation and resume-time morphing
# See docs/guides/TIMESTEP_DISTRIBUTION_MORPH_DESIGN.md
# ============================================================

SAMPLER_EXPR_VERSION = 1

#: distribution -> {param: default}. The canonical key set per distribution;
#: anything else in a config is irrelevant to the law and is dropped before an
#: equality check, so an omitted default and its explicit value compare equal.
_DISTRIBUTION_PARAMS: Dict[str, Dict[str, Any]] = {
    "uniform": {},
    "normal": {"mean": 0.5, "std": 0.2},
    "logit_normal": {"mean": 0.0, "std": 1.0},
    "beta": {"alpha": 2.0, "beta": 2.0},
    "custom": {"custom_weights": None},
}

_DISTRIBUTION_ALIASES = {
    "lognormal": "logit_normal",
    "logit-normal": "logit_normal",
    "logitnormal": "logit_normal",
}

MORPH_CURVES = ("cosine", "linear")
MORPH_INTERPOLATIONS = ("quantile", "mixture")

#: A fifth in-flight retarget flattens instead of nesting deeper.
MAX_MORPH_NESTING = 4
_FLATTEN_TABLE_POINTS = 4097
_FLATTEN_SAMPLES = 262144
_FLATTEN_SEED = 20260913


def normalize_distribution_name(name: Any) -> str:
    key = str(name or "uniform").lower()
    return _DISTRIBUTION_ALIASES.get(key, key)


def _round(value: float) -> float:
    # Canonical configs are compared for equality and round-tripped through
    # JSON/YAML; 12 digits is far below any meaningful sampler difference and
    # above the representation noise those two formats introduce.
    return float(round(float(value), 12))


def canonicalize_timestep_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """The law a config describes, with aliases resolved and defaults materialised.

    Two configs describing the same distribution canonicalise to equal dicts, so
    a resume that merely rewrote ``lognormal`` as ``logit_normal`` does not start
    a morph. ``morph`` is dropped: it describes how to reach a law, not the law.
    """
    if not isinstance(config, dict):
        raise ValueError(f"timestep_sampling config must be a dict, got {type(config).__name__}")
    distribution = normalize_distribution_name(config.get("distribution", "uniform"))
    if distribution not in _DISTRIBUTION_PARAMS:
        raise ValueError(
            f"Unknown timestep distribution: '{config.get('distribution')}'. "
            f"Supported: {', '.join(sorted(_DISTRIBUTION_PARAMS))}"
        )
    canon: Dict[str, Any] = {
        "distribution": distribution,
        "min_timestep": _round(config.get("min_timestep", 0.0)),
        "max_timestep": _round(config.get("max_timestep", 1.0)),
    }
    for key, default in _DISTRIBUTION_PARAMS[distribution].items():
        value = config.get(key, default)
        if key == "custom_weights":
            if not value:
                raise ValueError("custom timestep distribution requires non-empty custom_weights")
            weights = [float(w) for w in value]
            if any(not math.isfinite(w) or w < 0 for w in weights):
                raise ValueError("custom_weights must contain only finite, non-negative values")
            total = float(sum(weights))
            if total <= 0:
                raise ValueError("custom_weights must sum to a positive value")
            canon[key] = [_round(w / total) for w in weights]
        else:
            canon[key] = _round(value)
    validate_timestep_config(canon)
    return canon


def validate_timestep_config(canon: Dict[str, Any]) -> None:
    """Refuse a config that would sample nothing, or nothing finite."""
    for key in ("min_timestep", "max_timestep"):
        if not math.isfinite(canon[key]):
            raise ValueError(f"{key} must be finite, got {canon[key]}")
    if not (0.0 <= canon["min_timestep"] < canon["max_timestep"] <= 1.0):
        raise ValueError(
            f"timestep range must satisfy 0 <= min < max <= 1, got "
            f"[{canon['min_timestep']}, {canon['max_timestep']}]"
        )
    if "std" in canon:
        if not math.isfinite(canon["std"]) or canon["std"] <= 0:
            raise ValueError(f"timestep std must be finite and > 0, got {canon['std']}")
    if "mean" in canon and not math.isfinite(canon["mean"]):
        raise ValueError(f"timestep mean must be finite, got {canon['mean']}")
    for key in ("alpha", "beta"):
        if key in canon and (not math.isfinite(canon[key]) or canon[key] <= 0):
            raise ValueError(f"Beta {key} must be finite and > 0, got {canon[key]}")


def validate_morph_config(morph: Dict[str, Any]) -> None:
    """Refuse a morph block the trainer could not carry out."""
    if not isinstance(morph, dict):
        raise ValueError(f"timestep_sampling.morph must be a dict, got {type(morph).__name__}")
    if not morph.get("enabled"):
        return
    steps = morph.get("steps", 0)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError(f"timestep_sampling.morph.steps must be a positive integer, got {steps!r}")
    curve = str(morph.get("curve", "cosine")).lower()
    if curve not in MORPH_CURVES:
        raise ValueError(f"timestep_sampling.morph.curve must be one of {MORPH_CURVES}, got {curve!r}")
    interpolation = str(morph.get("interpolation", "quantile")).lower()
    if interpolation not in MORPH_INTERPOLATIONS:
        raise ValueError(
            f"timestep_sampling.morph.interpolation must be one of "
            f"{MORPH_INTERPOLATIONS}, got {interpolation!r}"
        )
    if morph.get("from") is not None:
        canonicalize_timestep_config(morph["from"])


def sampler_to_config(sampler: 'TimestepSampler') -> Dict[str, Any]:
    """The canonical config of a plain (non-morph) sampler instance."""
    config: Dict[str, Any] = {
        "min_timestep": sampler.min_timestep,
        "max_timestep": sampler.max_timestep,
    }
    if isinstance(sampler, UniformTimestepSampler):
        config["distribution"] = "uniform"
    elif isinstance(sampler, LogitNormalTimestepSampler):
        config.update(distribution="logit_normal", mean=sampler.mean, std=sampler.std)
    elif isinstance(sampler, NormalTimestepSampler):
        config.update(distribution="normal", mean=sampler.mean, std=sampler.std)
    elif isinstance(sampler, BetaTimestepSampler):
        config.update(distribution="beta", alpha=sampler.alpha, beta=sampler.beta)
    elif isinstance(sampler, CustomTimestepSampler):
        config.update(distribution="custom",
                      custom_weights=[float(w) for w in sampler.weights.tolist()])
    else:
        raise TypeError(f"{type(sampler).__name__} has no plain config form")
    return canonicalize_timestep_config(config)


def sampler_expr(sampler: 'TimestepSampler') -> Dict[str, Any]:
    """Serialise a sampler (plain, morphing or flattened) for checkpoint state."""
    if hasattr(sampler, "expr"):
        return sampler.expr()
    return {"version": SAMPLER_EXPR_VERSION, "kind": "config",
            "config": sampler_to_config(sampler)}


def build_sampler_from_expr(expr: Dict[str, Any]) -> 'TimestepSampler':
    """Rebuild a sampler from ``sampler_expr``. Refuses a version it cannot read."""
    if not isinstance(expr, dict):
        raise ValueError(f"sampler expression must be a dict, got {type(expr).__name__}")
    version = int(expr.get("version", 0))
    if version != SAMPLER_EXPR_VERSION:
        raise ValueError(
            f"sampler expression version {version} is not readable by this build "
            f"(expected {SAMPLER_EXPR_VERSION})"
        )
    kind = expr.get("kind")
    if kind == "config":
        return TimestepSampler.from_config(dict(expr["config"]))
    if kind == "quantile_table":
        return QuantileTableSampler(
            table=list(expr["table"]),
            min_timestep=float(expr["min_timestep"]),
            max_timestep=float(expr["max_timestep"]),
            seed=expr.get("seed"),
        )
    if kind == "morph":
        sampler = MorphingTimestepSampler(
            source=build_sampler_from_expr(expr["source"]),
            target=build_sampler_from_expr(expr["target"]),
            steps=int(expr["steps"]),
            curve=str(expr["curve"]),
            interpolation=str(expr["interpolation"]),
            start_update=int(expr["start_update"]),
        )
        if expr.get("frozen_lam") is not None:
            sampler = sampler.freeze(float(expr["frozen_lam"]))
        return sampler
    raise ValueError(f"unknown sampler expression kind: {kind!r}")


def sampler_depth(sampler: 'TimestepSampler') -> int:
    """Morph nesting depth; 0 for a plain sampler."""
    if isinstance(sampler, MorphingTimestepSampler):
        return 1 + max(sampler_depth(sampler.source), sampler_depth(sampler.target))
    return 0


def quantile_table(sampler: 'TimestepSampler', points: int = 1025,
                   seed: int = _FLATTEN_SEED) -> torch.Tensor:
    """``points`` quantiles of ``sampler`` on a uniform u-grid, exact where it has an icdf.

    Falls back to a deterministically seeded sample quantile for a sampler with
    no quantile function (Beta, mixture). The fallback runs inside an isolated
    CPU RNG context: it never advances the training stream and never touches the
    CUDA generator.
    """
    u = torch.linspace(0.0, 1.0, points, dtype=torch.float64)
    u = u.clamp(0.5 / _FLATTEN_SAMPLES, 1.0 - 0.5 / _FLATTEN_SAMPLES)
    try:
        return sampler.icdf(u.to(torch.float32)).to(torch.float64)
    except NotImplementedError:
        pass
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        draws = sampler.sample(_FLATTEN_SAMPLES, torch.device("cpu")).to(torch.float64)
    finally:
        torch.random.set_rng_state(rng_state)
    return torch.quantile(draws, u)


class QuantileTableSampler(TimestepSampler):
    """A law pinned as a piecewise-linear quantile table.

    Used when a morph source would nest deeper than ``MAX_MORPH_NESTING``. The
    TABLE is persisted, not the seed that produced it, so a later build
    reproduces the same law even if the sampling path changes.
    """

    def __init__(self, table, min_timestep: float = 0.0, max_timestep: float = 1.0,
                 seed: Any = None):
        super().__init__(min_timestep, max_timestep)
        values = torch.tensor([float(v) for v in table], dtype=torch.float32)
        if values.numel() < 2:
            raise ValueError("quantile table needs at least 2 points")
        self.table = torch.cummax(values, dim=0).values
        self.seed = seed

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        n = self.table.numel()
        table = self.table.to(device=u.device, dtype=u.dtype)
        pos = u.clamp(0.0, 1.0) * (n - 1)
        lo = pos.floor().long().clamp(0, n - 1)
        hi = (lo + 1).clamp(0, n - 1)
        frac = pos - lo.to(pos.dtype)
        return table[lo] + frac * (table[hi] - table[lo])

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.icdf(torch.rand(batch_size, device=device))

    def expr(self) -> Dict[str, Any]:
        return {
            "version": SAMPLER_EXPR_VERSION,
            "kind": "quantile_table",
            "table": [float(v) for v in self.table.tolist()],
            "min_timestep": self.min_timestep,
            "max_timestep": self.max_timestep,
            "seed": self.seed,
        }


def flatten_sampler(sampler: 'TimestepSampler',
                    points: int = _FLATTEN_TABLE_POINTS,
                    seed: int = _FLATTEN_SEED) -> QuantileTableSampler:
    """Approximate ``sampler`` by a quantile table (see ``QuantileTableSampler``)."""
    table = quantile_table(sampler, points=points, seed=seed)
    return QuantileTableSampler(
        table=table.tolist(),
        min_timestep=sampler.min_timestep,
        max_timestep=sampler.max_timestep,
        seed=seed,
    )


def morph_lambda(update_step: int, start_update: int, steps: int,
                 curve: str = "cosine") -> float:
    """λ in [0,1] at a successful-optimizer-update position."""
    import math

    steps = int(steps)
    if steps <= 0:
        return 1.0
    p = (int(update_step) - int(start_update)) / float(steps)
    p = min(1.0, max(0.0, p))
    if str(curve).lower() == "linear":
        return p
    return 0.5 * (1.0 - math.cos(math.pi * p))


class MorphingTimestepSampler(TimestepSampler):
    """Moves the sampled law from ``source`` to ``target`` over ``steps`` optimizer updates.

    ``quantile`` interpolates the two quantile functions (the Wasserstein
    geodesic: equally ranked mass is coupled and moves between the endpoints),
    which stays monotone in ``u`` and therefore keeps ``icdf`` -- and with it
    ``sample_stratified`` -- available. ``mixture`` draws from one endpoint or
    the other with probability λ; its quantile has no closed form, so it
    deliberately exposes none.

    The position axis is SUCCESSFUL OPTIMIZER UPDATES, set from outside by
    ``set_optimizer_update_step``; see the design doc for why ``global_step`` is
    not usable here.
    """

    def __init__(self, source: 'TimestepSampler', target: 'TimestepSampler',
                 steps: int, curve: str = "cosine", interpolation: str = "quantile",
                 start_update: int = 0):
        super().__init__(
            min(source.min_timestep, target.min_timestep),
            max(source.max_timestep, target.max_timestep),
        )
        if int(steps) <= 0:
            raise ValueError(f"morph steps must be positive, got {steps}")
        curve = str(curve).lower()
        if curve not in MORPH_CURVES:
            raise ValueError(f"morph curve must be one of {MORPH_CURVES}, got {curve!r}")
        interpolation = str(interpolation).lower()
        if interpolation not in MORPH_INTERPOLATIONS:
            raise ValueError(
                f"morph interpolation must be one of {MORPH_INTERPOLATIONS}, "
                f"got {interpolation!r}")

        self.source = source
        self.target = target
        self.steps = int(steps)
        self.curve = curve
        self.requested_interpolation = interpolation
        self.start_update = int(start_update)
        self._frozen_lam: Any = None
        self._update_step = int(start_update)
        self.fallback_reason: Any = None

        self.interpolation = interpolation
        if interpolation == "quantile" and not self._endpoints_have_icdf():
            self.interpolation = "mixture"
            self.fallback_reason = (
                "quantile interpolation needs a quantile function on both endpoints; "
                f"source={type(source).__name__}, target={type(target).__name__}"
            )
        self._mixture_cdf_tables: Any = None

    def _endpoints_have_icdf(self) -> bool:
        probe = torch.tensor([0.5])
        for endpoint in (self.source, self.target):
            try:
                endpoint.icdf(probe)
            except Exception:
                return False
        return True

    # -- position -------------------------------------------------------

    def set_optimizer_update_step(self, update_step: int) -> None:
        self._update_step = int(update_step)

    @property
    def update_step(self) -> int:
        return self._update_step

    @property
    def lam(self) -> float:
        if self._frozen_lam is not None:
            return float(self._frozen_lam)
        return morph_lambda(self._update_step, self.start_update, self.steps, self.curve)

    def lam_at(self, update_step: int) -> float:
        if self._frozen_lam is not None:
            return float(self._frozen_lam)
        return morph_lambda(update_step, self.start_update, self.steps, self.curve)

    def is_finished(self) -> bool:
        if self._frozen_lam is not None:
            return False
        return self._update_step >= self.start_update + self.steps

    def freeze(self, lam: Any = None) -> 'MorphingTimestepSampler':
        """A copy pinned at λ -- the law currently in force, usable as a new source."""
        frozen = MorphingTimestepSampler(
            source=self.source, target=self.target, steps=self.steps,
            curve=self.curve, interpolation=self.requested_interpolation,
            start_update=self.start_update,
        )
        frozen._frozen_lam = float(self.lam if lam is None else lam)
        frozen._update_step = self._update_step
        return frozen

    # -- sampling -------------------------------------------------------

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.is_finished():
            return self.target.sample(batch_size, device)
        if self.interpolation == "quantile":
            return self.icdf(torch.rand(batch_size, device=device))
        lam = self.lam
        pick = torch.rand(batch_size, device=device) < lam
        drawn = self.source.sample(batch_size, device)
        if bool(pick.any()):
            drawn = torch.where(pick, self.target.sample(batch_size, device), drawn)
        return drawn

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        if self.is_finished():
            return self.target.icdf(u)
        if self.interpolation != "quantile":
            raise NotImplementedError(
                "mixture morph interpolation has no closed-form quantile function")
        lam = self.lam
        return (1.0 - lam) * self.source.icdf(u) + lam * self.target.icdf(u)

    # -- diagnostics ----------------------------------------------------

    def median(self) -> float:
        """The active law's median, for the grad-t-cosine probe's split."""
        if self.is_finished():
            try:
                return float(self.target.icdf(torch.tensor([0.5])).item())
            except NotImplementedError:
                return float(quantile_table(self.target)[512].item())
        if self.interpolation == "quantile":
            return float(self.icdf(torch.tensor([0.5])).item())
        if self._mixture_cdf_tables is None:
            self._mixture_cdf_tables = (
                quantile_table(self.source), quantile_table(self.target))
        lam = self.lam
        source_table, target_table = self._mixture_cdf_tables

        def mixed_cdf(t: float) -> float:
            def cdf(table: torch.Tensor) -> float:
                return float(torch.searchsorted(
                    table, torch.tensor([t], dtype=table.dtype)
                ).item()) / float(table.numel())
            return (1.0 - lam) * cdf(source_table) + lam * cdf(target_table)

        lo, hi = self.min_timestep, self.max_timestep
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if mixed_cdf(mid) < 0.5:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def describe(self) -> str:
        return (f"{describe_sampler(self.source)} -> {describe_sampler(self.target)} "
                f"over {self.steps} updates from {self.start_update} "
                f"({self.curve}, {self.interpolation})")

    # -- serialisation --------------------------------------------------

    def expr(self) -> Dict[str, Any]:
        return {
            "version": SAMPLER_EXPR_VERSION,
            "kind": "morph",
            "source": sampler_expr(self.source),
            "target": sampler_expr(self.target),
            "steps": self.steps,
            "curve": self.curve,
            "interpolation": self.requested_interpolation,
            "start_update": self.start_update,
            "frozen_lam": (None if self._frozen_lam is None else float(self._frozen_lam)),
        }

    def state(self) -> Dict[str, Any]:
        """The ``timestep_morph`` record saved beside a checkpoint."""
        return {
            "version": SAMPLER_EXPR_VERSION,
            "start_update": self.start_update,
            "steps": self.steps,
            "curve": self.curve,
            "interpolation": self.requested_interpolation,
            "effective_interpolation": self.interpolation,
            "from": sampler_expr(self.source),
            "to": sampler_expr(self.target),
        }


def describe_sampler(sampler: 'TimestepSampler') -> str:
    """Short human-readable form, e.g. ``logit_normal(mean=0.5, std=1.0)``."""
    if isinstance(sampler, MorphingTimestepSampler):
        lam = sampler.lam
        return f"[morph {sampler.describe()} @ lam={lam:.3f}]"
    if isinstance(sampler, QuantileTableSampler):
        return f"quantile_table({sampler.table.numel()} points)"
    try:
        config = sampler_to_config(sampler)
    except TypeError:
        return type(sampler).__name__
    params = ", ".join(f"{k}={v}" for k, v in config.items()
                       if k not in ("distribution", "min_timestep", "max_timestep"))
    body = f"{config['distribution']}({params})" if params else config["distribution"]
    if (config["min_timestep"], config["max_timestep"]) != (0.0, 1.0):
        body += f"[{config['min_timestep']}, {config['max_timestep']}]"
    return body
