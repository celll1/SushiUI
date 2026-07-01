"""First Block Cache (FBCache) — dynamic per-step caching for DiT inference.

FBCache exploits the temporal redundancy of the denoising trajectory: across
adjacent steps the transformer's hidden-state residual changes slowly. Instead of a
fixed skip schedule (that is Spectrum, ``spectrum_forecaster.py``), FBCache decides
DYNAMICALLY, per step, from a cheap indicator:

  1. Run only the FIRST transformer block and take its residual
     ``r1 = hidden_after_block0 - hidden_before_block0`` (on the image stream).
  2. If the relative L1 change of ``r1`` vs the previous step's ``r1`` is below a
     threshold, REUSE the cached full-transformer residual and SKIP all remaining
     blocks: ``hidden_out = hidden_before_block0 + cached_full_residual``.
  3. Otherwise run the remaining blocks and refresh the cached full residual
     ``cached_full_residual = hidden_out - hidden_before_block0``.

Reference (MIT): chengzeyi/Comfy-WaveSpeed and chengzeyi/ParaAttention (first-block
cache). This is a clean-room reimplementation of the published algorithm; only the
first-block-residual indicator + relative-L1 threshold + residual reuse are used.

Model-agnostic: the per-architecture forward decides WHICH tensor is the indicator
(the image-stream first-block residual) and WHAT object to cache (the final residual,
possibly a tuple for dual-stream models). This module only owns the decision + store.

FBCache and Spectrum both target the same trajectory redundancy and are treated as
MUTUALLY EXCLUSIVE (the pipeline enables at most one); combining them would compound
error and feed FBCache-approximated outputs into Spectrum's polynomial fit.
"""

import torch


class FirstBlockCache:
    """Owns the FBCache decision + cached residual across denoising steps.

    One instance per generation. ``use_cache`` is called once per step with the
    first-block image residual; ``store``/``get`` hold the reusable full residual.
    """

    def __init__(self, threshold: float, warmup_steps: int = 0):
        self.threshold = float(threshold)
        self.warmup_steps = int(warmup_steps)
        self._prev_indicator = None   # previous step's first-block residual
        self._cache = None            # cached full residual (arch-defined object)
        self.n_hits = 0
        self.n_miss = 0

    @staticmethod
    def _rel_l1(cur: torch.Tensor, prev: torch.Tensor) -> float:
        """Relative L1 change ||cur-prev||_1 / ||prev||_1 (mean-reduced), as a float."""
        denom = prev.abs().mean()
        if float(denom) == 0.0:
            return float("inf")
        return float((cur - prev).abs().mean() / denom)

    def use_cache(self, indicator: torch.Tensor, step_idx: int) -> bool:
        """Decide whether to reuse the cached residual and skip the remaining blocks.

        ``indicator`` is the first block's image-stream residual for this step.
        Returns True only after warmup, when a previous indicator + a cache exist and
        the relative change is below the threshold. Always records ``indicator`` as
        the new previous value (compared next step)."""
        can = (
            step_idx >= self.warmup_steps
            and self._prev_indicator is not None
            and self._cache is not None
            and self._rel_l1(indicator, self._prev_indicator) < self.threshold
        )
        self._prev_indicator = indicator
        if can:
            self.n_hits += 1
        else:
            self.n_miss += 1
        return can

    def store(self, cache_obj) -> None:
        """Store the freshly-computed full residual (reused on future cache hits)."""
        self._cache = cache_obj

    def get(self):
        """Return the cached full residual."""
        return self._cache


def fbcache_active(params) -> bool:
    """Whether FBCache should run: enabled with a positive threshold. Mutually
    exclusive with Spectrum -- the caller must not enable both (guarded upstream)."""
    return bool(params.get("fbcache_enable", False)) and float(params.get("fbcache_threshold", 0.0)) > 0.0


def build_fbcache(params, label: str = ""):
    """Build a FirstBlockCache from generation params, or None when inactive.

    Params: ``fbcache_enable`` (bool), ``fbcache_threshold`` (relative-L1 residual
    threshold; higher = more skips/faster, lower = safer), ``fbcache_warmup_steps``
    (always-compute the first N steps)."""
    if not fbcache_active(params):
        return None
    fb = FirstBlockCache(
        threshold=float(params.get("fbcache_threshold", 0.12)),
        warmup_steps=int(params.get("fbcache_warmup_steps", 1)),
    )
    print(f"[FBCache] {label} enabled: threshold={fb.threshold}, warmup={fb.warmup_steps}")
    return fb
