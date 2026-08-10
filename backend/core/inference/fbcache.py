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
(the visual-stream first-block residual) and WHAT object to cache (the final residual,
possibly a tuple for dual-stream models). This module only owns the decision + store.

FBCache and Spectrum both target the same trajectory redundancy and are treated as
MUTUALLY EXCLUSIVE (the pipeline enables at most one); combining them would compound
error and feed FBCache-approximated outputs into Spectrum's polynomial fit.
"""

import torch


class FirstBlockCache:
    """Owns the FBCache decision + cached residual across denoising steps.

    One instance per generation. ``use_cache`` is called once per step with the
    first-block visual residual; ``store``/``get`` hold the reusable full residual.
    """

    def __init__(
        self,
        threshold: float,
        warmup_steps: int = 0,
        *,
        max_consecutive_hits: int | None = None,
        total_steps: int | None = None,
        tail_steps: int = 0,
    ):
        self.threshold = float(threshold)
        self.warmup_steps = int(warmup_steps)
        self.max_consecutive_hits = (
            None if max_consecutive_hits is None else int(max_consecutive_hits)
        )
        self.total_steps = None if total_steps is None else int(total_steps)
        self.tail_steps = int(tail_steps)
        self._prev_indicator = None   # previous step's first-block residual
        self._prev_guard_indicator = None
        self._cache = None            # cached full residual (arch-defined object)
        self._consecutive_hits = 0
        self.n_hits = 0
        self.n_miss = 0

    @staticmethod
    def _rel_l1(cur: torch.Tensor, prev: torch.Tensor) -> float:
        """Relative L1 change ||cur-prev||_1 / ||prev||_1 (mean-reduced), as a float."""
        denom = prev.abs().mean()
        if float(denom) == 0.0:
            return float("inf")
        return float((cur - prev).abs().mean() / denom)

    @staticmethod
    def _max_group_rel_l1(cur: torch.Tensor, prev: torch.Tensor) -> float:
        """Largest relative-L1 change across ``cur``'s leading groups."""
        if cur.ndim < 2 or cur.shape != prev.shape:
            raise ValueError(
                "FBCache guard indicators must have matching [groups, ...] shapes, "
                f"got {tuple(cur.shape)} and {tuple(prev.shape)}.")
        reduce_dims = tuple(range(1, cur.ndim))
        denom = prev.abs().mean(dim=reduce_dims)
        numer = (cur - prev).abs().mean(dim=reduce_dims)
        rel = torch.where(denom > 0, numer / denom, torch.full_like(denom, float("inf")))
        return float(rel.max())

    def use_cache(
        self,
        indicator: torch.Tensor,
        step_idx: int,
        *,
        guard_indicator: torch.Tensor | None = None,
    ) -> bool:
        """Decide whether to reuse the cached residual and skip the remaining blocks.

        ``indicator`` is the first block's visual-stream residual for this step.
        Returns True only after warmup, when a previous indicator + a cache exist and
        the relative change is below the threshold. Always records ``indicator`` as
        the new previous value (compared next step)."""
        tail_start = (
            self.total_steps - self.tail_steps
            if self.total_steps is not None else None
        )
        eligible = (
            step_idx >= self.warmup_steps
            and self._prev_indicator is not None
            and self._cache is not None
            and (tail_start is None or step_idx < tail_start)
            and (
                self.max_consecutive_hits is None
                or self._consecutive_hits < self.max_consecutive_hits
            )
        )
        can = False
        if eligible:
            global_diff = self._rel_l1(indicator, self._prev_indicator)
            guard_diff = 0.0
            if guard_indicator is not None:
                guard_diff = (
                    self._max_group_rel_l1(guard_indicator, self._prev_guard_indicator)
                    if self._prev_guard_indicator is not None else float("inf")
                )
            can = max(global_diff, guard_diff) < self.threshold
        self._prev_indicator = indicator
        self._prev_guard_indicator = guard_indicator
        if can:
            self.n_hits += 1
            self._consecutive_hits += 1
        else:
            self.n_miss += 1
            self._consecutive_hits = 0
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


def build_fbcache(
    params,
    label: str = "",
    *,
    max_consecutive_hits: int | None = None,
    total_steps: int | None = None,
    tail_steps: int = 0,
):
    """Build a FirstBlockCache from generation params, or None when inactive.

    Params: ``fbcache_enable`` (bool), ``fbcache_threshold`` (relative-L1 residual
    threshold; higher = more skips/faster, lower = safer), ``fbcache_warmup_steps``
    (always-compute the first N steps)."""
    if not fbcache_active(params):
        return None
    fb = FirstBlockCache(
        threshold=float(params.get("fbcache_threshold", 0.12)),
        warmup_steps=int(params.get("fbcache_warmup_steps", 1)),
        max_consecutive_hits=max_consecutive_hits,
        total_steps=total_steps,
        tail_steps=tail_steps,
    )
    safeguards = ""
    if max_consecutive_hits is not None or tail_steps:
        safeguards = (
            f", max_consecutive_hits={max_consecutive_hits}, tail_steps={tail_steps}"
        )
    print(
        f"[FBCache] {label} enabled: threshold={fb.threshold}, "
        f"warmup={fb.warmup_steps}{safeguards}"
    )
    return fb
