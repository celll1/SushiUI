"""Proactive per-bucket activation-offload dispatcher for training.

Aspect-ratio bucketing mixes buckets (e.g. 3072x1536) at per-bucket batch sizes.
Activation memory scales as ``static + coef * (bs * latent_h * latent_w)`` (the
coefficient is aspect-independent within ~3% across buckets), so the training
peak can be PREDICTED before the forward pass and the activation CPU offload can
be dispatched only where it pays off.

This matters on Windows WDDM, where exceeding dedicated VRAM does not raise
``torch.cuda.OutOfMemoryError`` -- the driver silently spills to shared host
memory and the step becomes 10-100x slower. Reactive OOM recovery never fires
there, so the decision must be made proactively from a memory prediction rather
than from a caught exception.

Three modes per (bucket, bs):
  - ``fast``     : predicted peak fits with margin -> offload OFF (no iter cost)
  - ``offload``  : doesn't fit, but fits WITH activation offload
  - ``escalate`` : even offload won't fit (caller decides what to do; currently
                   runs with offload and logs a warning)

The predictor starts from a conservative seed coefficient and is refined by
PASSIVE online calibration: the measured peak of each executed step is recorded
under the mode it ran in. No extra calibration step is ever run (that would
pollute gradients and waste compute), so the cache fills in as buckets recur.

Validated by tmp/test_B2..B5; see docs/VRAM_OVERFLOW_PREVENTION_DESIGN.md sec 11.
"""

import contextlib

import torch

_GB = 1024 ** 3


@contextlib.contextmanager
def offload_activations(enabled: bool, threshold_bytes: int = 4 * 1024 * 1024):
    """Offload large saved activations to pinned CPU during forward, restore in
    backward. Synchronous (blocking) copies -- value-exact, so training results
    are unchanged. ``enabled=False`` is a no-op (zero overhead).

    Must wrap BOTH the forward and the backward pass: ``saved_tensors_hooks``
    packs on save (forward) and unpacks on use (backward).
    """
    if not enabled:
        yield
        return

    def pack(t: torch.Tensor):
        if (t.is_cuda and t.is_floating_point() and not t.is_leaf
                and t.numel() * t.element_size() >= threshold_bytes):
            cpu = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
            cpu.copy_(t, non_blocking=False)
            return ("cpu", cpu, t.device)
        return ("gpu", t)

    def unpack(p):
        return p[1].to(p[2], non_blocking=False) if p[0] == "cpu" else p[1]

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


class ActivationDispatcher:
    """Predicts per-bucket training peak and dispatches activation offload.

    All sizes are in GB. The dispatcher holds no GPU state; it only reads
    ``torch.cuda.mem_get_info()`` / ``max_memory_allocated()`` via the caller.
    """

    def __init__(
        self,
        budget_gb: float,
        margin_gb: float = 1.0,
        seed_coef: float = 24.0e-6,
        residual_frac: float = 0.85,
        threshold_bytes: int = 4 * 1024 * 1024,
    ):
        """
        Args:
            budget_gb: Total VRAM usable by this process (captured once as
                allocated + driver-free at startup = total minus system reserve).
                Decisions compare predicted activation against (budget - resident).
            margin_gb: Safety headroom kept free to avoid WDDM spill.
            seed_coef: Initial GB per (bs * latent-pixel); self-calibrated from
                measured peaks so it generalises across the many aspect buckets.
            residual_frac: Fraction of activation that remains on GPU WITH
                offload under gradient checkpointing (offload removes ~15-20%).
            threshold_bytes: Min saved-tensor size to offload.
        """
        self.budget = budget_gb
        self.margin = margin_gb
        self.seed_coef = seed_coef
        self.residual_frac = residual_frac
        self.threshold_bytes = threshold_bytes
        # (lh, lw, bs) -> measured base (non-offloaded) activation in GB. PRIMARY
        # predictor: exact per-bucket measurement (cannot run away globally).
        self._act_cache = {}
        # Per-sample activation samples (latent_area, activation_GB_per_sample) from
        # measured steps, used to fit a 2-TERM model for UNSEEN buckets:
        #     act_per_sample ~= a + b * latent_area
        # The constant `a` captures the per-sample overhead that does NOT scale with
        # image area (text/vision-encoder activations, fixed buffers); `b` is the
        # genuine per-pixel cost. A single per-pixel coefficient (act/area) wrongly
        # folds `a` into the slope -> tiny buckets (where `a` dominates) inflate it
        # to absurd values (e.g. 240e-6 -> 130GB predicted for a large bucket) and
        # cause needless splitting of mid buckets that actually fit. seed_coef is the
        # slope fallback until >=2 samples exist.
        self._samples = []  # list of (area, act_per_sample_gb)

    def mark_overflow(self, lh: int, lw: int, bs: int) -> None:
        """A step at this bucket overflowed (raised OOM). Its true activation
        exceeds the headroom, so cache a value large enough to force 'escalate'
        next time -- otherwise the bucket would full-attempt and OOM-retry on
        every occurrence. Use a fixed large GB so it doesn't depend on a budget."""
        key = (lh, lw, bs)
        self._act_cache[key] = max(self._act_cache.get(key, 0.0), 1.0e6)

    def _fit(self):
        """Least-squares fit of act_per_sample = a + b*area over recent samples.
        Returns (a, b), both clamped >= 0. Falls back to (0, seed_coef) until two
        distinct-area samples exist."""
        pts = self._samples
        n = len(pts)
        if n < 2:
            return 0.0, self.seed_coef
        sx = sum(a for a, _ in pts)
        sy = sum(y for _, y in pts)
        sxx = sum(a * a for a, _ in pts)
        sxy = sum(a * y for a, y in pts)
        denom = n * sxx - sx * sx
        if denom <= 0:                      # all same area -> use mean as constant
            return max(0.0, sy / n), self.seed_coef
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        b = max(0.0, b)
        a = max(0.0, a)
        return a, b

    def base_act(self, lh: int, lw: int, bs: int) -> float:
        """Predicted base (non-offloaded) activation GB: exact if the bucket was
        measured, else the 2-term fit (bs * (a + b*area))."""
        cached = self._act_cache.get((lh, lw, bs))
        if cached is not None:
            return cached
        a, b = self._fit()
        return bs * (a + b * lh * lw)

    def decide(self, lh: int, lw: int, bs: int, headroom_gb: float) -> str:
        """Return 'fast' / 'offload' / 'escalate'. headroom_gb = GB this process can
        still allocate right now (driver-free + reusable cache - margin), computed
        live by the caller so it adapts to co-located processes (e.g. inference)."""
        act = self.base_act(lh, lw, bs)
        if act <= headroom_gb:
            return "fast"
        if act * self.residual_frac <= headroom_gb:
            return "offload"
        return "escalate"

    def plan_micro_bs(self, lh: int, lw: int, bs: int, headroom_gb: float) -> int:
        """Largest micro-batch M in [1, bs] whose offloaded activation fits the
        live headroom. The batch is split into ceil(bs/M) chunks with gradient
        accumulation, keeping the effective (gradient) batch = bs."""
        headroom = headroom_gb
        per_sample = (self.base_act(lh, lw, bs) / bs) * self.residual_frac
        if per_sample <= 0:
            return bs
        if headroom <= 0:
            return 1
        m = int(headroom // per_sample)
        return max(1, min(bs, m))

    def record(self, lh: int, lw: int, bs: int, mode: str, peak_gb: float,
               resident_gb: float, executed_bs: int = None) -> None:
        """Cache the measured base activation for this bucket.

        activation = peak - resident_at_dispatch. ``mode`` is "base" (ran without
        offload) or "offload" (divide out residual_frac to recover base cost).
        For a micro-split step the peak reflects ``executed_bs`` samples, so scale
        up to the full ``bs`` (activation is ~linear in batch) -- this lets a split
        bucket learn it actually fits and stop splitting next time.
        """
        eb = executed_bs or bs
        area = lh * lw
        act = peak_gb - resident_gb
        if act <= 0 or eb <= 0 or area <= 0:
            return
        if mode == "offload" and self.residual_frac > 0:
            act = act / self.residual_frac           # recover non-offloaded cost
        # Feed (area, per-sample activation) into the 2-term fit (unseen-bucket
        # predictor). KEEP spilled measurements: on WDDM a spill makes
        # max_memory_allocated report the real (unified) peak, the very signal that
        # this shape needs offload/split next time.
        aps = act / eb                               # activation per sample
        self._samples.append((area, aps))
        if len(self._samples) > 128:
            self._samples.pop(0)
        # Exact per-bucket cache: full-batch activation = bs * per-sample.
        self._act_cache[(lh, lw, bs)] = aps * bs
