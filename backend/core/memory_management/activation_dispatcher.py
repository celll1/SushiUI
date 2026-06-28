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
        self.coef = seed_coef          # running per-(bs*pixel) activation coef (GB)
        self.residual_frac = residual_frac
        self.threshold_bytes = threshold_bytes

    def _headroom(self, resident_gb: float) -> float:
        """Bytes available for this step's activation. Uses memory_allocated()
        (true live bytes) NOT driver-free, so PyTorch's reusable reserved cache
        counts as available -- otherwise headroom collapses to ~0 once the cache
        fills and every bucket falsely escalates."""
        return self.budget - resident_gb - self.margin

    def _act_pred(self, lh: int, lw: int, bs: int) -> float:
        return self.coef * bs * lh * lw

    def decide(self, lh: int, lw: int, bs: int, resident_gb: float) -> str:
        """Return 'fast' / 'offload' / 'escalate'. resident_gb = memory_allocated()
        at dispatch time (static weights+grad+optimizer; activations already freed)."""
        headroom = self._headroom(resident_gb)
        act = self._act_pred(lh, lw, bs)
        if act <= headroom:
            return "fast"
        if act * self.residual_frac <= headroom:
            return "offload"
        return "escalate"

    def plan_micro_bs(self, lh: int, lw: int, bs: int, resident_gb: float) -> int:
        """Largest micro-batch M in [1, bs] whose offloaded activation fits the
        headroom. The batch is split into ceil(bs/M) chunks with gradient
        accumulation, keeping the effective (gradient) batch = bs."""
        headroom = self._headroom(resident_gb)
        per_sample = self.coef * lh * lw * self.residual_frac
        if per_sample <= 0:
            return bs
        if headroom <= 0:
            return 1
        m = int(headroom // per_sample)
        return max(1, min(bs, m))

    def record(self, lh: int, lw: int, bs: int, mode: str, peak_gb: float,
               resident_gb: float) -> None:
        """Self-calibrate the activation coefficient from a measured step.

        activation = peak - resident_at_dispatch. Back out per-(bs*pixel) cost and
        update the running coef conservatively (grow fast to avoid under-predicting
        and spilling, relax slowly). ``mode`` is "base" (ran without offload) or
        "offload" (ran with offload -> divide out residual_frac to get base cost).
        """
        denom = bs * lh * lw
        act = peak_gb - resident_gb
        if act <= 0 or denom <= 0:
            return
        # A peak at/above the budget means the step spilled (WDDM) or nearly OOMed;
        # max_memory_allocated is then an unreliable lower bound, so don't let it
        # inflate the coefficient (which would over-predict and over-split forever).
        if peak_gb >= self.budget:
            return
        implied = act / denom
        if mode == "offload" and self.residual_frac > 0:
            implied = implied / self.residual_frac
        if implied > self.coef:
            self.coef = implied                      # grow immediately (safe side)
        else:
            self.coef = 0.9 * self.coef + 0.1 * implied   # relax slowly
