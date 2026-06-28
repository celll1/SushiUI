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
from typing import Dict, Optional, Tuple

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
        static_gb: float,
        margin_gb: float = 1.0,
        seed_coef: float = 24.0e-6,
        residual_frac: float = 0.85,
        threshold_bytes: int = 4 * 1024 * 1024,
    ):
        """
        Args:
            static_gb: Resolution-independent footprint (weights + grad +
                optimizer state). Used as the prediction baseline.
            margin_gb: Safety headroom kept free to avoid WDDM spill.
            seed_coef: Initial GB per (bs * latent-pixel) for unseen buckets.
            residual_frac: Fraction of activation that remains on GPU WITH
                offload under gradient checkpointing (offload removes ~15-20%).
            threshold_bytes: Min saved-tensor size to offload.
        """
        self.static = static_gb
        self.margin = margin_gb
        self.seed_coef = seed_coef
        self.residual_frac = residual_frac
        self.threshold_bytes = threshold_bytes
        # key=(lat_h, lat_w, bs) -> {"base": peak_gb, "offload": peak_gb}
        self._cache: Dict[Tuple[int, int, int], Dict[str, float]] = {}

    def update_static(self, static_gb: float) -> None:
        """Refresh the static baseline (e.g. once optimizer state is allocated)."""
        self.static = static_gb

    def _predict(self, lh: int, lw: int, bs: int, mode: str) -> float:
        entry = self._cache.get((lh, lw, bs))
        if entry is not None and mode in entry:
            return entry[mode]
        act = self.seed_coef * bs * lh * lw
        if mode == "offload":
            act *= self.residual_frac
        return self.static + act

    def decide(self, lh: int, lw: int, bs: int, free_gb: float) -> str:
        """Return one of 'fast' / 'offload' / 'escalate' for this bucket."""
        avail = free_gb - self.margin
        if self._predict(lh, lw, bs, "base") <= avail:
            return "fast"
        if self._predict(lh, lw, bs, "offload") <= avail:
            return "offload"
        return "escalate"

    def plan_micro_bs(self, lh: int, lw: int, bs: int, free_gb: float) -> int:
        """For an 'escalate' bucket (won't fit even with offload at the full batch),
        return the largest micro-batch M in [1, bs] whose predicted offloaded peak
        fits the budget. The batch is then split into ceil(bs/M) chunks processed
        with gradient accumulation, keeping the effective (gradient) batch = bs.

        Activation scales ~linearly with batch, so this inverts the predictor:
        M = floor((avail - static) / (coef * lat_area * residual_frac)).
        Returns bs when nothing needs splitting, 1 when even one sample is tight.
        """
        avail = free_gb - self.margin
        per_sample = self.seed_coef * lh * lw * self.residual_frac
        if per_sample <= 0:
            return bs
        room = avail - self.static
        if room <= 0:
            return 1
        m = int(room // per_sample)
        return max(1, min(bs, m))

    def record(self, lh: int, lw: int, bs: int, mode: str, peak_gb: float) -> None:
        """Passively calibrate from an executed step's measured peak.

        ``mode`` is the canonical bucket cost class -- "base" when the step ran
        without offload, "offload" when it ran with offload (escalate also runs
        with offload, so it records under "offload").
        """
        self._cache.setdefault((lh, lw, bs), {})[mode] = peak_gb
