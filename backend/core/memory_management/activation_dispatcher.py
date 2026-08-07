"""Proactive per-bucket activation-offload dispatcher for training.

Aspect-ratio bucketing mixes buckets (e.g. 3072x1536) at per-bucket batch sizes.
Activation memory scales as ``static + coef * (bs * latent_volume)`` (the
coefficient is aspect-independent within ~3% across buckets), so the training
peak can be PREDICTED before the forward pass and the activation CPU offload can
be dispatched only where it pays off.

LATENT VOLUME, NOT AREA. The regression variable and the bucket key are the FULL
latent extent ``latent_h * latent_w * latent_t``, where ``latent_t`` is the
temporal extent of a 5-D video latent ``[B, C, T, H', W']`` and is 1 for a 4-D
image latent ``[B, C, H', W']``. Activation is linear in the transformer's packed
sequence length, and for a video arch the clip length moves that length directly:
MiniMax-H3 at 384x640 measures 2.36 GB at a 22-frame clip and 8.90 GB at a
124-frame one (S = 1947 vs 9487; ``act = 0.632 + 8.72e-4*S`` GB, R^2 1.0000).
Keying on area alone put those two in the SAME bucket, so a measurement cached
from one answered for the other and mis-sized it by 3.8x. The latent volume is
the quantity available generically at the dispatch site that is proportional to
the sequence length's dominant (video-row) term; the sequence's constant text /
audio rows are absorbed by the fit's intercept. LTX-2.3 uses the same 5-D layout
and had the same defect. For image architectures ``latent_t`` is identically 1,
so the volume IS the area and every prediction is arithmetically unchanged.

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
def offload_activations(enabled: bool, threshold_bytes: int = 4 * 1024 * 1024,
                        use_pinned: bool = False, stats: dict = None):
    """Offload large saved activations to CPU during forward, restore in backward.
    Synchronous (blocking) copies -- value-exact, so training results are unchanged.
    ``enabled=False`` is a no-op (zero overhead).

    Must wrap BOTH the forward and the backward pass: ``saved_tensors_hooks``
    packs on save (forward) and unpacks on use (backward).

    ``use_pinned`` defaults to False (PAGEABLE CPU) on purpose. The copies here are
    synchronous (``non_blocking=False``), so page-locked (pinned) memory gives NO
    transfer-speed benefit -- and PyTorch's CachingHostAllocator keeps every freed
    pinned block cached per size and never returns it to the OS (empty_cache does not
    free host memory). With aspect-ratio bucketing the offloaded activations take on
    many distinct shapes (bucket x bs x per-layer x variable caption length), so the
    pinned cache accumulates one block set per shape and "shared GPU memory" grows
    monotonically and never shrinks for small batches -- risking a host-RAM crash.
    Pageable CPU tensors are freed back to the normal allocator and do not accumulate
    as shared GPU memory. Pinning only pays off with async (non_blocking=True) DMA on
    a dedicated stream, which is a separate follow-up; re-enable use_pinned there.

    ``stats`` (optional): a dict whose ``"bytes"`` key is incremented by the byte
    volume actually packed to CPU. This is the MEASURED offloadable volume for this
    step, fed back into ``ActivationDispatcher.record`` so the per-bucket offloadable
    fit is calibrated from real transfers instead of a fixed residual fraction.
    """
    if not enabled:
        yield
        return

    def pack(t: torch.Tensor):
        if (t.is_cuda and t.is_floating_point() and not t.is_leaf
                and t.numel() * t.element_size() >= threshold_bytes):
            cpu = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=use_pinned)
            cpu.copy_(t, non_blocking=False)
            if stats is not None:
                stats["bytes"] = stats.get("bytes", 0) + t.numel() * t.element_size()
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
            seed_coef: Initial GB per (bs * latent-voxel), i.e. per
                ``lh*lw*lt``; self-calibrated from measured peaks so it
                generalises across the many aspect (and clip-length) buckets.
            residual_frac: COLD-START FALLBACK ONLY. Fraction of activation assumed
                to remain on GPU WITH offload, used to estimate offloadable volume
                (= base_act * (1 - residual_frac)) ONLY until >=2 measured offload
                samples exist for the 2-term offloadable fit. Once measured samples
                accumulate, the calibrated per-bucket offloadable model supersedes it.
            threshold_bytes: Min saved-tensor size to offload.
        """
        self.budget = budget_gb
        self.margin = margin_gb
        self.seed_coef = seed_coef
        self.residual_frac = residual_frac
        self.threshold_bytes = threshold_bytes
        # (lh, lw, lt, bs) -> measured base (non-offloaded) activation in GB. PRIMARY
        # predictor: exact per-bucket measurement (cannot run away globally).
        # `lt` (latent temporal extent, 1 for image archs) is part of the key because
        # activation is linear in it: without it two clip lengths share a bucket and
        # the cached measurement of one silently answers for the other.
        self._act_cache = {}
        # Per-sample activation samples (latent_volume, activation_GB_per_sample) from
        # measured steps, used to fit a 2-TERM model for UNSEEN buckets:
        #     act_per_sample ~= a + b * latent_volume        (volume = lh*lw*lt)
        # The constant `a` captures the per-sample overhead that does NOT scale with
        # the latent extent (text/vision-encoder activations, fixed buffers, and for
        # video the constant text/audio rows of the packed sequence); `b` is the
        # genuine per-voxel cost. A single per-voxel coefficient (act/volume) wrongly
        # folds `a` into the slope -> tiny buckets (where `a` dominates) inflate it
        # to absurd values (e.g. 240e-6 -> 130GB predicted for a large bucket) and
        # cause needless splitting of mid buckets that actually fit. seed_coef is the
        # slope fallback until >=2 samples exist.
        self._samples = []  # list of (volume, act_per_sample_gb)
        # Measured OFFLOADABLE volume per sample (volume, offloadable_GB_per_sample)
        # from steps that actually ran WITH offload. Fit with the same 2-term model as
        # the base activation so unseen buckets get a calibrated offloadable estimate
        # instead of the flat residual_frac assumption. Empty until the first offload
        # step reports a measured packed-byte volume.
        self._offload_samples = []  # list of (volume, offloadable_per_sample_gb)
        # Exact per-bucket measured offloadable volume (lh, lw, lt, bs) -> GB.
        self._offload_cache = {}

    # ``lt`` is a TRAILING keyword on every public method with a default of 1 so
    # that existing positional call sites (all of which are image-shaped, or
    # predate the temporal term) keep working unchanged and keep meaning exactly
    # what they meant: a latent with no temporal extent.
    @staticmethod
    def _bucket(lh: int, lw: int, bs: int, lt: int = 1):
        """The bucket key. Must contain every factor the prediction scales with."""
        return (int(lh), int(lw), max(1, int(lt)), int(bs))

    @staticmethod
    def _volume(lh: int, lw: int, lt: int = 1) -> int:
        """Full per-sample latent extent. ``lt=1`` -> the latent AREA, so image
        architectures produce the identical number they always did."""
        return int(lh) * int(lw) * max(1, int(lt))

    def mark_overflow(self, lh: int, lw: int, bs: int, lt: int = 1) -> None:
        """A step at this bucket overflowed (raised OOM). Its true activation
        exceeds the headroom, so cache a value large enough to force 'escalate'
        next time -- otherwise the bucket would full-attempt and OOM-retry on
        every occurrence. Use a fixed large GB so it doesn't depend on a budget.

        The flag is per CLIP LENGTH as well as per spatial bucket: a 124-frame
        clip overflowing says nothing about a 22-frame one at the same canvas."""
        key = self._bucket(lh, lw, bs, lt)
        self._act_cache[key] = max(self._act_cache.get(key, 0.0), 1.0e6)

    @staticmethod
    def _fit_2term(pts, seed_slope):
        """Least-squares fit of ``y = a + b*volume`` over ``pts`` = [(volume, y), ...].
        Returns (a, b), both clamped >= 0. Falls back to (0, seed_slope) until two
        distinct-volume samples exist. Shared by the base-activation fit and the
        offloadable-volume fit so both predictors use identical machinery."""
        n = len(pts)
        if n < 2:
            return 0.0, seed_slope
        sx = sum(a for a, _ in pts)
        sy = sum(y for _, y in pts)
        sxx = sum(a * a for a, _ in pts)
        sxy = sum(a * y for a, y in pts)
        denom = n * sxx - sx * sx
        if denom <= 0:                      # all same area -> use mean as constant
            return max(0.0, sy / n), seed_slope
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        b = max(0.0, b)
        a = max(0.0, a)
        return a, b

    def _fit(self):
        """Base-activation fit (kept for compatibility); delegates to _fit_2term."""
        return self._fit_2term(self._samples, self.seed_coef)

    def base_act(self, lh: int, lw: int, bs: int, lt: int = 1) -> float:
        """Predicted base (non-offloaded) activation GB: exact if the bucket was
        measured, else the 2-term fit (bs * (a + b*volume))."""
        cached = self._act_cache.get(self._bucket(lh, lw, bs, lt))
        if cached is not None:
            return cached
        a, b = self._fit_2term(self._samples, self.seed_coef)
        return bs * (a + b * self._volume(lh, lw, lt))

    def predicted_offloadable(self, lh: int, lw: int, bs: int, lt: int = 1) -> float:
        """Predicted GB that activation offload can move to CPU for this bucket.

        - Exact per-bucket measurement if this (lh, lw, lt, bs) ran with offload.
        - Else the calibrated 2-term offloadable fit once >=2 offload samples exist.
        - Else (cold start, no measured offload sample yet) the residual_frac
          fallback: base_act * (1 - residual_frac). This is the ONLY remaining use
          of residual_frac and is superseded as soon as measurements accumulate.
        """
        cached = self._offload_cache.get(self._bucket(lh, lw, bs, lt))
        if cached is not None:
            return cached
        if len(self._offload_samples) < 2:
            return self.base_act(lh, lw, bs, lt) * max(0.0, 1.0 - self.residual_frac)
        # Seed slope 0: until the samples have volume spread, the fit degenerates to
        # a volume-independent constant (mean offloadable per sample). Extrapolation
        # to larger buckets stays flat, i.e. UNDER-predicts offloadable, which errs
        # on the safe (escalate) side rather than promising an offload that fails.
        a, b = self._fit_2term(self._offload_samples, 0.0)
        return bs * (a + b * self._volume(lh, lw, lt))

    def decide(self, lh: int, lw: int, bs: int, headroom_gb: float, lt: int = 1) -> str:
        """Return 'fast' / 'offload' / 'escalate'. headroom_gb = GB this process can
        still allocate right now (driver-free + reusable cache - margin), computed
        live by the caller so it adapts to co-located processes (e.g. inference)."""
        act = self.base_act(lh, lw, bs, lt)
        if act <= headroom_gb:
            return "fast"
        # Prefer pure offload whenever the calibrated post-offload footprint fits:
        # peak_with_offload ~= base_act - offloadable. Micro-batching (escalate) is
        # only chosen when even offload leaves the bucket over headroom, because
        # splitting serializes the batch and lowers per-image throughput.
        offloadable = self.predicted_offloadable(lh, lw, bs, lt)
        if act - offloadable <= headroom_gb:
            return "offload"
        return "escalate"

    def plan_micro_bs(self, lh: int, lw: int, bs: int, headroom_gb: float,
                      lt: int = 1) -> int:
        """Largest micro-batch M in [1, bs] whose offloaded activation fits the
        live headroom. The batch is split into ceil(bs/M) chunks with gradient
        accumulation, keeping the effective (gradient) batch = bs."""
        headroom = headroom_gb
        # Per-sample post-offload (resident) activation = base - offloadable, using
        # the calibrated offloadable model (residual_frac only at cold start).
        base_ps = self.base_act(lh, lw, bs, lt) / bs
        off_ps = self.predicted_offloadable(lh, lw, bs, lt) / bs
        per_sample = base_ps - off_ps
        if per_sample <= 0:
            return bs
        if headroom <= 0:
            return 1
        m = int(headroom // per_sample)
        return max(1, min(bs, m))

    def record(self, lh: int, lw: int, bs: int, mode: str, peak_gb: float,
               resident_gb: float, executed_bs: int = None,
               offloaded_gb: float = None,
               measured_threshold_bytes: int = None, lt: int = 1) -> None:
        """Cache the measured base activation and offloadable volume for this bucket.

        The GPU-resident activation is ``peak - resident_at_dispatch``. ``mode`` is
        "base" (ran without offload -> resident activation IS the base cost) or
        "offload" (resident activation is the part that stayed on GPU).

        ``offloaded_gb`` (from ``offload_activations`` stats) is a WHOLE-STEP
        quantity: the total volume packed to CPU across the entire nominal batch
        ``bs`` (the offload context stays active across all micro-chunks of a
        split step). It is therefore normalized by ``bs`` -- never by
        ``executed_bs`` -- both for the offloadable fit and for the per-chunk
        share added back when recovering the base cost. A measured 0 (offload ran
        but nothing exceeded the threshold) is a valid measurement and is
        distinguished from ``None`` (no measurement -> residual_frac fallback).
        Base-cost recovery is value-exact for full-batch offload; for a
        micro-split it assumes packed volume is uniform across chunks.

        ``measured_threshold_bytes``: the ``threshold_bytes`` the step actually
        ran with. The offloadable volume depends on the threshold, so
        measurements taken at a LOWERED threshold (fused-escalate rung, reactive
        offload retry) must NOT calibrate the default-threshold predictor --
        otherwise decide() would promise an offload the default threshold cannot
        deliver and the bucket would oscillate offload->OOM->escalate. Such
        measurements still recover the base cost (base = resident + offloaded
        holds at any threshold) but are excluded from the offloadable fit/cache.
        ``None`` means "measured at the default threshold" (calibrating).

        ``lt``: latent temporal extent of the step's latents (1 for image archs).
        It multiplies the regression variable and joins the cache key, so a
        measurement taken on one clip length never answers for another.

        For a micro-split step the peak reflects ``executed_bs`` samples, so scale
        up to the full ``bs`` (activation is ~linear in batch) -- this lets a split
        bucket learn it actually fits and stop splitting next time.
        """
        eb = executed_bs or bs
        volume = self._volume(lh, lw, lt)
        act = peak_gb - resident_gb                  # GPU-resident activation
        if act <= 0 or eb <= 0 or volume <= 0:
            return
        if mode == "offload":
            if offloaded_gb is not None and offloaded_gb >= 0:
                # Base cost = per-chunk resident activation + this chunk's share of
                # the whole-step packed volume (offloaded_gb covers all bs samples).
                act = act + offloaded_gb * (eb / bs)
                # Calibrate the offloadable fit only from default-threshold runs.
                at_default = (measured_threshold_bytes is None
                              or measured_threshold_bytes == self.threshold_bytes)
                if at_default:
                    off_ps = offloaded_gb / bs       # whole-step volume / full batch
                    self._offload_samples.append((volume, off_ps))
                    if len(self._offload_samples) > 128:
                        self._offload_samples.pop(0)
                    self._offload_cache[self._bucket(lh, lw, bs, lt)] = off_ps * bs
            elif self.residual_frac > 0:
                act = act / self.residual_frac       # recover non-offloaded cost
        # Feed (volume, per-sample activation) into the 2-term fit (unseen-bucket
        # predictor). KEEP spilled measurements: on WDDM a spill makes
        # max_memory_allocated report the real (unified) peak, the very signal that
        # this shape needs offload/split next time.
        aps = act / eb                               # activation per sample
        self._samples.append((volume, aps))
        if len(self._samples) > 128:
            self._samples.pop(0)
        # Exact per-bucket cache: full-batch activation = bs * per-sample.
        self._act_cache[self._bucket(lh, lw, bs, lt)] = aps * bs
