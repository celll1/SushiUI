"""First Block Cache (FBCache) block-feature mode for SDXL / SD1.5 UNet2DConditionModel.

This is the U-Net analogue of the DiT FBCache in ``fbcache.py``. It reuses the SAME
per-block monkey-patch interception as Spectrum's block mode (``spectrum_unet.py``,
``SpectrumBlockController``): the deep blocks ``down_blocks[branch+1:]`` + ``mid_block``
are wrapped so their compute can be skipped, and instead of Spectrum's SCHEDULED
Chebyshev forecast, FBCache decides DYNAMICALLY per step from a cheap indicator.

Mechanism (no diffusers fork; per-block wrappers only):
  - branch = max(1, min(cache_branch, n_down-1)); the shallow down blocks
    ``down_blocks[:branch]`` and the indicator block ``down_blocks[branch]`` and ALL
    up blocks always run for real, so high-res detail and skip connections are exact.
  - Indicator block ``down_blocks[branch]`` ALWAYS runs; its output is USED fresh. The
    block's OUTPUT (``sample.detach()``) is the FBCache indicator -- compared across
    STEPS, not output-minus-input: a down block downsamples, so its output and input
    have different spatial shapes and ``(out - in)`` is not computable. If the
    relative-L1 change of this step's indicator vs the previous step's is below a
    threshold (after warmup, with a cache present), REUSE the cached deep features and
    SKIP ``down_blocks[branch+1:]`` + ``mid_block``.
  - On a MISS the reused region runs for real and CAPTURES its outputs (deep res_samples
    + post-mid sample), which are packed into one flat feature vector and become the new
    cache. On a HIT the wrappers return the cached tensors WITHOUT computing.

Only the deep (low-res) features are cached, so per-step memory is small.

Reference (MIT): chengzeyi/Comfy-WaveSpeed / ParaAttention (first-block cache). This is
a clean-room reimplementation of the published algorithm (first-block-residual
indicator + relative-L1 threshold + residual reuse).
"""

import torch

from core.inference.fbcache import FirstBlockCache, fbcache_active
from core.inference.spectrum_unet import _Packer


class FBCacheBlockController:
    """Drives dynamic FBCache capture/reuse across the wrapped deep UNet blocks.

    Usage per sampling step i:
        controller.begin_step(i)              # installs wrappers, resets decision
        noise = unet(...)                     # indicator decides; deep wrappers reuse/capture
        controller.end_step()                 # restores wrappers, updates cache on a miss
    """

    def __init__(self, unet, threshold, warmup_steps=1, cache_branch=1):
        self.unet = unet
        self.n_down = len(unet.down_blocks)
        # Indicator block is down_blocks[branch]; reused region is down_blocks[branch+1:]
        # + mid. Keep >=1 shallow block (:branch) real for the highest-res skips.
        self.branch = max(1, min(int(cache_branch), self.n_down - 1)) if self.n_down > 1 else 1
        self.threshold = float(threshold)
        self.warmup_steps = int(warmup_steps)
        # Steps forced to a real (miss) forward regardless of the indicator, so the
        # in-loop hard-flatten sees a genuine (non-reused) x0 on its injection steps.
        self.force_real_steps = set()
        self._packer = _Packer()
        self._prev_indicator = None   # previous step's indicator (residual of indicator block)
        self._cache_flat = None       # packed deep features from the last miss
        self._step = 0
        self._reuse_this_step = None  # None until the indicator block runs; then bool
        self._capture = []            # tensors captured this (miss) step
        self._reuse_items = None      # unpacked cached tensors to return this (hit) step
        self._reuse_cursor = 0
        self._installed = False
        self._orig_down = {}
        self._orig_mid = None
        self._device = None
        self._dtype = None
        self.n_hits = 0
        self.n_miss = 0

    @staticmethod
    def _rel_l1(cur, prev):
        return FirstBlockCache._rel_l1(cur, prev)

    # ---- step control -------------------------------------------------------
    def begin_step(self, step_idx):
        """Reset the per-step decision and install the wrappers for one U-Net call.

        The decision (``_reuse_this_step``) is undecided until the indicator block
        (down_blocks[branch]) runs. Because down blocks run in index order and mid runs
        after all down blocks, the indicator wrapper always executes BEFORE the reuse
        wrappers (down_blocks[branch+1:], mid), so the decision is set by the time they
        run. Wrappers are removed in end_step() so an exception cannot leave them on."""
        self._step = step_idx
        self._reuse_this_step = None
        self._capture = []
        self._reuse_items = None
        self._reuse_cursor = 0
        self.install()

    def end_step(self):
        """Remove wrappers. On a miss, pack the captured deep features into the cache;
        on a hit, keep the previous cache."""
        self.restore()
        if self._reuse_this_step is False and self._capture:
            if self._device is None:
                self._device = self._capture[0].device
                self._dtype = self._capture[0].dtype
            self._cache_flat = self._packer.pack(self._capture)
            self._capture = []

    # ---- wrappers -----------------------------------------------------------
    def _wrap_indicator(self, orig):
        """down_blocks[branch]: ALWAYS runs real; its OUTPUT is the FBCache indicator.

        We compare the block's output across STEPS (rel-L1 of this step's output vs the
        previous step's), not output-minus-input: a U-Net down block downsamples, so its
        output and input have different spatial shapes and (out - in) is not computable.
        The step-to-step change of the block output is the shape-safe analogue of the
        DiT first-block-residual signal -- it measures how fast the deep feature evolves."""
        def wrapper(*args, **kwargs):
            sample, res = orig(*args, **kwargs)
            indicator = sample.detach()
            reuse = (
                self._step >= self.warmup_steps
                and self._step not in self.force_real_steps
                and self._prev_indicator is not None
                and self._cache_flat is not None
                and self._rel_l1(indicator, self._prev_indicator) < self.threshold
            )
            self._prev_indicator = indicator
            self._reuse_this_step = reuse
            if reuse:
                self.n_hits += 1
                # Prepare cached tensors for the reuse wrappers that follow.
                self._reuse_items = self._packer.unpack(
                    self._cache_flat, self._device, self._dtype
                )
                self._reuse_cursor = 0
            else:
                self.n_miss += 1
            return sample, res
        return wrapper

    def _wrap_down(self, idx, orig):
        def wrapper(*args, **kwargs):
            if self._reuse_this_step:
                n_res = len(self.unet.down_blocks[idx].resnets) + (
                    1 if getattr(self.unet.down_blocks[idx], "downsamplers", None) else 0
                )
                sample = self._next_reuse()
                res = tuple(self._next_reuse() for _ in range(n_res))
                return sample, res
            sample, res = orig(*args, **kwargs)
            self._capture.append(sample)
            for r in res:
                self._capture.append(r)
            return sample, res
        return wrapper

    def _wrap_mid(self, orig):
        def wrapper(*args, **kwargs):
            if self._reuse_this_step:
                return self._next_reuse()
            sample = orig(*args, **kwargs)
            self._capture.append(sample)
            return sample
        return wrapper

    def _next_reuse(self):
        item = self._reuse_items[self._reuse_cursor]
        self._reuse_cursor += 1
        return item

    def install(self):
        if self._installed:
            return
        # Indicator block always runs real (wrapped only to compute the indicator).
        blk = self.unet.down_blocks[self.branch]
        self._orig_down[self.branch] = blk.forward
        blk.forward = self._wrap_indicator(blk.forward)
        # Reused region: down_blocks[branch+1:] + mid.
        for idx in range(self.branch + 1, self.n_down):
            blk = self.unet.down_blocks[idx]
            self._orig_down[idx] = blk.forward
            blk.forward = self._wrap_down(idx, blk.forward)
        if self.unet.mid_block is not None:
            self._orig_mid = self.unet.mid_block.forward
            self.unet.mid_block.forward = self._wrap_mid(self.unet.mid_block.forward)
        self._installed = True

    def restore(self):
        if not self._installed:
            return
        for idx, fn in self._orig_down.items():
            self.unet.down_blocks[idx].forward = fn
        if self._orig_mid is not None:
            self.unet.mid_block.forward = self._orig_mid
        self._orig_down = {}
        self._orig_mid = None
        self._installed = False


def build_unet_fbcache_controller(unet, params, label=""):
    """Build an FBCacheBlockController from generation params, or None when inactive.

    Params: ``fbcache_enable`` (bool), ``fbcache_threshold`` (relative-L1 indicator
    threshold; higher = more skips/faster), ``fbcache_warmup_steps`` (always compute
    the first N steps), ``fbcache_cache_branch`` (indicator = down_blocks[branch];
    reused region = down_blocks[branch+1:] + mid)."""
    if not fbcache_active(params):
        return None
    ctrl = FBCacheBlockController(
        unet,
        threshold=float(params.get("fbcache_threshold", 0.12)),
        warmup_steps=int(params.get("fbcache_warmup_steps", 1)),
        cache_branch=int(params.get("fbcache_cache_branch", 1)),
    )
    print(f"[FBCache] {label} enabled (U-Net block mode): threshold={ctrl.threshold}, "
          f"warmup={ctrl.warmup_steps}, cache_branch={ctrl.branch}/{ctrl.n_down} "
          f"(indicator=down[{ctrl.branch}], reuse=down[{ctrl.branch + 1}:]+mid)")
    return ctrl
