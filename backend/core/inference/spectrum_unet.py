"""Spectrum block-feature mode for SDXL UNet2DConditionModel (paper-faithful variant).

The black-box mode (see spectrum_forecaster.py wired in custom_sampling) forecasts the
final U-Net output epsilon, which is rough over time, so far-extrapolation is noisy.
This module instead forecasts the SMOOTH, low-resolution DEEP features and recomputes
the detail-bearing path every step (DeepCache-style), which is what the Spectrum paper
relies on (forecasting intermediate features, not the final output).

Mechanism (no full-forward replication; per-block wrappers only):
  - The deep blocks ``down_blocks[cache_branch:]`` and ``mid_block`` are wrapped.
  - On ANCHOR steps the wrappers compute normally and CAPTURE their outputs (the deep
    res_samples + the post-mid sample), which are packed into one flat feature vector
    and recorded into a SpectrumForecaster (Chebyshev fit over time).
  - On FORECAST steps the wrappers return the FORECAST of those captured tensors
    WITHOUT computing the deep blocks, so their compute is skipped. The shallow down
    blocks ``down_blocks[:cache_branch]`` and ALL up blocks still run for real, so the
    high-resolution detail and skip connections are always exact.

Only the deep (low-res) features are cached, so the per-step memory is small (a 32x32
or 64x64 feature, not the 128x128 shallow ones).
"""

import torch


class _Packer:
    """Flatten an ordered list of tensors into one vector and back (shapes fixed)."""

    def __init__(self):
        self.shapes = None
        self.sizes = None

    def pack(self, tensors):
        if self.shapes is None:
            self.shapes = [t.shape for t in tensors]
            self.sizes = [t.numel() for t in tensors]
        return torch.cat([t.reshape(-1).float() for t in tensors], dim=0)

    def unpack(self, flat, device, dtype):
        out = []
        off = 0
        for shape, size in zip(self.shapes, self.sizes):
            out.append(flat[off:off + size].reshape(shape).to(device=device, dtype=dtype))
            off += size
        return out


class SpectrumBlockController:
    """Drives deep-feature capture/forecast across the wrapped UNet blocks.

    Usage per sampling step i:
        controller.begin_step(i)              # decides anchor vs forecast
        noise = unet(...)                     # wrappers capture or short-circuit
    The wrappers read controller state; install()/restore() manage the monkey-patch.
    """

    def __init__(self, unet, forecaster, cache_branch=1):
        self.unet = unet
        self.forecaster = forecaster
        self.n_down = len(unet.down_blocks)
        # deep down blocks to forecast: [cache_branch, n_down). Clamp to keep >=1 shallow
        # block real (so the highest-res skip connections are always exact) and >=1 deep
        # block forecast (otherwise there is nothing to skip).
        self.branch = max(1, min(int(cache_branch), self.n_down - 1)) if self.n_down > 1 else 1
        self._packer = _Packer()
        self._mode = "anchor"
        self._step = 0
        self._capture = []          # list of tensors captured this (anchor) step
        self._forecast_items = None  # list of tensors to return this (forecast) step
        self._forecast_cursor = 0
        self._installed = False
        self._orig_down = {}
        self._orig_mid = None
        self._device = None
        self._dtype = None

    # ---- step control -------------------------------------------------------
    def begin_step(self, step_idx):
        """Set mode, (on forecast) precompute the forecast, and install the wrappers.

        Wrappers are installed only for the duration of one U-Net call and removed in
        end_step(), so an exception during the forward cannot leave them on the module.
        """
        self._step = step_idx
        if self.forecaster.is_anchor(step_idx):
            self._mode = "anchor"
            self._capture = []
        else:
            self._mode = "forecast"
            flat = self.forecaster.forecast(step_idx)  # [F]
            self._forecast_items = self._packer.unpack(flat, self._device, self._dtype)
            self._forecast_cursor = 0
        self.install()

    def end_step(self):
        """Remove the wrappers and, on anchor steps, record the captured deep features."""
        self.restore()
        if self._mode == "anchor" and self._capture:
            if self._device is None:
                self._device = self._capture[0].device
                self._dtype = self._capture[0].dtype
            flat = self._packer.pack(self._capture)
            self.forecaster.record(self._step, flat)
            self._capture = []

    # ---- wrappers -----------------------------------------------------------
    def _wrap_down(self, idx, orig):
        def wrapper(*args, **kwargs):
            if self._mode == "forecast":
                # Return (sample, res_samples) from the forecast without computing.
                n_res = len(self.unet.down_blocks[idx].resnets) + (
                    1 if getattr(self.unet.down_blocks[idx], "downsamplers", None) else 0
                )
                sample = self._next_forecast()
                res = tuple(self._next_forecast() for _ in range(n_res))
                return sample, res
            sample, res = orig(*args, **kwargs)
            # capture: sample first, then each res_sample (order must match forecast)
            self._capture.append(sample)
            for r in res:
                self._capture.append(r)
            return sample, res
        return wrapper

    def _wrap_mid(self, orig):
        def wrapper(*args, **kwargs):
            if self._mode == "forecast":
                return self._next_forecast()
            sample = orig(*args, **kwargs)
            self._capture.append(sample)
            return sample
        return wrapper

    def _next_forecast(self):
        item = self._forecast_items[self._forecast_cursor]
        self._forecast_cursor += 1
        return item

    def install(self):
        if self._installed:
            return
        for idx in range(self.branch, self.n_down):
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
