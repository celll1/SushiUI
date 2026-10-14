"""Calibrated Wiener estimate of the diffusion noise from x_t alone (VAE_SWAP_NOISE_LEAK.md)."""
import torch
from torch import nn

POLICY = "wiener_radial_v1"
BINS = 48


def _radial_index(height, width, device):
    fy = torch.fft.fftfreq(height, device=device)[:, None]
    fx = torch.fft.fftfreq(width, device=device)[None, :]
    radius = (fy.square() + fx.square()).sqrt()
    # Normalised frequency in cycles per latent pixel, so the prior holds across buckets.
    return torch.clamp((radius / (0.5 * 2 ** 0.5) * (BINS - 1)).round().long(), 0, BINS - 1)


class RadialNoisePrior(nn.Module):
    """Per-channel mean and radially averaged power spectrum of clean latents.

    ``estimate`` is E[eps | x_t] for x_t = (1 - sigma) x0 + sigma eps under that Gaussian
    prior: a fixed linear filter of x_t, so it carries no information x_t does not.
    """

    def __init__(self, channels):
        super().__init__()
        self.register_buffer("mean", torch.zeros(channels))
        self.register_buffer("power", torch.ones(channels, BINS))
        self.register_buffer("calibrated", torch.tensor(False))
        self._ready = False
        self.register_load_state_dict_post_hook(self._restore_ready)

    def _apply(self, fn, recurse=True):
        originals = {name: self._buffers[name] for name in ("mean", "power")}
        super()._apply(fn, recurse)
        for name, value in originals.items():
            self._buffers[name] = value.to(device=self._buffers[name].device, dtype=torch.float32)
        return self

    def _restore_ready(self, module, incompatible_keys):
        if not (torch.isfinite(self.mean).all() and torch.isfinite(self.power).all() and (self.power > 0).all()):
            raise ValueError("Noise prior checkpoint contains invalid statistics")
        self._ready = bool(self.calibrated)

    @torch.no_grad()
    def capture(self, latents):
        """Fit from clean, normalised target latents ([N, C, H, W] each, any spatial size)."""
        values = [value.float().reshape(-1, *value.shape[-3:]) for value in latents]
        if not values or any(value.shape[1] != self.mean.numel() for value in values):
            raise ValueError("Noise prior calibration latents disagree with the codec channels")
        mean = torch.cat([value.mean(dim=(2, 3)) for value in values]).mean(0)
        total = torch.zeros_like(self.power, dtype=torch.float64)
        count = torch.zeros(BINS, dtype=torch.float64)
        for value in values:
            centered = value - mean[None, :, None, None]
            spectrum = torch.fft.fft2(centered).abs().square().double() / (value.shape[-2] * value.shape[-1])
            index = _radial_index(value.shape[-2], value.shape[-1], "cpu").flatten()
            total.index_add_(1, index, spectrum.sum(0).flatten(1))
            count.index_add_(0, index, torch.full_like(index, value.shape[0], dtype=torch.float64))
        filled = count > 0
        power = total[:, filled] / count[filled]
        # Bins a calibration grid never reaches take the nearest measured value.
        positions = torch.arange(BINS)[filled]
        nearest = (torch.arange(BINS)[:, None] - positions[None]).abs().argmin(1)
        self.mean.copy_(mean)
        self.power.copy_(power[:, nearest].clamp_min(1e-6).float())
        self.calibrated.fill_(True)
        self._ready = True

    def estimate(self, x_t, sigma):
        if not self._ready:
            raise RuntimeError("Noise prior has no calibrated statistics")
        frame = x_t.ndim == 5
        value = x_t.squeeze(2) if frame else x_t
        level = sigma.float().reshape(-1, 1, 1, 1)
        with torch.autocast(device_type=value.device.type, enabled=False):
            centered = value.float() - (1 - level) * self.mean[None, :, None, None]
            power = self.power[:, _radial_index(value.shape[-2], value.shape[-1], value.device)]
            gain = level / ((1 - level).square() * power[None] + level.square())
            result = torch.fft.ifft2(torch.fft.fft2(centered) * gain).real
        return result.unsqueeze(2) if frame else result
