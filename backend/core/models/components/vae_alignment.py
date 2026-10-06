"""Trainable spatial latent bridges, independent of the backbone architecture."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatialLatentBridge(nn.Module):
    """Apply a calibrated 3x3 affine map in a declared pixel-unshuffle domain.

    Frames are independent; this is not a temporal convolution. Callers must
    supply diffusion sigma explicitly rather than folding clean bias into IO.
    """

    def __init__(self, source_channels: int, target_channels: int, *,
                 source_pack: int = 1, target_pack: int = 1):
        super().__init__()
        values = (source_channels, target_channels, source_pack, target_pack)
        if any(type(n) is not int or n < 1 for n in values):
            raise ValueError("Channel counts and packing factors must be positive integers")
        self.source_channels, self.target_channels = source_channels, target_channels
        self.source_pack, self.target_pack = source_pack, target_pack
        self.projection = nn.Conv2d(source_channels * source_pack**2,
                                    target_channels * target_pack**2, 3, padding=1)
        # Uncalibrated bridges must not masquerade as useful initialization.
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("bridge_spec", torch.tensor(
            [1, source_channels, target_channels, source_pack, target_pack], dtype=torch.int64))
        self._ready = False
        self.register_load_state_dict_post_hook(self._restore_ready)

    def _restore_ready(self, module, incompatible_keys):
        expected = [1, self.source_channels, self.target_channels, self.source_pack, self.target_pack]
        if self.bridge_spec.tolist() != expected:
            raise ValueError("Spatial bridge checkpoint geometry/version does not match")
        self._ready = bool(self.calibrated)

    @torch.no_grad()
    def initialize(self, weight: torch.Tensor, bias: torch.Tensor):
        """Import row-vector coefficients fitted with channel-major F.unfold."""
        incoming, outgoing = self.projection.in_channels, self.projection.out_channels
        if weight.shape != (incoming * 9, outgoing) or bias.shape != (outgoing,):
            raise ValueError("Calibration shape does not match the declared bridge geometry")
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("Calibration contains nonfinite coefficients")
        self.projection.weight.copy_(weight.T.reshape(outgoing, incoming, 3, 3))
        self.projection.bias.copy_(bias)
        self.calibrated.fill_(True)
        self._ready = True
        return self

    def geometry(self):
        return dict(source_channels=self.source_channels, target_channels=self.target_channels,
                    source_pack=self.source_pack, target_pack=self.target_pack)

    def _flatten(self, latent):
        if latent.ndim not in (4, 5) or latent.shape[1] != self.source_channels:
            raise ValueError("Bridge requires BCHW or independent-frame BCFHW with declared channels")
        if any(n % self.source_pack for n in latent.shape[-2:]):
            raise ValueError("Source grid is not divisible by the packing factor")
        if latent.ndim == 5:
            b, c, frames, h, w = latent.shape
            latent = latent.permute(0, 2, 1, 3, 4).reshape(b * frames, c, h, w)
        return F.pixel_unshuffle(latent, self.source_pack)

    def _restore(self, mapped, original):
        mapped = F.pixel_shuffle(mapped, self.target_pack)
        if original.ndim == 5:
            b, _, frames, _, _ = original.shape
            mapped = mapped.reshape(b, frames, self.target_channels, *mapped.shape[-2:])
            mapped = mapped.permute(0, 2, 1, 3, 4)
        return mapped

    def _project(self, latent, *, bias):
        if not self._ready:
            raise RuntimeError("Spatial bridge has no paired calibration")
        values = self._flatten(latent)
        return self._restore(F.conv2d(values, self.projection.weight,
                            self.projection.bias if bias else None, padding=1), latent)

    def _bias_like(self, latent):
        values = self._flatten(latent)
        bias = self.projection.bias[None, :, None, None].expand(
            values.shape[0], -1, *values.shape[-2:])
        return self._restore(bias, latent)

    @staticmethod
    def _sigma(sigma, latent):
        sigma = torch.as_tensor(sigma, device=latent.device, dtype=latent.dtype)
        if sigma.ndim == 0:
            return sigma
        if sigma.shape != (latent.shape[0],):
            raise ValueError("Sigma must be scalar or one value per batch item")
        return sigma.reshape(-1, *([1] * (latent.ndim - 1)))

    def forward(self, latent):
        """Clean latent map, including calibrated mean offset."""
        return self._project(latent, bias=True)

    def flow_state(self, latent, sigma):
        """Map the clean part's bias only: y_s=A*x_s+(1-s)*b."""
        sigma = torch.as_tensor(sigma, device=latent.device, dtype=latent.dtype)
        return self.diffusion_state(latent, alpha=1-sigma)

    def diffusion_state(self, latent, *, alpha):
        """Map x_t=alpha*x0+sigma*noise without treating VP alpha as 1-sigma."""
        coefficient = self._sigma(alpha, latent)
        return self._project(latent, bias=False) + coefficient * self._bias_like(latent)

    def prediction(self, value, *, kind, sigma=None):
        if kind == "epsilon":
            return self.epsilon(value)
        if kind == "x0":
            return self(value)
        if kind in ("eps_minus_x0", "x0_minus_eps"):
            return self.flow_velocity(value, convention=kind)
        if kind == "v" and sigma is not None:
            return self._project(value, bias=False) - self._sigma(sigma, value) * self._bias_like(value)
        raise ValueError("Prediction kind requires a declared diffusion convention (and sigma for VP v)")

    def flow_velocity(self, velocity, *, convention):
        if convention not in ("eps_minus_x0", "x0_minus_eps"):
            raise ValueError("Unknown flow velocity convention")
        sign = -1 if convention == "eps_minus_x0" else 1
        return self._project(velocity, bias=False) + sign * self._bias_like(velocity)

    def epsilon(self, noise):
        """Noise prediction has no clean-latent mean offset."""
        return self._project(noise, bias=False)


class SpatialBridgeCalibrator:
    """Stream normalized paired latents; retain statistics, never dataset images."""

    def __init__(self, bridge: SpatialLatentBridge, *, ridge: float):
        if not 0 < ridge < float("inf"):
            raise ValueError("Ridge must be finite and positive")
        self.bridge, self.ridge = bridge, float(ridge)
        incoming = bridge.projection.in_channels * 9
        outgoing = bridge.projection.out_channels
        self.count = 0
        self.xsum = torch.zeros(incoming, dtype=torch.float64)
        self.ysum = torch.zeros(outgoing, dtype=torch.float64)
        self.xx = torch.zeros(incoming, incoming, dtype=torch.float64)
        self.xy = torch.zeros(incoming, outgoing, dtype=torch.float64)

    @torch.no_grad()
    def add(self, source: torch.Tensor, target: torch.Tensor):
        if source.ndim != target.ndim or source.shape[0] != target.shape[0]:
            raise ValueError("Calibration pairs must have the same batch/frame structure")
        if source.ndim == 5 and source.shape[2] != target.shape[2]:
            raise ValueError("Calibration frame counts differ")
        if target.ndim not in (4, 5) or target.shape[1] != self.bridge.target_channels:
            raise ValueError("Target channels disagree with calibration geometry")
        source = self.bridge._flatten(source.detach().cpu().double())
        target = target.detach().cpu().double()
        if target.ndim == 5:
            b, c, frames, h, w = target.shape
            target = target.permute(0, 2, 1, 3, 4).reshape(b * frames, c, h, w)
        if any(n % self.bridge.target_pack for n in target.shape[-2:]):
            raise ValueError("Target grid is not divisible by packing factor")
        target = F.pixel_unshuffle(target, self.bridge.target_pack)
        if source.shape[0:1] + source.shape[2:] != target.shape[0:1] + target.shape[2:]:
            raise ValueError("Calibration pairs require equal packed grids")
        if not torch.isfinite(source).all() or not torch.isfinite(target).all():
            raise ValueError("Calibration pairs contain nonfinite values")
        if not source.shape[0] or not source.shape[-1] or not source.shape[-2]:
            raise ValueError("Calibration pairs must be nonempty")
        # Unfold one independent frame at a time to bound temporary feature memory.
        for x, y in zip(source, target):
            x = F.unfold(x[None], 3, padding=1)[0].T
            y = y.permute(1, 2, 0).reshape(-1, y.shape[0])
            self.count += len(x)
            self.xsum.add_(x.sum(0))
            self.ysum.add_(y.sum(0))
            self.xx.add_(x.T @ x)
            self.xy.add_(x.T @ y)

    @torch.no_grad()
    def finish(self):
        if self.count < 2:
            raise ValueError("Paired calibration has insufficient observations")
        xm, ym = self.xsum / self.count, self.ysum / self.count
        gram = self.xx / self.count - torch.outer(xm, xm)
        cross = self.xy / self.count - torch.outer(xm, ym)
        penalty = self.ridge * gram.diag().mean().clamp_min(1e-12)
        weight = torch.linalg.solve(gram + penalty * torch.eye(len(xm), dtype=gram.dtype), cross)
        return self.bridge.initialize(weight, ym - xm @ weight)
