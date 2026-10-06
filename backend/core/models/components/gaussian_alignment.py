"""Frozen noise complements for spatial codec migration."""
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F


POLICY = "gaussian_row_orthogonal_v1"


class GaussianNoiseAlignment(nn.Module):
    def __init__(self, declaration):
        super().__init__()
        old, new = declaration["old_channels"], declaration["new_channels"]
        if old > new:
            raise ValueError("Row-orthogonal codec noise requires target channels >= source channels")
        self.old_channels, self.new_channels = old, new
        self.register_buffer("incoming", torch.zeros(old, new, 3, 3))
        self.register_buffer("outgoing", torch.zeros(new, old, 3, 3))
        self.register_buffer("white", torch.zeros(old, new, 1, 1))
        identity = hashlib.sha256(json.dumps(declaration, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).digest()
        self._identity = list(identity)
        self.register_buffer("identity", torch.tensor(self._identity, dtype=torch.uint8))
        self.register_buffer("calibrated", torch.tensor(False))
        self._ready = False
        self.register_load_state_dict_post_hook(self._restore_ready)

    def _apply(self, fn, recurse=True):
        originals = {name: self._buffers[name] for name in ("incoming", "outgoing", "white")}
        super()._apply(fn, recurse)
        # These frozen coefficients are also the sampler's FP32 noise definition.
        for name in ("incoming", "outgoing", "white"):
            self._buffers[name] = originals[name].to(device=self._buffers[name].device, dtype=torch.float32)
        return self

    def _restore_ready(self, module, incompatible_keys):
        if self.identity.cpu().tolist() != self._identity:
            raise ValueError("Gaussian alignment checkpoint codec identity differs")
        if not all(torch.isfinite(value).all() for value in (self.incoming, self.outgoing, self.white)):
            raise ValueError("Gaussian alignment checkpoint contains nonfinite coefficients")
        self._ready = bool(self.calibrated)

    @torch.no_grad()
    def capture(self, incoming, outgoing):
        old = self.old_channels
        matrix = incoming.projection.weight[:old, :, 1, 1].detach().cpu().double()
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        if not torch.isfinite(singular).all() or singular[-1] <= singular[0] * 1e-12:
            raise ValueError("Gaussian alignment requires a full-row-rank calibrated center matrix")
        self.incoming.copy_(incoming.projection.weight[:old].float())
        self.outgoing.copy_(outgoing.projection.weight[:, :old].float())
        self.white.copy_((u @ vh).float().reshape_as(self.white))
        self.calibrated.fill_(True)
        self._ready = True

    def _flatten(self, noise):
        if not self._ready:
            raise RuntimeError("Gaussian alignment has no calibrated noise coefficients")
        if noise.ndim not in (4, 5) or noise.shape[1] != self.new_channels:
            raise ValueError("Gaussian alignment requires the declared external noise grid")
        value = noise.detach()
        if value.ndim == 5:
            b, c, frames, h, w = value.shape
            value = value.permute(0, 2, 1, 3, 4).reshape(b * frames, c, h, w)
        return value.float()

    @staticmethod
    def _restore(value, noise):
        if noise.ndim == 5:
            b, _, frames, h, w = noise.shape
            value = value.reshape(b, frames, -1, h, w).permute(0, 2, 1, 3, 4)
        return value

    def input_delta(self, noise):
        value = self._flatten(noise)
        with torch.autocast(device_type=value.device.type, enabled=False):
            white = F.conv2d(value, self.white)
            delta = white - F.conv2d(value, self.incoming, padding=1)
        return self._restore(delta, noise)

    def residual(self, noise):
        value = self._flatten(noise)
        with torch.autocast(device_type=value.device.type, enabled=False):
            projected = F.conv2d(F.conv2d(value, self.white), self.outgoing, padding=1)
        return self._restore(value - projected, noise)
