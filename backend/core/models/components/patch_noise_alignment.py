"""Frozen patch-feature noise for codec migrations that change patch size."""
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F


POLICY = "patch_covariance_v1"


def validate_declaration(declaration):
    keys = ("channels", "bridge_channels", "hidden_channels", "patch_size")
    if not isinstance(declaration, dict) or any(
            type(declaration.get(key)) is not int or declaration[key] < 1 for key in keys):
        raise ValueError("Patch noise alignment requires positive integer geometry")
    if declaration.get("policy") != POLICY or set(declaration) != {*keys, "policy"}:
        raise ValueError("Unsupported patch noise alignment declaration")
    c, b, h, p = (declaration[key] for key in keys)
    if b*c*9 + h*(b+c)*p*p > 10_000_000:
        raise ValueError("Patch noise geometry exceeds the coefficient limit")
    return declaration


class PatchNoiseAlignment(nn.Module):
    def __init__(self, declaration):
        super().__init__()
        validate_declaration(declaration)
        keys = ("channels", "bridge_channels", "hidden_channels", "patch_size")
        self.declaration = dict(declaration)
        c, b, h, p = (declaration[key] for key in keys)
        self.register_buffer("white", torch.zeros(h, c, p, p))
        self.register_buffer("incoming", torch.zeros(b, c, 3, 3))
        self.register_buffer("projection", torch.zeros(h, b, p, p))
        self.register_buffer("covariance_error", torch.zeros(()))
        self.register_buffer("rank", torch.zeros((), dtype=torch.int64))
        identity = hashlib.sha256(json.dumps(declaration, sort_keys=True).encode()).digest()
        self._identity = list(identity)
        self.register_buffer("identity", torch.tensor(self._identity, dtype=torch.uint8))
        self.register_buffer("calibrated", torch.tensor(False))
        self._ready = False
        self.register_load_state_dict_post_hook(self._restore_ready)

    def _apply(self, fn, recurse=True):
        saved = {key: self._buffers[key] for key in ("white", "incoming", "projection", "covariance_error")}
        super()._apply(fn, recurse)
        for key, value in saved.items():
            self._buffers[key] = value.to(device=self._buffers[key].device, dtype=torch.float32)
        return self

    def _restore_ready(self, module, incompatible_keys):
        if self.identity.cpu().tolist() != self._identity:
            raise ValueError("Patch noise checkpoint geometry differs")
        if not all(torch.isfinite(value).all() for value in (self.white, self.incoming, self.projection, self.covariance_error)):
            raise ValueError("Patch noise checkpoint contains nonfinite coefficients")
        maximum = min(self.white.shape[0], self.white[0].numel())
        if bool(self.calibrated) and not 1 <= int(self.rank) <= maximum:
            raise ValueError("Patch noise checkpoint has an invalid covariance rank")
        self._ready = bool(self.calibrated)

    @torch.no_grad()
    def capture(self, native_weight, bridge, projection):
        p = self.declaration["patch_size"]
        if (type(bridge) is not nn.Conv2d or type(projection) is not nn.Conv2d or
                bridge.weight.shape != self.incoming.shape or bridge.stride != (1, 1) or
                bridge.padding != (1, 1) or bridge.dilation != (1, 1) or bridge.groups != 1 or
                projection.weight.shape != self.projection.shape or projection.stride != (p, p) or
                projection.padding != (0, 0) or projection.dilation != (1, 1) or projection.groups != 1 or
                native_weight.ndim != 4 or native_weight.shape[0] != self.white.shape[0]):
            raise ValueError("Patch noise capture requires declared plain convolution boundaries")
        if not all(torch.isfinite(value).all() for value in (native_weight, bridge.weight, projection.weight)):
            raise ValueError("Patch noise capture requires finite weights")
        matrix = native_weight.detach().cpu().double().flatten(1)
        covariance = matrix @ matrix.T
        values, vectors = torch.linalg.eigh(covariance)
        count = min(matrix.shape[0], self.white[0].numel())
        positive = values > values[-1].clamp_min(1e-30) * 1e-12
        count = min(count, int(positive.sum()))
        if count == 0:
            raise ValueError("Patch noise capture requires a nonzero source covariance")
        factor = vectors[:, -count:] * values[-count:].clamp_min(0).sqrt()
        seed = int.from_bytes(bytes(self._identity[:8]), "little") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(self.white[0].numel(), count,
            dtype=torch.float64, generator=generator), mode="reduced")
        white = factor @ q.T
        self.white.copy_(white.reshape_as(self.white))
        self.incoming.copy_(bridge.weight)
        self.projection.copy_(projection.weight)
        self.covariance_error.copy_((white @ white.T - covariance).abs().max())
        self.rank.fill_(count)
        self.calibrated.fill_(True)
        self._ready = True

    def correct(self, output, noise, sigma):
        if not self._ready:
            raise RuntimeError("Patch noise alignment has no captured coefficients")
        p = self.declaration["patch_size"]
        if (noise.ndim != 4 or noise.shape[1] != self.declaration["channels"] or
                noise.shape[0] != output.shape[0] or any(size % p for size in noise.shape[-2:])):
            raise ValueError("Patch noise requires this exposure's divisible BCHW grid")
        if not torch.is_tensor(sigma) or sigma.shape != (noise.shape[0],):
            raise ValueError("Patch noise requires one actual diffusion coefficient per image")
        with torch.autocast(device_type=noise.device.type, enabled=False):
            value = noise.detach().float()
            current = F.conv2d(F.conv2d(value, self.incoming, padding=1), self.projection, stride=p)
            desired = F.conv2d(value, self.white, stride=p)
            if desired.shape != output.shape:
                raise ValueError("Patch noise output does not share the declared token grid")
            return (output.float() + sigma.float().reshape(-1, 1, 1, 1) * (desired-current)).to(output.dtype)
