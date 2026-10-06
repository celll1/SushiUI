"""Codec-independent spatial alignment on the replacement codec's native grid."""

from dataclasses import dataclass
from math import lcm
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .vae_alignment import SpatialLatentBridge, SpatialBridgeCalibrator
from .vae_registry import normalize, denormalize


def topology_alignment(spec):
    factors, height, width = {}, 1, 1
    for region in (spec or {}).get("regions", []):
        ph, pw = factors.get(region["parent"], (1, 1))
        h, w = region["down_factor"]
        factors[region["id"]] = ph*h, pw*w
        height, width = lcm(height, ph*h), lcm(width, pw*w)
    return height, width


@dataclass(frozen=True)
class ImageCodecContract:
    latent_channels: int
    spatial_scale: int
    latent_rank: int
    normalization: str
    normalization_pack: int
    pixel_channels: int

    @classmethod
    def from_resolved(cls, resolved):
        config = resolved.config
        value = cls(resolved.latent_channels, resolved.scale_factor, resolved.ndim,
                    resolved.norm, resolved.norm_pack,
                    config.get("in_channels", config.get("input_channels", 3)))
        if value.latent_rank not in (4, 5) or value.pixel_channels not in (3, 4):
            raise ValueError("Spatial swap requires a codec that encodes individual RGB/RGBA images")
        if any(type(n) is not int or n < 1 for n in
               (value.latent_channels, value.spatial_scale, value.normalization_pack)):
            raise ValueError("Image codec geometry is missing")
        if value.normalization not in ("identity", "shift_scale", "per_channel", "batchnorm"):
            raise ValueError("Image codec normalization is unsupported")
        return value

    def declaration(self, *, source, source_identity, target_identity, old_channels, old_scale):
        return dict(validate_declaration(dict(version=1, method="paired_spatial", token_policy="target_native",
            source=source, source_identity=source_identity, target_identity=target_identity,
            old_channels=old_channels, old_scale=old_scale, new_channels=self.latent_channels,
            new_scale=self.spatial_scale, ndim=self.latent_rank, norm=self.normalization,
            norm_pack=self.normalization_pack, input_channels=self.pixel_channels)))


def validate_declaration(value):
    if not isinstance(value, dict) or value.get("version") != 1 or value.get("method") != "paired_spatial":
        raise ValueError("Unsupported VAE alignment declaration")
    if value.get("token_policy") != "target_native":
        raise ValueError("VAE alignment must use the target VAE token grid")
    for key in ("old_channels", "new_channels", "old_scale", "new_scale", "input_channels", "norm_pack"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError(f"Invalid VAE alignment {key}")
    if value["input_channels"] not in (3, 4) or value.get("ndim") not in (4, 5):
        raise ValueError("Spatial alignment requires an RGB/RGBA image-capable codec")
    if value["old_scale"] % value["new_scale"] and value["new_scale"] % value["old_scale"]:
        raise ValueError("VAE alignment needs integer spatial packing")
    if not isinstance(value.get("source"), str) or not value["source"]:
        raise ValueError("VAE alignment source identity is missing")
    if value.get("norm") not in ("shift_scale", "per_channel", "batchnorm", "identity"):
        raise ValueError("VAE alignment normalization is missing")
    if value["norm_pack"] not in (1, 2) or (value["norm_pack"] != 1 and value["norm"] != "batchnorm"):
        raise ValueError("Only BatchNorm codecs use the supported 2x2 normalization packing")
    return value


@dataclass(frozen=True)
class SpatialSwapPlan:
    external_channels: int
    backbone_channels: int
    spatial_scale: int
    pixel_alignment: int

    @classmethod
    def build(cls, declaration, *, patch_size, downsample_factors=()):
        value = validate_declaration(declaration)
        factors = (patch_size, *downsample_factors)
        if any(type(n) is not int or n < 1 for n in factors):
            raise ValueError("Patch/downsample factors must be positive integers")
        # Factors are cumulative grid divisors, not stage-local reductions.
        divisor = lcm(1, *downsample_factors)
        return cls(value["new_channels"], max(value["old_channels"], value["new_channels"]),
                   value["new_scale"], value["new_scale"] * lcm(patch_size * divisor, value["norm_pack"]))


class SpatialAlignmentPair(nn.Module):
    def __init__(self, declaration):
        super().__init__()
        self.declaration = dict(validate_declaration(declaration))
        old, new = declaration["old_channels"], declaration["new_channels"]
        width = max(old, new)
        self.input = SpatialLatentBridge(new, width)
        self.output = SpatialLatentBridge(width, new)

    @property
    def ready(self):
        return self.input._ready and self.output._ready

    @torch.no_grad()
    def untrained_flow_noise_residual(self, noise, *, convention):
        """Calibrated noise complement for frozen original-channel heads; not a sampler policy."""
        if convention not in ("eps_minus_x0", "x0_minus_eps"):
            raise ValueError("Noise residual requires an explicit flow velocity convention")
        incoming = self.input.epsilon(noise.to(self.input.projection.weight.dtype))
        incoming[:, self.declaration["old_channels"]:] = 0
        projected = self.output.epsilon(incoming.to(self.output.projection.weight.dtype))
        residual = noise.float() - projected.float()
        if convention == "x0_minus_eps":
            residual = -residual
        return residual

    @torch.no_grad()
    def initialize_from_pair(self, incoming, outgoing):
        if not incoming._ready or not outgoing._ready:
            raise ValueError("Both calibration directions must be fitted before installation")
        old, new = self.declaration["old_channels"], self.declaration["new_channels"]
        if (incoming.source_channels, incoming.target_channels, outgoing.source_channels, outgoing.target_channels) != (new, old, old, new):
            raise ValueError("Calibration directions disagree with the alignment declaration")
        if any(bridge.source_pack != 1 or bridge.target_pack != 1 for bridge in (incoming, outgoing)):
            raise ValueError("Target-native calibration must not repack the token grid")
        width = max(old, new)
        iw = incoming.projection.weight.new_zeros(width, new, 3, 3)
        ow = outgoing.projection.weight.new_zeros(new, width, 3, 3)
        ib = incoming.projection.bias.new_zeros(width)
        iw[:old], ib[:old] = incoming.projection.weight, incoming.projection.bias
        ow[:, :old] = outgoing.projection.weight
        for c in range(old, width):
            iw[c, c, 1, 1] = ow[c, c, 1, 1] = 1
        self.input.initialize(iw.flatten(1).T, ib)
        self.output.initialize(ow.flatten(1).T, outgoing.projection.bias)


def calibration_grid(latent, shape):
    """Resample the teacher only; runtime never reconstructs its larger grid."""
    if latent.ndim not in (4, 5) or len(shape) != 2 or any(n < 1 for n in shape):
        raise ValueError("Calibration requires BCHW or independent-frame BCFHW latents")
    original = latent.shape
    if latent.ndim == 5:
        b, c, frames, h, w = original
        latent = latent.permute(0, 2, 1, 3, 4).reshape(b * frames, c, h, w)
    if tuple(shape) != tuple(latent.shape[-2:]):
        if all(n <= old for n, old in zip(shape, latent.shape[-2:])):
            latent = F.interpolate(latent.float(), size=shape, mode="area")
        else:
            latent = F.interpolate(latent.float(), size=shape, mode="bilinear", align_corners=False)
    if len(original) == 5:
        latent = latent.reshape(b, frames, c, *shape).permute(0, 2, 1, 3, 4)
    return latent


class PairedSpatialCalibration:
    def __init__(self, alignment, *, ridge):
        self.alignment = alignment
        d = alignment.declaration
        # Fitted scratch kernels must not advance the training RNG.
        with torch.random.fork_rng(devices=[]):
            self.incoming = SpatialLatentBridge(d["new_channels"], d["old_channels"])
            self.outgoing = SpatialLatentBridge(d["old_channels"], d["new_channels"])
        self.input_fit = SpatialBridgeCalibrator(self.incoming, ridge=ridge)
        self.output_fit = SpatialBridgeCalibrator(self.outgoing, ridge=ridge)

    def add(self, source, target):
        source = calibration_grid(source, target.shape[-2:])
        self.input_fit.add(target, source)
        self.output_fit.add(source, target)

    def finish(self):
        self.input_fit.finish()
        self.output_fit.finish()
        self.alignment.initialize_from_pair(self.incoming, self.outgoing)


class _Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
    __setattr__ = dict.__setitem__


class _FramePosterior:
    def __init__(self, distributions, codec):
        self.distributions, self.codec = distributions, codec

    def _value(self, method, **kwargs):
        frames = []
        for distribution in self.distributions:
            raw = getattr(distribution, method)(**kwargs)
            if raw.ndim == 5:
                if raw.shape[2] != 1:
                    raise ValueError("Image codec changed the independent frame count")
                raw = raw.squeeze(2)
            if raw.ndim != 4 or raw.shape[1] != self.codec.declaration["new_channels"]:
                raise ValueError("Image codec output disagrees with its declared latent channels")
            frames.append(normalize(raw, self.codec.inner, self.codec.norm_spec))
        return torch.stack(frames, dim=2)

    def mode(self):
        return self._value("mode")

    def sample(self, generator=None):
        return self._value("sample", generator=generator)


class NormalizedFrameVAE(nn.Module):
    """Expose normalized independent RGBA frames from RGB/RGBA image codecs."""
    def __init__(self, module, declaration):
        super().__init__()
        self.inner = module
        self.declaration = dict(validate_declaration(declaration))
        self.norm_spec = SimpleNamespace(vae_norm=declaration["norm"], vae_norm_pack=declaration["norm_pack"])
        self.spatial_scale_factor = declaration["new_scale"]
        self.input_channels = 4
        channels = declaration["new_channels"]
        self.config = _Config(dict(module.config), input_channels=4, z_dim=channels, latent_channels=channels,
            scaling_factor=1.0, shift_factor=0.0, temperal_downsample=[],
            latents_mean=[0.] * channels, latents_std=[1.] * channels)

    @property
    def device(self):
        return next(self.inner.parameters()).device

    @property
    def dtype(self):
        return next(self.inner.parameters()).dtype

    def encode(self, pixels, **kwargs):
        if pixels.ndim != 5 or pixels.shape[1] != 4 or pixels.shape[2] < 1:
            raise ValueError("Aligned image VAE requires independent RGBA frames")
        distributions = []
        for frame in pixels.unbind(2):
            frame = frame[:, :self.declaration["input_channels"]]
            frame = codec_pixels(self.inner, frame, encode=True)
            if self.declaration["ndim"] == 5:
                frame = frame.unsqueeze(2)
            distributions.append(self.inner.encode(frame, **kwargs).latent_dist)
        return SimpleNamespace(latent_dist=_FramePosterior(distributions, self))

    def decode(self, latents, return_dict=True, **kwargs):
        if latents.ndim != 5 or latents.shape[1] != self.declaration["new_channels"] or latents.shape[2] < 1:
            raise ValueError("Aligned image decode requires declared independent-frame latents")
        decoded = []
        for frame in latents.unbind(2):
            raw = denormalize(frame, self.inner, self.norm_spec)
            if self.declaration["ndim"] == 5:
                raw = raw.unsqueeze(2)
            result = self.inner.decode(raw, return_dict=False, **kwargs)[0]
            if result.ndim == 5:
                if result.shape[2] != 1:
                    raise ValueError("Image codec changed the independent frame count")
                result = result.squeeze(2)
            result = codec_pixels(self.inner, result, encode=False)
            if result.shape[1] == 3:
                result = torch.cat((result, torch.ones_like(result[:, :1])), dim=1)
            if result.shape[1] != 4:
                raise ValueError("Aligned image decoder must return RGB or RGBA")
            decoded.append(result)
        value = torch.stack(decoded, dim=2)
        return SimpleNamespace(sample=value) if return_dict else (value,)

    def save_pretrained(self, path, **kwargs):
        return self.inner.save_pretrained(path, **kwargs)


def codec_pixels(module, pixels, *, encode):
    codec_class = module.config.get("_class_name") or type(module).__name__
    if codec_class == "AutoencoderKLMiniMaxH3":
        from core.models.minimax_h3.vae_contract import transform_pixels
        return transform_pixels(pixels, encode=encode)
    return pixels


class _ImagePosterior:
    def __init__(self, frames):
        self.frames = frames

    def mode(self):
        return self.frames.mode().squeeze(2)

    @property
    def mean(self):
        return self.mode()

    def sample(self, generator=None):
        return self.frames.sample(generator=generator).squeeze(2)


class NormalizedImageVAE(NormalizedFrameVAE):
    """BCHW/RGB boundary for backbones that do not expose a frame axis."""
    def __init__(self, module, declaration):
        super().__init__(module, declaration)
        self.config.input_channels = self.config.in_channels = 3
        self.input_channels = 3
        scale = self.spatial_scale_factor
        if scale & (scale - 1) == 0:
            # Diffusers pipelines infer the spatial ratio from this configuration.
            self.config.block_out_channels = [1] * scale.bit_length()

    def encode(self, pixels, **kwargs):
        if pixels.ndim != 4 or pixels.shape[1] not in (3, 4):
            raise ValueError("Aligned image VAE requires BCHW RGB/RGBA pixels")
        if pixels.shape[1] == 3:
            pixels = torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=1)
        result = super().encode(pixels.unsqueeze(2), **kwargs)
        return SimpleNamespace(latent_dist=_ImagePosterior(result.latent_dist))

    def decode(self, latents, return_dict=True, **kwargs):
        if latents.ndim != 4:
            raise ValueError("Aligned image VAE decode requires BCHW latents")
        result = super().decode(latents.unsqueeze(2), return_dict=False, **kwargs)[0]
        rgb = result[:, :3, 0]
        return SimpleNamespace(sample=rgb) if return_dict else (rgb,)
