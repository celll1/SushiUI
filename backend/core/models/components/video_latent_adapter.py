"""Target-native video codec geometry and spatial flow boundaries."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from .latent_adapter import ImageCodecContract, _Config, PairedSpatialCalibration, calibration_grid
from .vae_registry import normalize, denormalize


@dataclass(frozen=True)
class VideoCodecContract:
    image: ImageCodecContract
    temporal_scale: int
    temporal_policy: str

    def __post_init__(self):
        if self.image.latent_rank not in (4, 5) or any(type(n) is not int or n < 1 for n in
                (self.image.latent_channels, self.image.spatial_scale, self.image.normalization_pack)):
            raise ValueError("Video codec geometry is missing")
        if self.image.normalization not in ("identity", "shift_scale", "per_channel", "batchnorm"):
            raise ValueError("Video codec normalization is unsupported")
        if self.image.normalization_pack not in (1, 2) or (
                self.image.normalization_pack != 1 and self.image.normalization != "batchnorm"):
            raise ValueError("Video codec normalization packing is unsupported")
        if type(self.temporal_scale) is not int or self.temporal_scale < 1:
            raise ValueError("Video codec temporal compression must be a positive integer")
        if self.temporal_policy not in ("causal_first_frame_v1", "independent_frames_v1"):
            raise ValueError("Video codec must declare its frame-count policy")
        if self.temporal_policy == "independent_frames_v1" and self.temporal_scale != 1:
            raise ValueError("Independent frames cannot have temporal compression")
        if self.image.latent_rank == 4 and self.temporal_policy != "independent_frames_v1":
            raise ValueError("A 2D codec cannot encode causal video")
        if self.image.pixel_channels != 3:
            raise ValueError("Video boundary requires an RGB codec")

    @classmethod
    def from_resolved(cls, resolved, *, temporal_policy):
        return cls(ImageCodecContract.from_resolved(resolved), resolved.scale_temporal, temporal_policy)

    def latent_frames(self, pixel_frames):
        self._positive(pixel_frames)
        if (pixel_frames - 1) % self.temporal_scale:
            raise ValueError("Clip frame count is incompatible with the codec's temporal compression")
        return (pixel_frames - 1) // self.temporal_scale + 1

    def pixel_frames(self, latent_frames):
        self._positive(latent_frames)
        return (latent_frames - 1) * self.temporal_scale + 1

    @staticmethod
    def _positive(value):
        if type(value) is not int or value < 1:
            raise ValueError("Frame count must be a positive integer")


class _VideoPosterior:
    def __init__(self, distribution, codec, expected, independent):
        self.distribution, self.codec = distribution, codec
        self.expected, self.independent = expected, independent

    def _value(self, method, **kwargs):
        raw = getattr(self.distribution, method)(**kwargs)
        if self.independent:
            b, c, t, h, w = self.expected
            if raw.shape != (b*t, c, h, w):
                raise ValueError("Image codec returned unexpected video geometry")
            raw = raw.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        if tuple(raw.shape) != self.expected:
            raise ValueError("Video codec returned unexpected latent geometry")
        return normalize(raw, self.codec.inner, self.codec.norm_spec)

    def mode(self):
        return self._value("mode")

    @property
    def mean(self):
        return self.mode()

    def sample(self, generator=None):
        return self._value("sample", generator=generator)


class NormalizedVideoVAE(nn.Module):
    """Encode full causal clips, or explicitly independent frames, exactly once."""

    def __init__(self, module, contract):
        super().__init__()
        self.inner, self.contract = module, contract
        image = contract.image
        self.norm_spec = SimpleNamespace(vae_norm=image.normalization, vae_norm_pack=image.normalization_pack)
        self.spatial_compression_ratio = image.spatial_scale
        self.temporal_compression_ratio = contract.temporal_scale
        self.config = _Config(dict(module.config), in_channels=3, input_channels=3,
            latent_channels=image.latent_channels, scaling_factor=1.0, shift_factor=0.0,
            timestep_conditioning=bool(module.config.get("timestep_conditioning", False)),
            latents_mean=[0.] * image.latent_channels, latents_std=[1.] * image.latent_channels)
        self.register_buffer("latents_mean", torch.zeros(image.latent_channels), persistent=False)
        self.register_buffer("latents_std", torch.ones(image.latent_channels), persistent=False)

    @property
    def device(self):
        return next(self.inner.parameters()).device

    @property
    def dtype(self):
        return next(self.inner.parameters()).dtype

    def encode(self, pixels, **kwargs):
        image = self.contract.image
        if pixels.ndim != 5 or pixels.shape[1] != 3 or any(n < 1 for n in pixels.shape):
            raise ValueError("Video encoding requires nonempty BCTHW RGB pixels")
        b, _, t, h, w = pixels.shape
        scale = image.spatial_scale
        if h % scale or w % scale:
            raise ValueError("Video dimensions are not divisible by the codec spatial compression")
        expected = (b, image.latent_channels, self.contract.latent_frames(t), h//scale, w//scale)
        independent = image.latent_rank == 4
        if independent:
            pixels = pixels.permute(0, 2, 1, 3, 4).reshape(b*t, 3, h, w)
        posterior = self.inner.encode(pixels, **kwargs).latent_dist
        return SimpleNamespace(latent_dist=_VideoPosterior(posterior, self, expected, independent))

    def decode(self, latents, *args, return_dict=True, **kwargs):
        if not self.config.timestep_conditioning:
            if args and (len(args) != 1 or args[0] is not None):
                raise ValueError("This video codec does not accept decode timesteps")
            args = ()
        image = self.contract.image
        if latents.ndim != 5 or latents.shape[1] != image.latent_channels or any(n < 1 for n in latents.shape):
            raise ValueError("Video decoding requires nonempty declared BCTHW latents")
        b, c, t, h, w = latents.shape
        expected = (b, 3, self.contract.pixel_frames(t), h*image.spatial_scale, w*image.spatial_scale)
        raw = denormalize(latents, self.inner, self.norm_spec)
        independent = image.latent_rank == 4
        if independent:
            raw = raw.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
        result = self.inner.decode(raw, *args, return_dict=False, **kwargs)[0]
        if independent:
            if tuple(result.shape) != (b*t, 3, expected[-2], expected[-1]):
                raise ValueError("Image codec returned unexpected video pixel geometry")
            result = result.reshape(b, t, 3, expected[-2], expected[-1]).permute(0, 2, 1, 3, 4)
        if tuple(result.shape) != expected:
            raise ValueError("Video codec returned unexpected pixel geometry")
        return SimpleNamespace(sample=result) if return_dict else (result,)

    def save_pretrained(self, path, **kwargs):
        return self.inner.save_pretrained(path, **kwargs)

    def enable_tiling(self, *args, **kwargs):
        return self.inner.enable_tiling(*args, **kwargs)

    def disable_tiling(self):
        return self.inner.disable_tiling()


def map_video_tokens(bridge, tokens, *, grid, sigma=None, velocity=False):
    """Unit-patch BNC flow boundary; token count and frame count stay unchanged."""
    if len(grid) != 3 or any(type(n) is not int or n < 1 for n in grid):
        raise ValueError("Video boundary requires a positive frame/height/width grid")
    t, h, w = grid
    if tokens.ndim != 3 or tokens.shape[1:] != (t*h*w, bridge.source_channels):
        raise ValueError("Video token sequence disagrees with the declared grid")
    if bridge.source_pack != 1 or bridge.target_pack != 1:
        raise ValueError("Video flow boundary must retain the target-native token grid")
    b = tokens.shape[0]
    latent = tokens.transpose(1, 2).reshape(b, bridge.source_channels, t, h, w)
    if velocity:
        mapped = bridge.flow_velocity(latent, convention="eps_minus_x0")
    else:
        if sigma is None:
            raise ValueError("Video flow state requires explicit sigma")
        coefficient = torch.as_tensor(sigma, device=latent.device, dtype=latent.dtype)
        if coefficient.ndim == 0:
            pass
        elif tuple(coefficient.shape) == (b,):
            coefficient = coefficient.reshape(b, 1, 1, 1, 1)
        elif tuple(coefficient.shape) == (b, t*h*w):
            coefficient = coefficient.reshape(b, 1, t, h, w)
        else:
            raise ValueError("Video sigma must be scalar, per-item, or per-token")
        mapped = bridge.epsilon(latent) + (1-coefficient)*bridge._bias_like(latent)
    return mapped.flatten(2).transpose(1, 2)


def calibration_video_grid(latent, shape):
    """Calibration-only teacher resampling; retain the causal first frame."""
    from torch.nn import functional as F
    if latent.ndim != 5 or len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Video calibration requires BCTHW and a positive target grid")
    if any(n < 1 for n in latent.shape):
        raise ValueError("Video calibration requires nonempty latents")
    frames, height, width = shape
    spatial = calibration_grid(latent, (height, width))
    if spatial.shape[2] == frames:
        return spatial
    if frames == 1:
        return spatial[:, :, :1]
    if spatial.shape[2] == 1:
        raise ValueError("A still-image teacher cannot calibrate a temporal clip")
    tail = F.interpolate(spatial[:, :, 1:].float(), size=(frames-1, height, width),
                         mode="trilinear", align_corners=False)
    return torch.cat((spatial[:, :, :1].float(), tail), 2)


class PairedVideoCalibration(PairedSpatialCalibration):
    def add(self, source, target):
        if source.ndim != 5 or target.ndim != 5 or source.shape[0] != target.shape[0]:
            raise ValueError("Paired video calibration requires matching clip batches")
        source = calibration_video_grid(source, tuple(target.shape[-3:]))
        super().add(source, target)
