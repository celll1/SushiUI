"""Audio codec geometry and calibrated target-native temporal boundaries."""

from dataclasses import dataclass
from math import prod

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AudioCodecContract:
    latent_channels: int
    sample_rate: int
    waveform_channels: int
    hop_length: int
    frame_policy: str

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in
               (self.latent_channels, self.sample_rate, self.waveform_channels, self.hop_length)):
            raise ValueError("Audio codec geometry must contain positive integers")
        if self.frame_policy != "aligned_hop_v1":
            raise ValueError("Audio codec must declare its sample-count policy")

    @classmethod
    def from_oobleck(cls, codec):
        config = codec.config
        strides = config.get("downsampling_ratios", ())
        if not strides or any(type(n) is not int or n < 2 or n % 2 for n in strides):
            raise ValueError("Aligned Oobleck geometry requires even downsampling strides")
        channels = config.get("decoder_input_channels")
        if type(channels) is not int or channels < 1 or config.get("encoder_hidden_size") != 2 * channels:
            raise ValueError("Oobleck posterior and decoder channel counts disagree")
        value = cls(channels, config.get("sampling_rate"), config.get("audio_channels"),
                    prod(strides), "aligned_hop_v1")
        if codec.hop_length != value.hop_length:
            raise ValueError("Oobleck runtime hop differs from its config")
        return value

    def latent_frames(self, samples):
        if type(samples) is not int or samples < self.hop_length or samples % self.hop_length:
            raise ValueError("Audio sample count must be a positive multiple of the codec hop")
        return samples // self.hop_length

    def waveform_samples(self, frames):
        if type(frames) is not int or frames < 1:
            raise ValueError("Audio latent frame count must be positive")
        return frames * self.hop_length

    def check_pair(self, target):
        if (self.sample_rate, self.waveform_channels) != (target.sample_rate, target.waveform_channels):
            raise ValueError("Audio calibration requires equal sample rates and waveform channels")


class TemporalLatentBridge(nn.Module):
    """Three-tap affine boundary; BCT sequence length is never changed."""

    def __init__(self, source_channels, target_channels):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (source_channels, target_channels)):
            raise ValueError("Temporal bridge channels must be positive integers")
        self.source_channels, self.target_channels = source_channels, target_channels
        self.projection = nn.Conv1d(source_channels, target_channels, 3, padding=1)
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("bridge_spec", torch.tensor([1, source_channels, target_channels]))
        self._ready = False
        self.register_load_state_dict_post_hook(self._restore_ready)

    def _restore_ready(self, module, incompatible_keys):
        if self.bridge_spec.tolist() != [1, self.source_channels, self.target_channels]:
            raise ValueError("Temporal bridge checkpoint geometry/version differs")
        self._ready = bool(self.calibrated)

    @torch.no_grad()
    def initialize(self, weight, bias):
        if weight.shape != (3 * self.source_channels, self.target_channels) or bias.shape != (self.target_channels,):
            raise ValueError("Temporal calibration coefficient shapes differ")
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("Temporal calibration coefficients must be finite")
        self.projection.weight.copy_(weight.T.reshape(self.target_channels, self.source_channels, 3))
        self.projection.bias.copy_(bias)
        self.calibrated.fill_(True)
        self._ready = True
        return self

    def _project(self, value):
        if not self._ready:
            raise RuntimeError("Temporal bridge has no paired calibration")
        if value.ndim != 3 or value.shape[1] != self.source_channels or min(value.shape) < 1:
            raise ValueError("Temporal boundary requires nonempty BCT latents")
        return F.conv1d(value, self.projection.weight, padding=1)

    def forward(self, value):
        return self._project(value) + self.projection.bias[None, :, None]

    def flow_state(self, value, sigma):
        coefficient = torch.as_tensor(sigma, device=value.device, dtype=value.dtype)
        if coefficient.ndim == 1 and coefficient.shape == (value.shape[0],):
            coefficient = coefficient[:, None, None]
        elif coefficient.ndim != 0:
            raise ValueError("Audio sigma must be scalar or one value per batch item")
        return self._project(value) + (1 - coefficient) * self.projection.bias[None, :, None]

    def flow_velocity(self, value):
        return self._project(value) - self.projection.bias[None, :, None]


class TemporalBridgeCalibrator:
    """CPU sufficient statistics; calibration never retains source waveforms."""

    def __init__(self, bridge, *, ridge):
        if not 0 < ridge < float("inf"):
            raise ValueError("Temporal calibration ridge must be finite and positive")
        self.bridge, self.ridge = bridge, float(ridge)
        n, m = 3 * bridge.source_channels, bridge.target_channels
        self.count = 0
        self.xsum, self.ysum = torch.zeros(n, dtype=torch.float64), torch.zeros(m, dtype=torch.float64)
        self.xx, self.xy = torch.zeros(n, n, dtype=torch.float64), torch.zeros(n, m, dtype=torch.float64)

    @torch.no_grad()
    def add(self, source, target):
        if source.ndim != 3 or target.ndim != 3 or source.shape[::2] != target.shape[::2]:
            raise ValueError("Temporal calibration needs equal nonempty batch/time grids")
        if source.shape[1] != self.bridge.source_channels or target.shape[1] != self.bridge.target_channels or min(source.shape) < 1:
            raise ValueError("Temporal calibration channel geometry differs")
        source, target = source.detach().cpu().double(), target.detach().cpu().double()
        if not torch.isfinite(source).all() or not torch.isfinite(target).all():
            raise ValueError("Temporal calibration latents must be finite")
        for x, y in zip(source, target):
            x = F.pad(x, (1, 1)).unfold(-1, 3, 1).permute(1, 0, 2).reshape(-1, 3 * x.shape[0])
            y = y.T
            self.count += len(x)
            self.xsum.add_(x.sum(0))
            self.ysum.add_(y.sum(0))
            self.xx.add_(x.T @ x)
            self.xy.add_(x.T @ y)

    @torch.no_grad()
    def finish(self):
        if self.count < 2:
            raise ValueError("Temporal calibration has insufficient observations")
        xm, ym = self.xsum / self.count, self.ysum / self.count
        gram = self.xx / self.count - torch.outer(xm, xm)
        cross = self.xy / self.count - torch.outer(xm, ym)
        penalty = self.ridge * gram.diag().mean().clamp_min(1e-12)
        weight = torch.linalg.solve(gram + penalty * torch.eye(len(xm), dtype=gram.dtype), cross)
        return self.bridge.initialize(weight, ym - xm @ weight)


def calibration_audio_grid(teacher, target_frames):
    """Resample the teacher only; diffusion uses the replacement codec's grid."""
    if teacher.ndim != 3 or min(teacher.shape) < 1 or type(target_frames) is not int or target_frames < 1:
        raise ValueError("Audio calibration requires nonempty BCT latents")
    if teacher.shape[-1] == target_frames:
        return teacher
    mode = "area" if target_frames < teacher.shape[-1] else "linear"
    kwargs = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(teacher.float(), size=target_frames, mode=mode, **kwargs)


class AudioAlignmentPair(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        source.check_pair(target)
        self.source, self.target = source, target
        self.backbone_channels = max(source.latent_channels, target.latent_channels)
        self.input = TemporalLatentBridge(target.latent_channels, self.backbone_channels)
        self.output = TemporalLatentBridge(self.backbone_channels, target.latent_channels)
        self.register_buffer("codec_spec", torch.tensor(self._codec_spec(), dtype=torch.int64))
        self.register_load_state_dict_post_hook(self._restore_geometry)

    def _codec_spec(self):
        return [1, self.source.latent_channels, self.source.sample_rate,
                self.source.waveform_channels, self.source.hop_length,
                self.target.latent_channels, self.target.sample_rate,
                self.target.waveform_channels, self.target.hop_length]

    def _restore_geometry(self, module, incompatible_keys):
        if self.codec_spec.tolist() != self._codec_spec():
            raise ValueError("Audio alignment checkpoint codec geometry/version differs")

    @property
    def ready(self):
        return self.input._ready and self.output._ready

    @torch.no_grad()
    def initialize_from_pair(self, incoming, outgoing):
        old, new = self.source.latent_channels, self.target.latent_channels
        if not incoming._ready or not outgoing._ready or (
                incoming.source_channels, incoming.target_channels,
                outgoing.source_channels, outgoing.target_channels) != (new, old, old, new):
            raise ValueError("Audio calibration directions disagree with codec channels")
        iw = incoming.projection.weight.new_zeros(self.backbone_channels, new, 3)
        ow = outgoing.projection.weight.new_zeros(new, self.backbone_channels, 3)
        ib = incoming.projection.bias.new_zeros(self.backbone_channels)
        iw[:old], ib[:old] = incoming.projection.weight, incoming.projection.bias
        ow[:, :old] = outgoing.projection.weight
        for c in range(old, self.backbone_channels):
            iw[c, c, 1] = ow[c, c, 1] = 1
        self.input.initialize(iw.flatten(1).T, ib)
        self.output.initialize(ow.flatten(1).T, outgoing.projection.bias)


class PairedAudioCalibration:
    def __init__(self, alignment, *, ridge):
        self.alignment = alignment
        old, new = alignment.source.latent_channels, alignment.target.latent_channels
        with torch.random.fork_rng(devices=[]):
            self.incoming, self.outgoing = TemporalLatentBridge(new, old), TemporalLatentBridge(old, new)
        self.input_fit = TemporalBridgeCalibrator(self.incoming, ridge=ridge)
        self.output_fit = TemporalBridgeCalibrator(self.outgoing, ridge=ridge)

    def add(self, source, target):
        source = calibration_audio_grid(source, target.shape[-1])
        self.input_fit.add(target, source)
        self.output_fit.add(source, target)

    def finish(self):
        self.input_fit.finish()
        self.output_fit.finish()
        self.alignment.initialize_from_pair(self.incoming, self.outgoing)
