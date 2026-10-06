"""Synthetic video geometry and independent flow-boundary oracles."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from core.models.components.latent_adapter import ImageCodecContract, SpatialAlignmentPair
from core.models.components.video_latent_adapter import (
    VideoCodecContract, NormalizedVideoVAE, map_video_tokens, calibration_video_grid, PairedVideoCalibration,
)
from core.models.components.vae_alignment import SpatialLatentBridge


def contract(rank, temporal, policy):
    return VideoCodecContract(ImageCodecContract(2, 2, rank, "shift_scale", 1, 3), temporal, policy)


class Codec(nn.Module):
    def __init__(self, rank, temporal):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.rank, self.temporal = rank, temporal
        self.config = dict(scaling_factor=2., shift_factor=.25)
        self.calls = []
        # Normalization reads config attributes rather than dict keys.
        from core.models.components.latent_adapter import _Config
        self.config = _Config(self.config)

    def encode(self, pixels):
        self.calls.append(tuple(pixels.shape))
        raw = pixels[:, :2, ..., ::2, ::2]
        if self.rank == 5:
            raw = raw[:, :, ::self.temporal]
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: raw, sample=lambda generator=None: raw))

    def decode(self, raw, return_dict=False):
        if self.rank == 5:
            raw = torch.cat((raw[:, :, :1], raw[:, :, 1:].repeat_interleave(self.temporal, 2)), 2)
        value = torch.cat((raw, raw[:, :1]), 1).repeat_interleave(2, -1).repeat_interleave(2, -2)
        return (value,)


@pytest.mark.parametrize("rank,temporal,policy", [(5, 4, "causal_first_frame_v1"),
    (5, 1, "independent_frames_v1"), (4, 1, "independent_frames_v1")])
def test_full_clip_geometry_normalization_and_frame_order(rank, temporal, policy):
    c = contract(rank, temporal, policy)
    raw = Codec(rank, temporal)
    vae = NormalizedVideoVAE(raw, c)
    pixels = torch.arange(2*3*9*8*12).reshape(2, 3, 9, 8, 12).float()/100
    encoded = vae.encode(pixels).latent_dist
    expected_raw = pixels[:, :2, ::temporal, ::2, ::2]
    expected = (expected_raw-.25)*2
    torch.testing.assert_close(encoded.mode(), expected)
    torch.testing.assert_close(encoded.sample(torch.Generator().manual_seed(8)), expected)
    assert len(raw.calls) == 1
    assert raw.calls[0] == ((18, 3, 8, 12) if rank == 4 else tuple(pixels.shape))
    decoded = vae.decode(expected).sample
    assert decoded.shape == pixels.shape
    torch.testing.assert_close(decoded[:, :2, ::temporal, ::2, ::2], expected_raw)
    assert vae.temporal_compression_ratio == temporal
    assert vae.spatial_compression_ratio == 2
    assert vae.config.scaling_factor == 1
    assert vae.latents_mean.count_nonzero() == 0
    assert torch.equal(vae.latents_std, torch.ones(2))


@pytest.mark.parametrize("rank,temporal,policy", [(4, 4, "causal_first_frame_v1"),
    (5, 4, "independent_frames_v1"), (5, 4, "unknown"), (5, 0, "causal_first_frame_v1")])
def test_incompatible_temporal_contract_rejected(rank, temporal, policy):
    with pytest.raises(ValueError):
        contract(rank, temporal, policy)


def test_invalid_pixel_geometry_fails_before_encoding():
    raw = Codec(5, 4)
    vae = NormalizedVideoVAE(raw, contract(5, 4, "causal_first_frame_v1"))
    for shape in [(1, 3, 8, 8, 12), (1, 3, 9, 9, 12), (1, 4, 9, 8, 12)]:
        with pytest.raises(ValueError):
            vae.encode(torch.zeros(shape))
    assert not raw.calls


def test_false_codec_geometry_is_rejected():
    raw = Codec(5, 1)
    vae = NormalizedVideoVAE(raw, contract(5, 4, "causal_first_frame_v1"))
    with pytest.raises(ValueError, match="latent geometry"):
        vae.encode(torch.zeros(1, 3, 9, 8, 12)).latent_dist.mode()
    with pytest.raises(ValueError, match="pixel geometry"):
        vae.decode(torch.zeros(1, 2, 3, 4, 6))


@pytest.mark.parametrize("sigma_kind", ["scalar", "item", "token"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_flow_boundary_matches_independent_convolution_and_gradients(sigma_kind, dtype):
    g = torch.Generator().manual_seed(47)
    bridge = SpatialLatentBridge(2, 4).to(dtype=dtype)
    bridge.initialize(torch.randn(18, 4, generator=g).to(dtype), torch.randn(4, generator=g).to(dtype))
    tokens = torch.randn(2, 3*4*6, 2, generator=g, dtype=dtype, requires_grad=True)
    sigma = {"scalar": torch.tensor(.3), "item": torch.tensor([0., 1.]),
        "token": torch.linspace(0., 1., 144).reshape(2, 72)}[sigma_kind].to(dtype)
    actual = map_video_tokens(bridge, tokens, grid=(3, 4, 6), sigma=sigma)
    frames = tokens.reshape(2, 3, 4, 6, 2).permute(0, 1, 4, 2, 3).reshape(6, 2, 4, 6)
    weight, bias = bridge.projection.weight, bridge.projection.bias
    linear = F.conv2d(frames, weight, padding=1).reshape(2, 3, 4, 4, 6).permute(0, 1, 3, 4, 2).reshape(2, 72, 4)
    coefficient = sigma if sigma_kind == "scalar" else sigma[:, None, None] if sigma_kind == "item" else sigma[:, :, None]
    torch.testing.assert_close(actual, linear+(1-coefficient)*bias)
    velocity = map_video_tokens(bridge, tokens, grid=(3, 4, 6), velocity=True)
    torch.testing.assert_close(velocity, linear-bias)
    (actual.float().square().mean()+velocity.float().square().mean()).backward()
    assert tokens.grad.isfinite().all()
    assert bridge.projection.weight.grad.abs().sum() > 0
    assert bridge.projection.bias.grad.abs().sum() > 0


def test_token_boundary_rejects_wrong_grid_and_sigma():
    bridge = SpatialLatentBridge(2, 4)
    tokens = torch.zeros(2, 72, 2)
    for grid in [(3, 4, 7), (0, 4, 6)]:
        with pytest.raises(ValueError):
            map_video_tokens(bridge, tokens, grid=grid, sigma=.5)
    with pytest.raises(ValueError, match="sigma"):
        map_video_tokens(bridge, tokens, grid=(3, 4, 6), sigma=None)
    with pytest.raises(ValueError, match="Sigma|sigma"):
        map_video_tokens(bridge, tokens, grid=(3, 4, 6), sigma=torch.zeros(72))


def test_calibration_preserves_first_frame_and_only_resamples_teacher():
    source = torch.arange(9.).reshape(1, 1, 9, 1, 1).expand(1, 1, 9, 8, 12).clone()
    source[:, :, 0] = 100
    before = source.clone()
    reduced = calibration_video_grid(source, (3, 4, 6))
    torch.testing.assert_close(reduced[:, :, 0], torch.full((1, 1, 4, 6), 100.))
    torch.testing.assert_close(reduced[0, 0, :, 0, 0], torch.tensor([100., 2.5, 6.5]))
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    expanded = calibration_video_grid(reduced, (9, 8, 12))
    torch.testing.assert_close(expanded[:, :, 0], source[:, :, 0])
    with pytest.raises(ValueError, match="still-image"):
        calibration_video_grid(source[:, :, :1], (3, 4, 6))


def test_paired_clip_calibration_fits_both_directions():
    declaration = contract(5, 4, "causal_first_frame_v1").image.declaration(
        source="synthetic", source_identity="source", target_identity="target", old_channels=2, old_scale=1)
    alignment = SpatialAlignmentPair(declaration)
    fit = PairedVideoCalibration(alignment, ridge=1e-8)
    source = torch.randn(2, 2, 9, 8, 12, generator=torch.Generator().manual_seed(5))
    teacher = calibration_video_grid(source, (3, 4, 6))
    target = teacher*2+.4
    fit.add(source, target)
    fit.finish()
    assert alignment.ready
    torch.testing.assert_close(alignment.input(target), teacher, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(alignment.output(teacher), target, atol=2e-6, rtol=2e-6)
