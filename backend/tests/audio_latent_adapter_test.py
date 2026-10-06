"""Synthetic audio geometry, regression and flow-boundary acceptance."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from core.models.components.audio_latent_adapter import (
    AudioCodecContract, AudioAlignmentPair, TemporalLatentBridge,
    TemporalBridgeCalibrator, PairedAudioCalibration, calibration_audio_grid,
)


def contract(channels=2, hop=4):
    return AudioCodecContract(channels, 48000, 2, hop, "aligned_hop_v1")


def test_real_oobleck_aligned_geometry():
    from diffusers import AutoencoderOobleck
    codec = AutoencoderOobleck(encoder_hidden_size=4, decoder_input_channels=2,
        decoder_channels=4, downsampling_ratios=[2, 4], channel_multiples=[1, 2],
        audio_channels=2, sampling_rate=48000).eval()
    geometry = AudioCodecContract.from_oobleck(codec)
    with torch.no_grad():
        for samples in (8, 16, 64):
            value = codec.encode(torch.zeros(1, 2, samples)).latent_dist.mode()
            assert value.shape == (1, 2, geometry.latent_frames(samples))
            assert codec.decode(value).sample.shape == (1, 2, samples)
    assert geometry.waveform_samples(8) == 64
    with pytest.raises(ValueError, match="multiple"):
        geometry.latent_frames(63)


def test_codec_contract_refuses_implicit_resampling_and_unknown_frame_policy():
    with pytest.raises(ValueError, match="sample rates"):
        contract().check_pair(replace(contract(), sample_rate=44100))
    with pytest.raises(ValueError, match="sample rates"):
        contract().check_pair(replace(contract(), waveform_channels=1))
    with pytest.raises(ValueError, match="sample-count"):
        replace(contract(), frame_policy="guess")
    codec = SimpleNamespace(config=dict(downsampling_ratios=[3], decoder_input_channels=2,
        encoder_hidden_size=4), hop_length=3)
    with pytest.raises(ValueError, match="even"):
        AudioCodecContract.from_oobleck(codec)


def test_three_tap_regression_and_flow_bias_oracle():
    torch.manual_seed(25)
    bridge = TemporalLatentBridge(2, 3)
    with pytest.raises(RuntimeError, match="calibration"):
        bridge(torch.zeros(1, 2, 4))
    x = torch.randn(3, 2, 200)
    w, bias = torch.randn(3, 2, 3), torch.tensor([.2, -.7, 1.1])
    y = F.conv1d(x, w, bias, padding=1)
    fit = TemporalBridgeCalibrator(bridge, ridge=1e-9)
    for a, b in zip(x, y):
        fit.add(a[None], b[None])
    fit.finish()
    torch.testing.assert_close(bridge(x), y, atol=2e-6, rtol=2e-6)
    noise, sigma = torch.randn_like(x), torch.tensor([0., .3, 1.])
    xt = (1-sigma[:, None, None])*x + sigma[:, None, None]*noise
    expected = (1-sigma[:, None, None])*y + sigma[:, None, None]*F.conv1d(noise, w, padding=1)
    torch.testing.assert_close(bridge.flow_state(xt, sigma), expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(bridge.flow_velocity(noise-x), F.conv1d(noise-x, w, padding=1)-bias[None,:,None], atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError, match="sigma"):
        bridge.flow_state(x, sigma[:, None])
    restored = TemporalLatentBridge(2, 3)
    restored.load_state_dict(bridge.state_dict())
    torch.testing.assert_close(restored(x), bridge(x))
    bad = {k: v.clone() for k, v in bridge.state_dict().items()}
    bad["bridge_spec"][0] = 9
    with pytest.raises(ValueError, match="checkpoint"):
        restored.load_state_dict(bad)


@pytest.mark.parametrize("new", [1, 4])
def test_target_grid_calibration_expansion_and_trainable_extra_channels(new, tmp_path):
    torch.manual_seed(10)
    alignment = AudioAlignmentPair(contract(2, 4), contract(new, 8))
    rng = torch.get_rng_state().clone()
    fit = PairedAudioCalibration(alignment, ridge=1e-4)
    assert torch.equal(rng, torch.get_rng_state())
    target = torch.randn(2, new, 32)
    source = torch.randn(2, 2, 64)
    fit.add(source, target)
    fit.finish()
    assert alignment.ready
    internal = alignment.input.flow_state(target, .5)
    assert internal.shape == (2, max(2, new), 32)
    if new > 2:
        torch.testing.assert_close(internal[:, 2:], target[:, 2:])
    prediction = alignment.output.flow_velocity(internal)
    assert prediction.shape == target.shape
    prediction.square().mean().backward()
    assert alignment.input.projection.weight.grad is not None
    assert alignment.output.projection.weight.grad is not None
    if new > 2:
        assert torch.count_nonzero(alignment.input.projection.weight.grad[2:]) > 0
    restored = AudioAlignmentPair(contract(2, 4), contract(new, 8))
    from safetensors.torch import load_file, save_file
    path = tmp_path / "alignment.safetensors"
    save_file(alignment.state_dict(), str(path))
    restored.load_state_dict(load_file(str(path)))
    assert restored.ready
    torch.testing.assert_close(restored.output(restored.input(target)), alignment.output(alignment.input(target)))
    torch.testing.assert_close(calibration_audio_grid(source, 32), F.avg_pool1d(source, 2))
    wrong_hop = AudioAlignmentPair(contract(2, 4), contract(new, 16))
    with pytest.raises(ValueError, match="codec geometry"):
        wrong_hop.load_state_dict(alignment.state_dict())


def test_invalid_calibration_does_not_mark_ready():
    bridge = TemporalLatentBridge(2, 1)
    fit = TemporalBridgeCalibrator(bridge, ridge=1e-3)
    with pytest.raises(ValueError, match="insufficient"):
        fit.finish()
    with pytest.raises(ValueError, match="finite"):
        fit.add(torch.full((1, 2, 4), float("nan")), torch.zeros(1, 1, 4))
    with pytest.raises(ValueError, match="grids"):
        fit.add(torch.zeros(1, 2, 5), torch.zeros(1, 1, 4))
    assert not bridge._ready
