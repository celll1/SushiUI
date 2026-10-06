"""Audio source resolution against real Oobleck headers and synthetic waveforms."""

import pytest
import torch

from core.models.common import vae_source as vs


def small_codec():
    from diffusers import AutoencoderOobleck
    return AutoencoderOobleck(encoder_hidden_size=6, decoder_input_channels=3,
        decoder_channels=4, downsampling_ratios=[2, 4], channel_multiples=[1, 2],
        audio_channels=2, sampling_rate=48000).eval()


def test_directory_resolves_and_loads_actual_audio_codec(tmp_path):
    codec = small_codec()
    codec.save_pretrained(tmp_path)
    resolved = vs.resolve_vae_source(f"file:{tmp_path}", load_weights=True)
    facts = resolved.facts()
    assert (facts["ndim"], facts["latent_channels"], facts["scale_factor"], facts["scale_temporal"]) == (3, 3, 1, 8)
    assert facts["norm"] == "identity"
    assert facts["domain"] == "audio"
    assert facts["sample_rate"] == 48000
    assert facts["waveform_channels"] == facts["output_channels"] == 2
    assert "pixel_channels" not in facts
    restored = resolved.load_module(torch.float32)
    waveform = torch.randn(1, 2, 256)
    with torch.no_grad():
        expected = codec.encode(waveform).latent_dist.mode()
        actual = restored.encode(waveform).latent_dist.mode()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(restored.decode(actual).sample,
                                   codec.decode(expected).sample, rtol=0, atol=0)
    assert not vs.check_vae_compatibility(facts, "sdxl")[0]


@pytest.mark.parametrize("change", [
    {"decoder_input_channels": 4}, {"audio_channels": 1},
    {"downsampling_ratios": [2, 3]}, {"sampling_rate": None},
    {"channel_multiples": []}, {"scaling_factor": 2.0},
])
def test_config_header_disagreement_is_rejected(change):
    codec = small_codec()
    config = dict(codec.config, _class_name="AutoencoderOobleck", **change)
    with pytest.raises(vs.VaeSourceError):
        vs._resolve_facts(source="file:synthetic", family="custom", declared={},
            config=config, shapes={k: tuple(t.shape) for k, t in codec.state_dict().items()})


def test_header_inspection_does_not_allocate_codec(tmp_path, monkeypatch):
    codec = small_codec()
    codec.save_pretrained(tmp_path)
    import diffusers
    monkeypatch.setattr(diffusers.AutoencoderOobleck, "from_pretrained",
                        lambda *a, **k: pytest.fail("inspection allocated codec"))
    resolved = vs.resolve_vae_source(f"file:{tmp_path}", load_weights=False)
    assert resolved.state_dict is None
    assert resolved.scale_temporal == 8


def test_audio_sample_rate_is_part_of_latent_identity():
    config = dict(small_codec().config, _class_name="AutoencoderOobleck")
    assert vs.latent_space_hash("sameweights", config) != vs.latent_space_hash(
        "sameweights", dict(config, sampling_rate=44100))
