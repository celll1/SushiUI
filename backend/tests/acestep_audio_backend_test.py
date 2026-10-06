"""Generation backend staging with actual tiny codecs and synthetic audio."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from core.pipeline_backends.acestep import AceStepMixin
from acestep_audio_training_test import loaded_training


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 8)

    def forward(self, input_ids):
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids))


class Backend(AceStepMixin):
    def __init__(self, components):
        self.acestep_components = components
        self.device = torch.device("cpu")
        self.is_acestep_model = True
        self.moves = []

    def _acestep_move(self, name, device):
        self.moves.append((name, str(device)))
        return super()._acestep_move(name, device)

    def _apply_or_clear_lora_acestep(self, loras):
        assert not loras

    def _acestep_runtime_int8(self, *a, **kw):
        pass


def backend(tmp_path):
    components, source, _ = loaded_training(tmp_path, 4, True, noninteger=True)
    components["dit"].config.timbre_fix_frame = 16
    from core.models.acestep.residual_unet import decoder_identity_config, restore_topology
    from core.rewiring.spec import build_spec, OPERATORS
    graph = build_spec("acestep", [{"name": "main", "block_count": 7}], [
        dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
             down_factor=[2, 1], **OPERATORS)], decoder_identity_config(components["dit"]))
    restore_topology(components["dit"], graph)
    components["text_encoder"] = Encoder()
    components["tokenizer"] = lambda *a, **kw: SimpleNamespace(input_ids=torch.ones(1, 3, dtype=torch.long),
                                                              attention_mask=torch.ones(1, 3, dtype=torch.long))
    return Backend(components), source


def test_silence_uses_original_codec_and_preserves_cached_asset(tmp_path):
    owner, source = backend(tmp_path)
    with patch.object(owner.acestep_components["vae"], "encode", side_effect=AssertionError("target cannot create RVQ silence")):
        silence = owner._acestep_ensure_silence_latent("cpu")
        again = owner._acestep_ensure_silence_latent("cpu")
    with torch.no_grad():
        expected = source.encode(torch.zeros(1, 2, 16 * source.hop_length)).latent_dist.mode().transpose(1, 2)
    torch.testing.assert_close(silence, expected, rtol=0, atol=0)
    torch.testing.assert_close(again, expected, rtol=0, atol=0)
    assert [name for name, device in owner.moves] == ["source_vae", "source_vae"]


def test_failed_source_encode_restores_codec_and_does_not_cache_silence(tmp_path):
    owner, source = backend(tmp_path)
    with patch.object(source, "encode", side_effect=RuntimeError("codec failure")):
        with pytest.raises(RuntimeError, match="codec failure"):
            owner._acestep_ensure_silence_latent("cpu")
    assert owner.moves[-1] == ("source_vae", "cpu")
    assert "silence_latent" not in owner.acestep_components


@pytest.mark.parametrize("mode", ["text", "cover", "repaint", "outpaint"])
def test_backend_generates_and_decodes_target_codec(mode, tmp_path):
    owner, source = backend(tmp_path)
    target = owner.acestep_components["vae"]
    reference = torch.randn(2, 389).clamp(-.5, .5)
    params = dict(prompt="synthetic", seed=91, inference_steps=1, audio_duration=.1,
                  mode=mode, repaint_start=.002, repaint_end=.005,
                  total_duration=.02, input_offset_sec=.004)
    with patch.object(target, "encode", side_effect=AssertionError("target cannot encode source conditions")), \
            patch.object(source, "decode", side_effect=AssertionError("source cannot decode generated latents")), \
            patch.object(owner, "_acestep_load_reference_audio", return_value=(reference, 48000)), \
            patch.object(owner, "_acestep_apply_repaint_waveform_splice", wraps=owner._acestep_apply_repaint_waveform_splice) as splice:
        if mode == "text":
            waveform, rate, seed = owner._generate_txt2aud_acestep(params)
            assert waveform.shape == (2, 4800)
        elif mode == "outpaint":
            waveform, rate, seed = owner._generate_audoutpaint_acestep(params, "synthetic")
            assert waveform.shape == (2, 960)
            torch.testing.assert_close(waveform[:, 192:192+389], reference, rtol=0, atol=0)
        else:
            waveform, rate, seed = owner._generate_aud2aud_acestep(params, "synthetic")
            assert waveform.shape == (2, 389)
            if mode == "repaint":
                assert splice.call_args.kwargs["start_sec"] == 96 / 48000
                assert splice.call_args.kwargs["end_sec"] == 240 / 48000
    assert rate == 48000 and seed == 91
    assert torch.isfinite(waveform).all()
    assert ("source_vae", "cpu") in owner.moves
    assert owner.moves[-1] == ("vae", "cpu")
    assert params["dit_rewiring_actual"]["input_frames"] < (600 if mode == "text" else 120 if mode == "outpaint" else 49)
