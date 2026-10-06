"""New ACE audio swaps with real tiny codecs and synthetic audio files."""

from types import SimpleNamespace

import pytest
import torch

from acestep_audio_artifact_test import codec
from acestep_vae_boundary_test import small_owner
from ltx2_vae_training_test import Trainer


def setup(tmp_path, new=4, rewired=True):
    from core.models.common.vae_source import resolve_vae_source
    from core.training.arch.acestep import AceStepArchHandler
    model = small_owner()
    model.config.timbre_fix_frame = 80
    if rewired:
        from core.models.acestep.residual_unet import native_config, restore_topology
        from core.rewiring.spec import build_spec, OPERATORS
        graph = build_spec("acestep", [{"name": "main", "block_count": 7}], [
            dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
                 down_factor=[2, 1], **OPERATORS)], native_config(model.config))
        restore_topology(model, graph)
    source = codec(2, [2, 2])
    target = codec(new, [2, 4])
    target.save_pretrained(tmp_path / "target")
    resolved = resolve_vae_source(f"file:{tmp_path / 'target'}", load_weights=True)
    trainer = Trainer(transformer=model, vae=source, vae_dtype=torch.float32,
        config={"vae_swap_new_channel_init": "auto", "vae_swap_calibration_samples": 2},
        device=torch.device("cpu"), text_encoder=torch.nn.Linear(8, 8),
        events=[], checks=0, stop_at=None, optimizer=None, log_prefix="[CPU]",
        acestep_dit_config=model.config, acestep_source_vae=None)
    trainer.arch = AceStepArchHandler(trainer)
    return trainer, resolved, source


def audio_files(tmp_path):
    import soundfile
    items = []
    for i in range(2):
        path = tmp_path / f"synthetic_{i}.wav"
        value = torch.randn(391, 2, generator=torch.Generator().manual_seed(i)).clamp(-1, 1)
        soundfile.write(path, value.numpy(), 48000, subtype="FLOAT")
        items.append({"audio_path": str(path)})
    return [SimpleNamespace(items=items)]


@pytest.mark.parametrize("new,rewired", [(1, False), (4, True)])
def test_new_swap_calibration_export_restore(new, rewired, tmp_path):
    from core.models.acestep.vae_training import PENDING
    from core.models.acestep.audio_artifact import export_bundle, verify_source_codec
    from core.models.acestep.loader import _build_dit
    from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
    from core.training.ops.acestep_ops import _build_silence_latent, vae_encode_audio
    trainer, resolved, source = setup(tmp_path, new, rewired)
    _build_silence_latent(trainer)
    source_state = {k: v.clone() for k, v in source.state_dict().items()}
    report = trainer.arch.apply_vae_swap(trainer, resolved)
    pair = trainer.transformer.decoder._audio_latent_alignment
    assert report.changed and not pair.ready
    assert trainer.wiring.latent_channels == new
    assert trainer.wiring.vae_scale_temporal == 8
    assert trainer.acestep_latent_frame_rate == 6000
    with pytest.raises(ValueError, match="uncalibrated"):
        export_bundle(trainer.transformer, trainer.vae)
    progress = []
    trainer._vae_alignment_progress_callback = lambda **data: progress.append(data)
    trainer.arch.calibrate_before_training(trainer, audio_files(tmp_path))
    assert pair.ready and not hasattr(trainer, PENDING)
    assert trainer.events == ["park", "restore"]
    assert [p["step"] for p in progress] == [1, 2, 3, 4]
    assert all(p["total"] == 4 for p in progress)
    verify_source_codec(trainer.transformer, source)
    for k, v in source.state_dict().items():
        torch.testing.assert_close(v, source_state[k], rtol=0, atol=0)
    path = tmp_path / "aligned.safetensors"
    AceStepFullParameterAdapter(trainer).write_checkpoint(7, 1, path)
    restored, _ = _build_dit(str(path), torch.float32)
    assert restored.decoder._audio_latent_alignment.ready
    assert getattr(restored, "_residual_unet_spec", None) == getattr(trainer.transformer, "_residual_unet_spec", None)
    actual = vae_encode_audio(trainer, torch.randn(2, 389))
    assert actual.shape == (1, 49, new)
    trainer.arch.apply_vae_swap(trainer, resolved, module=trainer.vae)
    with torch.no_grad():
        next(source.parameters()).add_(1)
    with pytest.raises(ValueError, match="identity"):
        export_bundle(trainer.transformer, trainer.vae, source)


def test_cancelled_calibration_keeps_pending_unready_and_restores_residency(tmp_path):
    from core.models.acestep.vae_training import PENDING
    trainer, resolved, source = setup(tmp_path)
    trainer.arch.apply_vae_swap(trainer, resolved)
    trainer.stop_at = 3
    with pytest.raises(RuntimeError, match="stop requested"):
        trainer.arch.calibrate_before_training(trainer, audio_files(tmp_path))
    assert hasattr(trainer, PENDING)
    assert not trainer.transformer.decoder._audio_latent_alignment.ready
    assert trainer.events == ["park", "restore"]
    assert all(p.device.type == "cpu" for p in trainer.vae.parameters())
    assert all(p.device.type == "cpu" for p in source.parameters())
    assert not hasattr(trainer.transformer, "_audio_source_codec_identity")
    trainer.stop_at = None
    trainer.arch.calibrate_before_training(trainer, audio_files(tmp_path))
    assert trainer.transformer.decoder._audio_latent_alignment.ready


def test_calibration_reader_limits_file_decode_to_native_timbre_window(tmp_path):
    from unittest.mock import patch
    from core.models.acestep.vae_training import calibration_waveform
    from core.models.components.audio_latent_adapter import AudioCodecContract
    import soundfile
    path = tmp_path / "long_synthetic.wav"
    soundfile.write(path, torch.zeros(48000, 2).numpy(), 48000, subtype="FLOAT")
    original = soundfile.SoundFile.read
    observed = []
    def read(handle, frames, **kwargs):
        observed.append(frames)
        return original(handle, frames, **kwargs)
    with patch.object(soundfile.SoundFile, "read", read):
        value = calibration_waveform(path, .01, AudioCodecContract(2, 48000, 2, 4, "aligned_hop_v1"))
    assert observed == [480]
    assert value.shape == (2, 480)


def test_changed_waveform_between_passes_is_rejected(tmp_path):
    trainer, resolved, _ = setup(tmp_path)
    trainer.arch.apply_vae_swap(trainer, resolved)
    datasets = audio_files(tmp_path)
    def progress(**data):
        if data["step"] == 2:
            import soundfile
            soundfile.write(datasets[0].items[0]["audio_path"], torch.ones(391, 2).numpy(), 48000, subtype="FLOAT")
    trainer._vae_alignment_progress_callback = progress
    with pytest.raises(ValueError, match="changed between"):
        trainer.arch.calibrate_before_training(trainer, datasets)
    assert not trainer.transformer.decoder._audio_latent_alignment.ready


def test_incompatible_audio_is_rejected_before_model_changes(tmp_path):
    from dataclasses import replace
    trainer, resolved, _ = setup(tmp_path)
    state = {k: v.clone() for k, v in trainer.transformer.state_dict().items()}
    bad = replace(resolved, config=dict(resolved.config, sampling_rate=44100))
    with pytest.raises(ValueError, match="48kHz"):
        trainer.arch.apply_vae_swap(trainer, bad)
    for k, v in trainer.transformer.state_dict().items():
        torch.testing.assert_close(v, state[k], rtol=0, atol=0)


def test_component_owner_connects_new_swap_before_full_adapter(tmp_path):
    from unittest.mock import patch
    from core.training.ops.acestep_ops import load_components
    from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
    from core.models.acestep.vae_training import PENDING
    trainer, resolved, source = setup(tmp_path, rewired=False)
    trainer.model_path = str(tmp_path / "native.safetensors")
    trainer.weight_dtype = torch.float32
    trainer.attention_backend = "sdpa"
    trainer.gradient_checkpointing = False
    trainer.blocks_to_swap = 0
    trainer.train_unet = True
    trainer.trains_base_weights = True
    trainer.config["vae_swap_source"] = resolved.source
    components = dict(dit=trainer.transformer, dit_config=trainer.transformer.config,
                      vae=source, text_encoder=trainer.text_encoder, tokenizer=object(),
                      sample_rate=48000, latent_frame_rate=12000, latent_channels=2)
    with patch("core.model_loader.ModelLoader.load_acestep_from_path", return_value=components), \
            patch("core.training.ops.acestep_ops._build_empty_lyrics"), \
            patch("core.training.ops.acestep_ops.setup_attention_backend"):
        load_components(trainer)
    assert trainer.acestep_source_vae is None
    assert getattr(trainer, PENDING) is source
    assert trainer.acestep_silence_latent.shape == (1, 80, 2)
    assert trainer.vae_latent_channels == 4
    assert trainer.vae_identity.identity_native is False
    adapter = AceStepFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    assert trainer.transformer.decoder._audio_latent_alignment.input.projection.weight.requires_grad
    assert not any(p.requires_grad for p in trainer.transformer.tokenizer.parameters())
    trainer.arch.calibrate_before_training(trainer, audio_files(tmp_path))
    assert trainer.transformer.decoder._audio_latent_alignment.ready
    from core.training.ops.acestep_ops import vae_encode_audio, train_step
    trainer.training_dtype = torch.float32
    trainer.mixed_precision = False
    trainer.timestep_sampler = None
    trainer.reconstruction_loss_weight = 0.
    latents = vae_encode_audio(trainer, torch.randn(2, 389))
    aux = dict(text_attention_mask=torch.ones(1, 3), lyric_hidden_states=torch.randn(1, 4, 8),
               lyric_attention_mask=torch.ones(1, 4))
    loss, _, _ = train_step(trainer, latents, torch.randn(1, 3, 8), aux, timesteps=torch.tensor([.5]))
    assert torch.isfinite(loss)
    loss.backward()
    pair = trainer.transformer.decoder._audio_latent_alignment
    assert torch.isfinite(pair.input.projection.weight.grad).all()
    assert torch.isfinite(pair.output.projection.weight.grad).all()
