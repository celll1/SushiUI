"""Restored audio codec ownership and real small native ACE training steps."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from acestep_audio_artifact_test import artifact


def loaded_training(tmp_path, new, rewired, *, noninteger=False, for_training=True):
    from core.models.acestep.loader import load_acestep_from_path
    _,source,_,path = artifact(tmp_path,new,rewired)
    if noninteger:
        from acestep_audio_artifact_test import codec
        from acestep_vae_boundary_test import small_owner
        from core.models.components.audio_latent_adapter import AudioCodecContract,AudioAlignmentPair,PairedAudioCalibration
        from core.models.acestep.vae_boundary import install_decoder_alignment
        from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
        source,target,owner = codec(2,[2,4]),codec(new,[2,6]),small_owner()
        pair = AudioAlignmentPair(AudioCodecContract.from_oobleck(source),AudioCodecContract.from_oobleck(target))
        fit = PairedAudioCalibration(pair,ridge=1e-3)
        fit.add(torch.randn(2,2,96),torch.randn(2,new,64))
        fit.finish()
        install_decoder_alignment(owner,pair)
        AceStepFullParameterAdapter(SimpleNamespace(transformer=owner,vae=target,acestep_source_vae=source)).write_checkpoint(1,1,path)
    layout = dict(dit=str(path), vae=str(tmp_path/"source.safetensors"),
        text_encoder=str(tmp_path/"te.safetensors"),root=str(tmp_path),tokenizer_source=str(tmp_path))
    encoder = nn.Linear(8,8)
    from safetensors.torch import save_file
    save_file(source.state_dict(),layout["vae"])
    save_file(encoder.state_dict(),layout["text_encoder"])
    with patch("core.models.acestep.loader.detect_acestep_layout",return_value=layout), \
            patch("core.models.acestep.loader._build_vae",return_value=source) as build_source, \
            patch("core.models.acestep.loader._build_text_encoder",return_value=(encoder,None)), \
            patch("transformers.AutoTokenizer.from_pretrained",return_value=object()):
        components = load_acestep_from_path(str(path),torch.float32,for_training=for_training)
        build_source.assert_called_once_with(layout["vae"],torch.float32)
    return components,source,path


@pytest.mark.parametrize("new,rewired,noninteger", [(1,False,False),(4,True,False),(4,False,True)])
def test_restored_target_encoding_and_real_ace_train_step(new,rewired,noninteger,tmp_path):
    from core.training.ops.acestep_ops import _build_silence_latent, vae_encode_audio, train_step
    components,source,_ = loaded_training(tmp_path,new,rewired,noninteger=noninteger)
    model = components["dit"]
    model.config.timbre_fix_frame = 16
    trainer = SimpleNamespace(vae=components["vae"],acestep_source_vae=components["source_vae"],
        transformer=model,acestep_dit_config=model.config,log_prefix="[CPU]",device=torch.device("cpu"),
        training_dtype=torch.float32,mixed_precision=False,timestep_sampler=None,reconstruction_loss_weight=0.)
    hop = 12 if noninteger else 8
    assert components["latent_frame_rate"] == 48000/hop
    assert components["latent_channels"] == new
    assert not any(p.requires_grad for p in trainer.acestep_source_vae.parameters())
    assert all(p.device.type == "cpu" for p in trainer.acestep_source_vae.parameters())
    _build_silence_latent(trainer)
    with torch.no_grad():
        expected = source.encode(torch.zeros(1,2,16*source.hop_length)).latent_dist.mode().transpose(1,2)
    torch.testing.assert_close(trainer.acestep_silence_latent,expected,rtol=0,atol=0)
    assert expected.shape == (1,16,2)
    waveform = torch.randn(2,389)
    with patch.object(trainer.acestep_source_vae,"encode",side_effect=AssertionError("Source codec stays unused after silence setup")):
        torch.manual_seed(215)
        latents = vae_encode_audio(trainer,waveform)
        padding = (-389) % hop
        assert latents.shape == (1,(389+padding)//hop,new)
        torch.manual_seed(215)
        with torch.no_grad():
            expected = trainer.vae.encode(F.pad(waveform[None],(0,padding))).latent_dist.sample().transpose(1,2)
        torch.testing.assert_close(latents,expected,rtol=0,atol=0)
        before = {key:value.clone() for key,value in model.state_dict().items()}
        model.decoder.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.decoder.parameters(),lr=1e-3)
        aux = dict(text_attention_mask=torch.ones(1,3),lyric_hidden_states=torch.randn(1,4,8),lyric_attention_mask=torch.ones(1,4))
        loss,_,_ = train_step(trainer,latents,torch.randn(1,3,8),aux,timesteps=torch.tensor([.5]))
        assert torch.isfinite(loss)
        loss.backward()
        assert model.decoder._audio_latent_alignment.input.projection.weight.grad is not None
        assert model.decoder._audio_latent_alignment.output.projection.weight.grad is not None
        optimizer.step()
        assert not torch.equal(before["decoder.proj_out.1.weight"],model.state_dict()["decoder.proj_out.1.weight"])
        for key,value in model.state_dict().items():
            if not key.startswith("decoder."):
                torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_loader_rejects_changed_source_before_external_text_loading(tmp_path):
    from core.models.acestep.loader import load_acestep_from_path
    _,source,_,path = artifact(tmp_path,4,False)
    with torch.no_grad():
        next(source.parameters()).add_(1)
    layout = dict(dit=str(path),vae="source",text_encoder="text",root=str(tmp_path))
    with patch("core.models.acestep.loader.detect_acestep_layout",return_value=layout), \
            patch("core.models.acestep.loader._build_vae",return_value=source), \
            patch("core.models.acestep.loader._build_text_encoder",side_effect=AssertionError("Source verification precedes TE loading")):
        with pytest.raises(ValueError,match="conditioning codec identity"):
            load_acestep_from_path(str(path),torch.float32,for_training=True)


def test_same_weights_different_audio_padding_separate_cache_identity(tmp_path):
    from core.training.vae_swap import module_latent_hash
    from core.models.components.audio_latent_adapter import AudioCodecContract,AudioAlignmentPair
    from core.models.acestep.vae_boundary import bind_encoding_contract
    components,_,_ = loaded_training(tmp_path,4,False)
    codec = components["vae"]
    first = module_latent_hash(codec)
    pair = components["dit"].decoder._audio_latent_alignment
    other = AudioAlignmentPair(AudioCodecContract(2,48000,2,12,"aligned_hop_v1"),pair.target)
    bind_encoding_contract(codec,other)
    second = module_latent_hash(codec)
    assert first and second and first == second
    codec._sushi_audio_encoding_contract.update(sample_alignment=24,policy="zero_tail_lcm_v1")
    assert module_latent_hash(codec) != first
    bind_encoding_contract(codec,pair)
    assert module_latent_hash(codec) == first


def test_training_owner_releases_source_codec_and_keeps_rvq_frozen(tmp_path):
    from core.training.ops.acestep_ops import load_components
    from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
    components,_,path = loaded_training(tmp_path,4,True)
    components["dit"].config.timbre_fix_frame = 16
    trainer = SimpleNamespace(model_path=str(path),weight_dtype=torch.float32,vae_dtype=torch.float32,
        device=torch.device("cpu"),log_prefix="[CPU]",attention_backend="sdpa",config={},
        gradient_checkpointing=False,blocks_to_swap=0,train_unet=True)
    with patch("core.model_loader.ModelLoader.load_acestep_from_path",return_value=components) as load, \
            patch("core.training.ops.acestep_ops._build_empty_lyrics"), \
            patch("core.training.ops.acestep_ops.setup_attention_backend"):
        load_components(trainer)
        load.assert_called_once_with(str(path),torch.float32,for_training=True)
    assert trainer.acestep_source_vae is None
    assert components["source_vae"] is None
    assert trainer.acestep_silence_latent.shape == (1,16,2)
    assert trainer.vae_latent_channels == 4
    assert trainer.acestep_latent_frame_rate == 48000/8
    adapter = AceStepFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    for name in ("encoder","tokenizer","detokenizer"):
        assert not getattr(trainer.transformer,name).training
        assert not any(p.requires_grad for p in getattr(trainer.transformer,name).parameters())
    assert not trainer.transformer.null_condition_emb.requires_grad
    assert any(p.requires_grad for p in trainer.transformer.decoder.parameters())
    assert not any(p.requires_grad for p in trainer.vae.parameters())
    output = tmp_path/"source_released.safetensors"
    adapter.write_checkpoint(10,2,output)
    assert output.is_file()
