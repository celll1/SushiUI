"""Small real ACE/Oobleck bundled artifact roundtrips, using synthetic audio."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open

from acestep_vae_boundary_test import small_owner, alignment


def codec(channels, ratios):
    from diffusers import AutoencoderOobleck
    return AutoencoderOobleck(encoder_hidden_size=2*channels, decoder_input_channels=channels,
        decoder_channels=4, downsampling_ratios=ratios, channel_multiples=[1,2],
        audio_channels=2, sampling_rate=48000).eval()


def artifact(tmp_path, new, rewired):
    from core.models.acestep.vae_boundary import install_decoder_alignment
    from core.models.acestep.residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec, OPERATORS
    from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
    torch.manual_seed(35)
    owner, source, target = small_owner(), codec(2,[2,2]), codec(new,[2,4])
    if rewired:
        graph = build_spec("acestep",[{"name":"main","block_count":7}],[
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,1],**OPERATORS)],native_config(owner.config))
        restore_topology(owner, graph)
    install_decoder_alignment(owner, alignment(new))
    trainer = SimpleNamespace(transformer=owner, vae=target, acestep_source_vae=source)
    path = tmp_path/"audio.safetensors"
    AceStepFullParameterAdapter(trainer).write_checkpoint(9,1,path)
    return owner, source, target, path


@pytest.mark.parametrize("new,rewired", [(1,False),(4,False),(1,True),(4,True)])
def test_real_model_and_codec_bundle_roundtrip(new, rewired, tmp_path):
    from core.models.acestep.loader import _build_dit
    from core.models.acestep.audio_artifact import verify_source_codec
    from core.models.acestep.residual_unet import inspect_artifact, model_topology_metadata, temporal_geometry
    from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter
    owner, source, target, path = artifact(tmp_path,new,rewired)
    with patch("core.models.acestep.vendor.AceStepConditionGenerationModel", side_effect=AssertionError("No allocation during inspection")):
        config, graph = inspect_artifact(path)
    assert config["audio_acoustic_hidden_dim"] == max(2,new)
    assert graph == getattr(owner,"_residual_unet_spec",None)
    assert temporal_geometry(config,graph)["frame_rate"] == 48000/8
    restored, config = _build_dit(str(path),torch.float32)
    assert config.audio_acoustic_hidden_dim == 2
    assert restored.decoder.config.audio_acoustic_hidden_dim == max(2,new)
    for key,value in owner.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
    assert model_topology_metadata(restored) == model_topology_metadata(owner)
    assert restored.decoder._audio_latent_alignment.ready
    assert not any(name.startswith("_audio_restored_codec") for name in restored.state_dict())
    restored_codec = restored._audio_restored_codec
    assert not any(p.requires_grad for p in restored_codec.parameters())
    for key,value in target.state_dict().items():
        torch.testing.assert_close(restored_codec.state_dict()[key],value,rtol=0,atol=0)
    waveform = torch.randn(1,2,384)
    with torch.no_grad():
        z = target.encode(waveform).latent_dist.mode()
        torch.testing.assert_close(restored_codec.encode(waveform).latent_dist.mode(),z,rtol=0,atol=0)
        torch.testing.assert_close(restored_codec.decode(z).sample,target.decode(z).sample,rtol=0,atol=0)
    x = torch.randn(2,48,new)
    context = torch.cat((torch.randn_like(x),torch.ones_like(x)),-1)
    kwargs = dict(hidden_states=x, context_latents=context, timestep=torch.tensor([.3,.8]),
        timestep_r=torch.tensor([.3,.8]), attention_mask=torch.ones(2,48), encoder_hidden_states=torch.randn(2,5,16),
        encoder_attention_mask=torch.ones(2,5),use_cache=False)
    owner.eval()
    # Loader-selected SDPA and fixture eager attention differ at FP32 roundoff.
    torch.testing.assert_close(restored.decoder(**kwargs)[0],owner.decoder(**kwargs)[0],rtol=2e-5,atol=2e-6)
    verify_source_codec(restored,source)
    with torch.no_grad():
        next(source.parameters()).add_(.1)
    with pytest.raises(ValueError,match="conditioning codec identity"):
        verify_source_codec(restored,source)
    second = tmp_path/"resaved.safetensors"
    AceStepFullParameterAdapter(SimpleNamespace(transformer=restored,vae=restored_codec)).write_checkpoint(10,1,second)
    reread,_ = _build_dit(str(second),torch.float32)
    for key,value in restored.state_dict().items():
        torch.testing.assert_close(reread.state_dict()[key],value,rtol=0,atol=0)


def test_bad_metadata_and_orphan_weights_refused_before_allocation(tmp_path):
    from core.models.acestep.loader import _build_dit
    from core.models.acestep.audio_artifact import KEY
    _,_,_,path = artifact(tmp_path,4,True)
    state = load_file(str(path))
    with safe_open(str(path),framework="pt") as handle:
        metadata = handle.metadata()
    for mutation in ("hop","orphan","uncalibrated","missing_codec","tensor_spec"):
        data, meta = {k:v.clone() for k,v in state.items()}, dict(metadata)
        if mutation == "hop":
            record = json.loads(meta[KEY])
            record["target"]["hop_length"] = 16
            meta[KEY] = json.dumps(record)
        elif mutation == "orphan":
            del meta[KEY]
        elif mutation == "uncalibrated":
            data["net.decoder._audio_latent_alignment.input.calibrated"].fill_(False)
        elif mutation == "missing_codec":
            data = {k:v for k,v in data.items() if not k.startswith("vae.")}
        else:
            data["net.decoder._audio_latent_alignment.codec_spec"][4] = 16
        bad = tmp_path/(mutation+".safetensors")
        save_file(data,str(bad),metadata=meta)
        with patch("core.models.acestep.vendor.AceStepConditionGenerationModel",side_effect=AssertionError("No model allocation")):
            with pytest.raises(ValueError):
                _build_dit(str(bad),torch.float32)


def test_bundled_codec_corruption_is_detected(tmp_path):
    from core.models.acestep.loader import _build_dit
    _,_,_,path = artifact(tmp_path,4,False)
    state = load_file(str(path))
    with safe_open(str(path),framework="pt") as handle:
        metadata = handle.metadata()
    key = next(k for k in state if k.startswith("vae."))
    state[key].add_(1)
    bad = tmp_path/"changed_codec.safetensors"
    save_file(state,str(bad),metadata=metadata)
    with pytest.raises(ValueError,match="weight identity"):
        _build_dit(str(bad),torch.float32)
