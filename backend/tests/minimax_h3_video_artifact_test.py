"""Synthetic H3 Full bundles: actual codec, native DiT, optimizer and U-net."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from minimax_h3_video_alignment_test import setup


@pytest.fixture(autouse=True)
def suppress_import_cuda_context(monkeypatch):
    monkeypatch.setattr(torch.cuda,"get_device_capability",lambda *a,**k:(8,9))
    monkeypatch.setattr(torch.cuda,"_lazy_init",lambda *a,**k:None)
    monkeypatch.setattr(torch._C,"_cuda_init",lambda *a,**k:None)


def fixture(tmp_path,new,dtype=torch.float32,rewired=True):
    from diffusers import AutoencoderKL
    from core.models.common.vae_source import resolve_vae_source
    from core.models.components.latent_adapter import ImageCodecContract
    from core.models.components.video_latent_adapter import VideoCodecContract,NormalizedVideoVAE
    raw = AutoencoderKL(in_channels=3,out_channels=3,latent_channels=new,block_out_channels=(4,8),
        down_block_types=("DownEncoderBlock2D",)*2,up_block_types=("UpDecoderBlock2D",)*2,
        norm_num_groups=1,scaling_factor=2.,shift_factor=.25).to(dtype).eval().requires_grad_(False)
    directory = tmp_path/"codec"
    raw.save_pretrained(directory)
    identity = resolve_vae_source("file:"+str(directory),arch="minimax_h3",load_weights=True,download=False)
    contract = VideoCodecContract(ImageCodecContract.from_resolved(identity),1,"independent_frames_v1")
    model,pair,_,_ = setup(new,rewired,dtype,contract)
    pair.declaration.update(source=identity.source,source_identity="synthetic_source",target_identity=identity.latent_hash)
    return SimpleNamespace(transformer=model,vae=NormalizedVideoVAE(raw,contract),vae_identity=identity,
                           bundle_vae=True,arch=SimpleNamespace(name="minimax_h3"))


def forward(model):
    from core.models.minimax_h3.h3_pipeline_ops import build_packed_layout
    from core.models.minimax_h3.residual_unet import forward_kwargs
    inner = model.transformer
    layout = build_packed_layout(3,3,8,12,3,keyframe_anchors=("first",),video_codec_contract=inner._video_codec_contract)
    generator = torch.Generator().manual_seed(35)
    labels = torch.zeros(layout["sequence_length"],dtype=torch.long)
    labels[layout["video_indices"][:24]] = 1
    return model(hidden_states=torch.randn(1,96,model.config.in_channels*4,generator=generator),
        audio_hidden_states=torch.randn(1,6,6,generator=generator),encoder_hidden_states=torch.randn(1,3,10,generator=generator),
        timestep=torch.tensor([.3,.8]),timestep_indices=labels,return_dict=False,
        **{key:layout[key] for key in ("token_tags","position_ids","video_indices","audio_indices","text_indices")},
        **forward_kwargs(model,layout))


@pytest.mark.parametrize("new",[1,4])
@pytest.mark.parametrize("sharded",[False,True])
@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_codec_bridge_native_weight_optimizer_roundtrip(tmp_path,new,sharded,dtype):
    from core.models.minimax_h3.video_artifact import save_checkpoint,load_checkpoint,preflight_checkpoint
    trainer = fixture(tmp_path,new,dtype)
    optimizer = torch.optim.AdamW(trainer.transformer.parameters(),lr=1e-3)
    sum(value.float().square().mean() for value in forward(trainer.transformer)).backward()
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    optimizer_path = tmp_path/"optimizer.pt"
    torch.save(optimizer.state_dict(),optimizer_path)
    path = save_checkpoint(trainer,tmp_path/"full.safetensors",step=9,epoch=1,
                           max_shard_bytes=24000 if sharded else 10000000)
    md,config,record,graph = preflight_checkpoint(path)
    restored,vae,identity = load_checkpoint(path,dtype)
    restored.requires_grad_(True)
    assert graph == restored.transformer._residual_unet_spec
    assert identity.latent_hash == trainer.vae_identity.latent_hash
    assert record["source_config"]["in_channels"] == 2
    assert config["in_channels"] == max(2,new)
    assert md["step"] == "9"
    for key,value in trainer.transformer.transformer.state_dict().items():
        actual = restored.transformer.state_dict()[key]
        assert value.dtype == actual.dtype
        torch.testing.assert_close(actual,value,rtol=0,atol=0)
    for left,right in zip(forward(trainer.transformer),forward(restored)):
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
    for key,value in trainer.vae.inner.state_dict().items():
        actual = vae.inner.state_dict()[key]
        assert actual.dtype == value.dtype
        torch.testing.assert_close(actual,value,rtol=0,atol=0)
    pixels = torch.randn(1,3,1,16,24,dtype=dtype)
    latent = trainer.vae.encode(pixels).latent_dist.mode()
    torch.testing.assert_close(vae.encode(pixels).latent_dist.mode(),latent,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(vae.decode(latent).sample,trainer.vae.decode(latent).sample,rtol=1e-5,atol=1e-6)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(),lr=1e-3)
    resumed_optimizer.load_state_dict(torch.load(optimizer_path,weights_only=True))
    for model,opt in ((trainer.transformer,optimizer),(restored,resumed_optimizer)):
        sum(value.float().square().mean() for value in forward(model)).backward()
        opt.step();opt.zero_grad(set_to_none=True)
    for (name,left),(other,right) in zip(trainer.transformer.named_parameters(),restored.named_parameters()):
        assert name == other
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
    again = save_checkpoint(SimpleNamespace(transformer=restored,vae=vae,vae_identity=identity,bundle_vae=True,
        arch=trainer.arch),tmp_path/"again.safetensors",step=10,epoch=1)
    assert preflight_checkpoint(again)[2] == record
    if sharded:
        assert load_checkpoint(tmp_path/"full.safetensors",dtype)[0].config.in_channels == new


def test_corrupt_headers_identity_and_calibration_refuse(tmp_path):
    from core.models.common.single_file_format import read_state_dict,save_single_file_state
    from core.models.minimax_h3.video_artifact import save_checkpoint,preflight_checkpoint,load_checkpoint,KEY
    trainer = fixture(tmp_path,4)
    path = save_checkpoint(trainer,tmp_path/"full.safetensors",step=0,epoch=0)
    state,md = read_state_dict(str(path))
    record = json.loads(md[KEY])
    wrong_clock = dict(record,position_policy="latent_index")
    for label,modified,metadata in (
            ("missing_record",state,{k:v for k,v in md.items() if k != KEY}),
            ("missing_codec",{k:v for k,v in state.items() if not k.startswith("vae.")},md),
            ("bad_clock",state,{**md,KEY:json.dumps(wrong_clock)}),
            ("bad_hash",state,{**md,"component.vae.hash":"corrupt"}),
            ("bad_codec_geometry",state,{**md,"component.vae.channels":"99"}),
            ("bad_bridge",{**state,"net._video_latent_alignment.input.projection.weight":torch.zeros(1,1,3,3)},md),
            ("missing_block",{k:v for k,v in state.items() if not k.startswith("net.transformer_blocks.2.")},md),
            ("orphan",{**state,"net.orphan":torch.zeros(1)},md)):
        bad = save_single_file_state(modified,metadata,str(tmp_path/(label+".safetensors")))
        with patch("core.models.common.single_file_format.read_state_dict",side_effect=AssertionError("tensor load too early")):
            with pytest.raises(ValueError):
                preflight_checkpoint(bad)
    for field,value,reason in (("calibrated",torch.tensor(False),"uncalibrated"),
                               ("bridge_spec",torch.tensor([1,4,4,2,1]),"geometry/version")):
        bad = save_single_file_state({**state,"net._video_latent_alignment.input."+field:value},md,
                                     str(tmp_path/(field+".safetensors")))
        with pytest.raises(ValueError,match=reason):
            load_checkpoint(bad,torch.float32)
    weights = dict(state)
    key = next(k for k in weights if k.startswith("vae.") and weights[k].is_floating_point())
    weights[key] = weights[key]+1
    bad = save_single_file_state(weights,md,str(tmp_path/"codec_corrupt.safetensors"))
    with pytest.raises(ValueError,match="weight identity"):
        load_checkpoint(bad,torch.float32)
    with torch.no_grad():
        next(trainer.vae.inner.parameters()).add_(1)
    with pytest.raises(ValueError,match="actual codec"):
        save_checkpoint(trainer,tmp_path/"invalid.safetensors",step=0,epoch=0)


def test_shard_metadata_and_ownership_are_verified_before_tensor_load(tmp_path):
    from safetensors.torch import load_file,save_file
    from safetensors import safe_open
    from core.models.minimax_h3.video_artifact import save_checkpoint,preflight_checkpoint
    trainer = fixture(tmp_path,4)
    index = save_checkpoint(trainer,tmp_path/"full.safetensors",step=0,epoch=0,max_shard_bytes=24000)
    manifest = json.loads(index.read_text())
    shard = index.parent/next(iter(manifest["weight_map"].values()))
    with safe_open(str(shard),framework="pt",device="cpu") as handle:
        md = handle.metadata()
    state = load_file(str(shard))
    save_file(state,str(shard),metadata={**md,"component.vae.channels":"99"})
    with pytest.raises(ValueError,match="metadata disagree"):
        preflight_checkpoint(index)
    save_file(state,str(shard),metadata=md)
    manifest["weight_map"][next(iter(state))] = "../elsewhere.safetensors"
    index.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="shard mapping"):
        preflight_checkpoint(index)


def test_codec_assign_preserves_storage_and_bundle_reader_loads_each_component_once(tmp_path):
    from diffusers import AutoencoderKL
    from core.models.common.vae_source import _WeightFile,resolve_vae_source
    from core.models.minimax_h3.video_artifact import save_checkpoint,load_checkpoint
    trainer = fixture(tmp_path,4,torch.bfloat16)
    path = save_checkpoint(trainer,tmp_path/"full.safetensors",step=0,epoch=0)
    identity = resolve_vae_source("model:"+str(path),arch="minimax_h3",load_weights=True,download=False)
    original = AutoencoderKL.load_state_dict
    observed = []
    def assign(module,state,**kwargs):
        observed.append(all(value.is_meta for value in module.parameters()))
        return original(module,state,**kwargs)
    with patch.object(AutoencoderKL,"load_state_dict",assign):
        codec = identity.load_module(assign=True)
    assert observed == [True]
    for key,value in codec.state_dict().items():
        assert value.data_ptr() == identity.state_dict[key].data_ptr()
        assert value.dtype == identity.state_dict[key].dtype
    reads = []
    read = _WeightFile.tensors
    def record(owner,keys):
        keys = tuple(keys)
        reads.append(keys)
        return read(owner,keys)
    with patch.object(_WeightFile,"tensors",record):
        load_checkpoint(path,torch.bfloat16)
    assert len(reads) == 2
    assert all(key.startswith("vae.") for key in reads[0])
    assert all(key.startswith("net.") for key in reads[1])


def test_causal_video_codec_restore_keeps_its_temporal_contract_and_pixels(tmp_path):
    from ltx2_video_artifact_test import fixture as ltx_fixture
    from core.models.minimax_h3.video_artifact import save_checkpoint,load_checkpoint
    trainer,_ = ltx_fixture(tmp_path,4,codec_kind="video")
    contract = trainer.vae.contract
    model,pair,_,_ = setup(4,True,torch.float32,contract)
    pair.declaration.update(source=trainer.vae_identity.source,source_identity="synthetic_source",
                            target_identity=trainer.vae_identity.latent_hash)
    trainer.transformer = model
    trainer.arch = SimpleNamespace(name="minimax_h3")
    path = save_checkpoint(trainer,tmp_path/"causal.safetensors",step=0,epoch=0,max_shard_bytes=24000)
    restored,vae,identity = load_checkpoint(path,torch.float32)
    restored.requires_grad_(True)
    assert vae.contract == contract
    assert identity.scale_temporal == 2
    for left,right in zip(forward(model),forward(restored)):
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
    pixels = torch.randn(1,3,5,16,24)
    latent = trainer.vae.encode(pixels).latent_dist.mode()
    assert latent.shape == (1,4,3,8,12)
    torch.testing.assert_close(vae.encode(pixels).latent_dist.mode(),latent,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(vae.decode(latent).sample,trainer.vae.decode(latent).sample,rtol=1e-5,atol=1e-6)


def test_dense_restore_and_invalid_depth_curve_refuse_before_model_construction(tmp_path):
    from core.models.common.single_file_format import read_state_dict,save_single_file_state
    from core.models.minimax_h3.video_artifact import save_checkpoint,load_checkpoint,preflight_checkpoint,KEY
    from core.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
    trainer = fixture(tmp_path,4,rewired=False)
    path = save_checkpoint(trainer,tmp_path/"dense.safetensors",step=0,epoch=0)
    restored,_,_ = load_checkpoint(path,torch.float32)
    restored.requires_grad_(True)
    assert preflight_checkpoint(path)[3] is None
    for left,right in zip(forward(trainer.transformer),forward(restored)):
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
    state,md = read_state_dict(str(path))
    for field in ("num_layers","adaln_curve_grid"):
        config,record = json.loads(md["transformer_config"]),json.loads(md[KEY])
        config[field] = record["source_config"][field] = 1000000
        bad = save_single_file_state(state,{**md,"transformer_config":json.dumps(config),KEY:json.dumps(record)},
                                     str(tmp_path/(field+".safetensors")))
        with patch.object(MiniMaxH3Transformer3DModel,"__new__",side_effect=AssertionError("model constructed")):
            with pytest.raises(ValueError,match="header"):
                preflight_checkpoint(bad)
