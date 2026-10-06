"""LTX bundled codec/bridge round trips using synthetic small native models."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def fixture(tmp_path, new, codec_kind='image'):
    from diffusers import AutoencoderKL, AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel
    from core.models.common.vae_source import resolve_vae_source
    from core.models.components.latent_adapter import ImageCodecContract, SpatialAlignmentPair
    from core.models.components.video_latent_adapter import VideoCodecContract, NormalizedVideoVAE
    from core.models.ltx2.video_alignment import install_video_alignment
    from core.models.ltx2.residual_unet import native_config, wrap_transformer
    from core.rewiring.spec import build_spec, OPERATORS
    inner = LTX2VideoTransformer3DModel(in_channels=3, out_channels=3,
        num_attention_heads=2, attention_head_dim=16, cross_attention_dim=32,
        audio_in_channels=4, audio_out_channels=4, audio_num_attention_heads=2,
        audio_attention_head_dim=8, audio_cross_attention_dim=16, num_layers=4,
        use_prompt_embeddings=False, cross_attn_mod=True, audio_cross_attn_mod=True,
        gated_attn=True, audio_gated_attn=True)
    root = tmp_path/'root'
    root.mkdir()
    names = ('text_encoder','tokenizer','connectors','vae','audio_vae','vocoder','scheduler','transformer')
    for name in names:
        (root/name).mkdir()
    (root/'model_index.json').write_text(json.dumps(dict(_class_name='LTX2Pipeline',
        **{name:['diffusers','Fixture'] for name in names})), encoding='utf-8')
    (root/'transformer'/'config.json').write_text(json.dumps(dict(inner.config)), encoding='utf-8')
    inner._ltx2_component_sources = dict(schema_version=1, root=str(root.resolve()))
    graph = build_spec('ltx2',[dict(name='main',block_count=4)],[
        dict(id='region', stream='main',parent=None,start_block=1,end_block=3,down_factor=[2,2],**OPERATORS)],
        native_config(inner.config))
    model = wrap_transformer(inner, graph)
    if codec_kind == 'image':
        raw = AutoencoderKL(in_channels=3,out_channels=3,latent_channels=new,
            block_out_channels=(4,8),down_block_types=('DownEncoderBlock2D',)*2,
            up_block_types=('UpDecoderBlock2D',)*2,norm_num_groups=1,
            scaling_factor=2.,shift_factor=.25)
    else:
        raw = AutoencoderKLLTX2Video(in_channels=3,out_channels=3,latent_channels=new,
            block_out_channels=(8,),down_block_types=('LTX2VideoDownBlock3D',),layers_per_block=(1,1),
            decoder_block_out_channels=(16,),decoder_layers_per_block=(1,1),
            spatio_temporal_scaling=(True,),decoder_spatio_temporal_scaling=(True,),
            downsample_type=('spatiotemporal',),upsample_type=('spatiotemporal',),
            upsample_factor=(1,),upsample_residual=(False,),decoder_inject_noise=(False,False),
            patch_size=1,patch_size_t=1,spatial_compression_ratio=2,temporal_compression_ratio=2,
            decoder_spatial_padding_mode='zeros')
        raw.latents_mean.fill_(.1)
        raw.latents_std.fill_(2.)
    raw.eval().requires_grad_(False)
    directory = tmp_path/'codec'
    raw.save_pretrained(directory)
    resolved = resolve_vae_source('file:'+str(directory), arch='ltx2',load_weights=True, download=False)
    image = ImageCodecContract.from_resolved(resolved)
    contract = VideoCodecContract(image, resolved.scale_temporal,
        'independent_frames_v1' if codec_kind=='image' else 'causal_first_frame_v1')
    declaration = image.declaration(source=resolved.source,source_identity='source',target_identity=resolved.latent_hash,
        old_channels=3,old_scale=32)
    pair = SpatialAlignmentPair(declaration)
    for bridge in (pair.input,pair.output):
        bridge.initialize(torch.randn(bridge.source_channels*9,bridge.target_channels)/10,
                          torch.randn(bridge.target_channels)/10)
    aligned = install_video_alignment(model,pair,contract)
    trainer = SimpleNamespace(transformer=aligned,vae=NormalizedVideoVAE(raw,contract),
        vae_identity=resolved,bundle_vae=True,arch=SimpleNamespace(name='ltx2'))
    return trainer,root


def forward(model):
    c = model.config.in_channels
    g = torch.Generator().manual_seed(15)
    return model(hidden_states=torch.randn(1,3*4*6,c,generator=g),
        audio_hidden_states=torch.randn(1,5,4,generator=g),
        encoder_hidden_states=torch.randn(1,4,32,generator=g),audio_encoder_hidden_states=torch.randn(1,4,16,generator=g),
        timestep=torch.tensor([500.]),audio_timestep=torch.tensor([600.]),
        sigma=torch.tensor([.5]),audio_sigma=torch.tensor([.6]),
        num_frames=3,height=4,width=6,audio_num_frames=5,return_dict=False)


@pytest.mark.parametrize('new',[2,4])
@pytest.mark.parametrize('sharded',[False,True])
@pytest.mark.parametrize('codec_kind',['image','video'])
def test_full_adapter_codec_bridge_optimizer_and_companion_roundtrip(tmp_path,new,sharded,codec_kind):
    from core.training.adapters.ltx2_adapter import Ltx2FullParameterAdapter
    from core.models.common.single_file_format import save_single_file_state,read_metadata,read_state_dict
    from core.models.ltx2.loader import load_ltx2_transformer_checkpoint,load_ltx2_from_path
    from core.models.ltx2.video_artifact import read_record,restored_codec
    trainer,root = fixture(tmp_path,new,codec_kind)
    optimizer = torch.optim.AdamW(trainer.transformer.parameters(),lr=1e-3)
    sum(v.square().mean() for v in forward(trainer.transformer)).backward()
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    state_path = tmp_path/'optimizer.pt'
    torch.save(optimizer.state_dict(),state_path)
    writer = lambda state,md,path: save_single_file_state(state,md,path,max_shard_bytes=24000 if sharded else 10000000)
    with patch('core.models.common.single_file_format.save_single_file_state',side_effect=writer):
        path = Ltx2FullParameterAdapter(trainer).write_checkpoint(1,0,tmp_path/'full.safetensors')
    metadata = read_metadata(str(path))
    state,_ = read_state_dict(str(path))
    restored = load_ltx2_transformer_checkpoint(str(path),torch.float32)
    restored.requires_grad_(True)
    restored.train(trainer.transformer.training)
    for key,value in trainer.transformer.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
    expected = forward(trainer.transformer)
    actual = forward(restored)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    next_optimizer = torch.optim.AdamW(restored.parameters(),lr=1e-3)
    next_optimizer.load_state_dict(torch.load(state_path,weights_only=False))
    for model,opt in ((trainer.transformer,optimizer),(restored,next_optimizer)):
        sum(v.square().mean() for v in forward(model)).backward()
        opt.step()
    for (name,a),(other,b) in zip(trainer.transformer.named_parameters(),restored.named_parameters()):
        assert name == other
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    record = read_record(metadata,json.loads(metadata['transformer_config']))
    codec,identity = restored_codec(str(path),metadata,record,torch_dtype=torch.float32)
    temporal = 1 if codec_kind=='image' else 2
    pixels = torch.randn(1,3,(3-1)*temporal+1,8,12)
    latent = codec.encode(pixels).latent_dist.mode()
    torch.testing.assert_close(latent,trainer.vae.encode(pixels).latent_dist.mode(),rtol=0,atol=0)
    assert codec.decode(latent,None,return_dict=False)[0].shape == pixels.shape
    assert identity.latent_hash == trainer.vae_identity.latent_hash
    captured = {}
    def assemble(root_path,**kwargs):
        from core.models.ltx2 import LTX2Pipeline
        captured.update(kwargs)
        return LTX2Pipeline(scheduler=None,vae=kwargs['vae'],audio_vae=None,text_encoder=None,
            tokenizer=None,connectors=None,transformer=kwargs['transformer'],vocoder=None)
    with patch('core.models.ltx2.LTX2Pipeline.from_pretrained',side_effect=assemble):
        loaded = load_ltx2_from_path(str(path),torch.float32)
    assert loaded['latent_channels'] == new
    assert loaded['vae_scale_factor_spatial'] == 2
    assert loaded['vae_scale_factor_temporal'] == temporal
    assert loaded['vae'].inner is captured['vae']
    assert loaded['pipeline'].vae_spatial_compression_ratio == 2
    assert loaded['pipeline'].vae_temporal_compression_ratio == temporal
    assert loaded['transformer'].config.in_channels == new
    assert any(key.startswith('vae.') for key in state)
    # Re-saving a frozen normalized facade must preserve the raw codec's normalization.
    trainer.transformer,trainer.vae,trainer.vae_identity = loaded['transformer'],loaded['vae'],identity
    again = Ltx2FullParameterAdapter(trainer).write_checkpoint(2,0,tmp_path/'again.safetensors')
    assert read_metadata(str(again))['component.vae.config'] == metadata['component.vae.config']
    source_config_path = root/'transformer'/'config.json'
    source_config = json.loads(source_config_path.read_text(encoding='utf-8'))
    source_config['audio_in_channels'] = 99
    source_config_path.write_text(json.dumps(source_config),encoding='utf-8')
    with patch('core.models.ltx2.loader.load_ltx2_transformer_checkpoint',side_effect=AssertionError('tensor load too early')):
        with pytest.raises(ValueError,match='companion root geometry'):
            load_ltx2_from_path(str(path),torch.float32)


def test_corrupt_alignment_and_missing_codec_fail_header_preflight(tmp_path):
    from core.training.adapters.ltx2_adapter import Ltx2FullParameterAdapter
    from core.models.common.single_file_format import read_state_dict,save_single_file_state
    from core.models.ltx2.residual_unet import preflight_checkpoint
    trainer,_ = fixture(tmp_path,4)
    path = Ltx2FullParameterAdapter(trainer).write_checkpoint(0,0,tmp_path/'full.safetensors')
    state,metadata = read_state_dict(str(path))
    for label,modified,md in (
        ('missing_codec',{k:v for k,v in state.items() if not k.startswith('vae.')},metadata),
        ('missing_record',state,{k:v for k,v in metadata.items() if k!='ltx2.video_alignment'}),
        ('bad_channels',state,{**metadata,'component.vae.channels':'99'}),
        ('bad_hash',state,{**metadata,'component.vae.hash':'changed'}),
        ('bad_bridge',{**state,'net._video_latent_alignment.input.projection.weight':torch.zeros(1,1,3,3)},metadata),
    ):
        target = save_single_file_state(modified,md,str(tmp_path/(label+'.safetensors')))
        with pytest.raises(ValueError):
            preflight_checkpoint(target)
    from core.models.ltx2.loader import load_ltx2_transformer_checkpoint
    uncalibrated = {**state, 'net._video_latent_alignment.input.calibrated': torch.tensor(False)}
    target = save_single_file_state(uncalibrated,metadata,str(tmp_path/'uncalibrated.safetensors'))
    with pytest.raises(ValueError,match='uncalibrated'):
        load_ltx2_transformer_checkpoint(target,torch.float32)


def test_shard_codec_metadata_mismatch_is_rejected(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    from core.training.adapters.ltx2_adapter import Ltx2FullParameterAdapter
    from core.models.common.single_file_format import read_state_dict,save_single_file_state
    from core.models.ltx2.residual_unet import preflight_checkpoint
    trainer,_ = fixture(tmp_path,4)
    original = Ltx2FullParameterAdapter(trainer).write_checkpoint(0,0,tmp_path/'original.safetensors')
    state,metadata = read_state_dict(str(original))
    index_path = save_single_file_state(state,metadata,str(tmp_path/'split.safetensors'),max_shard_bytes=24000)
    index = json.loads(open(index_path,encoding='utf-8').read())
    shard = tmp_path/next(iter(index['weight_map'].values()))
    with safe_open(str(shard),framework='pt',device='cpu') as handle:
        md = dict(handle.metadata())
    config = json.loads(md['component.vae.config'])
    config['sample_size'] = 128
    md['component.vae.config'] = json.dumps(config)
    save_file(load_file(str(shard)),str(shard),metadata=md)
    with pytest.raises(ValueError,match='codec metadata disagree'):
        preflight_checkpoint(index_path)
