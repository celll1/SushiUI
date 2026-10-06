"""Aligned LTX component directories through production quantized export."""
import json
from unittest.mock import patch

import pytest
import torch

from ltx2_video_artifact_test import fixture,forward


@pytest.mark.parametrize('rewired',[False,True])
@pytest.mark.parametrize('sharded',[False,True])
def test_quantized_directory_preserves_codec_and_graph(tmp_path,rewired,sharded):
    from core.models.ideogram4.vendor.int8_linear import Int8Linear,quantize_weight_to_int8
    from core.models.common.quantized_export import export_quantized_transformer,layout_unwrap
    from core.models.ltx2.loader import load_ltx2_from_path
    from core.models.ltx2.video_alignment import aligned_wrapper
    from core.models.common.single_file_format import read_state_dict
    trainer,root = fixture(tmp_path,4)
    inner = trainer.transformer.transformer
    inner._ltx2_vae_identity = trainer.vae_identity
    if not rewired:
        del inner._residual_unet_spec
        trainer.transformer = aligned_wrapper(inner)
    layer = inner.transformer_blocks[0].attn1.to_q
    quant = Int8Linear(layer.in_features,layer.out_features,layer.bias is not None,torch.float32)
    quant.weight,quant.weight_scale = quantize_weight_to_int8(layer.weight.detach())
    if layer.bias is not None:
        quant.bias = layer.bias.detach().clone()
    inner.transformer_blocks[0].attn1.to_q = quant
    assert layout_unwrap('ltx2',trainer.transformer) is inner
    dest = tmp_path/'export'/'transformer'/'diffusion_pytorch_model.safetensors'
    def siblings(source,destination,names):
        # No junctions are needed for these config-only companion fixtures.
        import shutil
        for name in names:
            path = root/name
            target = tmp_path/'export'/name
            if path.is_file():
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target)
            elif path.is_dir():
                target.mkdir(parents=True,exist_ok=True)
        return list(names)
    with patch('core.models.common.quantized_export.link_siblings',side_effect=siblings):
        summary = export_quantized_transformer(trainer.transformer,'ltx2',str(dest),
            config=dict(trainer.transformer.config),latent_codec=trainer.vae,link_siblings_from=str(root),
            max_shard_bytes=24000 if sharded else 2**30)
    state,metadata = read_state_dict(summary['output_path'])
    from api.quantized_export_job import _source_root
    assert _source_root('ltx2',summary['output_path']) == str(root.resolve())
    if sharded:
        assert _source_root('ltx2',str(dest)) == str(root.resolve())
    assert metadata.get('ltx2.video_alignment')
    from core.rewiring.artifacts import read_artifact_topology
    assert (read_artifact_topology(summary['output_path']) is not None) == rewired
    assert 'vae.encoder.conv_in.weight' in state
    assert {item['component'] for item in summary['components']} == {'transformer','vae'}
    assert state['transformer_blocks.0.attn1.to_q.weight'].dtype == torch.int8
    captured={}
    def assemble(path,**kwargs):
        from core.models.ltx2 import LTX2Pipeline
        captured.update(kwargs)
        return LTX2Pipeline(scheduler=None,vae=kwargs['vae'],audio_vae=None,text_encoder=None,
            tokenizer=None,connectors=None,transformer=kwargs['transformer'],vocoder=None)
    with patch('core.models.ltx2.LTX2Pipeline.from_pretrained',side_effect=assemble):
        loaded = load_ltx2_from_path(str(tmp_path/'export'),torch.float32)
    model = loaded['transformer']
    assert loaded['vae'].inner is captured['vae']
    assert loaded['latent_channels'] == 4 and loaded['vae_scale_factor_spatial'] == 2
    assert getattr(model.transformer,'_residual_unet_spec',None) == getattr(inner,'_residual_unet_spec',None)
    for key,value in inner.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)
    with torch.no_grad():
        before,after = forward(trainer.transformer),forward(model)
    for expected,actual in zip(before,after):
        torch.testing.assert_close(expected,actual,atol=2e-6,rtol=2e-5)
    pixels=torch.randn(1,3,3,8,12)
    torch.testing.assert_close(trainer.vae.encode(pixels).latent_dist.mode(),
                               loaded['vae'].encode(pixels).latent_dist.mode(),atol=0,rtol=0)
    config_path = tmp_path/'export'/'transformer'/'config.json'
    config=json.loads(config_path.read_text())
    config['audio_in_channels']=99
    config_path.write_text(json.dumps(config))
    with patch('core.models.ltx2.loader.load_ltx2_transformer_checkpoint',side_effect=AssertionError('early tensor load')):
        with pytest.raises(ValueError,match='directory config'):
            load_ltx2_from_path(str(tmp_path/'export'),torch.float32)


def test_aligned_export_requires_its_codec_before_writing(tmp_path):
    from core.models.common.quantized_export import export_quantized_transformer
    trainer,_ = fixture(tmp_path,4)
    with patch('core.models.common.quantized_export.combined_linear_inventory',
               return_value=dict(int8=1,e4m3=0,plain=1)):
        with pytest.raises(ValueError,match='restored codec'):
            export_quantized_transformer(trainer.transformer,'ltx2',str(tmp_path/'bad.safetensors'))
    assert not (tmp_path/'bad.safetensors').exists()
