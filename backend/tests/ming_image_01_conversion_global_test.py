"""Offline DiT conversion preserves independent global weights and native topology."""
from types import SimpleNamespace
import json
import pytest
import torch
from safetensors.torch import save_file
from ming_image_01_conversion_test import make_source,digest_tree
from core.models.ming_image_01.conversion import convert_package
from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter,write_payload


def source_with_global(tmp_path,profile,dtype):
    root=tmp_path/'source'
    model,topology=make_source(root,profile)
    branch=MingPartitionGlobalAdapter(model.in_channels,model.dim,rank=8,summary_tokens=3,dtype=dtype)
    torch.nn.init.normal_(branch.out.weight,std=.02)
    write_payload(SimpleNamespace(ming_partition_global_adapter=branch),root)
    return root,branch,topology


@pytest.mark.parametrize('profile,format_,dtype',[
    ('generation_edit','int8',torch.float32),
    ('layer_decompose','int8',torch.bfloat16),
    ('generation_edit','int8_convrot',torch.bfloat16),
    ('layer_decompose','int8_convrot',torch.float32),
])
def test_conversion_preserves_global_precision_native_payload_and_unet(tmp_path,profile,format_,dtype):
    root,branch,topology=source_with_global(tmp_path,profile,dtype)
    before=digest_tree(root)
    result=convert_package(root,tmp_path/'converted',format_=format_,max_shard_bytes=4096)
    from core.models.ming_image_01.loader import load_components
    loaded=load_components(result['manifest'],dtype=torch.float32,conditioner_attention='eager')
    payload=loaded.transformer._ming_partition_global_payload
    assert payload['metadata']==branch.metadata()
    assert all(v.dtype==dtype and torch.equal(v,branch.export_tensors()[k]) for k,v in payload['state'].items())
    assert loaded.transformer._residual_unet_spec==topology
    assert not hasattr(loaded.transformer,'ming_partition_global_adapter')
    for name in ('mllm','connector','mlp','vae','scheduler'):
        assert digest_tree(root/name)==digest_tree(tmp_path/'converted'/name)
    assert digest_tree(root)==before
    assert sum(result['report']['converted'].values())>0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage',['missing','metadata','nonfinite'])
def test_invalid_global_payload_refused_before_conversion(tmp_path,damage):
    root,branch,_=source_with_global(tmp_path,'generation_edit',torch.float32)
    directory=root/'partition_global'
    if damage=='missing':
        (directory/'adapter.safetensors').unlink()
    elif damage=='metadata':
        info=branch.metadata();info['schema_version']=True
        (directory/'config.json').write_text(json.dumps(info))
    else:
        state=branch.export_tensors();state['out.weight'][0,0]=float('nan')
        save_file(state,str(directory/'adapter.safetensors'))
    before=digest_tree(root);progress=[]
    with pytest.raises(ValueError):
        convert_package(root,tmp_path/'converted',format_='int8',progress_callback=lambda *args:progress.append(args))
    assert not progress and not (tmp_path/'converted').exists()
    assert not list(tmp_path.glob('.converted.ming-*'))
    assert digest_tree(root)==before
    assert not torch.cuda.is_initialized()
