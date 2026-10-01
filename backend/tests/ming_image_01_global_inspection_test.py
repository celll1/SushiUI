"""Header-only sidecar preflight and unique loaded global-branch ownership."""
import json
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from ming_image_01_conditioner_quantized_test import source
from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter,write_payload


def package(tmp_path,profile,dtype=torch.float32):
    root=source(tmp_path,profile)
    branch=MingPartitionGlobalAdapter(16,24,rank=8,summary_tokens=3,dtype=dtype)
    write_payload(SimpleNamespace(ming_partition_global_adapter=branch),root)
    return root,branch


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_header_only_global_structure_and_loaded_stash(tmp_path,monkeypatch,profile,dtype):
    root,branch=package(tmp_path,profile,dtype)
    from core.models.ming_image_01.artifact import inspect_artifact
    from core.rewiring.inspect import inspect_model
    from core.models.ming_image_01.loader import load_components
    from core.models.ming_image_01.inspection import loaded_model_info
    def forbidden(*a,**k):raise AssertionError('Inspection read a weight payload')
    with monkeypatch.context() as patch:
        patch.setattr('safetensors.torch.load_file',forbidden)
        artifact=inspect_artifact(root)
        shape=inspect_model(root)
    n=sum(p.numel() for p in branch.parameters());size=n*(4 if dtype==torch.float32 else 2)
    assert artifact.global_summary['metadata']==branch.metadata()
    assert artifact.global_summary['parameters']==n
    assert artifact.summary()['components']['partition_global']['storage_bytes']==size
    assert shape['ming_partition_global']=={'metadata':branch.metadata(),'parameters':n,'storage_bytes':size,'loaded_in_model':False}
    loaded=load_components(root,dtype=torch.float32,conditioner_attention='eager')
    info=loaded_model_info(loaded)
    assert info['ming_partition_global']==shape['ming_partition_global']
    assert 'partition_global' not in info['component_parameter_counts']
    assert info['serialized_storage']['partition_global']['storage_bytes']==size
    assert info['component_parameter_counts']['dit']['parameters']==info['dit_rewiring_parameter_counts']['total']
    json.dumps(info)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
def test_registered_global_is_counted_once_separately_from_native_dit(tmp_path,profile):
    root,branch=package(tmp_path,profile)
    from core.models.ming_image_01.loader import load_components
    from core.models.ming_image_01.inspection import loaded_model_info
    from core.models.ming_image_01.partition_global import prepare_full_training
    loaded=load_components(root,dtype=torch.float32,conditioner_attention='eager')
    before=loaded_model_info(loaded)
    owner=SimpleNamespace(transformer=loaded.transformer,weight_dtype=torch.float32,train_unet=True,
        config={'dit_partition_training_enabled':True,'dit_partition_global_adapter_enabled':True,
            'dit_partition_global_rank':8,'dit_partition_global_tokens':3})
    prepare_full_training(owner)
    info=loaded_model_info(loaded);n=sum(p.numel() for p in branch.parameters())
    assert info['ming_partition_global']['loaded_in_model']
    assert info['component_parameter_counts']['partition_global']['parameters']==n
    assert info['component_parameter_counts']['partition_global']['trainable_parameters']==n
    assert info['component_parameter_counts']['dit']['parameters']==before['component_parameter_counts']['dit']['parameters']
    assert info['total_loaded_parameters']==before['total_loaded_parameters']+n
    assert sum(row['parameters'] for row in info['component_parameter_counts'].values())==info['total_loaded_parameters']
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage',['missing','metadata','dtype','keys','shape','duplicate_json'])
def test_bad_global_header_refused_before_any_weight_payload(tmp_path,monkeypatch,damage):
    root,branch=package(tmp_path,'generation_edit')
    directory=root/'partition_global';state=branch.export_tensors()
    if damage=='missing':(directory/'adapter.safetensors').unlink()
    elif damage=='metadata':
        meta=branch.metadata();meta['schema_version']=True;(directory/'config.json').write_text(json.dumps(meta))
    elif damage=='duplicate_json':
        text=json.dumps(branch.metadata());(directory/'config.json').write_text(text[:-1]+',"rank":8}')
    else:
        if damage=='dtype':state['out.weight']=state['out.weight'].to(torch.int8)
        elif damage=='keys':state['extra']=torch.zeros(1)
        else:state['out.weight']=state['out.weight'][:1]
        save_file(state,str(directory/'adapter.safetensors'))
    from core.models.ming_image_01.artifact import inspect_artifact
    from core.models.ming_image_01.loader import load_components
    def forbidden(*a,**k):raise AssertionError('Invalid sidecar reached weight loading')
    monkeypatch.setattr('safetensors.torch.load_file',forbidden)
    monkeypatch.setattr('core.models.ming_image_01.quantization.prepare_quantized_transformer',forbidden)
    with pytest.raises(ValueError):inspect_artifact(root)
    with pytest.raises(ValueError):load_components(root,dtype=torch.float32,conditioner_attention='eager')
    assert not torch.cuda.is_initialized()
