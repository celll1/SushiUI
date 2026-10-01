"""Request-scoped global LoRA validation, ranges, stacking and cleanup."""
import json
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from core.models.ming_image_01.generation_adapters import generation_adapters
from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter,GLOBAL_NAME,generation_global_residual
from core.models.ming_image_01.model_definition import create_transformer
from core.training.partition import PartitionBox
from ming_image_01_rewiring_test import config


def package(tmp_path,padding):
    torch.manual_seed(79)
    model=create_transformer(config(padding),device='cpu',dtype=torch.float32)
    if padding=='learned':
        torch.nn.init.constant_(model.x_pad_token,.02);torch.nn.init.constant_(model.cap_pad_token,.03)
    branch=MingPartitionGlobalAdapter(16,24,rank=4,summary_tokens=3)
    torch.nn.init.normal_(branch.out.weight,std=.02)
    state={'lora_unet_layers__0__attention__to_q.lora_down.weight':torch.randn(2,24)*.01,
        'lora_unet_layers__0__attention__to_q.lora_up.weight':torch.randn(24,2)*.01,
        'lora_unet_layers__0__attention__to_q.alpha':torch.tensor(2.)}
    state.update({GLOBAL_NAME+'.'+k:v for k,v in branch.export_tensors().items()});state[GLOBAL_NAME+'.alpha']=torch.tensor(2.)
    profile='layer_decompose' if padding=='learned' else 'generation_edit'
    metadata={'model_type':'ming_image_01','ming_image_01_profile':profile,'lora_rank':'2','lora_alpha':'2',
        GLOBAL_NAME:json.dumps(branch.metadata())}
    path=tmp_path/'a.safetensors';save_file(state,str(path),metadata=metadata)
    return SimpleNamespace(transformer=model,artifact=SimpleNamespace(profile=profile)),branch,state,metadata,path


@pytest.mark.parametrize('padding',['zero_masked','learned'])
def test_global_generation_stack_step_ranges_native_use_and_cleanup(tmp_path,padding):
    components,branch,state,meta,path=package(tmp_path,padding)
    second=tmp_path/'b.safetensors';save_file(state,str(second),metadata=meta)
    model=components.transformer;original=model.layers[0].attention.to_q
    snapshot={k:v.clone() for k,v in model.state_dict().items()}
    frames=3 if padding=='learned' else 1
    noisy=torch.randn(1,16,frames,16,16);box=PartitionBox(2,2,6,6)
    expected=branch(noisy,box).detach()
    with generation_adapters(components,[{'path':str(path),'strength':.5,'step_range':[0,500]},
        {'path':str(second),'strength':.25,'step_range':[500,1000]}],resolve_path=str) as session:
        assert session.has_step_range
        session.set_step(2,4)
        torch.testing.assert_close(generation_global_residual(model,noisy,box),.75*expected)
        session.set_step(3,4)
        residual=generation_global_residual(model,noisy,box)
        torch.testing.assert_close(residual,.25*expected)
        from core.models.ming_image_01.partition import MingPartitionAdapter
        output=model(list(noisy[...,4:12,4:12].unbind()),torch.tensor([.4]),[torch.randn(5,8)],
            cap_feats_2=[torch.randn(3,24)],target_spatial_position_ids=MingPartitionAdapter().position_ids(8,8,box),
            target_input_residual=residual).sample[0]
        assert output.shape==(16,frames,8,8) and output.isfinite().all()
    assert model.layers[0].attention.to_q is original
    assert not hasattr(model,'_ming_partition_global_generation')
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in snapshot.items())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage',['missing','version','nan'])
def test_bad_global_file_prevents_partial_adapter_install(tmp_path,damage):
    from core.adapters import AdapterIncompatible
    components,branch,state,meta,path=package(tmp_path,'zero_masked')
    original=components.transformer.layers[0].attention.to_q
    if damage=='missing':state.pop(GLOBAL_NAME+'.out.weight')
    elif damage=='version':info=json.loads(meta[GLOBAL_NAME]);info['schema_version']=True;meta[GLOBAL_NAME]=json.dumps(info)
    else:state[GLOBAL_NAME+'.out.weight'][0,0]=float('nan')
    bad=tmp_path/'bad.safetensors';save_file(state,str(bad),metadata=meta)
    with pytest.raises(AdapterIncompatible):
        with generation_adapters(components,[{'path':str(path)},{'path':str(bad)}],resolve_path=str):pass
    assert components.transformer.layers[0].attention.to_q is original
    assert not hasattr(components.transformer,'_ming_partition_global_generation')


@pytest.mark.parametrize('failure',['body','topology_cleanup'])
def test_global_lifetime_restored_on_all_exit_paths(tmp_path,monkeypatch,failure):
    import importlib
    module=importlib.import_module('core.models.ming_image_01.generation_adapters')
    components,_,_,_,path=package(tmp_path,'zero_masked')
    original=components.transformer.layers[0].attention.to_q
    previous=[object()];components.transformer._ming_partition_global_generation=previous
    if failure=='topology_cleanup':
        real=module.restore_topology;calls=[]
        def cleanup(model,spec):
            calls.append(spec)
            if len(calls)==2:raise RuntimeError('deliberate cleanup')
            return real(model,spec)
        monkeypatch.setattr(module,'restore_topology',cleanup)
    with pytest.raises(RuntimeError,match='deliberate'):
        with generation_adapters(components,[{'path':str(path)}],resolve_path=str):
            if failure=='body':raise RuntimeError('deliberate body')
    assert components.transformer._ming_partition_global_generation is previous
    assert components.transformer.layers[0].attention.to_q is original
