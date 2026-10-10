from copy import deepcopy

import pytest
import torch

from hybrid_rewiring_test import declaration,execute
from core.rewiring.hybrid_modules import HybridModules
from core.rewiring.hybrid_spec import (build_hybrid_spec,legacy_topology,hybrid_metadata,
    read_hybrid_metadata,hybrid_parameter_counts,hybrid_preview)
from core.rewiring.spec import digest,preview_plan


def fusion_spec(storage='gpu'):
    base=declaration()
    return build_hybrid_spec(legacy_topology(base),stream_widths=base['stream_widths'],
        boundaries=[dict(item,fusion='full_rank') for item in base['boundaries']],
        refiners=base['refiners'],skips=[dict(item,storage=storage) for item in base['skips']])


def run(modules,value,mode='none',groups=1):
    return modules.forward_stream(value,'main',8*groups,12,3,None,execute,lambda *a:None,
        grid_groups=groups,checkpoint_modules=mode!='none',checkpoint_offload=mode=='cpu')


def test_factory_metadata_counts_zero_identity_and_old_lineage():
    old=declaration();before=deepcopy(old);source=HybridModules(old)
    rng=torch.get_rng_state().clone();spec=fusion_spec();modules=HybridModules(spec)
    assert torch.equal(rng,torch.get_rng_state()) and old==before
    assert read_hybrid_metadata(hybrid_metadata(old))==old
    assert all('fusion' not in item for item in old['boundaries']) and not source.features
    assert read_hybrid_metadata(hybrid_metadata(spec))==spec and digest(spec)!=digest(old)
    expected_added=hybrid_parameter_counts(old)['added']+2*3*8**2
    assert hybrid_parameter_counts(spec)['added']==sum(p.numel() for p in modules.parameters())==expected_added
    modules.load_state_dict({**source.state_dict(),**{k:v for k,v in modules.state_dict().items() if k.startswith('features.')}},strict=True)
    value=torch.randn(1,99,8)
    torch.testing.assert_close(run(modules,value),run(source,value),rtol=0,atol=0)
    preview=hybrid_preview(spec,8,12,batch_size=2,element_bytes=2)
    work={item['region']:item['elements'] for item in preview['feature_fusion_workspace']}
    assert work=={'outer':6*8*12*8,'inner':6*4*6*8}
    assert preview_plan(spec,8,12)['feature_fusion_workspace']==preview['feature_fusion_workspace']


@pytest.mark.parametrize('bad',[None,True,0,'concat',[],{}])
def test_invalid_operator_refused(bad):
    base=declaration()
    with pytest.raises(ValueError,match='fusion'):
        build_hybrid_spec(legacy_topology(base),stream_widths=base['stream_widths'],
            boundaries=[dict(item,fusion=bad) for item in base['boundaries']],refiners=base['refiners'],skips=base['skips'])


@pytest.mark.parametrize('storage',['gpu','cpu','recompute'])
@pytest.mark.parametrize('mode',['none','normal','cpu'])
def test_public_nested_grouped_gradient_parity(storage,mode):
    torch.manual_seed(63)
    modules=HybridModules(fusion_spec(storage))
    for module in modules.features.values():module.weight.data.normal_(std=.01)
    reference=deepcopy(modules)
    value=torch.randn(1,195,8,requires_grad=True);other=value.detach().clone().requires_grad_()
    expected=run(reference,other,groups=2);expected.square().mean().backward()
    actual=run(modules,value,mode,2);actual.square().mean().backward()
    torch.testing.assert_close(actual,expected);torch.testing.assert_close(value.grad,other.grad)
    for name,p in modules.named_parameters():
        torch.testing.assert_close(p.grad,dict(reference.named_parameters())[name].grad)
    assert all(p.grad.abs().sum()>0 for p in modules.features.parameters())


def test_weight_and_optimizer_continuation(tmp_path):
    from safetensors.torch import save_file,load_file
    from core.adapters.full_module import FullModulePayload
    torch.manual_seed(63)
    spec=fusion_spec();model=HybridModules(spec)
    optim=torch.optim.AdamW(model.parameters(),lr=.001)
    value=torch.randn(1,99,8)
    def step(module,optimizer):
        optimizer.zero_grad(set_to_none=True);run(module,value,'normal').square().mean().backward();optimizer.step()
    step(model,optim)
    payload=FullModulePayload(model)
    assert any(name.startswith('features.') for name in payload.branch_tensors())
    path=tmp_path/'structural.safetensors';save_file(model.state_dict(),str(path),metadata=hybrid_metadata(spec))
    saved_optimizer=deepcopy(optim.state_dict())
    restored=HybridModules(spec);restored.load_state_dict(load_file(str(path)),strict=True)
    resume=torch.optim.AdamW(restored.parameters(),lr=.001);resume.load_state_dict(saved_optimizer)
    step(model,optim);step(restored,resume)
    for name,value in model.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
    assert all(int(row['step'])==2 for row in resume.state.values())
    with pytest.raises(RuntimeError,match='Unexpected key'):
        HybridModules(declaration()).load_state_dict(load_file(str(path)),strict=True)
