from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from hybrid_rewiring_test import declaration,execute
from core.rewiring.hybrid_modules import HybridModules,HybridExecution
from core.rewiring.hybrid_spec import build_hybrid_spec,legacy_topology
from core.experiments.boundary_features import FeatureFusion,build_fusions,boundary_features_probe
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import build_spec,OPERATORS


def run(modules,value,mode,groups=1):
    return modules.forward_stream(value,'main',8*groups,12,3,None,execute,lambda *a:None,
        grid_groups=groups,checkpoint_modules=mode!='none',checkpoint_offload=mode=='cpu')


def test_three_feature_oracle_and_gradients():
    module=FeatureFusion(2).double()
    saved=torch.randn(1,2,4,6,dtype=torch.float64,requires_grad=True)
    incoming=torch.randn(1,2,2,3,dtype=torch.float64,requires_grad=True)
    outgoing=torch.randn_like(incoming,requires_grad=True)
    torch.testing.assert_close(module(saved,incoming,outgoing),torch.zeros_like(saved),rtol=0,atol=0)
    with torch.no_grad():module.weight.normal_(std=.1)
    expected=torch.einsum('oc,bchw->bohw',module.weight[:,:,0,0],torch.cat([
        saved,F.interpolate(incoming,size=(4,6),mode='bilinear',align_corners=False),
        F.interpolate(outgoing,size=(4,6),mode='bilinear',align_corners=False)],dim=1))
    torch.testing.assert_close(module(saved,incoming,outgoing),expected)
    expected.square().sum().backward()
    for value in (saved,incoming,outgoing):assert value.grad.abs().sum()>0


def test_runtime_hook_receives_actual_coarse_fields_before_decoder_block():
    spec=build_spec('anima',[dict(name='main',block_count=4)],
        [dict(id='r',stream='main',parent=None,start_block=1,end_block=3,down_factor=[2,2],**OPERATORS)],{})
    value=torch.randn(1,24,2)
    class Extension:
        def down(self,region,grid,pooled):return pooled
        def up(self,region,update):return 3*update
        def before(self,index,hidden,*args):return hidden
        def after(self,index,hidden,*args):return hidden
        def boundary_features(self,region,saved,low_in,low_out,update,height,width):
            torch.testing.assert_close(saved,value+1)
            expected=F.avg_pool2d((value+1).transpose(1,2).reshape(1,2,4,6),2).flatten(2).transpose(1,2)
            torch.testing.assert_close(low_in,expected)
            torch.testing.assert_close(low_out,expected+5)
            torch.testing.assert_close(update,torch.full_like(update,15))
            assert (height,width)==(4,6)
            return torch.zeros_like(update)
    actual=run_rewired_stream(value,spec,'main',4,6,0,None,
        lambda index,hidden,context:hidden+index+1,lambda *a:None,extensions=Extension())
    torch.testing.assert_close(actual,value+20)


@pytest.mark.parametrize('storage',['gpu','cpu','recompute'])
@pytest.mark.parametrize('mode',['none','normal','cpu'])
def test_nested_grouped_checkpoint_and_storage(storage,mode):
    torch.manual_seed(63)
    spec=declaration()
    spec=build_hybrid_spec(legacy_topology(spec),stream_widths=spec['stream_widths'],
        boundaries=spec['boundaries'],refiners=spec['refiners'],
        skips=[dict(item,storage=storage) for item in spec['skips']])
    modules=HybridModules(spec);fusions=build_fusions(modules)
    for module in fusions.values():module.weight.data.normal_(std=.01)
    reference,other_fusions=deepcopy(modules),deepcopy(fusions)
    value=torch.randn(1,3+16*12,8,requires_grad=True)
    other=value.detach().clone().requires_grad_()
    with boundary_features_probe(reference,other_fusions):
        expected=run(reference,other,'none',2);expected.square().mean().backward()
    with boundary_features_probe(modules,fusions):
        actual=run(modules,value,mode,2);actual.square().mean().backward()
    torch.testing.assert_close(actual,expected);torch.testing.assert_close(value.grad,other.grad)
    for name,p in fusions.named_parameters():
        assert p.grad.abs().sum()>0 and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad,dict(other_fusions.named_parameters())[name].grad)


def test_zero_identity_source_rng_isolation_cleanup_and_reload():
    modules=HybridModules(declaration()).double();other=deepcopy(modules)
    rng=torch.get_rng_state().clone();fusions=build_fusions(modules)
    assert torch.equal(rng,torch.get_rng_state())
    value=torch.randn(1,3+8*12,8,dtype=torch.float64)
    expected=run(modules,value,'none')
    with pytest.raises(RuntimeError,match='sentinel'):
        with boundary_features_probe(modules,fusions):
            torch.testing.assert_close(run(modules,value,'cpu'),expected,rtol=0,atol=0)
            torch.testing.assert_close(run(other,value,'none'),expected,rtol=0,atol=0)
            with pytest.raises(ValueError):
                with boundary_features_probe(modules,fusions):pass
            raise RuntimeError('sentinel')
    assert not hasattr(HybridExecution,'boundary_features') and not hasattr(modules,'_boundary_features_probe')
    with torch.no_grad():
        for module in fusions.values():module.weight.normal_(std=.1)
    restored=build_fusions(modules);restored.load_state_dict(fusions.state_dict(),strict=True)
    for name,p in fusions.named_parameters():torch.testing.assert_close(p,dict(restored.named_parameters())[name],rtol=0,atol=0)


def test_layout_refusal_transfer_failure_and_existing_hook(monkeypatch):
    module=FeatureFusion(2)
    with pytest.raises(ValueError):module(torch.ones(1,2,4,4),torch.ones(1,3,2,2),torch.ones(1,3,2,2))
    modules=HybridModules(declaration());fusions=build_fusions(modules)
    transfer=fusions.to
    def fail(*args,**kwargs):raise RuntimeError('transfer sentinel')
    with monkeypatch.context() as patch:
        patch.setattr(fusions,'to',fail)
        with pytest.raises(RuntimeError,match='transfer sentinel'):
            with boundary_features_probe(modules,fusions):pass
    assert not hasattr(HybridExecution,'boundary_features')
    fusions.to=transfer
    with monkeypatch.context() as patch:
        patch.setattr(HybridExecution,'boundary_features',None,raising=False)
        with boundary_features_probe(modules,fusions):pass
        assert HybridExecution.boundary_features is None
    with monkeypatch.context() as patch:
        patch.setattr(HybridExecution,'boundary_features',lambda *a:torch.zeros(1,1,1,1),raising=False)
        with pytest.raises(ValueError,match='restored grid layout'):run(modules,torch.randn(1,99,8),'none')
        with pytest.raises(ValueError,match='exclusive'):
            with boundary_features_probe(modules,fusions):pass
