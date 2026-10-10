from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from hybrid_rewiring_test import declaration,execute
from core.rewiring.hybrid_modules import HybridModules,HybridExecution
from core.rewiring.hybrid_spec import build_hybrid_spec,legacy_topology
from core.experiments.boundary_detail import DetailGate,build_gates,boundary_detail_probe


def run(modules,value,mode,groups=1):
    return modules.forward_stream(value,'main',8*groups,12,3,None,execute,lambda *a:None,
        grid_groups=groups,checkpoint_modules=mode!='none',checkpoint_offload=mode=='cpu')


def test_detail_oracle_and_both_input_gradients():
    gate=DetailGate(2)
    saved=torch.randn(1,2,4,6,requires_grad=True)
    low=torch.randn(1,2,2,3,requires_grad=True)
    torch.testing.assert_close(gate(saved,low),torch.zeros_like(saved),rtol=0,atol=0)
    with torch.no_grad():gate.gain.fill_(.25)
    expected=.25*(saved-F.interpolate(low,size=(4,6),mode='bilinear',align_corners=False))
    torch.testing.assert_close(gate(saved,low),expected)
    gate(saved,low).square().sum().backward()
    for value in (saved,low):assert torch.isfinite(value.grad).all() and value.grad.abs().sum()>0


@pytest.mark.parametrize('storage',['gpu','cpu','recompute'])
@pytest.mark.parametrize('mode',['none','normal','cpu'])
def test_nested_storage_checkpoint_and_group_gradients(storage,mode):
    torch.manual_seed(63)
    spec=declaration()
    spec=build_hybrid_spec(legacy_topology(spec),stream_widths=spec['stream_widths'],
        boundaries=spec['boundaries'],refiners=spec['refiners'],
        skips=[dict(item,storage=storage) for item in spec['skips']])
    modules=HybridModules(spec)
    gates=build_gates(modules)
    for gate in gates.values():gate.gain.data.normal_(std=.1)
    reference,other_gates=deepcopy(modules),deepcopy(gates)
    value=torch.randn(1,3+16*12,8,requires_grad=True)
    other=value.detach().clone().requires_grad_()
    with boundary_detail_probe(reference,other_gates):
        expected=run(reference,other,'none',2)
        expected.square().mean().backward()
    with boundary_detail_probe(modules,gates):
        actual=run(modules,value,mode,2)
        actual.square().mean().backward()
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(value.grad,other.grad)
    for name,parameter in gates.named_parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
        torch.testing.assert_close(parameter.grad,dict(other_gates.named_parameters())[name].grad)
    for name,parameter in modules.named_parameters():
        torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad)


def test_zero_identity_rng_dtype_other_model_and_failure_cleanup():
    modules=HybridModules(declaration())
    for module in modules.up.values():module.expand.weight.data.normal_(std=.03)
    rng=torch.get_rng_state().clone()
    gates=build_gates(modules)
    assert torch.equal(rng,torch.get_rng_state())
    modules.double()
    other=deepcopy(modules)
    value=torch.randn(1,3+8*12,8,dtype=torch.float64)
    expected=run(modules,value,'none')
    with pytest.raises(RuntimeError,match='intentional'):
        with boundary_detail_probe(modules,gates):
            assert all(p.dtype==torch.float64 for p in gates.parameters())
            torch.testing.assert_close(run(modules,value,'cpu'),expected,rtol=0,atol=0)
            torch.testing.assert_close(run(other,value,'none'),expected,rtol=0,atol=0)
            with pytest.raises((ValueError,RuntimeError)):
                with boundary_detail_probe(modules,gates):pass
            raise RuntimeError('intentional')
    assert not hasattr(modules,'_boundary_detail_probe')
    assert not hasattr(HybridExecution,'boundary_fusion')


def test_bad_layout_is_rejected():
    gate=DetailGate(2)
    for low in (torch.ones(1,3,2,2),torch.ones(1,2,5,5),torch.ones(2,2,2,2)):
        with pytest.raises(ValueError):gate(torch.ones(1,2,4,4),low)


def test_setup_transfer_failure_releases_exclusive_owner(monkeypatch):
    modules=HybridModules(declaration())
    gates=build_gates(modules)
    original=gates.to
    def fail(*args,**kwargs):raise RuntimeError('transfer failed')
    monkeypatch.setattr(gates,'to',fail)
    with pytest.raises(RuntimeError,match='transfer failed'):
        with boundary_detail_probe(modules,gates):pass
    assert not hasattr(HybridExecution,'boundary_fusion')
    assert not hasattr(modules,'_boundary_detail_probe')
    monkeypatch.setattr(gates,'to',original)
    with boundary_detail_probe(modules,gates):pass


def test_runtime_rejects_wrong_boundary_correction_shape(monkeypatch):
    modules=HybridModules(declaration())
    monkeypatch.setattr(HybridExecution,'boundary_fusion',
        lambda *args:torch.zeros(1,1,1,1),raising=False)
    with pytest.raises(ValueError,match='restored grid layout'):
        run(modules,torch.randn(1,3+8*12,8),'none')


def test_explicitly_absent_hook_is_preserved_for_other_model(monkeypatch):
    modules=HybridModules(declaration())
    other=deepcopy(modules)
    gates=build_gates(modules)
    value=torch.randn(1,3+8*12,8)
    expected=run(other,value,'none')
    monkeypatch.setattr(HybridExecution,'boundary_fusion',None,raising=False)
    with boundary_detail_probe(modules,gates):
        torch.testing.assert_close(run(other,value,'none'),expected,rtol=0,atol=0)
    assert HybridExecution.boundary_fusion is None
