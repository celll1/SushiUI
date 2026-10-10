from copy import deepcopy

import pytest
import torch

from hybrid_rewiring_test import declaration, execute
from core.rewiring.hybrid_modules import HybridModules
from core.rewiring.hybrid_spec import build_hybrid_spec, legacy_topology
from core.experiments.full_width_skip import FullWidthSkip, build_corrections, full_width_probe


def run(modules, value, mode):
    return modules.forward_stream(value, 'main', 8, 12, 3, None, execute, lambda *a: None,
        checkpoint_modules=mode != 'none', checkpoint_offload=mode == 'cpu')


@pytest.mark.parametrize('storage', ['gpu', 'cpu', 'recompute'])
@pytest.mark.parametrize('mode', ['none', 'normal', 'cpu'])
def test_full_width_storage_and_checkpoint_gradients(storage, mode):
    torch.manual_seed(63)
    spec = declaration()
    spec = build_hybrid_spec(legacy_topology(spec), stream_widths=spec['stream_widths'],
        boundaries=spec['boundaries'], refiners=spec['refiners'],
        skips=[dict(item, storage=storage) for item in spec['skips']])
    modules = HybridModules(spec)
    for skip in modules.skips.values():torch.nn.init.normal_(skip.restore.weight, std=.02)
    corrections = build_corrections(modules)
    for correction in corrections.values():torch.nn.init.normal_(correction.weight, std=.02)
    reference, other_corrections = deepcopy(modules), deepcopy(corrections)
    value = torch.randn(1, 3+8*12, 8, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    with full_width_probe(reference, other_corrections):
        expected = run(reference, other, 'none')
        expected.square().mean().backward()
    with full_width_probe(modules, corrections):
        observed = run(modules, value, mode)
        observed.square().mean().backward()
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(value.grad, other.grad)
    for name, parameter in corrections.named_parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
        torch.testing.assert_close(parameter.grad, dict(other_corrections.named_parameters())[name].grad)
    for name, parameter in modules.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)


def test_zero_preservation_parameter_cost_rng_and_cleanup():
    modules = HybridModules(declaration())
    for skip in modules.skips.values():torch.nn.init.normal_(skip.restore.weight, std=.1)
    rng = torch.get_rng_state().clone()
    corrections = build_corrections(modules)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in corrections.parameters()) == len(modules.skips)*8*9
    originals = dict(modules.skips.items())
    value = torch.randn(1, 3+8*12, 8)
    expected = run(modules, value, 'none')
    with pytest.raises(RuntimeError, match='intentional'):
        with full_width_probe(modules, corrections):
            torch.testing.assert_close(run(modules,value,'normal'),expected,rtol=0,atol=0)
            with pytest.raises(ValueError):
                with full_width_probe(modules,corrections):pass
            raise RuntimeError('intentional')
    assert all(modules.skips[name] is base for name,base in originals.items())


def test_full_channel_information_and_dtype_move():
    modules = HybridModules(declaration())
    corrections = build_corrections(modules)
    modules.double()
    with full_width_probe(modules,corrections):
        wrapper = next(iter(modules.skips.values()))
        assert isinstance(wrapper,FullWidthSkip)
        x = torch.randn(1,8,4,6,dtype=torch.float64,requires_grad=True)
        captured = wrapper.capture_grid(x)
        torch.testing.assert_close(captured[:,wrapper.rank:],x,rtol=0,atol=0)
        with torch.no_grad():
            wrapper.base.restore.weight.zero_()
            wrapper.correction.weight[:,:,1,1].fill_(1.)
        y = wrapper.restore_grid(captured)
        torch.testing.assert_close(y,x,rtol=0,atol=0)
        y.square().sum().backward()
        torch.testing.assert_close(x.grad,2*x)


def test_bad_correction_contract_is_rejected_before_mutation():
    modules = HybridModules(declaration())
    corrections = build_corrections(modules)
    name = next(iter(corrections))
    corrections[name] = torch.nn.Conv2d(8,8,3,padding=1,bias=False)
    originals = dict(modules.skips.items())
    with pytest.raises(ValueError):
        with full_width_probe(modules,corrections):pass
    assert all(modules.skips[name] is base for name,base in originals.items())
