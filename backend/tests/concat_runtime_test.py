from copy import deepcopy

import pytest
import torch

from hybrid_rewiring_test import declaration, execute
from core.rewiring.hybrid_modules import HybridModules, HybridExecution
from core.rewiring.hybrid_spec import build_hybrid_spec, legacy_topology
from core.experiments.concat_runtime import build_corrections, concat_probe


def run(modules, value, mode):
    return modules.forward_stream(value, 'main', 8, 12, 3, None, execute, lambda *a: None,
        checkpoint_modules=mode != 'none', checkpoint_offload=mode == 'cpu')


@pytest.mark.parametrize('storage', ['gpu', 'cpu', 'recompute'])
@pytest.mark.parametrize('mode', ['none', 'normal', 'cpu'])
def test_storage_checkpoint_gradients_match(storage, mode):
    torch.manual_seed(63)
    spec = declaration()
    spec = build_hybrid_spec(legacy_topology(spec), stream_widths=spec['stream_widths'],
        boundaries=spec['boundaries'], refiners=spec['refiners'],
        skips=[dict(item, storage=storage) for item in spec['skips']])
    modules = HybridModules(spec)
    for skip in modules.skips.values():torch.nn.init.normal_(skip.restore.weight, std=.02)
    corrections = build_corrections(modules)
    for correction in corrections.values():torch.nn.init.normal_(correction.output.weight, std=.02)
    reference = deepcopy(modules)
    other_corrections = deepcopy(corrections)
    value = torch.randn(1, 3+8*12, 8, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    with concat_probe(reference, other_corrections):
        expected = run(reference, other, 'none')
        expected.square().mean().backward()
    with concat_probe(modules, corrections):
        observed = run(modules, value, mode)
        observed.square().mean().backward()
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(value.grad, other.grad)
    for name, parameter in corrections.named_parameters():
        counterpart = dict(other_corrections.named_parameters())[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, counterpart.grad)
    for name, parameter in modules.named_parameters():
        counterpart = dict(reference.named_parameters())[name]
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, counterpart.grad)


def test_zero_equivalence_rng_and_failure_cleanup():
    modules = HybridModules(declaration())
    rng = torch.get_rng_state().clone()
    corrections = build_corrections(modules)
    assert torch.equal(rng, torch.get_rng_state())
    value = torch.randn(1, 3+8*12, 8)
    baseline = run(modules, value, 'none')
    original = HybridExecution.before
    with pytest.raises(RuntimeError, match='intentional'):
        with concat_probe(modules, corrections):
            torch.testing.assert_close(run(modules, value, 'normal'), baseline, rtol=0, atol=0)
            with pytest.raises((RuntimeError, ValueError)):
                with concat_probe(modules, corrections):pass
            raise RuntimeError('intentional')
    assert HybridExecution.before is original
    assert not hasattr(modules, '_concat_probe_corrections')
    with concat_probe(modules, corrections):pass


def test_context_aligns_corrections_after_root_dtype_move():
    modules = HybridModules(declaration())
    corrections = build_corrections(modules)
    modules.double()
    with concat_probe(modules, corrections):
        assert all(p.dtype == torch.float64 for p in corrections.parameters())
        observed = run(modules, torch.randn(1, 3+8*12, 8, dtype=torch.float64), 'normal')
        assert observed.dtype == torch.float64 and torch.isfinite(observed).all()
