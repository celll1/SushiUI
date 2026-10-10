from copy import deepcopy

import pytest
import torch

from hybrid_rewiring_test import declaration, execute
from core.experiments.correction_capacity import widen_boundaries
from core.rewiring.hybrid_modules import HybridModules
from core.rewiring.hybrid_spec import hybrid_parameter_counts


def source():
    torch.manual_seed(19)
    modules = HybridModules(declaration())
    with torch.no_grad():
        for family in (modules.down, modules.up, modules.local):
            for module in family.values():
                module.expand.weight.normal_(std=.05)
    return modules


def ranks(modules):
    return {item['id']: min(8, item['rank'] * 2) for item in modules.declaration['boundaries']}


def test_fitted_output_rng_and_source_state_preserved():
    original = source()
    saved = deepcopy(original.state_dict())
    rng = torch.get_rng_state().clone()
    candidate = widen_boundaries(original, ranks(original))
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    value = torch.randn(2, 8, 8, 12)
    for family in ('down', 'up'):
        for name, module in getattr(original, family).items():
            torch.testing.assert_close(getattr(candidate, family)[name](value), module(value))


@pytest.mark.parametrize('checkpoint', [False, True])
def test_stream_and_input_gradient_parity(checkpoint):
    original = source()
    candidate = widen_boundaries(original, ranks(original))
    value = torch.randn(1, 3 + 8*12, 8, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    kwargs = dict(grid_groups=1, checkpoint_modules=checkpoint)
    expected = original.forward_stream(value, 'main', 8, 12, 3, None, execute, lambda *a: None, **kwargs)
    actual = candidate.forward_stream(other, 'main', 8, 12, 3, None, execute, lambda *a: None, **kwargs)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(other.grad, value.grad)


def test_added_coordinates_receive_staged_gradients_and_factory_roundtrip():
    original = source()
    candidate = widen_boundaries(original, ranks(original))
    name = next(iter(candidate.down))
    module = candidate.down[name]
    old_rank = original.down[name].reduce.out_channels
    value = torch.randn(2, 8, 8, 12)
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    module(value).square().mean().backward()
    assert module.expand.weight.grad[:, old_rank:].abs().sum() > 0
    assert torch.count_nonzero(module.reduce.weight.grad[old_rank:]) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module(value).square().mean().backward()
    assert module.reduce.weight.grad[old_rank:].abs().sum() > 0
    restored = HybridModules(candidate.declaration)
    restored.load_state_dict(candidate.state_dict(), strict=True)
    torch.testing.assert_close(restored.down[name](value), module(value), rtol=0, atol=0)
    assert sum(p.numel() for p in candidate.parameters()) == hybrid_parameter_counts(candidate.declaration)['added']
    assert candidate.declaration['module_layout_sha256'] != original.declaration['module_layout_sha256']


@pytest.mark.parametrize('mapping', [{}, {'missing': 4}, {'b0': True}, {'b0': 1}, {'b0': 999}])
def test_bad_capacity_does_not_modify_source(mapping):
    original = source()
    name = next(iter(original.down))
    mapping = {name if k == 'b0' else k: v for k, v in mapping.items()}
    saved = deepcopy(original.state_dict())
    with pytest.raises(ValueError):
        widen_boundaries(original, mapping)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


def test_private_extra_state_is_refused():
    original = source()
    original.register_parameter('private_gain', torch.nn.Parameter(torch.ones(8)))
    with pytest.raises(ValueError, match='unextended public factory'):
        widen_boundaries(original, ranks(original))
