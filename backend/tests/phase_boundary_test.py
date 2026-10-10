from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from hybrid_rewiring_test import declaration, execute
from core.experiments.phase_boundary import PhaseDown, PhaseUp, build_branches, phase_boundary_probe
from core.rewiring.hybrid_modules import HybridModules, HybridExecution
from core.rewiring.hybrid_spec import build_hybrid_spec, legacy_topology


def run(modules, value, mode, groups=1):
    return modules.forward_stream(value, 'main', 8*groups, 12, 3, None, execute, lambda *a: None,
        grid_groups=groups, checkpoint_modules=mode != 'none', checkpoint_offload=mode == 'cpu')


def test_phase_layout_oracles_and_staged_gradients():
    down = PhaseDown(2, 1)
    value = torch.randn(2, 2, 4, 6, requires_grad=True)
    with torch.no_grad():
        down.reduce.weight.fill_(.2)
        down.expand.weight.fill_(.3)
    normalized = F.layer_norm(value.movedim(1, -1), (2,)).movedim(-1, 1)
    packed = torch.stack([normalized[:, channel, dy::2, dx::2] for channel in range(2) for dy in range(2) for dx in range(2)], dim=1)
    expected = .3*F.silu(.2*packed.sum(1, keepdim=True)).expand(-1, 2, -1, -1)
    torch.testing.assert_close(down(value), expected)
    up = PhaseUp(1, 1)
    with torch.no_grad():
        up.reduce.weight.fill_(1.)
        up.expand.weight.copy_(torch.arange(1., 5.).reshape(4, 1, 1, 1))
    coarse = torch.ones(1, 1, 2, 3)
    actual = up(coarse)
    for dy in range(2):
        for dx in range(2):
            torch.testing.assert_close(actual[:, :, dy::2, dx::2], F.silu(coarse)*(1+2*dy+dx))
    for module, data in ((PhaseDown(2, 1), value.detach()), (PhaseUp(2, 1), value.detach())):
        optimizer = torch.optim.SGD(module.parameters(), lr=.1)
        output = module(data)
        assert torch.count_nonzero(output) == 0
        output.sum().backward()
        assert module.expand.weight.grad.abs().sum() > 0
        assert torch.count_nonzero(module.reduce.weight.grad) == 0
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        module(data).sum().backward()
        assert module.reduce.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('storage', ['gpu', 'cpu', 'recompute'])
@pytest.mark.parametrize('mode', ['none', 'normal', 'cpu'])
def test_nested_grouped_checkpoint_and_storage_gradients(storage, mode):
    spec = declaration()
    spec = build_hybrid_spec(legacy_topology(spec), stream_widths=spec['stream_widths'],
        boundaries=spec['boundaries'], refiners=spec['refiners'],
        skips=[dict(item, storage=storage) for item in spec['skips']])
    modules = HybridModules(spec)
    down, up = build_branches(modules, 3)
    for family in (down, up):
        for module in family.values():
            module.expand.weight.data.normal_(std=.05)
    other = deepcopy(modules); other_down, other_up = deepcopy(down), deepcopy(up)
    value = torch.randn(1, 3+16*12, 8, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    with phase_boundary_probe(other, other_down, other_up):
        expected = run(other, reference, 'none', 2)
        expected.square().mean().backward()
    with phase_boundary_probe(modules, down, up):
        actual = run(modules, value, mode, 2)
        actual.square().mean().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(value.grad, reference.grad)
    for name, param in modules.named_parameters():
        torch.testing.assert_close(param.grad, dict(other.named_parameters())[name].grad)
    for family, other_family in ((down, other_down), (up, other_up)):
        for name, param in family.named_parameters():
            assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
            torch.testing.assert_close(param.grad, dict(other_family.named_parameters())[name].grad)


def test_zero_identity_rng_and_other_root_isolation():
    modules = HybridModules(declaration())
    for family in (modules.down, modules.up, modules.local):
        for module in family.values():
            module.expand.weight.data.normal_(std=.1)
    value = torch.randn(1, 3+8*12, 8)
    reference = run(modules, value, 'none')
    rng = torch.get_rng_state().clone()
    down, up = build_branches(modules, 3)
    assert torch.equal(rng, torch.get_rng_state())
    other = deepcopy(modules)
    with phase_boundary_probe(modules, down, up):
        torch.testing.assert_close(run(modules, value, 'normal'), reference, rtol=0, atol=0)
        torch.testing.assert_close(run(other, value, 'none'), reference, rtol=0, atol=0)
    assert not hasattr(modules, '_phase_up_probe')
    assert not hasattr(HybridExecution, 'up_from_coarse')


def test_exception_cleanup_and_existing_none_hook():
    modules = HybridModules(declaration()); originals = dict(modules.down.items())
    down, up = build_branches(modules, 3)
    HybridExecution.up_from_coarse = None
    try:
        with pytest.raises(RuntimeError, match='sentinel'):
            with phase_boundary_probe(modules, down, up):
                raise RuntimeError('sentinel')
        assert HybridExecution.up_from_coarse is None
        assert dict(modules.down.items()) == originals
        assert not hasattr(modules, '_phase_up_probe')
        with phase_boundary_probe(modules, down, up):
            pass
    finally:
        del HybridExecution.up_from_coarse


def test_failed_transfer_releases_ownership(monkeypatch):
    modules = HybridModules(declaration()); down, up = build_branches(modules, 3)
    with monkeypatch.context() as patch:
        def fail(*a, **k):
            raise RuntimeError('transfer')
        patch.setattr(down, 'to', fail)
        with pytest.raises(RuntimeError, match='transfer'):
            with phase_boundary_probe(modules, down, up):
                pass
    with phase_boundary_probe(modules, down, up):
        pass


def test_invalid_layouts_and_callback_shape(monkeypatch):
    with pytest.raises(ValueError):
        PhaseDown(2, 1)(torch.ones(1, 2, 3, 4))
    with pytest.raises(ValueError):
        PhaseUp(2, 1)(torch.ones(1, 3, 2, 2))
    modules = HybridModules(declaration())
    monkeypatch.setattr(HybridExecution, 'up_from_coarse', lambda *a: torch.zeros(1), raising=False)
    with pytest.raises(ValueError, match='Coarse up correction'):
        run(modules, torch.randn(1, 3+8*12, 8), 'none')
