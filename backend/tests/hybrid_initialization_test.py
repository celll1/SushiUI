"""Teacher fitting must never change the native model's training ownership."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch import nn

from core.rewiring.hybrid_modules import HybridModules
from core.rewiring.initialization import frozen_native_teacher
from hybrid_rewiring_test import declaration


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Conv2d(8, 8, 1)
        self._residual_unet_spec = declaration()
        self.rewiring_modules = HybridModules(self._residual_unet_spec)
        self.fail_native = False

    def forward(self, value):
        if self._residual_unet_spec is None and self.fail_native:
            raise RuntimeError("teacher failure")
        output = self.base(value)
        if self._residual_unet_spec is not None:
            output = output + self.rewiring_modules.local["entry"](value)
        return output


def test_fitting_changes_only_added_weights_and_restores_all_ownership():
    model = TinyModel()
    model.base.bias.requires_grad_(False)
    model.rewiring_modules.requires_grad_(False)
    model.base.eval()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    modes = [module.training for module in model.modules()]
    native = {name: value.clone() for name, value in model.base.state_dict().items()}
    topology = model._residual_unet_spec
    value = torch.randn(2, 8, 8, 10)
    rng = torch.get_rng_state().clone()
    with frozen_native_teacher(model) as teacher:
        assert all(not parameter.requires_grad for parameter in model.base.parameters())
        assert all(parameter.requires_grad for parameter in teacher.parameters)
        optimizer = torch.optim.SGD(teacher.parameters, lr=0.1)
        for _ in range(3):
            target = teacher(value)
            assert not target.requires_grad
            torch.testing.assert_close(target, model.base(value))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(value)
            (prediction-target-0.1).square().mean().backward()
            optimizer.step()
            torch.rand(9)
        model.train()
        assert model.rewiring_modules.local["entry"].expand.weight.abs().max() > 0
    assert model._residual_unet_spec is topology
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert [module.training for module in model.modules()] == modes
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in model.base.state_dict().items():
        assert torch.equal(value, native[name])
    assert teacher.model is None and teacher.parameters == ()
    with pytest.raises(RuntimeError, match="outside"):
        teacher(value)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failure_restores_topology_flags_modes_and_rng(failure):
    model = TinyModel()
    topology = model._residual_unet_spec
    flags = [parameter.requires_grad for parameter in model.parameters()]
    model.base.eval()
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    with pytest.raises(failure):
        with frozen_native_teacher(model) as teacher:
            model.train()
            torch.rand(8)
            model.fail_native = True
            with pytest.raises(RuntimeError, match="teacher failure"):
                teacher(torch.zeros(1, 8, 8, 8))
            assert model._residual_unet_spec is topology
            raise failure("cancel or fail")
    assert model._residual_unet_spec is topology
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert not hasattr(model, "_hybrid_teacher_owner")


def test_existing_training_gradients_are_never_discarded():
    model = TinyModel()
    model.base.weight.grad = torch.ones_like(model.base.weight)
    gradient = model.base.weight.grad
    with pytest.raises(ValueError, match="existing training gradients"):
        with frozen_native_teacher(model):
            pass
    assert model.base.weight.grad is gradient


def test_nested_teacher_and_unexpected_topology_are_refused():
    model = TinyModel()
    topology = model._residual_unet_spec
    with frozen_native_teacher(model) as teacher:
        with pytest.raises(ValueError, match="already active"):
            with frozen_native_teacher(model):
                pass
        model._residual_unet_spec = dict(topology, architecture="ming_image_01")
        with pytest.raises(RuntimeError, match="exclusive ownership"):
            teacher(torch.zeros(1, 8, 8, 8))
    assert model._residual_unet_spec is topology


def test_native_shared_parameters_and_block_offload_are_refused():
    model = TinyModel()
    model.native_alias = next(model.rewiring_modules.parameters())
    with pytest.raises(ValueError, match="owned separately"):
        with frozen_native_teacher(model):
            pass
    del model.native_alias
    model._block_offloader = object()
    with pytest.raises(ValueError, match="resident native blocks"):
        with frozen_native_teacher(model):
            pass
