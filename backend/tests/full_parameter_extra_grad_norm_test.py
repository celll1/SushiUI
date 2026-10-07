"""Optimizer-owned components outside the conventional model roots."""
from types import SimpleNamespace

import pytest
import torch

from core.training.base_trainer import BaseTrainer


@pytest.mark.parametrize("fused", [False, True])
def test_chimera_bridge_and_future_components_are_counted_once(fused):
    unet = torch.nn.Linear(2, 2)
    bridge = torch.nn.Linear(2, 3)
    extra = torch.nn.Parameter(torch.ones(2))
    unet.requires_grad_(False)
    for parameter in bridge.parameters():
        parameter.grad = torch.full_like(parameter, 3.)
    extra.grad = torch.full_like(extra, 4.)
    groups = [dict(params=list(bridge.parameters()), component="text_encoder"),
              dict(params=[extra], component="unet")]
    optimizers = ([torch.optim.SGD([group], lr=.1) for group in groups] if fused else
                  [torch.optim.SGD(groups, lr=.1)])
    trainer = SimpleNamespace(unet=unet, transformer_original=unet, text_encoder=None,
        optimizer=optimizers[0], fused_optimizer_groups=SimpleNamespace(optimizers=optimizers) if fused else None,
        _full_parameter_grad_components=lambda: {}, log_prefix="[test]")
    if fused:
        from core.training.optimizers.fused_grad_norm import FusedGradNormAccumulator
        trainer._fused_grad_norm = FusedGradNormAccumulator()
        trainer._fused_grad_norm.enabled = True
        for parameter in (*bridge.parameters(), extra):
            trainer._fused_grad_norm.record(parameter)
            parameter.grad = None
    trainer._grad_norm_parameter_entries = lambda: BaseTrainer._grad_norm_parameter_entries(trainer)
    total, te, te1, te2, main, vision = BaseTrainer._calculate_grad_norms(trainer)
    assert total == pytest.approx((9*9+16*2)**.5)
    assert te == pytest.approx(9.)
    assert main == pytest.approx((16*2)**.5)
    assert te1 == te2 == vision == 0
    entries = trainer._grad_norm_parameter_entries()
    assert len(entries) == len({id(parameter) for parameter, _ in entries})
    assert {id(parameter) for parameter in bridge.parameters()} <= {id(parameter) for parameter, _ in entries}


def test_adapter_component_override_wins_for_optimizer_only_parameter():
    parameter = torch.nn.Parameter(torch.ones(3))
    parameter.grad = torch.full_like(parameter, 2.)
    trainer = SimpleNamespace(optimizer=torch.optim.SGD([parameter], lr=.1),
        _full_parameter_grad_components=lambda: {id(parameter): "text_encoder_2"},log_prefix="[test]")
    total, te, te1, te2, main, vision = BaseTrainer._calculate_grad_norms(trainer)
    assert total == te == te2 == pytest.approx(12**.5)
    assert te1 == main == vision == 0
