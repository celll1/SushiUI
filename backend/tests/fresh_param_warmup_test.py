"""Tensor-specific resume warmup without GPU work or per-tensor groups."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import Adafactor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.optimizers.adafactor_fused import patch_adafactor_fused
from core.training.optimizers.fresh_param_warmup import (
    arm_fresh_param_warmup,
    fresh_param_warmup_factor,
    parameter_warmup_lr,
)


def _scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def test_context_scales_only_fresh_parameter_and_restores_group_lr():
    old, new = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([old, new], lr=2e-4)
    scheduler = _scheduler(optimizer)
    scheduler.last_epoch = 100
    arm_fresh_param_warmup(optimizer, scheduler, {id(new)}, 100, 20)
    group = optimizer.param_groups[0]

    with parameter_warmup_lr(optimizer, old, group):
        assert group["lr"] == pytest.approx(2e-4)
    with parameter_warmup_lr(optimizer, new, group):
        assert group["lr"] == 0.0
    assert group["lr"] == pytest.approx(2e-4)

    scheduler.last_epoch = 110
    with parameter_warmup_lr(optimizer, new, group):
        assert group["lr"] == pytest.approx(1e-4)
    scheduler.last_epoch = 120
    assert fresh_param_warmup_factor(optimizer, new) == 1.0
    assert not hasattr(optimizer, "_sushi_fresh_param_warmup")


def test_first_restored_tensor_after_the_end_clears_the_fast_path():
    old, new = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([old, new], lr=2e-4)
    scheduler = _scheduler(optimizer)
    arm_fresh_param_warmup(optimizer, scheduler, {id(new)}, 10, 5)
    scheduler.last_epoch = 15

    assert fresh_param_warmup_factor(optimizer, old) == 1.0
    assert not hasattr(optimizer, "_sushi_fresh_param_warmup")


def test_fused_adafactor_updates_restored_tensor_but_zero_warms_fresh_tensor():
    old = torch.nn.Parameter(torch.ones(4))
    new = torch.nn.Parameter(torch.ones(4))
    optimizer = Adafactor(
        [{"params": [old, new]}], lr=1e-2, relative_step=False,
        scale_parameter=False, warmup_init=False, weight_decay=0.0,
    )
    patch_adafactor_fused(optimizer)
    scheduler = _scheduler(optimizer)
    scheduler.last_epoch = 50
    arm_fresh_param_warmup(optimizer, scheduler, {id(new)}, 50, 10)
    old.grad = torch.ones_like(old)
    new.grad = torch.ones_like(new)

    optimizer.step_param(old, optimizer.param_groups[0])
    optimizer.step_param(new, optimizer.param_groups[0])

    assert not torch.equal(old, torch.ones_like(old))
    assert torch.equal(new, torch.ones_like(new))
    assert optimizer.state[new]["step"] == 1
