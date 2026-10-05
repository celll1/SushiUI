"""Numerical gates for transfer deduplication and coalesced update waits."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.memory_management import offload_activations
from core.training.base_trainer import BaseTrainer
from core.training.optimizers.bnb_step_sync import coalesce_adamw8bit_step
from core.training.optimizers.fused_grad_clip import FusedGradClipper


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_coalesced_bnb_matches_updates_state_hooks_and_resume(dtype):
    import bitsandbytes as bnb

    torch.manual_seed(67)
    a = [torch.nn.Parameter(torch.randn(n, device="cuda", dtype=dtype)) for n in (32, 8192, 4096)]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    original = bnb.optim.AdamW8bit(a, lr=.001)
    optimized = coalesce_adamw8bit_step(bnb.optim.AdamW8bit(b, lr=.001))
    schedules = [torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1 / (step + 1))
                 for opt in (original, optimized)]
    hooks = []
    optimized.register_step_pre_hook(lambda *args: hooks.append("pre"))
    optimized.register_step_post_hook(lambda *args: hooks.append("post"))
    for step in range(4):
        for x, y in zip(a, b):
            x.grad = torch.randn_like(x)
            y.grad = x.grad.clone()
        if step == 3:
            a[0].grad = b[0].grad = None
        before = original.step(lambda: a[0].square().mean())
        after = optimized.step(lambda: b[0].square().mean())
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
            for key, value in original.state[x].items():
                other = optimized.state[y][key]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, other, rtol=0, atol=0)
                else:
                    assert value == other
        if step == 1:
            saved = copy.deepcopy(optimized.state_dict())
            optimized.load_state_dict(saved)
        for schedule in schedules:
            schedule.step()
    assert hooks == ["pre", "post"] * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_transfer_dedup_preserves_gradients_and_versioned_saves():
    def run(offload):
        leaf = torch.linspace(-1, 1, 4096, device="cuda", requires_grad=True)
        value = leaf * 2
        stats = {}
        with offload_activations(offload, threshold_bytes=1, stats=stats):
            first = (value * value).sum()
            if offload:
                value.add_(1)
            else:
                value = value + 1
            second = (value * value).sum()
            (first + second).backward()
        return leaf.grad.cpu(), stats
    expected, _ = run(False)
    actual, stats = run(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert stats["deduplicated_bytes"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_dedup_preserves_gradients_and_separate_views():
    leaf = torch.linspace(-1, 1, 4096, device="cuda", requires_grad=True)
    stats = {}
    with offload_activations(True, threshold_bytes=1, stats=stats,
                            transfer_mode="async", pinned_budget_bytes=65536):
        value = leaf * 2
        reversed_value = value.flip(0)
        loss = (value * value).sum() + (value * reversed_value).sum()
        loss.backward()
    # The symmetric cross term cancels the square term's gradient.
    torch.testing.assert_close(leaf.grad, torch.zeros_like(leaf), rtol=0, atol=1e-6)
    assert stats["bytes"] == 2 * value.numel() * value.element_size()
    assert stats["deduplicated_bytes"] == 2 * value.numel() * value.element_size()


def test_nonfused_probe_reuses_norm_without_changing_gradients(monkeypatch):
    params = [torch.nn.Parameter(torch.ones(17)), torch.nn.Parameter(torch.ones(31))]
    for parameter in params:
        parameter.grad = torch.linspace(-1, 1, parameter.numel())
    optimizer = torch.optim.AdamW(params)
    trainer = SimpleNamespace(optimizer=optimizer, log_prefix="[test]",
        _fused_grad_clipper=FusedGradClipper(factor=0, warmup_steps=2, monitor_only=True))
    monkeypatch.setattr(BaseTrainer, "_grad_norm_parameter_entries", lambda self: [(p, "unet") for p in params])
    original = torch.linalg.vector_norm
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.linalg, "vector_norm", counted)
    before = [p.grad.clone() for p in params]
    BaseTrainer._calculate_grad_norms(trainer)
    BaseTrainer._clip_nonfused_parameters(trainer)
    assert len(calls) == len(params)
    for parameter, grad in zip(params, before):
        torch.testing.assert_close(parameter.grad, grad, rtol=0, atol=0)
