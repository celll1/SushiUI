"""CUDA correctness and lifetime gates for bounded async activation offload."""

import sys
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.memory_management import AsyncActivationOffloader, offload_activations


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _step(mode: str, *, budget: int = 1024 * 1024):
    torch.manual_seed(41)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 16),
    ).cuda().float()
    x = torch.randn(8, 32, device="cuda", requires_grad=True)
    stats = {"bytes": 0}
    engine = AsyncActivationOffloader(budget) if mode == "async" else None
    with offload_activations(
        True,
        threshold_bytes=1,
        transfer_mode=mode,
        async_engine=engine,
        pinned_budget_bytes=budget,
        stats=stats,
    ):
        loss = model(x).square().mean()
        loss.backward()
    torch.cuda.synchronize()
    return (
        loss.detach().cpu(),
        x.grad.detach().cpu(),
        [parameter.grad.detach().cpu() for parameter in model.parameters()],
        stats,
        engine,
    )


def test_async_activation_offload_matches_synchronous_gradients():
    sync = _step("sync")
    async_result = _step("async")
    torch.testing.assert_close(async_result[0], sync[0], rtol=0, atol=0)
    torch.testing.assert_close(async_result[1], sync[1], rtol=0, atol=0)
    for actual, expected in zip(async_result[2], sync[2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert async_result[3]["async_bytes"] > 0
    assert async_result[3].get("sync_fallback_bytes", 0) == 0
    assert async_result[4].arena_bytes == 1024 * 1024


def test_async_activation_offload_falls_back_at_fixed_arena_limit():
    result = _step("async", budget=512)
    assert result[3]["async_bytes"] <= 512
    assert result[3]["sync_fallback_bytes"] > 0
    assert result[4].arena_bytes == 512


def test_async_activation_offloader_is_reusable_after_exception():
    engine = AsyncActivationOffloader(1024 * 1024)
    with pytest.raises(RuntimeError, match="injected"):
        with offload_activations(
            True, threshold_bytes=1, transfer_mode="async", async_engine=engine
        ):
            value = torch.randn(16, 16, device="cuda", requires_grad=True)
            (value * value).sum()
            raise RuntimeError("injected")
    with offload_activations(
        True, threshold_bytes=1, transfer_mode="async", async_engine=engine
    ):
        value = torch.randn(16, 16, device="cuda", requires_grad=True)
        (value * value).sum().backward()

