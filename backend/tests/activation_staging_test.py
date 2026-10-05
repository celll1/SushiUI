"""CUDA gates for bounded pageable staging and speculative restore ordering."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.memory_management.activation_dispatcher import AsyncActivationOffloader, offload_activations

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def run(mode, *, retained=False):
    leaf = torch.linspace(-.8, .8, 262144, device="cuda", requires_grad=True)
    stats = {}
    engine = AsyncActivationOffloader(8*1024**2)
    with offload_activations(mode != "none", threshold_bytes=1, transfer_mode="async" if mode == "async" else "sync",
                            async_engine=engine, stats=stats):
        value = leaf*2
        for _ in range(12):
            value = value.sin()
        loss = value.square().sum()
        loss.backward(retain_graph=retained)
        first = leaf.grad.detach().clone()
        if retained:
            leaf.grad = None
            loss.backward()
            torch.testing.assert_close(leaf.grad, first, rtol=0, atol=0)
    return loss.detach().cpu(), first.cpu(), stats, engine


def test_staging_matches_no_offload_and_uses_bounded_lookahead():
    expected = run("none")
    actual = run("async", retained=True)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert actual[2]["staged_bytes"] > 0
    assert actual[2].get("sync_fallback_bytes", 0) == 0
    assert actual[2]["staged_gpu_peak_bytes"] <= 2*1024**2
    assert actual[2]["prefetch_hits"] > 0
    assert actual[3].arena_bytes == 8*1024**2
    assert actual[3].staging is None


def test_staging_worker_error_is_propagated_and_engine_reusable(monkeypatch):
    from core.memory_management.activation_staging import ActivationStaging
    engine = AsyncActivationOffloader(8*1024**2)
    original = ActivationStaging._pack
    def fail(*args):
        raise RuntimeError("injected staging error")
    with monkeypatch.context() as patch:
        patch.setattr(ActivationStaging, "_pack", fail)
        with pytest.raises(RuntimeError, match="injected staging error"):
            with offload_activations(True, threshold_bytes=1, transfer_mode="async", async_engine=engine):
                leaf = torch.ones(262144, device="cuda", requires_grad=True)
                value = leaf*2
                for _ in range(12): value = value.sin()
                value.sum().backward()
    assert not engine._active and engine.staging is None
    assert ActivationStaging._pack is original
    with offload_activations(True, threshold_bytes=1, transfer_mode="async", async_engine=engine):
        leaf = torch.ones(262144, device="cuda", requires_grad=True)
        leaf.sin().square().sum().backward()


def test_oversize_and_noncontiguous_pageable_fallback_remains_exact():
    grads = []
    for mode in ("sync", "async"):
        leaf = torch.linspace(-1, 1, 1048576, device="cuda", requires_grad=True)
        stats = {}
        with offload_activations(True, threshold_bytes=1, transfer_mode=mode,
                                pinned_budget_bytes=8*1024**2, stats=stats):
            value = (leaf*2).reshape(1024, 1024).t()
            (value.sin().cos().square().sum()).backward()
        grads.append(leaf.grad.cpu())
    torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)
    assert stats["sync_fallback_bytes"] > 0


def test_async_mutation_refuses_silent_snapshot_corruption():
    with pytest.raises(RuntimeError, match="modified in place"):
        with offload_activations(True, threshold_bytes=1, transfer_mode="async", pinned_budget_bytes=8*1024**2):
            leaf = torch.ones(262144, device="cuda", requires_grad=True)
            value = leaf*2
            loss = value.sin().sum()
            value.add_(1)
            loss.backward()
