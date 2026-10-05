"""CPU gates for independent transfer queues and bounded delayed lookahead."""
from concurrent.futures import Future
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
import sys
from pathlib import Path
import weakref

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.memory_management.activation_staging import ActivationStaging


@pytest.fixture
def staging(monkeypatch):
    stream = SimpleNamespace(wait_event=lambda *_: None, record_event=lambda: None,
                             synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: nullcontext())
    result = ActivationStaging(torch.empty(8*1024**2, dtype=torch.uint8), "cpu", stream, stream, {})
    yield result
    result.close()


def test_restore_queue_is_not_blocked_by_unrelated_pack(staging, monkeypatch):
    entered, release = Event(), Event()
    def block(*_):
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(staging, "_pack", block)
    saved = SimpleNamespace(version=0)
    staging.pack(torch.ones(256), saved)
    assert entered.wait(1)
    try:
        independent = SimpleNamespace(future=None, cpu=torch.ones(8), d2h_ready=None)
        restored, _ = staging.h2d_worker.submit(staging._restore, independent).result(timeout=1)
        torch.testing.assert_close(restored, independent.cpu)
    finally:
        release.set()


def test_unexpected_prefetch_is_used_later_without_discard(staging):
    unexpected = SimpleNamespace(cpu=torch.zeros(8))
    class Saved:
        pass
    wanted = Saved()
    future = Future()
    staging.prefetched = (weakref.ref(wanted), future)
    assert staging._take_prefetch(unexpected) is None
    assert staging.prefetched is not None and not future.done()
    future.set_result((torch.ones(8), None))
    result, _ = staging._take_prefetch(wanted)
    assert result.sum() == 8
    assert staging.stats["prefetch_hits"] == 1
    assert staging.stats["prefetch_deferred"] == 1
    assert not staging.stats.get("prefetch_wasted_bytes", 0)


def test_strided_pack_accounts_for_scratch_and_preserves_values(staging):
    value = torch.arange(262144, dtype=torch.float32).reshape(512, 512).t()
    saved = SimpleNamespace(version=value._version)
    assert staging.pack(value, saved)
    saved.future.result(timeout=2)
    torch.testing.assert_close(saved.cpu, value, rtol=0, atol=0)
    assert staging.stats["staged_gpu_peak_bytes"] == 2*value.numel()*value.element_size()
    assert staging.stats["staged_gpu_peak_bytes"] <= staging.gpu_budget


def test_large_strided_pack_falls_back_without_allocating(staging):
    value = torch.empty(1024, 1024).t()
    assert not staging.pack(value, SimpleNamespace(version=value._version))
    assert staging.stats["fallback_layout_bytes"] == value.numel()*value.element_size()
    assert not staging.pending


def test_restore_failure_still_shuts_down_both_workers(staging, monkeypatch):
    class Saved:
        pass
    saved = Saved()
    def fail(*_):
        raise RuntimeError("injected restore failure")
    monkeypatch.setattr(staging, "_restore", fail)
    staging.prefetched = (weakref.ref(saved), staging.h2d_worker.submit(staging._restore, saved))
    with pytest.raises(RuntimeError, match="injected restore failure"):
        staging.close()
    assert staging.prefetched is None
    assert all(not thread.is_alive() for thread in staging.h2d_worker._threads)
    assert all(not thread.is_alive() for thread in staging.d2h_worker._threads)
