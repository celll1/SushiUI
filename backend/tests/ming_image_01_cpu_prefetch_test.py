"""Frozen native CPU worker ordering, ownership, errors and cleanup."""

import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def forbidden_cuda(*a, **k): raise AssertionError("Prefetch attempted CUDA initialization")
torch.cuda.get_device_capability=lambda *a, **k:(8,9)
torch.cuda.is_available=lambda:False
torch.cuda._lazy_init=forbidden_cuda
torch._C._cuda_init=forbidden_cuda

from core.models.ming_image_01.cpu_prefetch import MingCpuPrefetch
from core.models.ming_image_01.training_buffer import MingBatchBuffer, batch_identity, close_training_buffer
from core.models.ming_image_01.dataset import load_shared_items, prepare_record_conditioning
from ming_image_01_training_buffer_test import setup


def owner_setup(tmp_path):
    value, owner, batches, old, _, policy = setup(tmp_path)
    old.close()
    owner.ming_batch_buffer = None
    policy.modes["text_encoding_mode"] = "cpu_prefetch"
    return value, owner, batches, policy


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefetched_native_streams_match_synchronous_and_drive_batch_buffer(tmp_path, dtype):
    value, owner, batches, _ = owner_setup(tmp_path)
    value.conditioner.to(dtype=dtype)
    expected = [prepare_record_conditioning(value, load_shared_items(owner, pairs, epoch=2, batch_position=5+i),
                resolution=32).for_step() for i, pairs in enumerate(batches)]
    buffer = MingBatchBuffer(owner, batches, epoch=2, batch_offset=5,
                            latent_interval=2, text_interval=3, prefetch_depth=1)
    owner.ming_batch_buffer = buffer
    worker = buffer.prefetcher
    try:
        for index, pairs in enumerate(batches):
            batch = buffer.prepare(index, pairs).for_step()
            torch.testing.assert_close(batch.conditioning.queries, expected[index].queries, rtol=0, atol=0)
            torch.testing.assert_close(batch.conditioning.direct, expected[index].direct, rtol=0, atol=0)
            assert batch.conditioning.direct.dtype == dtype and batch.conditioning.direct.device.type == "cpu"
            assert not batch.conditioning.direct.requires_grad
            assert worker.queue.qsize() <= 1
    finally:
        close_training_buffer(owner)
    assert not worker.thread.is_alive() and worker.done.is_set() and not worker.conditioner._forward_pre_hooks
    assert not torch.cuda.is_initialized()


def test_full_queue_stop_joins_and_releases_exclusive_forward_guard(tmp_path, monkeypatch):
    from ming_image_01_conditioning_test import inputs
    value, owner, batches, _ = owner_setup(tmp_path)
    original, calls, two_encoded = value.conditioner.forward, [], threading.Event()
    def forward(*a, **k):
        result = original(*a, **k)
        calls.append(1)
        if len(calls) == 2: two_encoded.set()
        return result
    monkeypatch.setattr(value.conditioner, "forward", forward)
    worker = MingCpuPrefetch(owner, batches, epoch=2, batch_offset=5, start=0, depth=1, identity_fn=batch_identity)
    try:
        assert two_encoded.wait(5)
        assert worker.queue.qsize() == 1 and len(calls) == 2
        with pytest.raises(RuntimeError, match="exclusively"):
            value.conditioner(**inputs())
    finally:
        worker.stop()
    assert not worker.thread.is_alive() and worker.queue.empty()
    value.conditioner(**inputs())


def test_worker_exception_reaches_consumer_and_hooks_are_removed(tmp_path, monkeypatch):
    value, owner, batches, _ = owner_setup(tmp_path)
    def fail(*a, **k): raise ValueError("native encoder failed")
    monkeypatch.setattr(value.conditioner, "forward", fail)
    worker = MingCpuPrefetch(owner, batches, epoch=2, batch_offset=5, start=0, depth=1, identity_fn=batch_identity)
    try:
        with pytest.raises(ValueError, match="native encoder failed"): worker.take(0)
    finally:
        worker.stop()
    assert worker.done.is_set() and not worker.thread.is_alive() and not value.conditioner._forward_pre_hooks


def test_trainable_cpu_prefetch_refuses_before_starting_worker(tmp_path):
    value, owner, batches, _ = owner_setup(tmp_path)
    value.conditioner.set_trainable({"projections"})
    with pytest.raises(ValueError, match="frozen"):
        MingCpuPrefetch(owner, batches, epoch=2, batch_offset=5, start=0, depth=1, identity_fn=batch_identity)


@pytest.mark.parametrize("change", ["weights", "precision", "mode", "trainability"])
def test_queued_result_refuses_producer_change(tmp_path, change):
    value, owner, batches, _ = owner_setup(tmp_path)
    worker = MingCpuPrefetch(owner, batches, epoch=2, batch_offset=5, start=0, depth=3, identity_fn=batch_identity)
    try:
        assert worker.done.wait(5)
        if change == "weights":
            with torch.no_grad(): next(value.conditioner.parameters()).add_(1)
        elif change == "precision": value.conditioner.to(dtype=torch.bfloat16)
        elif change == "mode": value.conditioner.train()
        else: value.conditioner.set_trainable({"projections"})
        with pytest.raises(RuntimeError, match="ownership changed"): worker.take(0)
    finally:
        worker.stop()


def test_changed_caption_restarts_native_worker_at_current_batch(tmp_path):
    _, owner, batches, _ = owner_setup(tmp_path)
    buffer = MingBatchBuffer(owner, batches, epoch=2, batch_offset=5,
                            latent_interval=2, text_interval=3, prefetch_depth=3)
    owner.ming_batch_buffer = buffer
    old = buffer.prefetcher
    try:
        assert old.done.wait(5)
        buffer.prepare(0, batches[0])
        batches[1][0][0]["caption"] = "word5 word6 word5"
        prepared = buffer.prepare(1, batches[1])
        assert buffer.prefetcher is not old and not old.thread.is_alive()
        assert prepared.for_step().conditioning.direct.shape[1] > 2
    finally:
        close_training_buffer(owner)


def test_stop_request_is_propagated_and_main_cleanup_joins(tmp_path):
    _, owner, batches, _ = owner_setup(tmp_path)
    buffer = MingBatchBuffer(owner, batches, epoch=2, batch_offset=5,
                            latent_interval=2, text_interval=3, prefetch_depth=1)
    owner.ming_batch_buffer = buffer
    worker = buffer.prefetcher
    def stop(): raise KeyboardInterrupt("stop")
    owner._check_stop_requested = stop
    try:
        with pytest.raises(KeyboardInterrupt): buffer.prepare(0, batches[0])
    finally:
        close_training_buffer(owner)
    assert not worker.thread.is_alive() and worker.queue.empty()
