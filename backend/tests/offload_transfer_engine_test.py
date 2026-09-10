import sys
from pathlib import Path

import torch


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.memory_management.offload_transfer_engine import (  # noqa: E402
    FrozenSequentialTransferEngine,
)


def _engine(keys=(10, 11, 12, 13, 14), ring_size=2):
    masters = {
        key: {
            torch.float32: torch.full((key - 7,), float(key)),
            torch.uint8: torch.full((2,), key, dtype=torch.uint8),
        }
        for key in keys
    }
    pointed = {}

    def point(key, bundle):
        pointed[key] = bundle

    engine = FrozenSequentialTransferEngine(
        keys=keys,
        masters=masters,
        ring_size=ring_size,
        device=torch.device("cpu"),
        point_bundle=point,
    )
    return engine, masters, pointed


def test_prime_acquire_and_release_use_bounded_multi_plane_slots():
    engine, masters, pointed = _engine()
    engine.prime()

    assert engine.loaded_key == [10, 11]
    engine.acquire(10)
    assert pointed[10][torch.float32].data_ptr() == engine.slots[0][torch.float32].data_ptr()
    assert torch.equal(pointed[10][torch.float32], masters[10][torch.float32])
    assert torch.equal(pointed[10][torch.uint8], masters[10][torch.uint8])

    engine.release(10)
    assert pointed[10][torch.float32].data_ptr() == masters[10][torch.float32].data_ptr()
    assert engine.loaded_key == [12, 11]
    assert len(engine.slots) == 2


def test_release_wraps_each_slot_to_the_first_key_assigned_to_it():
    engine, _, _ = _engine()
    engine.prime()
    for key in engine.keys:
        engine.acquire(key)
        engine.release(key)

    assert engine.loaded_key == [10, 11]
    before = engine.stats()
    engine.acquire(10)
    engine.acquire(11)
    assert engine.stats().acquire_misses == before.acquire_misses


def test_variable_item_sizes_copy_only_the_registered_prefix():
    engine, masters, pointed = _engine(keys=(10, 14), ring_size=1)
    engine.prime()
    engine.acquire(10)
    assert pointed[10][torch.float32].numel() == masters[10][torch.float32].numel()
    engine.release(10)
    engine.acquire(14)
    assert pointed[14][torch.float32].numel() == masters[14][torch.float32].numel()
    assert torch.equal(pointed[14][torch.float32], masters[14][torch.float32])


def test_stats_count_logical_plane_copies_and_bytes():
    engine, masters, _ = _engine(keys=(10, 11), ring_size=2)
    engine.prime()
    expected = sum(
        tensor.numel() * tensor.element_size()
        for bundle in masters.values()
        for tensor in bundle.values()
    )
    stats = engine.stats()
    assert stats.h2d_bytes == expected
    assert stats.h2d_submissions == 4
    assert stats.acquire_misses == 0


def test_close_restores_all_cpu_masters_and_releases_slots():
    engine, masters, pointed = _engine()
    engine.prime()
    engine.acquire(10)
    engine.close()

    assert engine.slots == []
    assert engine.loaded_key == [None, None]
    for key in engine.keys:
        assert pointed[key][torch.float32].data_ptr() == masters[key][torch.float32].data_ptr()
