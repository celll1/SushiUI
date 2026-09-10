import sys
from pathlib import Path

import torch
import torch.nn as nn


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.memory_management.offload_transfer_engine import (  # noqa: E402
    FrozenSequentialTransferEngine,
)
from core.memory_management.block_offloading import TransformerBlockOffloader  # noqa: E402


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


class SidecarLinear(nn.Linear):
    def __init__(self, value):
        super().__init__(2, 2, bias=False)
        self.weight.data.fill_(value)
        self.register_buffer("weight_scale", torch.full((2,), value + 10, dtype=torch.float32))


def test_transformer_h2d_path_packs_weight_and_sidecar_planes():
    blocks = nn.ModuleList([SidecarLinear(1), SidecarLinear(2), SidecarLinear(3)])
    offloader = TransformerBlockOffloader(
        blocks=blocks,
        blocks_to_swap=3,
        device=torch.device("cpu"),
        h2d_only=True,
        ring_size=2,
    )
    offloader.prepare_block_devices_before_forward()
    offloader.wait_for_block(0)

    assert offloader.h2d_only
    assert torch.equal(blocks[0].weight, torch.ones_like(blocks[0].weight))
    assert torch.equal(blocks[0].weight_scale, torch.full_like(blocks[0].weight_scale, 11))
    weight_ptr = blocks[0].weight.data_ptr()
    scale_ptr = blocks[0].weight_scale.data_ptr()
    assert weight_ptr != offloader.h2d_masters[0][0][torch.float32].data_ptr()
    assert scale_ptr != offloader.h2d_masters[0][0][torch.float32].data_ptr()

    offloader.submit_move_blocks_forward(0)
    assert blocks[0].weight.data_ptr() == offloader.h2d_masters[0][0][torch.float32].data_ptr()
    offloader.cleanup()
