"""Tile RoPE must equal its full-canvas slice at every U-Net depth."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from core.models.anima.anima_models import VideoRopePosition3DEmb
from core.models.anima.partition import AnimaPartitionAdapter, region_rotary
from core.training.partition import PartitionBox


@pytest.mark.parametrize("factor", [(1, 1), (2, 2), (4, 4), (2, 4)])
def test_global_tile_rotary_matches_full_canvas_slice(factor):
    ah, aw = factor
    embedder = VideoRopePosition3DEmb(head_dim=24, len_h=32, len_w=40, len_t=1)
    box = PartitionBox(4, 8, 20, 32)
    full = torch.zeros(1, 1, 24 // ah, 40 // aw, 24)
    tile = torch.zeros(1, 1, box.height // ah, box.width // aw, 24)
    expected = embedder(full).reshape(1, 24 // ah, 40 // aw, 1, 1, 24)
    expected = expected[:, box.top//ah:box.bottom//ah, box.left//aw:box.right//aw].reshape(-1, 1, 1, 24)
    torch.testing.assert_close(region_rotary(embedder, tile, box, factor=factor), expected, rtol=0, atol=0)


def test_origin_zero_preserves_native_rotary():
    embedder = VideoRopePosition3DEmb(head_dim=24, len_h=16, len_w=16, len_t=1)
    grid = torch.zeros(2, 1, 8, 12, 24)
    torch.testing.assert_close(region_rotary(embedder, grid, PartitionBox(0, 0, 8, 12)),
                               embedder(grid), rtol=0, atol=0)


def test_invalid_region_is_rejected():
    adapter = AnimaPartitionAdapter()
    with pytest.raises(ValueError, match="outside"):
        adapter.position_ids(8, 8, PartitionBox(0, 0, 9, 8))
    embedder = VideoRopePosition3DEmb(head_dim=24, len_h=16, len_w=16, len_t=1)
    with pytest.raises(ValueError, match="align"):
        region_rotary(embedder, torch.zeros(1, 1, 2, 2, 24), PartitionBox(1, 1, 5, 5), factor=(2, 2))
    with pytest.raises(ValueError, match="range"):
        embedder.generate_embeddings((1, 1, 8, 8, 24), spatial_origin=(12, 0))


def test_region_position_ids_retain_origin():
    actual = AnimaPartitionAdapter().position_ids(12, 16, PartitionBox(2, 4, 4, 7))
    assert actual.tolist() == [[2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6]]
