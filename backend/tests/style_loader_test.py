"""Style batches: collate shapes, masked views, key-determined views, worker parity (design §7)."""

from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_ingest import _ItemPathsBuilder
from core.style_encoder.style_loader import (
    StyleImageDataset, StyleSampleKey, make_loader, style_collate,
)
from core.style_encoder.style_views import masked_patch_count, train_view_entropy
from style_training_fixture import MAX_PATCHES, PATCH, image_processor

PATCH_DIM = 3 * PATCH * PATCH


def _dataset(tmp_path, sizes, ratios, corrupt=()):
    builder = _ItemPathsBuilder()
    for i, size in enumerate(sizes):
        path = tmp_path / f"{i}.png"
        if i in corrupt:
            path.write_bytes(b"broken")
        else:
            rng = np.random.default_rng(i)
            Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
        builder.append(100 + i, 1000 + i, 1, str(path))
    return StyleImageDataset(builder.finish(), np.asarray(ratios, dtype=np.float64), image_processor(), MAX_PATCHES)


def _keys(rows, epoch=0, step=0):
    return [StyleSampleKey(r, train_view_entropy(7, epoch, step, p)) for p, r in enumerate(rows)]


def test_collate_shapes_and_masked_views(tmp_path):
    ratios = [0.3, 0.5, 0.75, 0.4]
    dataset = _dataset(tmp_path, [(8, 8), (16, 4), (6, 10), (8, 8)], ratios, corrupt={3})
    batch = style_collate([dataset[k] for k in _keys([0, 1, 2, 3])])

    assert batch.pixel_values.shape == (3, MAX_PATCHES, PATCH_DIM)
    assert batch.attention_mask.shape == batch.masked_attention_mask.shape == (3, MAX_PATCHES)
    assert batch.rows.tolist() == [0, 1, 2]
    for b in range(3):
        h, w = batch.spatial_shapes[b].tolist()
        valid = batch.attention_mask[b] != 0
        kept = batch.masked_attention_mask[b] != 0
        assert valid.sum() == h * w and valid[:h * w].all()
        assert not (kept & ~valid).any()
        assert int((valid & ~kept).sum()) == masked_patch_count(h * w, ratios[b])
        assert torch.all(batch.pixel_values[b, h * w:] == 0)


def test_masked_view_is_a_function_of_the_key(tmp_path):
    dataset = _dataset(tmp_path, [(8, 8)], [0.5])
    first, again = dataset[_keys([0])[0]], pickle.loads(pickle.dumps(dataset))[_keys([0])[0]]
    assert torch.equal(first.masked_attention_mask, again.masked_attention_mask)
    others = {tuple(dataset[StyleSampleKey(0, train_view_entropy(7, 0, s, 0))].masked_attention_mask.tolist())
              for s in range(8)}
    assert len(others) > 1


def test_spawned_worker_produces_the_same_batches(tmp_path):
    dataset = _dataset(tmp_path, [(8, 8), (16, 4), (6, 10), (8, 8)], [0.3, 0.5, 0.6, 0.4])
    batches = [_keys([0, 1], step=0), _keys([2, 3], step=1)]
    serial = list(make_loader(dataset, batches, 0))
    spawned = list(make_loader(dataset, batches, 1))
    for a, b in zip(serial, spawned, strict=True):
        for x, y in zip(a, b):
            assert torch.equal(x, y)
