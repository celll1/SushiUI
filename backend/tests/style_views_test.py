"""Masked-view block masks and per-artist mask ratio (design §7)."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_views import apply_block_masks, artist_mask_ratios, block_mask, masked_patch_count


def test_masked_count_reaches_ratio_and_keeps_one_patch():
    assert masked_patch_count(10, 0.3) == 3
    assert masked_patch_count(10, 0.31) == 4
    assert masked_patch_count(10, 1.0) == 9
    assert masked_patch_count(1, 0.75) == 0
    rng = np.random.default_rng(0)
    for h, w in [(1, 1), (1, 7), (5, 3), (16, 16), (9, 31)]:
        for r in (0.0, 0.3, 0.5, 0.75, 1.0):
            mask = block_mask(h, w, r, rng)
            assert mask.shape == (h * w,) and mask.dtype == bool
            assert int(mask.sum()) == masked_patch_count(h * w, r)
            assert mask.sum() < h * w


def test_block_mask_is_deterministic_per_generator():
    a = block_mask(16, 16, 0.5, np.random.default_rng(11))
    b = block_mask(16, 16, 0.5, np.random.default_rng(11))
    c = block_mask(16, 16, 0.5, np.random.default_rng(12))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_apply_block_masks_touches_only_valid_patches():
    mask = torch.zeros(3, 30, dtype=torch.int32)
    shapes = torch.tensor([[4, 5], [5, 6], [1, 1]])
    for b, (h, w) in enumerate(shapes.tolist()):
        mask[b, :h * w] = 1
    out = apply_block_masks(mask, shapes, np.array([0.5, 0.75, 0.9]), np.random.default_rng(5))
    assert out.dtype == mask.dtype and out.shape == mask.shape
    assert torch.all(out <= mask)
    assert out.sum(dim=1).tolist() == [20 - 10, 30 - 23, 1]
    again = apply_block_masks(mask, shapes, np.array([0.5, 0.75, 0.9]), np.random.default_rng(5))
    assert torch.equal(out, again)

    broken = mask.clone()
    broken[0, 25] = 1
    with pytest.raises(ValueError):
        apply_block_masks(broken, shapes, np.array([0.5, 0.5, 0.5]), np.random.default_rng(0))


def test_artist_mask_ratio_interpolates_confound():
    np.testing.assert_allclose(artist_mask_ratios(np.array([0.0, 0.5, 1.0]), 0.3, 0.75), [0.3, 0.525, 0.75])
    with pytest.raises(ValueError):
        artist_mask_ratios(np.array([0.5]), 0.8, 0.3)
    with pytest.raises(ValueError):
        artist_mask_ratios(np.array([1.5]), 0.3, 0.75)
