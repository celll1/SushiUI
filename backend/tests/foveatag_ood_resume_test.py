"""Native reservoir sampling must resume its owned RNG, not just its samples."""

import numpy as np
import pytest

from core.tagger.ood_embedding_accumulator import OodEmbeddingAccumulator


def test_reservoir_resume_continues_identical_sampling(tmp_path):
    samples = np.random.default_rng(19).normal(size=(240, 8)).astype(np.float32)
    uninterrupted = OodEmbeddingAccumulator(max_samples=17, seed=23)
    partial = OodEmbeddingAccumulator(max_samples=17, seed=23)
    uninterrupted.update(samples)
    partial.update(samples[:83])
    path = str(tmp_path / "reservoir.npz")
    partial.save_reservoir(path)
    restored = OodEmbeddingAccumulator(max_samples=17, seed=999)
    assert restored.restore_from_reservoir(path, strict=True)
    restored.update(samples[83:])
    assert restored.n_seen == uninterrupted.n_seen
    np.testing.assert_array_equal(restored.reservoir, uninterrupted.reservoir)
    assert restored._rng.getstate() == uninterrupted._rng.getstate()


def test_native_refuses_missing_rng_while_legacy_remains_readable(tmp_path):
    path = str(tmp_path / "legacy.npz")
    np.savez(path, reservoir=np.ones((10, 8)), n_seen=20)
    accumulator = OodEmbeddingAccumulator()
    assert accumulator.restore_from_reservoir(path)
    with pytest.raises(ValueError, match="native OOD"):
        accumulator.restore_from_reservoir(path, strict=True)


def test_native_capacity_mismatch_does_not_mutate_live_reservoir(tmp_path):
    partial = OodEmbeddingAccumulator(max_samples=17)
    partial.update(np.ones((30, 8)))
    path = str(tmp_path / "reservoir.npz")
    partial.save_reservoir(path)
    restored = OodEmbeddingAccumulator(max_samples=19)
    with pytest.raises(ValueError, match="native OOD"):
        restored.restore_from_reservoir(path, strict=True)
    assert restored.n_seen == 0 and restored.reservoir == []
