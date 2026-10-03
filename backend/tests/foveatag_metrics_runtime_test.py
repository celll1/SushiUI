"""Exact histogram continuity and unknown-label masks for native training."""

import numpy as np
import pytest
import torch

from core.tagger.tag_metrics_accumulator import TagMetricsAccumulator
from core.tagger.foveatag.metrics_runtime import (
    HISTOGRAMS, VECTORS, COUNTERS, save_metrics_runtime, restore_metrics_runtime,
    save_f1_runtime, restore_f1_runtime, masked_metric_inputs,
)


def populated():
    accumulator = TagMetricsAccumulator(4, n_bins=10)
    rng = np.random.default_rng(13)
    for count in (9, 7, 3):
        accumulator.update(torch.from_numpy(rng.random((count, 4))),
                           torch.from_numpy(rng.random((count, 4)) > .6))
        if count != 3:
            accumulator.rotate_epoch()
    return accumulator


def equal(left, right):
    for key in HISTOGRAMS + VECTORS:
        np.testing.assert_array_equal(getattr(left, key), getattr(right, key))
    for key in COUNTERS:
        assert getattr(left, key) == getattr(right, key)
    for boundary in (False, True):
        for key, value in left.compute_metrics(boundary).items():
            np.testing.assert_array_equal(value, right.compute_metrics(boundary)[key])


@pytest.mark.parametrize("boundary", [False, True])
def test_resume_restores_slots_exposure_and_next_updates_exactly(tmp_path, boundary):
    uninterrupted = populated()
    path = tmp_path / "state.npz"
    before = {key: getattr(uninterrupted, key).copy() for key in HISTOGRAMS + VECTORS}
    save_metrics_runtime(uninterrupted, path, step=18, epoch_boundary=boundary)
    for key, value in before.items():
        np.testing.assert_array_equal(value, getattr(uninterrupted, key))
    if boundary:
        uninterrupted.rotate_epoch()
    resumed = TagMetricsAccumulator(4, n_bins=10)
    restore_metrics_runtime(resumed, path, step=18)
    equal(uninterrupted, resumed)
    preds, labels = torch.rand(5, 4), torch.rand(5, 4) > .5
    for accumulator in (uninterrupted, resumed):
        accumulator.update(preds, labels)
        accumulator.rotate_epoch()
    equal(uninterrupted, resumed)


def test_masked_entries_are_not_false_negatives_or_observed_negatives():
    accumulator = TagMetricsAccumulator(2, n_bins=10)
    accumulator.update(torch.tensor([[.9, .8], [.3, .7]]),
        torch.tensor([[0., 1.], [1., 0.]]), observation_mask=torch.tensor([[False, False], [True, False]]))
    assert accumulator.total_hist_cur.sum(axis=1).tolist() == [1, 0]
    assert accumulator.pos_hist_cur.sum(axis=1).tolist() == [1, 0]
    assert accumulator.tag_count.tolist() == [1, 0]


def test_vocab_reorder_rename_growth_and_merge_are_explicit(tmp_path):
    original = populated()
    path = tmp_path / "state.npz"
    save_metrics_runtime(original, path, step=2, epoch_boundary=False)
    restored = TagMetricsAccumulator(5, n_bins=10)
    result = restore_metrics_runtime(restored, path, step=2,
        saved_vocab={"a": 0, "b": 1, "c": 2, "d": 3},
        current_vocab={"b": 0, "renamed": 1, "new": 2, "merged": 3, "d": 4},
        lineage={"renamed": ["a"], "merged": ["a", "c"]})
    assert result == {"reset_rows": 2}
    for key in HISTOGRAMS + VECTORS:
        before, after = getattr(original, key), getattr(restored, key)
        np.testing.assert_array_equal(after[0], before[1])
        np.testing.assert_array_equal(after[1], before[0])
        np.testing.assert_array_equal(after[4], before[3])
        assert not after[2:4].any()


def test_wrong_step_or_row_mapping_does_not_mutate_live_state(tmp_path):
    path = tmp_path / "state.npz"
    save_metrics_runtime(populated(), path, step=2, epoch_boundary=False)
    accumulator = TagMetricsAccumulator(5, n_bins=10)
    for step in (1, 2):
        with pytest.raises(ValueError):
            restore_metrics_runtime(accumulator, path, step=step)
        assert accumulator.total_images_all == 0 and not accumulator.pos_hist_cur.any()


def test_rolling_f1_variable_batches_and_threshold_roundtrip(tmp_path):
    from collections import deque
    buffer = deque(maxlen=3)
    for count in (2, 1, 3):
        buffer.append((torch.rand(count, 4).half(), torch.rand(count, 4) > .5))
    path = tmp_path / "f1.npz"
    save_f1_runtime(buffer, path, step=2, threshold=.41, vocab_size=4)
    restored = deque(maxlen=3)
    assert restore_f1_runtime(restored, path, step=2, vocab_size=4) == .41
    for first, second in zip(buffer, restored):
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_unknowns_do_not_change_native_f1_at_probability_thresholds():
    from core.tagger.tagger_trainer import _compute_all_metrics
    predictions, labels = masked_metric_inputs(torch.tensor([[.9, .99], [.99, .9]]),
        torch.tensor([[1., 0.], [0., 1.]]), torch.eye(2))
    for threshold in (0., .35, .8):
        metrics = _compute_all_metrics(predictions, labels, threshold)
        assert all(value == 1. for value in metrics.values())


def test_native_validation_ignores_unknown_and_new_vocabulary_columns():
    from core.tagger.tagger_trainer import TaggerTrainer, _compute_all_metrics
    class Model(torch.nn.Module):
        architecture = "foveatag"
        def forward(self, values, *args):
            return values
    logits = torch.full((2, 3), 4.)
    batch = (logits, torch.zeros(2, 0), torch.zeros(2, 0),
             torch.eye(2), torch.eye(2))
    trainer = TaggerTrainer.__new__(TaggerTrainer)
    predictions, labels = trainer._collect_val_preds(Model(), [batch], torch.device("cpu"), None)
    assert predictions[:, 2].tolist() == [-1., -1.]
    assert all(value == 1. for value in _compute_all_metrics(predictions, labels).values())


def test_rolling_f1_mapping_masks_new_and_merged_rows(tmp_path):
    from collections import deque
    buffer = deque([(torch.tensor([[.8, .2]], dtype=torch.float16), torch.tensor([[True, False]]))], maxlen=2)
    path = tmp_path / "f1.npz"
    save_f1_runtime(buffer, path, step=2, threshold=.5, vocab_size=2)
    restored = deque(maxlen=2)
    restore_f1_runtime(restored, path, step=2, vocab_size=4,
        saved_vocab={"a": 0, "b": 1}, current_vocab={"b": 0, "renamed": 1, "new": 2, "merged": 3},
        lineage={"renamed": ["a"], "merged": ["a", "b"]})
    torch.testing.assert_close(restored[0][0][:, :2], buffer[0][0][:, [1, 0]])
    assert restored[0][0][:, 2:].tolist() == [[-1., -1.]]
    assert not restored[0][1][:, 2:].any()


def test_bad_f1_runtime_keeps_existing_buffer_intact(tmp_path):
    from collections import deque
    buffer = deque([(torch.ones(2, 4).half(), torch.ones(2, 4, dtype=torch.bool))], maxlen=2)
    path = tmp_path / "f1.npz"
    save_f1_runtime(buffer, path, step=2, threshold=.5, vocab_size=4)
    for capacity, step in ((1, 2), (2, 3)):
        restored = deque(buffer, maxlen=capacity)
        with pytest.raises(ValueError):
            restore_f1_runtime(restored, path, step=step, vocab_size=4)
        assert len(restored) == 1
        torch.testing.assert_close(restored[0][0], buffer[0][0])
