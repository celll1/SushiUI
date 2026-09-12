import numpy as np
import torch

from backend.core.tagger.tag_metrics_accumulator import TagMetricsAccumulator


def _legacy_update(accumulator, preds, labels):
    probabilities = preds.float()
    positives = labels > 0.5
    batch_size, vocab_size = probabilities.shape
    bin_idx = (probabilities * accumulator.n_bins).long().clamp(
        0, accumulator.n_bins - 1
    )

    accumulator.tag_count += positives.sum(dim=0).numpy().astype(np.int32)
    accumulator.total_images_all += batch_size
    accumulator.total_images_cur += batch_size

    flat_idx = (
        torch.arange(vocab_size, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        * accumulator.n_bins
        + bin_idx
    ).reshape(-1)
    total_flat = torch.zeros(vocab_size * accumulator.n_bins, dtype=torch.int32)
    total_flat.scatter_add_(
        0, flat_idx, torch.ones(batch_size * vocab_size, dtype=torch.int32)
    )
    accumulator.total_hist_cur += total_flat.reshape(
        vocab_size, accumulator.n_bins
    ).numpy()

    pos_b, pos_v = torch.where(positives)
    if len(pos_b) > 0:
        pos_flat_idx = pos_v * accumulator.n_bins + bin_idx[pos_b, pos_v]
        pos_flat = torch.zeros(vocab_size * accumulator.n_bins, dtype=torch.int32)
        pos_flat.scatter_add_(
            0, pos_flat_idx.long(), torch.ones(len(pos_b), dtype=torch.int32)
        )
        accumulator.pos_hist_cur += pos_flat.reshape(
            vocab_size, accumulator.n_bins
        ).numpy()


def _assert_same_state(actual, expected):
    for name in (
        "tag_count",
        "pos_hist_cur",
        "total_hist_cur",
        "pos_hist_prev",
        "total_hist_prev",
        "pos_hist_pp",
        "total_hist_pp",
        "tag_count_epoch_start",
        "last_epoch_delta",
    ):
        assert np.array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.total_images_all == expected.total_images_all
    assert actual.total_images_cur == expected.total_images_cur
    for name, values in actual.compute_metrics().items():
        assert np.array_equal(values, expected.compute_metrics()[name], equal_nan=True)


def test_update_matches_previous_histograms_and_metrics():
    actual = TagMetricsAccumulator(vocab_size=5, n_bins=10)
    expected = TagMetricsAccumulator(vocab_size=5, n_bins=10)
    batches = [
        (
            torch.tensor(
                [[0.0, 0.09, 0.1, 0.999, 1.0], [0.51, 0.49, 0.25, 0.75, 0.5]],
                dtype=torch.float16,
            ),
            torch.tensor(
                [[False, True, False, True, True], [True, False, True, False, True]]
            ),
        ),
        (
            torch.tensor([[0.33, 0.66, 0.42, 0.88, 0.12]], dtype=torch.float16),
            torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]]),
        ),
    ]

    for predictions, labels in batches:
        actual.update(predictions, labels)
        _legacy_update(expected, predictions, labels)

    _assert_same_state(actual, expected)


def test_growth_rebuilds_offsets_without_changing_existing_state():
    actual = TagMetricsAccumulator(vocab_size=2, n_bins=4)
    expected = TagMetricsAccumulator(vocab_size=2, n_bins=4)
    first_predictions = torch.tensor([[0.1, 0.9]], dtype=torch.float16)
    first_labels = torch.tensor([[False, True]])
    actual.update(first_predictions, first_labels)
    _legacy_update(expected, first_predictions, first_labels)

    actual.grow(4)
    expected.grow(4)
    second_predictions = torch.tensor([[0.2, 0.4, 0.6, 0.8]], dtype=torch.float16)
    second_labels = torch.tensor([[True, False, True, False]])
    actual.update(second_predictions, second_labels)
    _legacy_update(expected, second_predictions, second_labels)

    _assert_same_state(actual, expected)
    assert actual._tag_bin_offsets.tolist() == [[0, 4, 8, 12]]
    assert actual._scatter_ones.numel() == 4
