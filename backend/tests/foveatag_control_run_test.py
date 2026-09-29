"""CPU tests for deterministic Phase-1 run construction."""

import torch

from core.tagger.foveatag.control_run import (
    domain_sample_weights,
    select_domain_splits,
)


class _Source:
    def __init__(self):
        self.rows = [
            (29, "ground-a.png"),
            (29, "ground-b.png"),
            (29, "ground-c.png"),
            (16, "photo-a.png"),
            (16, "photo-b.png"),
            (16, "photo-c.png"),
        ]

    def __len__(self):
        return len(self.rows)

    def dataset_id_for_index(self, index):
        return self.rows[index][0]

    def source_path_for_index(self, index):
        return self.rows[index][1]


def test_domain_splits_are_disjoint_stable_and_exclude_holdout():
    source = _Source()
    first = select_domain_splits(
        source,
        train_counts={29: 1, 16: 1},
        validation_counts={29: 1, 16: 1},
        excluded_paths=("ground-a.png",),
        seed=7,
    )
    second = select_domain_splits(
        source,
        train_counts={29: 1, 16: 1},
        validation_counts={29: 1, 16: 1},
        excluded_paths=("ground-a.png",),
        seed=7,
    )
    train, validation, available = first
    assert first == second
    assert set(train).isdisjoint(validation)
    assert 0 not in train and 0 not in validation
    assert available == {16: 3, 29: 2}


def test_domain_weights_assign_requested_probability_mass():
    source = _Source()
    indices = [0, 1, 2, 3]
    weights = domain_sample_weights(source, indices, {29: 0.75, 16: 0.25})

    assert weights.dtype == torch.double
    torch.testing.assert_close(weights[:3].sum(), torch.tensor(0.75, dtype=torch.double))
    torch.testing.assert_close(weights[3:].sum(), torch.tensor(0.25, dtype=torch.double))
