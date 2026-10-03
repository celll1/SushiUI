"""SupCon positive weights (design §8.1, test T5)."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_dataset import SplitParams, build_style_index
from core.style_encoder.style_labels import StyleLabels
from core.style_encoder.style_loss import positive_weights


def test_positive_weights_hand_computed():
    artists = [1, 1, 2, 1, 2, 1]
    content = [[1, 2], [2, 3], [], [], [], [2, 1]]
    w = positive_weights(artists, content, view_positive_weight=1.5, positive_weight_min=0.3)
    n = len(artists)
    image = torch.tensor([
        [1.5, 2 / 3, 0.0, 1.0, 0.0, 0.3],
        [2 / 3, 1.5, 0.0, 1.0, 0.0, 2 / 3],
        [0.0, 0.0, 1.5, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.5, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.5, 0.0],
        [0.3, 2 / 3, 0.0, 1.0, 0.0, 1.5],
    ])
    expected = image.repeat(2, 2)
    expected.fill_diagonal_(0.0)
    assert w.shape == (2 * n, 2 * n)
    torch.testing.assert_close(w, expected)


def test_copyright_and_character_with_the_same_name_are_distinct_content():
    records = [
        (1, 101, StyleLabels(frozenset({"a"}), frozenset(), frozenset({"x"}), False)),
        (2, 102, StyleLabels(frozenset({"a"}), frozenset({"x"}), frozenset(), False)),
    ]
    split = SplitParams.create(0.0, 0.0, 1, 0)
    index = build_style_index(records, split, pinned_keys=np.zeros(0, dtype=np.int64))
    w = positive_weights(index.item_artist[[0, 1]].tolist(), [index.content_ids(0), index.content_ids(1)],
                         view_positive_weight=1.0, positive_weight_min=0.3)
    assert w[0, 1].item() == 1.0
