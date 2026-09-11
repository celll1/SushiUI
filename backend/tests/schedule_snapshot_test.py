from __future__ import annotations

import os
import sys

import torch

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.inference.schedule_utils import snapshot_schedule_scalars


def test_tensor_snapshot_matches_per_element_item_values():
    schedule = torch.tensor([1.0, 0.33333334, 0.0], dtype=torch.float32)

    assert snapshot_schedule_scalars(schedule) == [
        float(value.item()) for value in schedule
    ]


def test_sequence_snapshot_preserves_scalar_values():
    schedule = [torch.tensor(1.0), 0.5, torch.tensor(0.0)]

    assert snapshot_schedule_scalars(schedule) == [1.0, 0.5, 0.0]
