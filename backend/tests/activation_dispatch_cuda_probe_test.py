"""CPU tests for pure helpers in the CUDA activation-dispatch probe."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.probes.activation_dispatch_cuda import _as_rows, _compare, _p95


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 7, 5), (14, 5)),
        ((2, 5, 3, 7), (42, 5)),
        ((2, 5, 3, 7, 11), (462, 5)),
    ],
)
def test_as_rows_keeps_channels_and_flattens_work(shape, expected):
    assert _as_rows(torch.zeros(shape)).shape == expected


def test_as_rows_refuses_unknown_rank():
    with pytest.raises(ValueError, match="Unsupported probe rank"):
        _as_rows(torch.zeros(2, 3))


def test_comparison_reports_exact_and_changed_gradients():
    off = {
        "loss": 1.0,
        "input_grad": torch.tensor([1.0]),
        "parameter_grads": {"weight": torch.tensor([2.0])},
    }
    exact = _compare(off, {
        "loss": 1.0,
        "input_grad": torch.tensor([1.0]),
        "parameter_grads": {"weight": torch.tensor([2.0])},
    })
    changed = _compare(off, {
        "loss": 1.0,
        "input_grad": torch.tensor([1.5]),
        "parameter_grads": {"weight": torch.tensor([2.0])},
    })
    assert exact["gradients_exact"] is True
    assert changed["gradient_exact_mismatches"] == ["input"]
    assert changed["gradients_within_bf16_tolerance"] is False
    assert changed["gradient_tolerance_mismatches"] == ["input"]
    assert changed["gradient_max_abs_error"] == 0.5


def test_p95_uses_nearest_rank():
    assert _p95(range(1, 21)) == 19
