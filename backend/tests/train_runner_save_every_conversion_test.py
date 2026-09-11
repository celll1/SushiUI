"""Save-cadence conversion behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.train_runner import _resolve_save_every_n_steps

@pytest.mark.parametrize(("unit", "cadence", "items", "batch_size", "expected"), [
    ("epochs", 3, 100, 10, 30),
    ("epochs", 2, 105, 10, 22),
    ("steps", 500, 0, 10, 500),
    ("epochs", 0, 105, 10, 0),
])
def test_resolve_save_every_n_steps(unit, cadence, items, batch_size, expected):
    assert _resolve_save_every_n_steps(unit, cadence, items, batch_size) == expected
