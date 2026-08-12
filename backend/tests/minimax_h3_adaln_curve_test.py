"""``sample_adaln_curve`` — the pruned checkpoint's tabulated timestep embedding.

The released ``*_pruned_*`` MiniMax-H3 checkpoints carry no timestep MLP: the
already-activated time-embedding curve is tabulated on a uniform grid over
``t in [0, 1]`` (``adaln_t_table``, ``[1025, 8]`` in the release) and read back
at continuous ``t``. Every AdaLN projection in all 50 blocks reads that one
vector, so a wrong grid row or a wrong blend weight is a silent, whole-model
bias — nothing raises and the shapes are unchanged.

The reference below is a deliberately naive scalar interpolation written from
the definition (per-element Python floats, ``math.floor``), so this pins the
shipped tensor expression to the SEMANTICS rather than to a paraphrase of
itself. Boundary cases are checked exactly: on a grid row the result must be
that row bit-for-bit, and ``t`` outside ``[0, 1]`` must saturate at the ends.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.minimax_h3.vendor.transformer_minimax_h3 import sample_adaln_curve  # noqa: E402


GRID = 17
DIM = 5


def _table() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(GRID, DIM, dtype=torch.float32)


def _scalar_reference(table: torch.Tensor, t: float) -> list:
    """Linear interpolation on a uniform [0, 1] grid, written from the definition."""
    rows = table.shape[0]
    t = min(max(float(t), 0.0), 1.0)
    coordinate = t * (rows - 1)
    lower = min(int(math.floor(coordinate)), rows - 2)
    frac = coordinate - lower
    return [
        float(table[lower][d]) * (1.0 - frac) + float(table[lower + 1][d]) * frac
        for d in range(table.shape[1])
    ]


@pytest.mark.parametrize("t", [
    0.0, 1.0,                                  # both ends
    -0.5, -1e-9, 1.0 + 1e-9, 3.7,              # out of range: must saturate
    0.5,                                       # exactly on an interior grid row (GRID = 17)
    1.0 / (GRID - 1), 2.0 / (GRID - 1),        # the first two interior rows
    0.01, 0.2499, 0.33333, 0.7071, 0.999999,   # between rows
])
def test_the_curve_lookup_matches_a_scalar_interpolation(t):
    table = _table()
    got = sample_adaln_curve(table, torch.tensor([t], dtype=torch.float32))
    assert got.shape == (1, DIM)
    expected = _scalar_reference(table, t)
    for d in range(DIM):
        # The reference runs in Python float64 and the shipped path in float32,
        # so equality is up to float32 rounding of an O(1) table entry.
        assert got[0, d].item() == pytest.approx(expected[d], rel=1e-6, abs=1e-6)


def test_every_grid_row_is_returned_exactly():
    """On a grid point the blend weight is 0 (or 1 at the very end), so the row
    must come back bit-for-bit — no drift into a neighbour."""
    table = _table()
    ts = torch.arange(GRID, dtype=torch.float32) / (GRID - 1)
    got = sample_adaln_curve(table, ts)
    assert torch.equal(got, table)


def test_out_of_range_t_saturates_at_the_curve_ends():
    table = _table()
    got = sample_adaln_curve(table, torch.tensor([-4.0, -1e-3, 1.0 + 1e-3, 250.0]))
    assert torch.equal(got[0], table[0])
    assert torch.equal(got[1], table[0])
    assert torch.equal(got[2], table[-1])
    assert torch.equal(got[3], table[-1])


def test_a_batch_of_timesteps_keeps_its_order():
    """One row out per timestep, in the caller's order: the packed sequence
    addresses this table by `timestep_indices`, so a permutation here would
    silently modulate the wrong rows."""
    table = _table()
    ts = torch.tensor([0.9, 0.1, 0.5, 0.1])
    got = sample_adaln_curve(table, ts)
    assert got.shape == (4, DIM)
    for i, t in enumerate(ts.tolist()):
        single = sample_adaln_curve(table, torch.tensor([t]))
        assert torch.equal(got[i], single[0])


def test_the_lookup_is_read_at_float32_from_a_lower_precision_timestep():
    """`timestep` reaches the model in the pipeline's dtype; the table is float32
    and the interpolation must not be dragged down to bfloat16."""
    table = _table()
    got = sample_adaln_curve(table, torch.tensor([0.3], dtype=torch.bfloat16))
    assert got.dtype == torch.float32
    expected = _scalar_reference(table, float(torch.tensor(0.3, dtype=torch.bfloat16)))
    for d in range(DIM):
        assert got[0, d].item() == pytest.approx(expected[d], rel=1e-6, abs=1e-6)
