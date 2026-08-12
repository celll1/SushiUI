"""MiniMax-H3: 362 is an ADVISORY trained-range top, not an enforced ceiling.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_temporal_ceiling_test.py -v

WHY THIS FILE EXISTS
--------------------
`MINIMAX_H3_TEMPORAL.max_frames` used to be 362 (corrected from 345: the
trained range is stated as "~124-362, longer is untested"; 362 is the grid
point AT that stated top, 345 undersold it by one grid step) and was an
ENFORCED production ceiling, lifted only by an opt-in env gate
(`SUSHI_TEMPORAL_UNCAPPED`).

362 is a DOCUMENTED trained-range endpoint, not a model limit: RoPE is
computed on the fly (no learned position table, no mask, no baked sequence
literal), so nothing structural stops a longer clip -- only the 17n+5 grid is
structural. `max_frames` is therefore `None` and `trained_max_frames` (362)
is ADVISORY only: a length past it is accepted and warned as untested,
UNCONDITIONALLY (no env gate to opt into the warning -- "stated, not silent"
no longer needs an opt-in on the caller's side). The 17n+5 grid itself is
untouched by this: an off-grid length is still snapped up (or refused, on an
arch that doesn't snap) exactly as before.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.components.wiring import MINIMAX_H3_TEMPORAL as spec  # noqa: E402
from api.generation_utils import validate_video_geometry  # noqa: E402


# 379 = 17*22 + 5, the next grid point above 362 (= 17*21 + 5).
NEXT_GRID_POINT_ABOVE_362 = 379


def test_362_is_the_documented_trained_top_and_on_the_grid():
    assert spec.trained_max_frames == 362
    assert spec.is_valid_length(362)
    assert (362 - spec.frame_offset) % spec.frame_multiple == 0


def test_max_frames_is_not_enforced():
    """The load-bearing fact this whole file exists to pin: `max_frames` is
    None, so `ceiling()` imposes no production top at all."""
    assert spec.max_frames is None
    assert spec.ceiling() is None


def test_379_is_the_next_grid_point_above_362():
    assert spec.is_valid_length(NEXT_GRID_POINT_ABOVE_362)
    assert NEXT_GRID_POINT_ABOVE_362 > spec.trained_max_frames
    assert (NEXT_GRID_POINT_ABOVE_362 - 17) == spec.trained_max_frames


def test_snap_length_no_longer_clamps_past_362():
    """`snap_length` only rounds UP onto the grid now; an on-grid value past
    362 passes through unchanged, and an off-grid one rounds up onto the
    grid, past 362 if that is where the next grid point lands."""
    assert spec.snap_length(NEXT_GRID_POINT_ABOVE_362) == NEXT_GRID_POINT_ABOVE_362
    assert spec.snap_length(400) == 413  # next 17n+5 point at/above 400


def test_suggested_lengths_still_stops_at_362():
    """The served clip-length menu still means something even though
    `max_frames` no longer bounds validity: it stops at the ADVISORY
    `trained_max_frames`, not open-ended."""
    lengths = spec.suggested_lengths(16)
    assert lengths[-1] == 362
    assert 379 not in lengths
    assert len(lengths) == 15  # 124..362 on the 17n+5 grid


def test_validator_accepts_and_warns_untested_for_an_on_grid_value_past_362():
    """No env gate anymore -- the warning fires unconditionally."""
    params = {
        "width": 512, "height": 512,
        "num_frames": NEXT_GRID_POINT_ABOVE_362,
        "frame_rate": spec.fps_fixed,
    }
    warnings = validate_video_geometry(params, "minimax_h3")
    # Not clamped back to 362 -- accepted as requested.
    assert params["num_frames"] == NEXT_GRID_POINT_ABOVE_362
    assert any("362" in w and "untested" in w for w in warnings)


def test_validator_snaps_an_off_grid_value_past_362_and_still_warns_untested():
    """400 is off-grid (nearest points are 396=17*23+5 and 379); the snapped
    result must not be clamped back to 362, and the untested-range warning
    must still fire on the snapped value."""
    params = {
        "width": 512, "height": 512,
        "num_frames": 400,
        "frame_rate": spec.fps_fixed,
    }
    warnings = validate_video_geometry(params, "minimax_h3")
    assert params["num_frames"] == 413
    assert any("not a length this model can generate" in w for w in warnings)
    assert any("362" in w and "untested" in w for w in warnings)


def test_validator_is_a_no_op_below_the_documented_ceiling():
    params = {
        "width": 512, "height": 512,
        "num_frames": 141,
        "frame_rate": spec.fps_fixed,
    }
    warnings = validate_video_geometry(params, "minimax_h3")
    assert params["num_frames"] == 141
    assert not any("untested" in w for w in warnings)


def test_validator_does_not_warn_untested_exactly_at_362():
    params = {
        "width": 512, "height": 512,
        "num_frames": 362,
        "frame_rate": spec.fps_fixed,
    }
    warnings = validate_video_geometry(params, "minimax_h3")
    assert params["num_frames"] == 362
    assert not any("untested" in w for w in warnings)
