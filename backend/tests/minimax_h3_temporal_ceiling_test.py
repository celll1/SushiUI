"""MiniMax-H3: the 124-362 production ceiling and its env-gated override.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_temporal_ceiling_test.py -v

WHY THIS FILE EXISTS
--------------------
`MINIMAX_H3_TEMPORAL.max_frames` was corrected from 345 to 362 (ComfyUI's node
states the trained range as "~124-362, longer is untested"; 362 is the grid
point AT that stated top, 345 undersold it by one grid step). A ceiling-side
override (`SUSHI_TEMPORAL_UNCAPPED`, the mirror of the existing floor-side
`SUSHI_TEMPORAL_SMOKE`) lets a deliberate caller probe past 362.

The override is only real if `TemporalSpec.snap_length`'s clamp bound
actually widens when it is set -- `validate_video_geometry` computes `hi` from
`spec.ceiling(uncapped)`, and a caller that forgets to thread `uncapped`
through would silently reintroduce the 362 clamp while everything else looks
wired. `test_override_actually_widens_the_snap_ceiling` is written to FAIL if
that threading regresses (it does not just check the validator's warnings; it
directly asserts `snap_length(..., uncapped=True)` stops clamping).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.components.wiring import MINIMAX_H3_TEMPORAL as spec  # noqa: E402
from api.generation_utils import validate_video_geometry  # noqa: E402


# 379 = 17*22 + 5, the next grid point above 362 (= 17*21 + 5).
NEXT_GRID_POINT_ABOVE_362 = 379


def test_362_is_the_documented_ceiling_and_on_the_grid():
    assert spec.max_frames == 362
    assert spec.is_valid_length(362)
    assert (362 - spec.frame_offset) % spec.frame_multiple == 0


def test_379_is_the_next_grid_point_above_362():
    assert spec.is_valid_length(NEXT_GRID_POINT_ABOVE_362)
    assert NEXT_GRID_POINT_ABOVE_362 > spec.max_frames
    assert (NEXT_GRID_POINT_ABOVE_362 - 17) == spec.max_frames


def test_snap_length_clamps_to_362_when_uncapped_is_false():
    assert spec.snap_length(NEXT_GRID_POINT_ABOVE_362) == 362
    assert spec.snap_length(NEXT_GRID_POINT_ABOVE_362, uncapped=False) == 362


def test_override_actually_widens_the_snap_ceiling():
    """The load-bearing assertion: `uncapped=True` must stop the clamp,
    not just be accepted as a no-op parameter."""
    assert spec.ceiling(uncapped=True) is None
    assert spec.snap_length(NEXT_GRID_POINT_ABOVE_362, uncapped=True) == NEXT_GRID_POINT_ABOVE_362
    # And the floor-side behaviour is untouched by the ceiling override.
    assert spec.snap_length(30, smoke=False, uncapped=True) == 124


def test_validator_clamps_to_362_with_a_warning_when_override_is_off(monkeypatch):
    monkeypatch.delenv(spec.max_override_env, raising=False)
    params = {
        "width": 512, "height": 512,
        "num_frames": NEXT_GRID_POINT_ABOVE_362,
        "frame_rate": spec.fps_fixed,
    }
    warnings = validate_video_geometry(params, "minimax_h3")
    assert params["num_frames"] == 362
    assert any("362" in w for w in warnings)
    assert not any(spec.max_override_env in w for w in warnings)


def test_validator_raises_the_effective_bound_when_override_is_on(monkeypatch):
    monkeypatch.setenv(spec.max_override_env, "1")
    params = {
        "width": 512, "height": 512,
        "num_frames": NEXT_GRID_POINT_ABOVE_362,
        "frame_rate": spec.fps_fixed,
    }
    warnings = validate_video_geometry(params, "minimax_h3")
    # The load-bearing check: with the override on, `num_frames` must NOT be
    # clamped back to 362 -- it only rounds onto the grid, and 379 already is.
    assert params["num_frames"] == NEXT_GRID_POINT_ABOVE_362
    assert any(spec.max_override_env in w and "untested" in w for w in warnings)


def test_validator_snap_path_also_raises_the_bound_when_override_is_on(monkeypatch):
    """Same claim as the test above, but through the SNAP branch, not the
    already-on-grid branch: 400 is off-grid, so `validate_video_geometry`
    must call `spec.snap_length(..., uncapped=True)` to answer 413 (the next
    grid point) rather than silently clamping to 362. A caller that forgot to
    thread `uncapped` into that specific `snap_length` call -- the exact
    regression this file exists to catch -- would answer 362 here while every
    other test in this file still passes, because they exercise the
    already-valid branch instead."""
    monkeypatch.setenv(spec.max_override_env, "1")
    params = {
        "width": 512, "height": 512,
        "num_frames": 400,
        "frame_rate": spec.fps_fixed,
    }
    warnings = validate_video_geometry(params, "minimax_h3")
    assert params["num_frames"] == 413
    assert any(spec.max_override_env in w for w in warnings)


def test_validator_snap_path_clamps_to_362_when_override_is_off(monkeypatch):
    monkeypatch.delenv(spec.max_override_env, raising=False)
    params = {
        "width": 512, "height": 512,
        "num_frames": 400,
        "frame_rate": spec.fps_fixed,
    }
    warnings = validate_video_geometry(params, "minimax_h3")
    assert params["num_frames"] == 362
    assert not any(spec.max_override_env in w for w in warnings)


def test_validator_warns_untested_for_an_on_grid_value_past_362_when_uncapped(monkeypatch):
    """A value that is already on the grid (no snap needed) still gets the
    untested-range warning when it exceeds the documented ceiling."""
    monkeypatch.setenv(spec.max_override_env, "1")
    params = {
        "width": 512, "height": 512,
        "num_frames": NEXT_GRID_POINT_ABOVE_362,
        "frame_rate": spec.fps_fixed,
    }
    warnings = validate_video_geometry(params, "minimax_h3")
    assert params["num_frames"] == NEXT_GRID_POINT_ABOVE_362
    assert any(spec.max_override_env in w for w in warnings)


def test_override_is_a_no_op_below_the_documented_ceiling(monkeypatch):
    monkeypatch.setenv(spec.max_override_env, "1")
    params = {
        "width": 512, "height": 512,
        "num_frames": 141,
        "frame_rate": spec.fps_fixed,
    }
    warnings = validate_video_geometry(params, "minimax_h3")
    assert params["num_frames"] == 141
    assert not any(spec.max_override_env in w for w in warnings)
