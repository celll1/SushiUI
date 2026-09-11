"""Contract for the user-overridable generation bound registry."""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.param_defaults import PARAM_BOUNDS


def test_param_bounds_registry_is_complete_and_well_formed():
    assert set(PARAM_BOUNDS) == {
        "image_width_max",
        "image_height_max",
        "steps_max",
        "cfg_scale_max",
        "video_frame_rate_max",
        "upscale_tile_size_max",
    }
    for spec in PARAM_BOUNDS.values():
        assert set(spec) >= {"builtin", "floor", "ceiling", "family", "label"}
        assert spec["floor"] <= spec["builtin"] <= spec["ceiling"]
