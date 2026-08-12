"""High-B (post-`a33347f9` memory audit): the RAM guard that refuses an
oversized spatial-mask video-inpaint request before any GPU work starts.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_mask_spatial_ram_guard_test.py -v

WHY THIS FILE EXISTS
--------------------
`/generate/inpaint/video`'s own decode RAM guard (`_refuse_if_decode_too_large`,
covered by `video_decode_bounds_test.py`) only bounds the raw decoded uint8
clip. The spatial-mask branch (`spatial_mask_manifest` set) allocates float32
buffers over the SAME clip length ON TOP OF that, in
`core.inference.video_mask_timeline.build_spatial_mask_plan` and
`composite_masked_frames` -- neither of which the decode guard's byte math
accounts for. Since MiniMax-H3's own `max_frames` ceiling was removed
(`a33347f9`), a clip length far past its old 362-frame trained-range top can
now reach this branch, so this is no longer a length question `plan_video_
inpaint_span` alone answers -- it is a second, TIGHTER resource question.

Two things are pinned here:

1. A regression guard on the MEASUREMENT the route's own budget constant
   (`api.param_defaults.SPATIAL_MASK_GENERATION_PEAK_MULTIPLIER`) is based
   on: `build_spatial_mask_plan`'s actual peak RAM, measured with
   `tracemalloc`, must stay within that documented multiplier of the raw
   uint8 clip's byte count. If a future change to `build_spatial_mask_plan`
   raises its real peak past the documented multiplier, THIS test catches it
   -- silently invalidating the route's own budget math would otherwise only
   surface as a production OOM.
2. The route's own guard function and call site exist and are wired in the
   right order (source-text check, matching this test suite's existing
   convention in `video_decode_bounds_test.py` -- exercising the multipart
   route directly would need a real upload + a mocked pipeline_manager for
   no additional coverage over asserting the wiring itself).
"""

from __future__ import annotations

import os
import sys
import tracemalloc

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from api.param_defaults import (  # noqa: E402
    MAX_SPATIAL_MASK_GENERATION_RAM_BYTES,
    SPATIAL_MASK_GENERATION_PEAK_MULTIPLIER,
)
from core.inference.video_mask_timeline import (  # noqa: E402
    MaskCanvas,
    MaskKeyframe,
    MaskTimelineManifest,
    MaskTransform,
    build_spatial_mask_plan,
)

_ROUTES_PATH = os.path.join(_BACKEND, "api", "routes.py")


def _routes_source() -> str:
    with open(_ROUTES_PATH, encoding="utf-8") as f:
        return f.read()


def _make_timeline(width: int, height: int, clip_frames: int) -> MaskTimelineManifest:
    canvas = MaskCanvas(width=width, height=height)
    kf0 = MaskKeyframe(id="k0", frame=0, mask_id="m", interpolation_to_next="hold", transform=MaskTransform())
    kf1 = MaskKeyframe(
        id="k1", frame=clip_frames - 1, mask_id="m", interpolation_to_next="hold", transform=MaskTransform()
    )
    return MaskTimelineManifest(
        version=1,
        coordinate_space="output_canvas",
        polarity="white_generates",
        canvas=canvas,
        keyframes=[kf0, kf1],
        composite_feather_px=0,
    )


# ---------------------------------------------------------------------------
# 1. The measurement the route's budget constant is based on must still hold.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clip_frames", [64, 362, 1000])
def test_build_spatial_mask_plan_peak_stays_within_the_documented_multiplier(clip_frames):
    width, height = 768, 1344  # MiniMax-H3's own maximum canvas.
    timeline = _make_timeline(width, height, clip_frames)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1.0

    tracemalloc.start()
    try:
        full_masks, _pinned = build_spatial_mask_plan(
            timeline,
            {"m": mask},
            clip_frames=clip_frames,
            start_frame=0,
            end_frame=clip_frames,
            latent_frame_spans=[(0, clip_frames)],
            spatial_scale=8,
            patch_h=2,
            patch_w=2,
        )
        _current, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    uint8_clip_bytes = clip_frames * width * height * 3
    # A SMALL margin over the raw measurement (not equality) -- this is a
    # regression guard on the route's own budget math, not a re-derivation
    # of the exact peak, which can vary slightly with numpy's allocator.
    assert peak_bytes <= uint8_clip_bytes * SPATIAL_MASK_GENERATION_PEAK_MULTIPLIER * 1.15, (
        f"build_spatial_mask_plan's peak ({peak_bytes / 1024**3:.3f} GiB) at "
        f"clip_frames={clip_frames} exceeded "
        f"{SPATIAL_MASK_GENERATION_PEAK_MULTIPLIER}x the raw uint8 clip's own "
        f"byte count ({uint8_clip_bytes / 1024**3:.3f} GiB) -- the route's RAM "
        "guard budget in api/param_defaults.py no longer reflects this "
        "function's real memory behavior and must be re-measured."
    )
    del full_masks


# ---------------------------------------------------------------------------
# 2. The route's own guard: wiring, not behavior (see module docstring for why).
# ---------------------------------------------------------------------------
def test_the_route_defines_and_calls_the_spatial_mask_ram_guard():
    source = _routes_source()
    anchor = source.index("def generate_inpaint_video(")
    section = source[anchor:anchor + 40000]

    assert "def _refuse_if_spatial_mask_generation_too_large(" in section, (
        "generate_inpaint_video must define a dedicated RAM guard for the "
        "spatial-mask branch, separate from _refuse_if_decode_too_large"
    )
    assert "MAX_SPATIAL_MASK_GENERATION_RAM_BYTES" in section
    assert "SPATIAL_MASK_GENERATION_PEAK_MULTIPLIER" in section
    assert (
        "_refuse_if_spatial_mask_generation_too_large(\n"
        "            clip_frames=int(trimmed_len)" in section
    ), "the guard must be called with the exact clip_frames build_spatial_mask_plan later uses"


def test_the_guard_is_called_only_when_a_spatial_mask_is_present_and_before_the_gpu_slot():
    source = _routes_source()
    anchor = source.index("def generate_inpaint_video(")
    section = source[anchor:anchor + 40000]

    guard_call_pos = section.index("_refuse_if_spatial_mask_generation_too_large(\n            clip_frames=")
    # Guarded by `if spatial_mask_timeline is not None:` immediately above the
    # call, not called unconditionally (a mask-free request must not pay for
    # or be refused by this check at all).
    preceding = section[:guard_call_pos]
    assert preceding.rstrip().endswith("if spatial_mask_timeline is not None:")

    gpu_slot_pos = section.index("_gen_id = start_generation(")
    assert guard_call_pos < gpu_slot_pos, (
        "the spatial-mask RAM guard must run before the GPU generation slot "
        "is reserved, the same ordering _refuse_if_decode_too_large uses"
    )


if __name__ == "__main__":
    import unittest

    unittest.main()
