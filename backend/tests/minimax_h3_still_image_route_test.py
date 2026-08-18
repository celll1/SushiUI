"""MiniMax-H3: `num_frames=1` still-image request (Phase 0 of the still-image
feature) at the route-validation and packed-layout layers.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_still_image_route_test.py -v

WHY THIS FILE EXISTS
--------------------
`num_frames == 1` is a still-image special case, spec-driven through
`TemporalSpec.allows_single_frame` (`core/models/components/wiring.py`):
`validate_video_geometry` reads the flag itself and exempts that one length
from the floor/grid rule (with a `warnings[]` entry, never silent), so
`/generate/txt2vid` needs no `if arch == "minimax_h3"` branch of its own --
it calls `is_still_image_video_request(params, arch)`
(`api/generation_utils.py`) only to decide whether to force `audio_enable`
off before validation runs. Without the flag, `num_frames=1` is neither on
the 17*n+5 grid nor at the 124 floor, so the ordinary path SNAPS it up to
124 -- silently turning a still-image request into a 124-frame one instead
of honoring it, which is exactly what LTX-2.3 (`allows_single_frame=False`)
does NOT need, because `1` is already a normal on-grid length there
(`frame_offset=1`) and represents a real 1-frame clip, not a shortcut around
the multi-chunk VAE's floor.

`build_packed_layout` is exercised directly at `num_latent_frames=1` (the
geometry `minimax_h3_latent_frames(1)` resolves to) as a cheap, CPU-only
regression guard for the "training already proves T=1 through this builder"
claim the still-image design relies on -- this pins actual row-count/shape
assertions to it rather than leaving it as an unverified claim.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.components.wiring import MINIMAX_H3_TEMPORAL as spec  # noqa: E402
from core.models.components.wiring import LTX2_TEMPORAL  # noqa: E402
from api.generation_utils import validate_video_geometry, is_still_image_video_request  # noqa: E402
from core.models.minimax_h3.loader import minimax_h3_latent_frames  # noqa: E402
from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402


def test_num_frames_1_is_off_the_ordinary_grid_and_below_the_floor():
    """Establishes WHY the exemption is needed: without it, num_frames=1 is
    neither valid nor left alone by the ordinary rule."""
    assert not spec.is_valid_length(1)
    assert 1 < spec.min_frames


def test_spec_marks_the_two_video_archs_correctly():
    """MiniMax-H3 has measured, shipped T=1 decode support; LTX-2.3 does not
    need the flag because 1 is already a normal on-grid length for it."""
    assert spec.allows_single_frame is True
    assert LTX2_TEMPORAL.allows_single_frame is False
    assert LTX2_TEMPORAL.is_valid_length(1)  # confirms WHY it doesn't need the flag


def test_is_still_image_video_request_matches_the_spec_flag():
    """The route's own predicate: true only for an arch with the flag set,
    and only at num_frames == 1 exactly."""
    assert is_still_image_video_request({"num_frames": 1}, "minimax_h3") is True
    assert is_still_image_video_request({"num_frames": 2}, "minimax_h3") is False
    assert is_still_image_video_request({"num_frames": 124}, "minimax_h3") is False
    assert is_still_image_video_request({"num_frames": 1}, "ltx2") is False
    assert is_still_image_video_request({"num_frames": 1}, "unknown_arch") is False
    assert is_still_image_video_request({}, "minimax_h3") is False


def test_num_frames_1_is_left_untouched_with_a_stated_warning():
    """The still-image special case: exempt from the clip-length rule
    entirely, leaving num_frames exactly as the client sent it -- but never
    silently, per this function's own "stated, not silent" convention."""
    params = {"width": 512, "height": 512, "num_frames": 1, "frame_rate": spec.fps_fixed}

    warnings = validate_video_geometry(params, "minimax_h3")

    assert params["num_frames"] == 1
    assert any("num_frames=1" in w and "still-image" in w for w in warnings)


def test_num_frames_2_is_still_snapped_up_to_the_floor():
    """The exemption is scoped to exactly num_frames == 1: a too-short value
    that is NOT the still-image sentinel is snapped exactly as before."""
    params = {"width": 512, "height": 512, "num_frames": 2, "frame_rate": spec.fps_fixed}

    warnings = validate_video_geometry(params, "minimax_h3")

    assert params["num_frames"] == 124
    assert any("num_frames=2" in w for w in warnings)


def test_still_image_request_still_enforces_spatial_and_frame_rate_rules():
    """The exemption is scoped to clip length ONLY: a bad canvas is still a
    hard 400, and a non-fixed frame_rate is still forced with a warning -- a
    still image is still a video-shaped request in every other respect."""
    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError):
        validate_video_geometry({"width": 511, "height": 512, "num_frames": 1}, "minimax_h3")

    params = {"width": 512, "height": 512, "num_frames": 1, "frame_rate": 30.0}
    warnings = validate_video_geometry(params, "minimax_h3")
    assert params["frame_rate"] == spec.fps_fixed
    assert any("frame_rate" in w for w in warnings)


def test_capability_payload_carries_the_still_image_flag():
    """A client building a clip-length control reads `allows_single_frame`
    from arch-capabilities rather than hardcoding which arch offers it."""
    from api.arch_capabilities import video_constraints_payload

    payload = video_constraints_payload()
    assert payload["minimax_h3"]["allows_single_frame"] is True
    assert payload["ltx2"]["allows_single_frame"] is False


def test_latent_frames_for_a_still_image_request_is_one():
    assert minimax_h3_latent_frames(1) == 1


def test_build_packed_layout_accepts_a_single_latent_frame():
    """Cheap, CPU-only regression guard for the claim the still-image design
    leans on: the packed-layout builder (RoPE position grid, row indices,
    modality tags) is generic in `num_latent_frames` and needs no T=1 branch
    of its own -- unlike the VAE's `_decode`, which does (see
    `minimax_h3_still_image_decode_test.py`)."""
    num_text_tokens = 32
    latent_height, latent_width = 4, 4
    num_audio_latents = ops.audio_latent_frames(1, fps=24.0)
    patch_size = (1, 2, 2)

    layout = ops.build_packed_layout(
        num_text_tokens, 1, latent_height, latent_width, num_audio_latents,
        patch_size=patch_size,
    )

    rows_per_frame = (latent_height // patch_size[1]) * (latent_width // patch_size[2])
    row_counts = ops.packed_row_counts(layout)
    assert row_counts["target_video"] == rows_per_frame
    assert row_counts["condition_video"] == 0
    assert row_counts["text"] == num_text_tokens
    assert layout["position_ids"].shape[0] == row_counts["total"]
    # No NaN/inf leaked into the rotary grid from a degenerate single-frame
    # temporal span (a division-by-`num_latent_frames - 1` bug would surface
    # here as a NaN in `video_position_ids`).
    assert torch.isfinite(layout["position_ids"]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
