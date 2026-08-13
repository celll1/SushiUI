"""A pinned overlap in the PLAN's frame arithmetic (design §7.2b defect 2).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_chain_pinned_overlap_plan_test.py -v

WHY THIS FILE EXISTS
--------------------
`pinned_tail` shares more than the anchor frame, so the continuation has to
generate a longer span -- and that span is rounded UP onto the arch's frame
grid. The rounding lands in the OUTPUT: measured on MiniMax-H3 (P-VC-1), a
continuation that adds 123 frames at the 1-frame anchor adds 136 / 132 / 124 at
overlaps 5 / 9 / 17. A plan built with the anchor-only arithmetic therefore
states frame ranges the chain will not have.

The checks here are all of the same shape: the manifest's own numbers versus
`plan_video_outpaint_placement`, which is what the GENERATION solves. They are
not two readings of the planner.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.arch_capabilities import (  # noqa: E402
    MINIMAX_H3_PINNED_TAIL_MIN_FRAMES,
    video_constraints_payload,
)
from api.generation_utils import plan_video_outpaint_placement  # noqa: E402
from core.inference.video_chain_context import (  # noqa: E402
    VIDEO_CHAIN_ANCHOR_FRAMES,
    VideoGridSpec,
    build_segment_spans,
    plan_chain_lengths,
)

SEGMENT = 124            # MiniMax-H3's shortest clip: one segment per request
TARGET = 500
OVERLAPS = (5, 9, 13, 17)


def _grid() -> VideoGridSpec:
    return VideoGridSpec.from_video_constraints(video_constraints_payload()["minimax_h3"])


# --------------------------------------------------------------------------
# 1. The plan states the length the chain reaches
# --------------------------------------------------------------------------

@pytest.mark.parametrize("overlap", OVERLAPS)
def test_every_segments_end_is_what_the_generation_would_return(overlap):
    """`owned_end_frame` == the placement planner's `total_frames`.

    One request at a time: feed the placement the accumulated length and the
    `total_frames` the queue sends (`requested_total_frames`), and its answer
    has to be the manifest's own range. This is the cross-check that would have
    caught the shipped defect -- the two layers disagreed by +13 frames per
    segment at overlap 5.
    """
    spans = build_segment_spans(_grid(), TARGET, SEGMENT, None, overlap)
    accumulated = spans[0].owned_end_frame
    for span in spans[1:]:
        placement = plan_video_outpaint_placement(
            {"total_frames": span.requested_total_frames, "input_offset_frames": 0},
            "minimax_h3", head_frames=accumulated, overlap_frames=overlap,
        )
        assert placement["generated_frames"] == span.generated_span_frames
        assert placement["total_frames"] == span.owned_end_frame
        assert span.owned_start_frame == accumulated
        accumulated = span.owned_end_frame


@pytest.mark.parametrize("overlap", OVERLAPS)
def test_the_ranges_tile_and_the_anchor_points_at_the_first_shared_frame(overlap):
    spans = build_segment_spans(_grid(), TARGET, SEGMENT, None, overlap)
    assert spans[0].owned_start_frame == 0
    assert spans[0].anchor_global_frame is None
    for previous, span in zip(spans, spans[1:]):
        assert span.owned_start_frame == previous.owned_end_frame
        # local index 0 is the FIRST shared frame, and the shared region is
        # exactly the overlap -- so the mapping stays exact in both directions.
        assert span.anchor_global_frame == span.owned_start_frame - overlap
        assert span.global_frame(0) == span.anchor_global_frame
        assert span.local_frame(span.owned_start_frame) == overlap
        assert span.owned_frames == span.generated_span_frames - overlap


def test_the_measured_overlap_of_the_shipped_ab_reproduces_its_frame_counts():
    """P-VC-1's own numbers: 123 / 136 / 132 / 124 new frames per continuation.

    Recorded because they are the evidence the plan was wrong, and because they
    pin the direction of the rounding: a WIDER pin does not monotonically add
    more frames (17 lands almost exactly back on the anchor's 124).
    """
    grid = _grid()
    gains = {}
    for overlap in (1, 5, 9, 17):
        spans = build_segment_spans(grid, TARGET, SEGMENT, None, overlap)
        gains[overlap] = spans[1].owned_end_frame - spans[1].owned_start_frame
        assert spans[1].requested_total_frames == 247  # the request is the same one
    assert gains == {1: 123, 5: 136, 9: 132, 17: 124}


@pytest.mark.parametrize("overlap", (1,) + OVERLAPS)
def test_the_final_length_is_the_last_range_not_the_last_request(overlap):
    grid = _grid()
    plan = plan_chain_lengths(grid, TARGET, SEGMENT, overlap)
    spans = build_segment_spans(grid, TARGET, SEGMENT, None, overlap)
    assert plan.final_frames == spans[-1].owned_end_frame
    assert plan.segments == len(spans)
    assert plan.final_frames >= TARGET


def test_the_anchor_default_is_byte_identical_to_the_shipped_planner():
    """NEGATIVE CONTROL: `boundary_frame` geometry did not move.

    The overlap is an added argument, not a changed default -- everything the
    1-frame anchor produced before must still come out, including the request
    totals being equal to the ranges there.
    """
    grid = _grid()
    default = build_segment_spans(grid, TARGET, SEGMENT)
    explicit = build_segment_spans(grid, TARGET, SEGMENT, None, VIDEO_CHAIN_ANCHOR_FRAMES)
    assert [s.to_dict() for s in default] == [s.to_dict() for s in explicit]
    for span in default[1:]:
        assert span.requested_total_frames == span.owned_end_frame
        assert span.anchor_global_frame == span.owned_start_frame - 1


# --------------------------------------------------------------------------
# 2. The same numbers over the wire (POST /video-chain/plan)
# --------------------------------------------------------------------------

def _app():
    from fastapi import FastAPI

    import api.routes as routes
    from api.error_handlers import register_error_handlers

    app = FastAPI()
    register_error_handlers(app)
    app.post("/video-chain/plan")(routes.plan_video_chain_route)
    return app


def _plan(**overrides):
    import asyncio

    import httpx

    body = {
        "architecture": "minimax_h3",
        "variant": "fl2va",
        "root_prompt": "a cat walks across a sunlit room",
        "target_frames": TARGET,
        "fps": 24.0,
        "requested_segment_frames": SEGMENT,
        "context_mode": "legacy_repeat",
        "continuation_mode": "boundary_frame",
    }
    body.update(overrides)

    async def run():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/video-chain/plan", json=body)
            return response.status_code, response.json()

    return asyncio.run(run())


def test_the_planned_manifest_carries_the_overlaps_own_geometry():
    status, payload = _plan(continuation_mode="pinned_tail", requested_overlap_frames=5)
    assert status == 200, payload
    manifest = payload["manifest"]
    segments = manifest["segments"]
    assert manifest["continuation_mode"] == "pinned_tail"
    assert segments[1]["owned_end_frame"] == 260          # not the requested 247
    assert segments[1]["requested_total_frames"] == 247   # ... which is still the request
    assert manifest["expected_final_frames"] == segments[-1]["owned_end_frame"]
    assert payload["frame_plan"]["expected_final_frames"] == manifest["expected_final_frames"]
    for segment in segments[1:]:
        assert segment["effective_overlap_frames"] == 5
        assert segment["requested_overlap_frames"] == 5
        assert segment["visual_context"]["shared_context_frames"] == 5
        assert segment["visual_context"]["mode"] == "pinned_tail"
    assert segments[0]["effective_overlap_frames"] == 0
    # The disclosure names the real per-segment gain rather than leaving the
    # reader to compare the numbers themselves.
    assert any("136" in issue["message"] for issue in payload["warnings"]), payload["warnings"]


def test_the_preview_rows_agree_with_the_manifest_ranges():
    status, payload = _plan(continuation_mode="pinned_tail", requested_overlap_frames=17)
    assert status == 200, payload
    for segment, preview in zip(payload["manifest"]["segments"], payload["segments"]):
        assert preview["global_frame_end"] == segment["owned_end_frame"]
        assert preview["new_output_frames"] == (
            segment["owned_end_frame"] - segment["owned_start_frame"])


@pytest.mark.parametrize("overlap", [0, 1, 2, 16])
def test_the_planner_refuses_the_overlaps_the_generation_would_refuse(overlap):
    """A plan must not promise a geometry `/generate/outpaint/video` rejects."""
    status, payload = _plan(continuation_mode="pinned_tail",
                            requested_overlap_frames=overlap)
    assert status == 400, payload
    assert "continuation_overlap_frames" in payload["error"]


def test_the_boundary_frame_plan_is_untouched_by_this_change():
    status, payload = _plan()
    assert status == 200, payload
    manifest = payload["manifest"]
    assert manifest["continuation_mode"] == "boundary_frame"
    for segment in manifest["segments"][1:]:
        assert segment["effective_overlap_frames"] == 0
        assert segment["owned_end_frame"] == segment["requested_total_frames"]
        assert segment["visual_context"]["shared_context_frames"] == 1
    assert manifest["expected_final_frames"] == manifest["segments"][-1]["owned_end_frame"]


def test_a_capability_floor_below_the_measurement_would_fail_here():
    """The floor is a fact this suite owns, not a value free to drift."""
    assert MINIMAX_H3_PINNED_TAIL_MIN_FRAMES == 5


# --------------------------------------------------------------------------
# 3. The queue sends the request the plan assumed
# --------------------------------------------------------------------------

def _videochain_ts() -> str:
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "frontend", "src", "utils", "videoChain.ts")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_the_queue_asks_for_requested_total_frames_and_drifts_against_the_range():
    """Source-anchored, like the sibling chain tests: with the two numbers now
    distinct, sending `owned_end_frame` as `total_frames` would make the
    generation solve a span the plan never planned."""
    source = _videochain_ts()
    assert "s.requested_total_frames ?? s.owned_end_frame" in source
    # `plannedSegment` is `manifestSegments[index]`, i.e. THIS continuation's
    # own manifest row.
    assert "const plannedSegment = manifestSegments?.[index];" in source
    assert "chainPlannedAccumulatedFrames: plannedSegment?.owned_end_frame" in source
