"""Shot-aligned, variable-length segments (design §7.2c).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_chain_shot_aligned_test.py -v

WHY THIS FILE EXISTS
--------------------
A fixed segment length puts the segment boundaries wherever the arithmetic
lands them, which has two consequences the design records:

  1. a shot whose timestamp does not coincide with a boundary made
     `assign_event_owners` raise, and the user cannot know the boundary frames
     before planning -- so the only way out was plan / error / edit timestamps /
     re-plan. The round trip is reproduced below (00:05.500 / 00:11.000 /
     00:16.000 against boundaries 124 / 247 / 370) and has to disappear;
  2. one segment is one long independent clip, so a prompt's shot-size
     instruction applies uniformly across it.

The mode is OPT-IN, so most of this file is about what did NOT change: an
explicitly requested segment length still produces byte-identical geometry, and
a prompt with no shot structure is still planned at fixed lengths.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.arch_capabilities import video_constraints_payload  # noqa: E402
from api.generation_utils import plan_video_outpaint_placement  # noqa: E402
from core.inference.video_chain_context import (  # noqa: E402
    ChainPlanRequest,
    ChainReference,
    TimelineEvent,
    VideoChainPlanError,
    VideoGridSpec,
    build_segment_geometry,
    build_segment_spans,
    grid_span_options,
    plan_h3_chain_from_prompt,
    plan_shot_aligned_spans,
    plan_video_chain_manifest,
)

FPS = 24.0
SEGMENT = 124            # MiniMax-H3's shortest clip
CAP = 362                # its longest documented one


def _grid() -> VideoGridSpec:
    return VideoGridSpec.from_video_constraints(video_constraints_payload()["minimax_h3"])


def _shot_event(index: int, start: int, end: int, text: str = "a shot") -> TimelineEvent:
    return TimelineEvent(
        id=f"shot_{index}", kind="shot", start_frame=start, end_frame=end,
        description=text, shot_number=index,
    )


def _cuts(spans):
    return [span.owned_end_frame for span in spans]


# --------------------------------------------------------------------------
# 1. The boundaries follow the shots
# --------------------------------------------------------------------------

def test_boundaries_land_exactly_on_shot_boundaries_when_the_grid_allows_it():
    """Zero crossings: both shot starts are reachable cuts, so both are used.

    124 and 247 are on H3's own reachable-cut lattice (`124`, then `+123` per
    continuation), so a planner that looks at the shots has no reason to cut
    anywhere else -- while the fixed plan, which knows only the cap, would cut
    at 362 and 723 and split both shots.
    """
    grid = _grid()
    warnings = []
    spans = plan_shot_aligned_spans(grid, 500, [124, 247], CAP, warnings)
    assert _cuts(spans) == [124, 247, 506]
    # ... and every one of those spans is a legal single request.
    options = set(grid_span_options(grid, CAP))
    assert all(span.generated_span_frames in options for span in spans)
    assert "2 of 2 segment boundaries fall on a shot boundary" in warnings[0]
    # No shot was split, so nothing is disclosed as split.
    assert not any("split across segments" in message for message in warnings)

    fixed = build_segment_spans(grid, 500, CAP)
    assert _cuts(fixed) == [362, 502]        # what the same request used to be:
    # one cut, at 362, which is inside the third shot and splits it.


def test_shots_shorter_than_one_segment_share_a_segment():
    """A shot below the floor cannot BE a segment, so several fill one.

    H3's floor is 124 frames; four shots of ~10 frames cannot each own a
    segment, and the design says to merge rather than to refuse.
    """
    grid = _grid()
    warnings = []
    spans = plan_shot_aligned_spans(grid, 400, [10, 20, 30, 40], SEGMENT, warnings)
    assert spans[0].owned_start_frame == 0
    # All four boundaries are inside segment 1: it owns every one of those shots.
    assert spans[0].owned_end_frame >= 40
    assert all(span.generated_span_frames >= grid.floor_frames for span in spans)


def test_a_shot_longer_than_one_segment_is_split_and_disclosed():
    grid = _grid()
    warnings = []
    # One 400-frame shot with a 124-frame cap: it CANNOT fit in a segment.
    spans = plan_shot_aligned_spans(grid, 400, [], SEGMENT, warnings)
    assert spans is None            # no boundary at all -> not this mode's case
    warnings = []
    spans = plan_shot_aligned_spans(grid, 500, [400], SEGMENT, warnings)
    assert spans is not None
    split = [m for m in warnings if "split across segments" in m]
    assert split, warnings
    assert "frames 0-400" in split[0]


def test_the_planner_never_moves_a_shot_timestamp():
    """The segment boundaries move; the timeline does not (design §7.2c)."""
    grid = _grid()
    boundaries = [132, 264, 384]
    spans = plan_shot_aligned_spans(grid, 500, list(boundaries), SEGMENT, [])
    events = [
        _shot_event(1, 0, 132, "a tram door opens"),
        _shot_event(2, 132, 264, "the courier reads an address"),
        _shot_event(3, 264, 384, "the courier crosses a flooded street"),
        _shot_event(4, 384, spans[-1].owned_end_frame, "the courier rings a doorbell"),
    ]
    manifest = plan_video_chain_manifest(
        ChainPlanRequest(
            architecture="minimax_h3", root_prompt="x", grid=grid, fps=FPS,
            target_frames=500, segment_frames=SEGMENT,
            segment_length_mode="shot_aligned", events=events, root_seed=1,
        )
    )
    assert [(e.start_frame, e.end_frame) for e in manifest.events] == [
        (0, 132), (132, 264), (264, 384), (384, manifest.expected_final_frames)
    ]


# --------------------------------------------------------------------------
# 2. The round trip the design says has to disappear
# --------------------------------------------------------------------------

REPORTED_TIMESTAMPS = [132, 264, 384]   # 00:05.500 / 00:11.000 / 00:16.000 @ 24fps
REPORTED_BOUNDARIES = [124, 247, 370]   # what the fixed plan cut at


def _reported_events(final_frames: int):
    starts = [0] + REPORTED_TIMESTAMPS
    ends = REPORTED_TIMESTAMPS + [final_frames]
    return [
        _shot_event(i + 1, s, e, f"shot {i + 1}") for i, (s, e) in enumerate(zip(starts, ends))
    ]


def _reported_request(grid, mode, final_frames):
    return ChainPlanRequest(
        architecture="minimax_h3", root_prompt="the reported chain", grid=grid, fps=FPS,
        target_frames=500, segment_frames=SEGMENT, segment_length_mode=mode,
        events=_reported_events(final_frames), root_seed=7,
    )


def test_the_reported_timestamps_are_a_hard_error_at_fixed_lengths():
    """The defect, stated as a test so the fix is not measured against nothing."""
    grid = _grid()
    fixed = build_segment_spans(grid, 500, SEGMENT)
    assert _cuts(fixed)[:3] == REPORTED_BOUNDARIES
    with pytest.raises(VideoChainPlanError) as excinfo:
        plan_video_chain_manifest(_reported_request(grid, "fixed", fixed[-1].owned_end_frame))
    assert "crosses the boundary" in str(excinfo.value)


def test_the_reported_timestamps_plan_without_an_error_when_shot_aligned():
    """The same input, planned. Splits are disclosed, not refused.

    Every shot here is longer than the 124-frame cap, so no boundary CAN be hit
    -- the value of the mode in this case is entirely that the planner owns the
    cut instead of demanding the user guess boundary frames it never showed them.
    """
    grid = _grid()
    manifest = plan_video_chain_manifest(
        _reported_request(grid, "shot_aligned", 616)
    )
    assert manifest.segment_length_mode == "shot_aligned"
    assert len(manifest.segments) >= 4
    assert manifest.expected_final_frames >= 500
    # Every event still has exactly one owner, and the splits are warnings.
    owners = [manifest.owner_of(event.id) for event in manifest.events]
    assert all(owner is not None for owner in owners)
    assert any("crosses the boundary" in message for message in manifest.warnings)
    assert any("split across segments" in message for message in manifest.warnings)


# --------------------------------------------------------------------------
# 3. Negative controls: what must NOT have changed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("segment_frames", [None, SEGMENT, 200, CAP])
@pytest.mark.parametrize("target", [300, 500, 1000])
def test_the_default_mode_is_byte_identical_to_the_shipped_geometry(segment_frames, target):
    """NEGATIVE CONTROL: an explicit segment length still means fixed lengths."""
    grid = _grid()
    before = build_segment_spans(grid, target, segment_frames)
    after = build_segment_geometry(grid, target, segment_frames).spans
    assert [s.to_dict() for s in after] == [s.to_dict() for s in before]
    assert build_segment_geometry(grid, target, segment_frames).segment_length_mode == "fixed"


def test_shot_alignment_declines_when_there_is_no_shot_structure():
    """A free-form prompt has nothing to align to: fixed lengths, and say so."""
    grid = _grid()
    warnings = []
    geometry = build_segment_geometry(grid, 500, SEGMENT, warnings, 1, "shot_aligned", [])
    assert geometry.segment_length_mode == "fixed"
    assert [s.to_dict() for s in geometry.spans] == [
        s.to_dict() for s in build_segment_spans(grid, 500, SEGMENT)
    ]
    assert any("no shot boundary" in message for message in warnings)


def test_a_target_that_fits_one_request_is_not_chained_either_way():
    grid = _grid()
    assert plan_shot_aligned_spans(grid, 300, [124], CAP, []) is None
    assert len(build_segment_geometry(grid, 300, CAP, None, 1, "shot_aligned", [124]).spans) == 1


def test_an_unknown_mode_is_refused():
    with pytest.raises(VideoChainPlanError):
        build_segment_geometry(_grid(), 500, SEGMENT, None, 1, "nearest_shot", [124])


# --------------------------------------------------------------------------
# 4. The pinned-tail overlap arithmetic still holds
# --------------------------------------------------------------------------

@pytest.mark.parametrize("overlap", [1, 5, 9, 17])
def test_each_shot_aligned_request_produces_the_span_the_plan_planned(overlap):
    """Cross-checked against the GENERATION's own placement solver.

    `continuation_generated_span` is reused rather than reimplemented, so the
    only thing worth proving is that the request the queue sends
    (`requested_total_frames`) comes back as the range the plan states -- at
    every overlap, not just the anchor.
    """
    grid = _grid()
    spans = plan_shot_aligned_spans(grid, 900, [124, 247, 500], CAP, [], overlap)
    assert spans is not None
    accumulated = spans[0].owned_end_frame
    for span in spans[1:]:
        placement = plan_video_outpaint_placement(
            {"total_frames": span.requested_total_frames, "input_offset_frames": 0},
            "minimax_h3", head_frames=accumulated, overlap_frames=overlap,
        )
        assert placement["generated_frames"] == span.generated_span_frames
        assert placement["total_frames"] == span.owned_end_frame
        assert span.owned_start_frame == accumulated
        assert span.anchor_global_frame == span.owned_start_frame - overlap
        accumulated = span.owned_end_frame
    assert accumulated >= 900


@pytest.mark.parametrize("overlap,boundaries", [(1, [124, 247]), (17, [124, 248])])
def test_a_wider_pin_moves_which_cuts_exist_and_alignment_follows_them(overlap, boundaries):
    """The pin changes the reachable cuts, so alignment is solved against them.

    A continuation adds `span - overlap` frames, so the lattice of reachable
    boundaries is overlap-dependent: 247 is reachable with the 1-frame anchor
    and 248 with a 17-frame pin. The planner has to hit whichever exists, which
    is why it solves the cuts rather than snapping a length.
    """
    spans = plan_shot_aligned_spans(_grid(), 500, boundaries, CAP, [], overlap)
    assert _cuts(spans)[:2] == boundaries


# --------------------------------------------------------------------------
# 5. Over the wire (POST /video-chain/plan), no server started
# --------------------------------------------------------------------------

H3_PROMPT = (
    "integrated_multimodal_description: [Shot 1] A courier steps off a tram into "
    "the rain.\n"
    "[Shot 2] At 00:05.500 The courier checks a paper address under a shop awning.\n"
    "[Shot 3] At 00:11.000 The courier runs across a flooded crossing.\n"
    "[Shot 4] At 00:16.000 The courier presses a doorbell and waits.\n\n"
    "overall_soundscape: Rain on canvas and passing tyres.\n\n"
    "non_diegetic_music: N/A"
)


def _app():
    from fastapi import FastAPI

    import api.routes as routes
    from api.error_handlers import register_error_handlers

    app = FastAPI()
    register_error_handlers(app)
    app.post("/video-chain/plan")(routes.plan_video_chain_route)
    app.post("/video-chain/validate")(routes.validate_video_chain_route)
    return app


def _post(path: str, body: dict):
    import asyncio

    import httpx

    async def run():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(path, json=body)
            return response.status_code, response.json()

    return asyncio.run(run())


def _plan(**overrides):
    body = {
        "architecture": "minimax_h3",
        "variant": "fl2va",
        "root_prompt": H3_PROMPT,
        "target_frames": 500,
        "fps": FPS,
        "requested_segment_frames": SEGMENT,
        "context_mode": "timeline",
    }
    body.update(overrides)
    return _post("/video-chain/plan", body)


def test_the_reported_request_now_plans_over_the_wire():
    """The exact shape the user hit: timestamps that miss 124 / 247 / 370."""
    status, fixed = _plan()
    assert status == 200, fixed
    assert fixed["success"] is False
    assert any("crosses the boundary" in issue["message"] for issue in fixed["errors"])

    status, aligned = _plan(segment_length_mode="shot_aligned")
    assert status == 200, aligned
    assert aligned["success"] is True, aligned["errors"]
    assert aligned["manifest"]["segment_length_mode"] == "shot_aligned"
    assert aligned["frame_plan"]["segment_count"] == len(aligned["manifest"]["segments"])
    assert aligned["frame_plan"]["segment_new_output_frames"] == [
        s["owned_end_frame"] - s["owned_start_frame"] for s in aligned["manifest"]["segments"]
    ]


def test_the_planned_manifest_survives_a_round_trip_through_validate():
    status, payload = _plan(segment_length_mode="shot_aligned")
    assert status == 200, payload
    manifest = payload["manifest"]
    status, result = _post(
        "/video-chain/validate", {"manifest": manifest, "recompute_plan_hash": True}
    )
    assert status == 200, result
    assert result["valid"] is True, result["errors"]
    # The mode is part of the plan, so it must survive AND keep the hash.
    assert result["manifest"]["segment_length_mode"] == "shot_aligned"
    assert result["plan_hash"] == manifest["plan_hash"]


def test_the_default_request_is_unchanged_over_the_wire():
    """NEGATIVE CONTROL: not sending the field is the shipped plan, exactly."""
    status, without = _plan(root_prompt="a courier runs through the rain",
                            context_mode="legacy_repeat")
    status2, explicit = _plan(root_prompt="a courier runs through the rain",
                              context_mode="legacy_repeat", segment_length_mode="fixed")
    assert (status, status2) == (200, 200)
    assert without["frame_plan"] == explicit["frame_plan"]
    assert without["manifest"]["segment_length_mode"] == "fixed"
    assert [s["owned_end_frame"] for s in without["manifest"]["segments"]] == [124, 247, 370, 493, 616]


def test_a_free_form_prompt_stays_fixed_length_even_when_alignment_is_asked_for():
    status, payload = _plan(root_prompt="a courier runs through the rain",
                            context_mode="legacy_repeat",
                            segment_length_mode="shot_aligned")
    assert status == 200, payload
    assert payload["manifest"]["segment_length_mode"] == "fixed"
    assert [s["owned_end_frame"] for s in payload["manifest"]["segments"]] == [124, 247, 370, 493, 616]
    assert any("no shot boundary" in issue["message"] for issue in payload["warnings"])


def test_an_unknown_segment_length_mode_is_a_400():
    status, payload = _plan(segment_length_mode="nearest_shot")
    assert status == 400, payload
    assert "segment_length_mode" in payload["error"]


def test_alignment_shows_in_the_plan_when_the_grid_allows_it():
    """A cap of 362 makes 124 and 247 reachable cuts, so they get used."""
    prompt = (
        "integrated_multimodal_description: [Shot 1] A lighthouse beam sweeps the bay.\n"
        f"[Shot 2] At 00:05.167 The keeper climbs the stair.\n"
        f"[Shot 3] At 00:10.292 The keeper opens the lamp room door.\n\n"
        "overall_soundscape: Wind and surf.\n\nnon_diegetic_music: N/A"
    )
    status, payload = _plan(root_prompt=prompt, requested_segment_frames=CAP,
                            target_frames=500, segment_length_mode="shot_aligned")
    assert status == 200, payload
    ends = [s["owned_end_frame"] for s in payload["manifest"]["segments"]]
    assert ends[:2] == [124, 247]
    assert payload["success"] is True, payload["errors"]
    # Each shot owns exactly one segment here, so nothing is reported as split.
    assert not any("split across segments" in i["message"] for i in payload["warnings"])


# --------------------------------------------------------------------------
# 6. The queue reads the per-segment plan (frontend contract)
# --------------------------------------------------------------------------

def _frontend(*parts: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, "frontend", "src", *parts), encoding="utf-8") as handle:
        return handle.read()


def test_advance_chain_uses_the_planned_length_instead_of_re_deriving_it():
    """Source-anchored, like the sibling chain tests.

    `nextVideoChainTotalFrames` re-derives a continuation's length from the cap,
    which is only correct while every segment IS the cap. A shot-aligned chain
    has to take the length from the item the plan froze it onto.
    """
    source = _frontend("utils", "videoChain.ts")
    assert "chainPlannedNewOutputFrames" in source
    assert "nextChainItem.chainPlannedNewOutputFrames" in source
    assert 'segment_length_mode === "shot_aligned"' in source
    # Segment 1 may be shorter than the cap under this mode.
    assert "segmentChainFirstFrames" in source
    for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
        assert "segmentChainFirstFrames" in _frontend("components", "generation", panel)


def test_the_plan_editor_shows_the_segment_count_and_lengths():
    """Design §7.2c: the cost side of the trade-off is the user's to judge."""
    dialog = _frontend("components", "common", "VideoChainConfirmDialog.tsx")
    assert "Frames added per segment" in dialog
    assert "segment_length_mode" in dialog
    assert "generation request" in dialog
