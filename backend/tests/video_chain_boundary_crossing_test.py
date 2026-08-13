"""`boundary_crossing_policy`: the explicit way through a boundary-crossing shot.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_chain_boundary_crossing_test.py -v

WHY THIS FILE EXISTS
--------------------
`assign_event_owners` has always been able to keep a boundary-crossing event
whole in the earlier segment, but only its `allow_boundary_split` argument
selected that, and nothing on the wire reached it. With
`segment_length_mode: fixed` the segment boundaries follow from the frame
arithmetic, so a caller cannot place timestamps on them before planning -- which
left a timeline with natural timestamps (the P-VC-2 set below) with no route
through the API at all: every plan came back as an error.

The choice is now a request field. What must NOT change is its default: a
fixed-length plan that does not ask for the other policy is still refused, with
the shot named. Both halves are tested here, at fixed lengths AND under shot
alignment. (Which length mode a plan runs under is a separate default, resolved
from the timeline; the fixed-length cases below name it.)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.arch_capabilities import video_constraints_payload  # noqa: E402
from api.param_defaults import VIDEO_CHAIN_DEFAULTS  # noqa: E402
from core.inference.video_chain_context import (  # noqa: E402
    BOUNDARY_CROSSING_POLICIES,
    ChainPlanRequest,
    TimelineEvent,
    VideoChainPlanError,
    VideoGridSpec,
    boundary_crossing_allows_ownership,
    build_segment_spans,
    plan_video_chain_manifest,
)

FPS = 24.0
SEGMENT = 124            # MiniMax-H3's shortest clip
TARGET = 500

# The timestamps the P-VC-2 runs actually used: 00:05.875 / 00:11.000 /
# 00:16.833 at 24 fps. None of them is a reachable fixed-length cut
# (124 / 247 / 370 / 493), which is the whole point.
P_VC_2_TIMESTAMPS = ["00:05.875", "00:11.000", "00:16.833"]
P_VC_2_FRAMES = [141, 264, 404]

H3_PROMPT = (
    "integrated_multimodal_description: [Shot 1] A courier steps off a tram into the rain.\n"
    f"[Shot 2] At {P_VC_2_TIMESTAMPS[0]} The courier checks a paper address under an awning.\n"
    f"[Shot 3] At {P_VC_2_TIMESTAMPS[1]} The courier runs across a flooded crossing.\n"
    f"[Shot 4] At {P_VC_2_TIMESTAMPS[2]} The courier presses a doorbell and waits.\n\n"
    "overall_soundscape: Rain on canvas and passing tyres.\n\n"
    "non_diegetic_music: N/A"
)

# One shot, no internal boundary: shot alignment has nothing to align to and
# falls back to fixed lengths, so this is where the policy still decides the
# outcome of a `shot_aligned` request.
SINGLE_SHOT_PROMPT = (
    "integrated_multimodal_description: [Shot 1] A lighthouse beam sweeps the bay "
    "while the keeper climbs the stair.\n\n"
    "overall_soundscape: Wind and surf.\n\nnon_diegetic_music: N/A"
)


def _grid() -> VideoGridSpec:
    return VideoGridSpec.from_video_constraints(video_constraints_payload()["minimax_h3"])


# --------------------------------------------------------------------------
# 1. The core message says which shot, which frames, and where it is cut
# --------------------------------------------------------------------------

def _crossing_request(mode: str, allow: bool):
    grid = _grid()
    final = build_segment_spans(grid, TARGET, SEGMENT)[-1].owned_end_frame
    starts = [0] + P_VC_2_FRAMES
    ends = P_VC_2_FRAMES + [final]
    events = [
        TimelineEvent(
            id=f"shot_{i + 1}", kind="shot", start_frame=s, end_frame=e,
            description=f"shot {i + 1}", shot_number=i + 1,
        )
        for i, (s, e) in enumerate(zip(starts, ends))
    ]
    return ChainPlanRequest(
        architecture="minimax_h3", root_prompt="the reported chain", grid=grid, fps=FPS,
        target_frames=TARGET, segment_frames=SEGMENT, segment_length_mode=mode,
        events=events, root_seed=7, allow_boundary_split=allow,
    )


def test_the_refusal_names_the_shot_its_frames_and_the_cut():
    with pytest.raises(VideoChainPlanError) as excinfo:
        plan_video_chain_manifest(_crossing_request("fixed", False))
    message = str(excinfo.value)
    assert "crosses the boundary" in message
    assert "shot 1" in message              # 0-141 crosses the 124 cut
    assert "(frames 0-141)" in message
    assert "at frame 124" in message
    assert "boundary_crossing_policy" in message


def test_the_disclosure_names_every_split_shot_with_its_frames():
    manifest = plan_video_chain_manifest(_crossing_request("fixed", True))
    crossings = [w for w in manifest.warnings if "crosses the boundary" in w]
    assert crossings
    # Shot 1 (0-141) crosses at 124, shot 2 (141-264) at 247, shot 3 (264-404) at 370.
    assert any("shot 1" in w and "(frames 0-141)" in w and "at frame 124" in w for w in crossings)
    assert any("shot 2" in w and "(frames 141-264)" in w and "at frame 247" in w for w in crossings)
    assert any("shot 3" in w and "(frames 264-404)" in w and "at frame 370" in w for w in crossings)
    # Kept whole, not cut in two: exactly one owner each, in the earlier segment.
    assert [manifest.owner_of(f"shot_{i}") for i in (1, 2, 3, 4)] == [0, 1, 2, 3]
    assert all("owns it whole" in w for w in crossings)


def test_the_policy_vocabulary_maps_to_the_one_internal_flag():
    assert BOUNDARY_CROSSING_POLICIES == ("refuse", "assign_to_earlier_segment")
    assert boundary_crossing_allows_ownership("refuse") is False
    assert boundary_crossing_allows_ownership("assign_to_earlier_segment") is True
    with pytest.raises(VideoChainPlanError):
        boundary_crossing_allows_ownership("split")


def test_the_default_is_the_refusal():
    assert VIDEO_CHAIN_DEFAULTS["boundary_crossing_policy"] == "refuse"
    assert boundary_crossing_allows_ownership(
        VIDEO_CHAIN_DEFAULTS["boundary_crossing_policy"]
    ) is False


# --------------------------------------------------------------------------
# 2. Over the wire (POST /video-chain/plan), no server started
# --------------------------------------------------------------------------

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
        "target_frames": TARGET,
        "fps": FPS,
        "requested_segment_frames": SEGMENT,
        "context_mode": "timeline",
    }
    body.update(overrides)
    return _post("/video-chain/plan", body)


def test_a_fixed_length_plan_is_still_refused_by_default():
    """NEGATIVE CONTROL: the shipped behaviour of a fixed-length plan, unchanged.

    `segment_length_mode` is named here because omitting it now resolves to shot
    alignment for this timeline, and then the planner owns the cut.
    """
    status, payload = _plan(segment_length_mode="fixed")
    assert status == 200, payload
    assert payload["success"] is False
    assert any("crosses the boundary" in issue["message"] for issue in payload["errors"])
    # ... and sending the default explicitly is the same request.
    status, explicit = _plan(segment_length_mode="fixed",
                             boundary_crossing_policy="refuse")
    assert status == 200, explicit
    assert explicit["success"] is False
    assert explicit["frame_plan"] == payload["frame_plan"]


def test_the_p_vc_2_timestamps_plan_at_fixed_lengths_when_the_policy_is_chosen():
    """The dead end the API had no way out of."""
    status, payload = _plan(segment_length_mode="fixed",
                            boundary_crossing_policy="assign_to_earlier_segment")
    assert status == 200, payload
    assert payload["success"] is True, payload["errors"]
    assert payload["manifest"]["segment_length_mode"] == "fixed"
    # The boundaries are untouched: this decides ownership, not geometry.
    assert [s["owned_end_frame"] for s in payload["manifest"]["segments"]] == [
        124, 247, 370, 493, 616
    ]
    crossings = [i["message"] for i in payload["warnings"] if "crosses the boundary" in i["message"]]
    assert crossings
    assert any("shot 1" in m and "(frames 0-141)" in m and "at frame 124" in m for m in crossings)
    # The shot timestamps themselves are not moved.
    assert [e["start_frame"] for e in payload["manifest"]["events"]] == [0] + P_VC_2_FRAMES


def test_the_planned_manifest_survives_a_round_trip_through_validate():
    status, payload = _plan(segment_length_mode="fixed",
                            boundary_crossing_policy="assign_to_earlier_segment")
    assert status == 200, payload
    manifest = payload["manifest"]
    status, result = _post(
        "/video-chain/validate", {"manifest": manifest, "recompute_plan_hash": True}
    )
    assert status == 200, result
    assert result["valid"] is True, result["errors"]
    # The policy decides whether a plan is produced, not what it contains, so it
    # is not part of the hash and a re-validate reproduces it.
    assert result["plan_hash"] == manifest["plan_hash"]


@pytest.mark.parametrize("segment_length_mode", ["fixed", "shot_aligned"])
def test_the_policy_decides_the_outcome_in_both_length_modes(segment_length_mode):
    """Shot alignment falls back to fixed lengths when there is nothing to align
    to, and then the crossing is back -- so the policy has to reach that path too."""
    refused = _plan(root_prompt=SINGLE_SHOT_PROMPT, segment_length_mode=segment_length_mode)
    assert refused[0] == 200, refused[1]
    assert refused[1]["success"] is False
    assert any("crosses the boundary" in i["message"] for i in refused[1]["errors"])

    allowed = _plan(
        root_prompt=SINGLE_SHOT_PROMPT,
        segment_length_mode=segment_length_mode,
        boundary_crossing_policy="assign_to_earlier_segment",
    )
    assert allowed[0] == 200, allowed[1]
    assert allowed[1]["success"] is True, allowed[1]["errors"]
    assert any(
        "shot 1" in i["message"] and "crosses the boundary" in i["message"]
        for i in allowed[1]["warnings"]
    )


def test_shot_aligned_still_keeps_its_own_splits_whole_without_the_field():
    """NEGATIVE CONTROL: alignment owns the cut it chose, as it did before."""
    status, payload = _plan(segment_length_mode="shot_aligned")
    assert status == 200, payload
    assert payload["success"] is True, payload["errors"]
    assert payload["manifest"]["segment_length_mode"] == "shot_aligned"


def test_an_unknown_policy_is_a_400():
    status, payload = _plan(boundary_crossing_policy="split_it")
    assert status == 400, payload
    assert "boundary_crossing_policy" in payload["error"]


# --------------------------------------------------------------------------
# 3. The editor exposes the choice and the interaction it has with the overlap
# --------------------------------------------------------------------------

def _frontend(*parts: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, "frontend", "src", *parts), encoding="utf-8") as handle:
        return handle.read()


def test_the_plan_editor_offers_the_policy_and_defaults_to_refusing():
    dialog = _frontend("components", "common", "VideoChainConfirmDialog.tsx")
    assert "boundary_crossing_policy: boundaryCrossingPolicy" in dialog
    assert 'useState<VideoChainBoundaryCrossingPolicy>("refuse")' in dialog
    assert 'value="assign_to_earlier_segment"' in dialog
    assert "boundary_crossing_policy?: VideoChainBoundaryCrossingPolicy;" in _frontend(
        "utils", "api.ts"
    )


def test_the_editor_discloses_the_alignment_overlap_interaction_before_planning():
    """Design §7.2c: the combination is stated where the modes are chosen."""
    dialog = _frontend("components", "common", "VideoChainConfirmDialog.tsx")
    # Shown for a shot-aligned plan AND for the timeline-resolved default, which
    # is also shot-aligned whenever there are shots.
    assert 'segmentLengthMode !== "fixed" && effectiveMode !== "boundary_frame"' in dialog
    # JSX prose is hard-wrapped, so compare on collapsed whitespace.
    prose = " ".join(dialog.split())
    assert "fewer shot starts are reachable" in prose
    assert "changes with the number of frames each continuation shares" in prose
