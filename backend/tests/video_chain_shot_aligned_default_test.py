"""Shot alignment is what a structured prompt gets when no mode is named.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_chain_shot_aligned_default_test.py -v

WHY THIS FILE EXISTS
--------------------
Shot alignment shipped opt-in. The A/B that followed (design §7.2d) could not
show a quality effect either way -- the spread inside each arm swamped the
difference between arms -- but it did measure the operational one: a timeline
with natural timestamps has no fixed-length plan at all (every plan came back
as a boundary-crossing error), and a shot-aligned plan of the same input
returns with its boundaries on the shots. That is what makes it the default
here, and nothing in this file claims anything about quality.

The condition is the TIMELINE, not `requested_segment_frames`. MiniMax-H3 has
no single-inference maximum, so a plan without a requested segment length is one
segment and never chains: a "default only when no length was asked for" rule
would never fire on the one architecture this planner exists for. A requested
length becomes the cap that alignment searches under.

`fixed` by name still means fixed, which is the negative control below.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.arch_capabilities import video_constraints_payload  # noqa: E402
from api.param_defaults import VIDEO_CHAIN_DEFAULTS  # noqa: E402
from core.inference.video_chain_context import (  # noqa: E402
    VideoGridSpec,
    build_segment_geometry,
    build_segment_spans,
    resolve_segment_length_mode,
)

FPS = 24.0
SEGMENT = 124            # MiniMax-H3's shortest clip
CAP = 362                # ... and its longest measured one
TARGET = 500

# P-VC-2's own timestamps (design §7.2d): 00:05.875 / 00:11.000 / 00:16.833 at
# 24 fps. None of them is a fixed-length cut (124 / 247 / 370 / 493).
P_VC_2_TIMESTAMPS = ["00:05.875", "00:11.000", "00:16.833"]
P_VC_2_FRAMES = [141, 264, 404]
FIXED_CUTS = [124, 247, 370, 493, 616]

H3_PROMPT = (
    "integrated_multimodal_description: [Shot 1] A courier steps off a tram into the rain.\n"
    f"[Shot 2] At {P_VC_2_TIMESTAMPS[0]} The courier checks a paper address under an awning.\n"
    f"[Shot 3] At {P_VC_2_TIMESTAMPS[1]} The courier runs across a flooded crossing.\n"
    f"[Shot 4] At {P_VC_2_TIMESTAMPS[2]} The courier presses a doorbell and waits.\n\n"
    "overall_soundscape: Rain on canvas and passing tyres.\n\n"
    "non_diegetic_music: N/A"
)

# Shots that DO land on reachable cuts with a 362-frame cap (124 and 247).
ALIGNABLE_PROMPT = (
    "integrated_multimodal_description: [Shot 1] A lighthouse beam sweeps the bay.\n"
    "[Shot 2] At 00:05.167 The keeper climbs the stair.\n"
    "[Shot 3] At 00:10.292 The keeper opens the lamp room door.\n\n"
    "overall_soundscape: Wind and surf.\n\nnon_diegetic_music: N/A"
)

FREE_FORM_PROMPT = "a courier runs through the rain and rings a doorbell"


def _grid() -> VideoGridSpec:
    return VideoGridSpec.from_video_constraints(video_constraints_payload()["minimax_h3"])


# --------------------------------------------------------------------------
# 1. The resolution rule itself
# --------------------------------------------------------------------------

def test_the_wire_default_is_unset_not_a_mode():
    """`None` has to survive to the planner: it is not the same as `fixed`."""
    assert VIDEO_CHAIN_DEFAULTS["segment_length_mode"] is None
    assert VIDEO_CHAIN_DEFAULTS["manifest_segment_length_mode"] == "fixed"


@pytest.mark.parametrize(
    "shots,expected",
    [
        ([], "fixed"),                      # free-form: nothing to align to
        ([0], "fixed"),                     # one shot, and it starts at 0
        ([0, TARGET], "fixed"),             # the clip end is not a boundary
        ([0] + P_VC_2_FRAMES, "shot_aligned"),
        ([264], "shot_aligned"),
    ],
)
def test_an_unset_mode_follows_the_timeline(shots, expected):
    assert resolve_segment_length_mode(None, shots, TARGET) == expected


@pytest.mark.parametrize("named", ["fixed", "shot_aligned"])
def test_a_named_mode_wins_in_both_directions(named):
    assert resolve_segment_length_mode(named, [0] + P_VC_2_FRAMES, TARGET) == named
    assert resolve_segment_length_mode(named, [], TARGET) == named


# --------------------------------------------------------------------------
# 2. NEGATIVE CONTROL: an explicit `fixed` is byte-identical to the shipped plan
# --------------------------------------------------------------------------

@pytest.mark.parametrize("segment_frames", [None, SEGMENT, 200, CAP])
@pytest.mark.parametrize("target", [300, 500, 1000])
def test_named_fixed_ignores_the_shots_entirely(segment_frames, target):
    """Same spans, field for field, as the length planner that predates alignment."""
    grid = _grid()
    shots = [0] + [f for f in P_VC_2_FRAMES if f < target]
    geometry = build_segment_geometry(grid, target, segment_frames, None, 1, "fixed", shots)
    assert geometry.segment_length_mode == "fixed"
    assert [s.to_dict() for s in geometry.spans] == [
        s.to_dict() for s in build_segment_spans(grid, target, segment_frames)
    ]


# --------------------------------------------------------------------------
# 3. Over the wire (POST /video-chain/plan), no server started
# --------------------------------------------------------------------------

def _app():
    from fastapi import FastAPI

    import api.routes as routes
    from api.error_handlers import register_error_handlers

    app = FastAPI()
    register_error_handlers(app)
    app.post("/video-chain/plan")(routes.plan_video_chain_route)
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


def test_the_p_vc_2_timeline_plans_without_being_asked_to_align():
    """The measured operational result of §7.2d, now the default.

    The same request used to come back as a plan error naming a shot cut at
    frame 124 -- a boundary the caller cannot know before planning. With the
    cap the run used, all three of its boundaries are reachable.
    """
    status, payload = _plan(requested_segment_frames=CAP)
    assert status == 200, payload
    assert payload["success"] is True, payload["errors"]
    assert payload["manifest"]["segment_length_mode"] == "shot_aligned"
    assert payload["frame_plan"]["segment_length_mode"] == "shot_aligned"
    # The timestamps stay where they were written; the boundaries moved onto them.
    assert [e["start_frame"] for e in payload["manifest"]["events"]] == [0] + P_VC_2_FRAMES
    ends = [s["owned_end_frame"] for s in payload["manifest"]["segments"]]
    assert ends[:3] == P_VC_2_FRAMES
    assert ends != FIXED_CUTS
    assert "3 of 3 segment boundaries" in _aligned_boundaries(payload)


def test_a_cap_too_short_to_hold_a_shot_still_plans_and_discloses_the_splits():
    """At the 124-frame floor no shot fits, so the geometry IS the fixed one.

    What the default changes here is not where the cuts are but who owns them:
    the planner cuts and says which shots it had to split, instead of refusing
    and asking the user for timestamps that avoid boundaries it never showed.
    """
    status, payload = _plan()
    assert status == 200, payload
    assert payload["success"] is True, payload["errors"]
    assert payload["manifest"]["segment_length_mode"] == "shot_aligned"
    assert [s["owned_end_frame"] for s in payload["manifest"]["segments"]] == FIXED_CUTS
    assert any("split across segments" in i["message"] for i in payload["warnings"])


def test_the_default_puts_the_boundaries_on_the_shots_when_the_grid_allows_it():
    status, payload = _plan(root_prompt=ALIGNABLE_PROMPT, requested_segment_frames=CAP)
    assert status == 200, payload
    assert payload["success"] is True, payload["errors"]
    assert payload["manifest"]["segment_length_mode"] == "shot_aligned"
    assert [s["owned_end_frame"] for s in payload["manifest"]["segments"]][:2] == [124, 247]


def test_a_requested_segment_length_is_a_cap_not_the_length_of_every_segment():
    status, payload = _plan(root_prompt=ALIGNABLE_PROMPT, requested_segment_frames=CAP)
    assert status == 200, payload
    spans = [s["generated_span_frames"] for s in payload["manifest"]["segments"]]
    assert all(span <= CAP for span in spans), spans
    # ... and it really is a bound, not the length: the aligned segments differ.
    assert len(set(spans)) > 1, spans


def test_a_free_form_prompt_is_still_planned_at_fixed_lengths():
    """Nothing to align to, so the default resolves the other way -- silently.

    `legacy_repeat` because a prompt with no `[Shot N]` structure has no
    timeline; the point is that the geometry is the shipped one, unchanged.
    """
    status, payload = _plan(root_prompt=FREE_FORM_PROMPT, context_mode="legacy_repeat")
    assert status == 200, payload
    assert payload["success"] is True, payload["errors"]
    assert payload["manifest"]["segment_length_mode"] == "fixed"
    assert [s["owned_end_frame"] for s in payload["manifest"]["segments"]] == FIXED_CUTS
    assert not any("no shot boundary" in i["message"] for i in payload["warnings"])


def test_naming_fixed_reproduces_the_shipped_plan_for_the_same_timeline():
    """NEGATIVE CONTROL over the wire: the refusal that used to be the default."""
    status, payload = _plan(segment_length_mode="fixed")
    assert status == 200, payload
    assert payload["success"] is False
    assert payload["manifest"]["segment_length_mode"] == "fixed"
    assert [s["owned_end_frame"] for s in payload["manifest"]["segments"]] == FIXED_CUTS
    assert any("crosses the boundary" in i["message"] for i in payload["errors"])


def test_null_is_accepted_and_means_the_same_as_omitting_the_field():
    # A frozen root seed: -1 is drawn per plan, and the seeds ride on the
    # segments being compared.
    status, omitted = _plan(root_seed=7)
    status2, explicit_null = _plan(root_seed=7, segment_length_mode=None)
    assert (status, status2) == (200, 200)
    assert explicit_null["frame_plan"] == omitted["frame_plan"]
    # `chain_id` is drawn per plan and the hash covers it, so the comparison is
    # of the planned content.
    assert explicit_null["manifest"]["segments"] == omitted["manifest"]["segments"]
    assert explicit_null["manifest"]["events"] == omitted["manifest"]["events"]


def test_an_unknown_mode_is_still_a_400():
    status, payload = _plan(segment_length_mode="nearest_shot")
    assert status == 400, payload
    assert "segment_length_mode" in payload["error"]


# --------------------------------------------------------------------------
# 4. The structural trade-off the default must not hide (design §7.2d)
# --------------------------------------------------------------------------

def _aligned_boundaries(payload) -> str:
    hits = [
        i["message"] for i in payload["warnings"]
        if "segment boundaries fall on a shot boundary" in i["message"]
    ]
    assert hits, payload["warnings"]
    return hits[0]


def test_the_plan_discloses_how_many_boundaries_landed_on_a_shot():
    status, payload = _plan(root_prompt=ALIGNABLE_PROMPT, requested_segment_frames=CAP)
    assert status == 200, payload
    assert "2 of 2 segment boundaries" in _aligned_boundaries(payload)


def test_a_pin_lowers_the_alignment_rate_and_the_plan_says_so():
    """§7.2d: at overlap != 1 the reachable cuts move, so shot starts drop out.

    The default must not make that silent -- the same disclosure the opt-in mode
    carried is what reports it.
    """
    aligned = _plan(root_prompt=ALIGNABLE_PROMPT, requested_segment_frames=CAP,
                    continuation_mode="pinned_tail", requested_overlap_frames=17)
    assert aligned[0] == 200, aligned[1]
    payload = aligned[1]
    assert payload["manifest"]["segment_length_mode"] == "shot_aligned"
    message = _aligned_boundaries(payload)
    hit, total = message.split("boundaries fall")[0].split()[-4:][0:3:2]
    assert int(hit) < int(total), message
