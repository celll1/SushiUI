"""The motion pre-roll in the PLAN (design §7.3), end to end over HTTP.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_chain_motion_preroll_plan_test.py -q

The generation-side counterparts (exact prefix, discard, anchor placement, the
refusals) are in `minimax_h3_motion_preroll_test.py`. This file is about the
manifest: that it fixes the pre-roll length, the anchor count and the anchor
positions; that its frame ranges are the lengths the chain reaches AFTER the
pre-roll is thrown away; that the extra cost is disclosed rather than implied;
and that a manifest survives `/video-chain/validate` with its `plan_hash`
intact, since a hash that moves on a round trip means the plan was not fixed.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND_ROOT = os.path.join(_REPO_ROOT, "backend")
for _path in (_REPO_ROOT, _BACKEND_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from api.generation_utils import plan_video_outpaint_placement  # noqa: E402
from core.inference.video_chain_context import (  # noqa: E402
    motion_preroll_anchor_frames,
)

PREROLL = 9
ANCHORS = 3
SEGMENT = 124
TARGET = 700


def _app():
    from fastapi import FastAPI

    import api.routes as routes
    from api.error_handlers import register_error_handlers

    app = FastAPI()
    register_error_handlers(app)
    app.post("/video-chain/plan")(routes.plan_video_chain_route)
    app.post("/video-chain/validate")(routes.validate_video_chain_route)
    return app


def _post(path: str, payload: dict):
    import httpx

    async def run():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(path, json=payload)
            return response.status_code, response.json()

    return asyncio.run(run())


def _plan_body(**overrides):
    body = {
        "architecture": "minimax_h3",
        "variant": "fl2va",
        "root_prompt": "a cat walks across a sunlit room",
        "target_frames": TARGET,
        "fps": 24.0,
        "requested_segment_frames": SEGMENT,
        "context_mode": "legacy_repeat",
        "continuation_mode": "motion_preroll",
        "requested_overlap_frames": PREROLL,
        "requested_anchor_count": ANCHORS,
    }
    body.update(overrides)
    return body


@pytest.fixture(scope="module")
def planned():
    status, payload = _post("/video-chain/plan", _plan_body())
    assert status == 200, payload
    assert not payload["errors"], payload["errors"]
    return payload


def test_the_manifest_fixes_the_preroll_the_count_and_the_positions(planned):
    manifest = planned["manifest"]
    assert manifest["continuation_mode"] == "motion_preroll"
    continuations = [s for s in manifest["segments"] if s["index"] > 0]
    assert continuations, "this target should need more than one segment"
    for segment in continuations:
        assert segment["requested_overlap_frames"] == PREROLL
        assert segment["effective_overlap_frames"] == PREROLL
        assert segment["requested_anchor_count"] == ANCHORS
        visual = segment["visual_context"]
        assert visual["mode"] == "motion_preroll"
        assert visual["shared_context_frames"] == PREROLL
        assert visual["anchor_count"] == ANCHORS
        # The positions are the shared arithmetic's, not a second derivation.
        assert visual["anchor_local_frames"] == list(
            motion_preroll_anchor_frames(PREROLL, ANCHORS))
        # Nothing is pinned, so no audio overlap is claimed.
        assert segment["effective_overlap_samples"] == 0
    # Segment 0 has no predecessor and therefore no pre-roll.
    first = manifest["segments"][0]
    assert first["visual_context"]["mode"] == "initial"
    assert first["requested_anchor_count"] == 0


def test_owned_end_frame_is_the_length_after_the_discard(planned):
    """The 17cfdb7a defect for this mode: the plan states the length that comes
    back, not the length the 1-frame-anchor arithmetic would have given.

    Checked against `plan_video_outpaint_placement`, which is what the
    GENERATION solves -- not against a second reading of the planner.
    """
    manifest = planned["manifest"]
    segments = manifest["segments"]
    accumulated = segments[0]["owned_end_frame"]
    for segment in segments[1:]:
        placement = plan_video_outpaint_placement(
            {"total_frames": segment["requested_total_frames"], "input_offset_frames": 0},
            "minimax_h3", head_frames=accumulated, overlap_frames=PREROLL,
        )
        assert placement["generated_frames"] == segment["generated_span_frames"]
        assert placement["total_frames"] == segment["owned_end_frame"]
        # The discard, in the manifest's own numbers.
        assert (segment["owned_end_frame"] - segment["owned_start_frame"]
                == segment["generated_span_frames"] - PREROLL)
        assert segment["anchor_global_frame"] == segment["owned_start_frame"] - PREROLL
        accumulated = segment["owned_end_frame"]
    assert manifest["expected_final_frames"] == segments[-1]["owned_end_frame"]
    assert manifest["expected_final_frames"] >= TARGET


def test_the_plan_discloses_the_frames_it_generates_and_throws_away(planned):
    """The cost is DISCLOSED, in facts: how many frames are dropped per
    continuation and that the anchors ride every step. A mode that costs more
    compute than the output shows may not be silent about it."""
    messages = " ".join(w["message"] for w in planned["warnings"])
    assert "motion_preroll" in messages
    assert "discarded" in messages
    assert f"last {PREROLL} frame(s)" in messages
    assert "conditioning rows to every denoise step" in messages
    # The anchors it named are the ones the manifest fixed.
    assert ", ".join(str(f) for f in motion_preroll_anchor_frames(PREROLL, ANCHORS)) in messages


def test_the_plan_survives_validation_with_the_same_hash(planned):
    """A round trip must not move `plan_hash`: the anchors are part of the plan,
    so a client that sends the manifest back has to get the same identity."""
    status, payload = _post(
        "/video-chain/validate",
        {"manifest": planned["manifest"], "recompute_plan_hash": True},
    )
    assert status == 200, payload
    assert not payload["errors"], payload["errors"]
    assert payload["manifest"]["plan_hash"] == planned["manifest"]["plan_hash"]
    returned = [s for s in payload["manifest"]["segments"] if s["index"] > 0]
    for segment in returned:
        assert segment["visual_context"]["anchor_local_frames"] == list(
            motion_preroll_anchor_frames(PREROLL, ANCHORS))
        assert segment["requested_anchor_count"] == ANCHORS


def test_the_same_plan_input_hashes_the_same_and_a_different_one_does_not():
    """Determinism, and that the anchors are actually IN the identity.

    `root_seed` is pinned because -1 is resolved to a real seed once per plan
    and IS hashed -- two plans from -1 are two different plans by design.
    """
    body = _plan_body(root_seed=1234)
    first = _post("/video-chain/plan", body)[1]["manifest"]["plan_hash"]
    again = _post("/video-chain/plan", body)[1]["manifest"]["plan_hash"]
    assert first == again
    other_count = _post("/video-chain/plan", _plan_body(root_seed=1234,
                                                        requested_anchor_count=2))[1]
    other_preroll = _post("/video-chain/plan", _plan_body(root_seed=1234,
                                                          requested_overlap_frames=5))[1]
    assert other_count["manifest"]["plan_hash"] != first
    assert other_preroll["manifest"]["plan_hash"] != first


def test_the_boundary_frame_plan_is_untouched():
    """NEGATIVE CONTROL: the default plan does not gain any of this."""
    status, payload = _post("/video-chain/plan", _plan_body(
        continuation_mode="boundary_frame", requested_overlap_frames=0,
        requested_anchor_count=0))
    assert status == 200, payload
    for segment in payload["manifest"]["segments"]:
        assert segment["requested_anchor_count"] == 0
        assert segment["effective_overlap_frames"] == 0
        assert segment["visual_context"].get("anchor_local_frames") is None
        if segment["index"] > 0:
            assert (segment["owned_end_frame"] - segment["owned_start_frame"]
                    == segment["generated_span_frames"] - 1)


@pytest.mark.parametrize("body,fragment", [
    (dict(requested_anchor_count=0), "2..4"),
    (dict(requested_anchor_count=9), "2..4"),
    (dict(requested_overlap_frames=1), "2..17"),
    (dict(requested_overlap_frames=18), "2..17"),
    (dict(requested_overlap_frames=2, requested_anchor_count=4), "anchors"),
    (dict(continuation_mode="pinned_tail", requested_overlap_frames=9), "mutually exclusive"),
    (dict(continuation_mode="boundary_frame", requested_overlap_frames=0), "mutually exclusive"),
])
def test_the_plan_refuses_what_the_generation_would_refuse(body, fragment):
    """A plan must not promise a geometry the generation rejects, so both go
    through the same resolver -- including the pin/anchor exclusivity."""
    status, payload = _post("/video-chain/plan", _plan_body(**body))
    assert status == 400, payload
    assert fragment in payload["detail"]


def test_a_negative_anchor_count_is_refused():
    status, payload = _post("/video-chain/plan", _plan_body(requested_anchor_count=-1))
    assert status == 400, payload
    assert "requested_anchor_count" in payload["detail"]
