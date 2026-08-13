"""`plan_hash` identifies the CONTENT of a plan, `chain_id` identifies the run.

Re-planning the same request used to produce a different hash every time,
because `chain_id` (a fresh uuid per call) was part of the hashed payload. Under
`seed_policy: "derived"` -- seeds are a function OF the hash -- that made a
chain impossible to reproduce even with an explicit `root_seed`. Pinned here,
together with the properties the fix must not trade away: `/validate` still
round-trips the hash unchanged, and a real content edit still moves it.

Also pinned: a plan that fails hard returns a geometry-only manifest, and that
manifest still carries a real `root_prompt_hash`.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_chain_plan_hash_test.py -v
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND_ROOT = os.path.join(_REPO_ROOT, "backend")
for _path in (_REPO_ROOT, _BACKEND_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.inference.video_chain_context import derive_segment_seed  # noqa: E402

FPS = 25.0
TARGET_FRAMES = 300
SEGMENT_FRAMES = 158

I2VA_INSTRUCTION = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)
PROMPT = (
    f"{I2VA_INSTRUCTION}\n\n"
    "integrated_multimodal_description: [Shot 1] The woman walks along the pier. "
    "[Shot 2] At 00:06.320 she stops and looks at the sea.\n\n"
    "overall_soundscape: waves against the pilings\n\n"
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


def _post(path: str, payload: dict) -> dict:
    import httpx

    async def run():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(path, json=payload)
            return response.status_code, response.json()

    status, body = asyncio.run(run())
    assert status == 200, body
    return body


def _request(**overrides) -> dict:
    payload = {
        "architecture": "minimax_h3",
        "variant": "i2va",
        "root_prompt": PROMPT,
        "target_frames": TARGET_FRAMES,
        "fps": FPS,
        "requested_segment_frames": SEGMENT_FRAMES,
        "seed_policy": "fixed",
        "root_seed": 7,
    }
    payload.update(overrides)
    return payload


def _plan(**overrides) -> dict:
    body = _post("/video-chain/plan", _request(**overrides))
    assert body["success"], body["errors"]
    return body["manifest"]


# --------------------------------------------------------------------------
# the same request plans to the same hash
# --------------------------------------------------------------------------


def test_replanning_the_same_request_reproduces_the_plan_hash():
    first = _plan()
    second = _plan()
    assert first["plan_hash"] == second["plan_hash"]
    # ... and the run identifier really did change, so this is not a stale-cache
    # tautology.
    assert first["chain_id"] != second["chain_id"]


def test_a_supplied_chain_id_does_not_move_the_hash():
    """The one field the fix removed, exercised directly through /validate."""
    manifest = _plan()
    renamed = copy.deepcopy(manifest)
    renamed["chain_id"] = "00000000-0000-4000-8000-000000000000"
    result = _post("/video-chain/validate", {"manifest": renamed, "recompute_plan_hash": True})
    assert result["valid"], result["errors"]
    assert result["manifest"]["plan_hash"] == manifest["plan_hash"]
    assert result["manifest"]["chain_id"] == "00000000-0000-4000-8000-000000000000"


def test_derived_seeds_reproduce_across_plans():
    first = _plan(seed_policy="derived")
    second = _plan(seed_policy="derived")
    seeds = [s["seed"] for s in first["segments"]]
    assert seeds == [s["seed"] for s in second["segments"]]
    assert seeds == [
        derive_segment_seed(7, first["plan_hash"], i) for i in range(len(seeds))
    ]
    # `derived` must still spread the segments apart; equality above would be
    # vacuous if every segment got the same number.
    assert len(set(seeds)) == len(seeds)


# --------------------------------------------------------------------------
# properties the fix must not trade away
# --------------------------------------------------------------------------


def test_validate_round_trip_keeps_the_hash():
    manifest = _plan()
    result = _post("/video-chain/validate", {"manifest": manifest, "recompute_plan_hash": True})
    assert result["valid"], result["errors"]
    assert result["manifest"]["plan_hash"] == manifest["plan_hash"]


def test_content_changes_still_move_the_hash():
    base = _plan()["plan_hash"]
    edited_prompt = PROMPT.replace("walks along the pier", "walks slowly along the pier")
    variants = {
        "prompt": _plan(root_prompt=edited_prompt)["plan_hash"],
        "root_seed": _plan(root_seed=8)["plan_hash"],
        "seed_policy": _plan(seed_policy="derived")["plan_hash"],
        "target_frames": _plan(target_frames=280)["plan_hash"],
        "negative_prompt": _plan(negative_prompt="blurry, low quality")["plan_hash"],
    }
    for name, digest in variants.items():
        assert digest != base, f"{name} left the plan hash unchanged"
    assert len(set(variants.values())) == len(variants)


def test_a_segment_prompt_edit_still_moves_the_hash():
    manifest = _plan()
    edited = copy.deepcopy(manifest)
    edited["segments"][0]["prompt"] += " The gulls circle overhead."
    result = _post("/video-chain/validate", {"manifest": edited, "recompute_plan_hash": True})
    assert result["manifest"]["plan_hash"] != manifest["plan_hash"]


# --------------------------------------------------------------------------
# the failure path still describes its own root prompt
# --------------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_root_prompt_hash_is_filled_on_a_planned_manifest():
    assert _plan()["root_prompt_hash"] == _sha256(PROMPT)


def test_root_prompt_hash_is_filled_on_the_geometry_only_manifest():
    """A shot crossing a segment boundary is a hard error (no silent split), and
    the geometry-only manifest returned with it must still be identifiable."""
    body = _post(
        "/video-chain/plan",
        _request(
            context_mode="manual",
            canonical_timeline={
                "events": [
                    {
                        "id": "e1",
                        "kind": "action",
                        "start_frame": 0,
                        "end_frame": 120,
                        "description": "The woman walks along the pier.",
                    },
                    {
                        "id": "e2",
                        "kind": "action",
                        "start_frame": 120,
                        "end_frame": 260,
                        "description": "She stops at the rail and looks out.",
                    },
                ]
            },
        ),
    )
    assert not body["success"], "expected the boundary-crossing shot to be refused"
    manifest = body["manifest"]
    assert manifest["segments"] and all(s["prompt"] == "" for s in manifest["segments"])
    assert manifest["root_prompt_hash"] == _sha256(PROMPT)
