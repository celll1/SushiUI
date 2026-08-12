"""Reference-token integrity across plan -> edit -> validate.

Two claims are pinned here, both about the ONE thing a token means:

* a `<Picture N>` inside a mode alignment instruction (i2va / l2va) is the
  mode's own keyframe, not a manifest reference. It must not imply a binding
  (`derive_token_bindings`) and must not be validated as reference usage. Both
  sides go through `split_alignment_instruction`, so there is one exemption
  rule, not two;
* every token a segment prompt carries must be inside that segment's LOCAL
  numbering (1..number of bound references of that kind), which is what
  `/video-chain/validate` checks after a plan-editor edit. Local numbering is
  the only numbering a manifest prompt has -- the root prompt's numbers were
  rewritten at plan time.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_chain_reference_token_test.py -v
"""

from __future__ import annotations

import asyncio
import copy
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND_ROOT = os.path.join(_REPO_ROOT, "backend")
for _path in (_REPO_ROOT, _BACKEND_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.inference.video_chain_context import (  # noqa: E402
    ChainPlanRequest,
    ChainReference,
    VideoGridSpec,
    plan_h3_chain_from_prompt,
    plan_video_chain_manifest,
    split_alignment_instruction,
)

H3_GRID = VideoGridSpec(
    frame_multiple=17,
    frame_offset=5,
    min_frames=22,
    min_decodable_frames=22,
    max_frames=None,
)

FPS = 25.0
TARGET_FRAMES = 300
SEGMENT_FRAMES = 158  # on grid (17*9+5) and 00:06.320 at 25 fps, so shot 2 starts
# exactly on the segment boundary and no event crosses it.

I2VA_INSTRUCTION = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)


def _i2va_prompt(shot_two_text: str) -> str:
    return (
        f"{I2VA_INSTRUCTION}\n\n"
        "integrated_multimodal_description: [Shot 1] The woman in <Picture 1> walks "
        f"along the pier. [Shot 2] At 00:06.320 {shot_two_text}\n\n"
        "overall_soundscape: waves against the pilings\n\n"
        "non_diegetic_music: N/A"
    )


def _plan(prompt: str, segment_indices):
    return plan_h3_chain_from_prompt(
        prompt=prompt,
        mode="i2va",
        grid=H3_GRID,
        fps=FPS,
        target_frames=TARGET_FRAMES,
        segment_frames=SEGMENT_FRAMES,
        references=[
            ChainReference(
                id="ref1",
                kind="image",
                label="woman",
                token="<Picture 1>",
                segment_indices=segment_indices,
            )
        ],
        root_seed=1,
    )


# --------------------------------------------------------------------------
# core: the alignment instruction is not reference usage
# --------------------------------------------------------------------------


def test_instruction_picture_token_does_not_widen_a_narrow_binding():
    # i2va repeats `<Picture 1>` in EVERY segment's alignment instruction. Only
    # segment 1's body mentions the reference, so the binding must stay [0].
    manifest = _plan(_i2va_prompt("She stops and looks at the sea."), [0])
    assert len(manifest.segments) == 2
    assert manifest.references[0].segment_indices == [0]
    assert manifest.references[0].binding_source == "explicit"
    assert not [w for w in manifest.warnings if "so the sentence stays intact" in w]
    assert manifest.segments[1].reference_ids == []


def test_a_token_in_the_body_still_widens_the_binding():
    # The exemption is the instruction only: a token the user put in the shot
    # text still binds the reference to that segment (commit ae5643cf).
    manifest = _plan(_i2va_prompt("The woman in <Picture 1> boards the ferry."), [0])
    assert manifest.references[0].segment_indices == [0, 1]
    assert manifest.references[0].binding_source == "token_implied"
    assert [w for w in manifest.warnings if "so the sentence stays intact" in w]


def _legacy_repeat_plan(body: str, segment_indices):
    return plan_video_chain_manifest(
        ChainPlanRequest(
            architecture="minimax_h3",
            root_prompt=(
                f"{I2VA_INSTRUCTION}\n\n"
                f"integrated_multimodal_description: [Shot 1] {body}\n\n"
                "overall_soundscape: waves against the pilings\n\n"
                "non_diegetic_music: N/A"
            ),
            grid=H3_GRID,
            fps=FPS,
            target_frames=TARGET_FRAMES,
            variant="i2va",
            segment_frames=SEGMENT_FRAMES,
            context_mode="legacy_repeat",
            root_seed=1,
            references=[
                ChainReference(
                    id="ref1",
                    kind="image",
                    label="woman",
                    token="<Picture 1>",
                    segment_indices=segment_indices,
                )
            ],
        )
    )


def test_legacy_repeat_applies_the_same_instruction_exemption():
    # legacy_repeat resends the root prompt verbatim, instruction included, so
    # without the exemption its `<Picture 1>` would widen an explicitly narrow
    # binding to every segment -- the asymmetry with the timeline path above.
    manifest = _legacy_repeat_plan("The woman walks along the pier.", [0])
    assert len(manifest.segments) == 2
    assert manifest.references[0].segment_indices == [0]
    assert manifest.references[0].binding_source == "explicit"
    assert manifest.segments[1].reference_ids == []


def test_legacy_repeat_still_widens_on_a_token_in_the_body():
    manifest = _legacy_repeat_plan("The woman in <Picture 1> walks along the pier.", [0])
    assert manifest.references[0].segment_indices == [0, 1]
    assert manifest.references[0].binding_source == "token_implied"


def test_split_alignment_instruction_only_matches_the_head():
    body = "integrated_multimodal_description: [Shot 1] x"
    head, rest = split_alignment_instruction(f"{I2VA_INSTRUCTION}\n\n{body}", I2VA_INSTRUCTION)
    assert head == I2VA_INSTRUCTION and rest == body
    # A prompt that does not open with the instruction is body in full.
    assert split_alignment_instruction(body, I2VA_INSTRUCTION) == ("", body)
    # No instruction for this mode: nothing is exempt.
    assert split_alignment_instruction(body, "") == ("", body)


# --------------------------------------------------------------------------
# the routes: plan -> edit -> validate
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


def _plan_request(prompt: str, segment_indices):
    return {
        "architecture": "minimax_h3",
        "variant": "i2va",
        "root_prompt": prompt,
        "target_frames": TARGET_FRAMES,
        "fps": FPS,
        "requested_segment_frames": SEGMENT_FRAMES,
        "root_seed": 1,
        "references": [
            {
                "id": "ref1",
                "kind": "image",
                "label": "woman",
                "token": "<Picture 1>",
                "segment_indices": segment_indices,
            }
        ],
    }


def _planned(prompt: str, segment_indices):
    body = _post("/video-chain/plan", _plan_request(prompt, segment_indices))
    assert body["success"], body["errors"]
    return body["manifest"]


def _validate(manifest: dict) -> dict:
    return _post("/video-chain/validate", {"manifest": manifest, "recompute_plan_hash": True})


def _codes(result: dict):
    return [issue["code"] for issue in result["errors"]]


def test_unedited_plan_validates_and_keeps_its_plan_hash():
    manifest = _planned(_i2va_prompt("She stops and looks at the sea."), [0])
    result = _validate(manifest)
    assert result["valid"], result["errors"]
    assert result["plan_hash"] == manifest["plan_hash"]
    # Idempotent: validating the returned manifest again is a fixed point.
    again = _validate(result["manifest"])
    assert again["valid"] and again["plan_hash"] == manifest["plan_hash"]


def test_restored_instruction_token_is_not_reported_as_unbound():
    # Segment 2 has NO reference bound, yet its prompt opens with the i2va
    # instruction and its `<Picture 1>`. That must not be an error.
    manifest = _planned(_i2va_prompt("She stops and looks at the sea."), [0])
    assert manifest["segments"][1]["reference_ids"] == []
    assert I2VA_INSTRUCTION in manifest["segments"][1]["prompt"]
    result = _validate(manifest)
    assert result["valid"], result["errors"]


def test_editing_in_an_unbound_token_is_an_error():
    manifest = _planned(_i2va_prompt("She stops and looks at the sea."), [0])
    edited = copy.deepcopy(manifest)
    # Segment 1 has exactly one image reference, so `<Picture 2>` names nothing.
    edited["segments"][0]["prompt"] = edited["segments"][0]["prompt"].replace(
        "walks along the pier", "walks along the pier towards <Picture 2>"
    )
    result = _validate(edited)
    assert not result["valid"]
    assert "reference_token_not_bound" in _codes(result)
    assert result["errors"][0]["segment_index"] == 0


def test_a_token_in_a_segment_with_no_reference_bound_is_an_error():
    manifest = _planned(_i2va_prompt("She stops and looks at the sea."), [0])
    edited = copy.deepcopy(manifest)
    edited["segments"][1]["prompt"] += " The woman in <Picture 1> waves."
    result = _validate(edited)
    assert not result["valid"]
    assert "reference_token_not_bound" in _codes(result)
    assert result["errors"][0]["segment_index"] == 1


def test_binding_the_reference_there_makes_the_same_edit_valid():
    manifest = _planned(_i2va_prompt("She stops and looks at the sea."), [0])
    edited = copy.deepcopy(manifest)
    edited["segments"][1]["prompt"] += " The woman in <Picture 1> waves."
    edited["references"][0]["segment_indices"] = [0, 1]
    edited["segments"][1]["reference_ids"] = ["ref1"]
    result = _validate(edited)
    assert result["valid"], result["errors"]
    # An edited prompt is a different plan, so the hash must move.
    assert result["plan_hash"] != manifest["plan_hash"]


def test_prose_only_edits_stay_valid():
    manifest = _planned(_i2va_prompt("She stops and looks at the sea."), [0])
    edited = copy.deepcopy(manifest)
    edited["segments"][0]["prompt"] = edited["segments"][0]["prompt"].replace(
        "walks along the pier", "walks slowly along the pier"
    )
    result = _validate(edited)
    assert result["valid"], result["errors"]
