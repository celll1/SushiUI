"""The seed a chain segment RUNS with is the manifest's, not a fresh draw.

`root_seed: -1` is resolved to a concrete value exactly once, at plan time, and
`segments[].seed` is frozen from it (design sec.8). Two ways that freeze can be
undone, both pinned here:

* the request path sends the panel's raw seed instead of the segment's frozen
  one, so `-1` makes every segment draw its own random noise in the backend and
  the seed the plan editor showed is not the seed that ran. The frontend side of
  the chain has no test runner in this repo, so the three send sites are pinned
  against the shipping sources (same approach as `video_chain_context_test`'s
  frontend-planner parity and `video_chain_lora_test`'s sender checks);
* `/video-chain/validate` recomputes `plan_hash` after an edit but leaves the
  seeds alone, which under `seed_policy: "derived"` (seeds are a function OF the
  hash) hands back a manifest whose seeds no longer match its own plan.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_chain_seed_freeze_test.py -v
"""

from __future__ import annotations

import asyncio
import copy
import os
import re
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


def _planned(seed_policy: str, root_seed: int) -> dict:
    body = _post(
        "/video-chain/plan",
        {
            "architecture": "minimax_h3",
            "variant": "i2va",
            "root_prompt": PROMPT,
            "target_frames": TARGET_FRAMES,
            "fps": FPS,
            "requested_segment_frames": SEGMENT_FRAMES,
            "seed_policy": seed_policy,
            "root_seed": root_seed,
        },
    )
    assert body["success"], body["errors"]
    return body["manifest"]


def _validate(manifest: dict, recompute: bool = True) -> dict:
    body = _post(
        "/video-chain/validate",
        {"manifest": manifest, "recompute_plan_hash": recompute},
    )
    assert body["valid"], body["errors"]
    return body["manifest"]


# --------------------------------------------------------------------------
# plan: -1 is resolved once and frozen
# --------------------------------------------------------------------------


def test_plan_freezes_a_concrete_seed_per_segment():
    manifest = _planned("fixed", -1)
    assert manifest["root_seed"] >= 0
    assert len(manifest["segments"]) == 2
    assert [s["seed"] for s in manifest["segments"]] == [manifest["root_seed"]] * 2


def test_derived_seeds_are_a_function_of_the_plan_hash():
    manifest = _planned("derived", 7)
    expected = [
        derive_segment_seed(7, manifest["plan_hash"], i)
        for i in range(len(manifest["segments"]))
    ]
    assert [s["seed"] for s in manifest["segments"]] == expected
    assert len(set(expected)) == len(expected)


# --------------------------------------------------------------------------
# validate: derived seeds are re-derived alongside the hash
# --------------------------------------------------------------------------


def test_editing_a_derived_plan_re_derives_its_seeds():
    manifest = _planned("derived", 7)
    edited = copy.deepcopy(manifest)
    edited["segments"][0]["prompt"] = edited["segments"][0]["prompt"].replace(
        "walks along the pier", "walks slowly along the pier"
    )
    result = _validate(edited)
    assert result["plan_hash"] != manifest["plan_hash"]
    assert [s["seed"] for s in result["segments"]] == [
        derive_segment_seed(7, result["plan_hash"], i)
        for i in range(len(result["segments"]))
    ]
    # i.e. the stale seeds really did move; this is not a no-op assertion.
    assert [s["seed"] for s in result["segments"]] != [
        s["seed"] for s in manifest["segments"]
    ]


def test_seeds_are_re_derived_even_when_the_hash_is_not_recomputed():
    manifest = _planned("derived", 7)
    tampered = copy.deepcopy(manifest)
    for segment in tampered["segments"]:
        segment["seed"] = 0
    result = _validate(tampered, recompute=False)
    assert [s["seed"] for s in result["segments"]] == [
        derive_segment_seed(7, manifest["plan_hash"], i)
        for i in range(len(result["segments"]))
    ]


def test_explicit_per_segment_seeds_survive_validation():
    manifest = _planned("derived", 7)
    edited = copy.deepcopy(manifest)
    edited["seed_policy"] = "explicit"
    for index, segment in enumerate(edited["segments"]):
        segment["seed"] = 1000 + index
    result = _validate(edited)
    assert [s["seed"] for s in result["segments"]] == [1000, 1001]


def test_an_unresolved_root_seed_is_never_drawn_at_validate_time():
    manifest = _planned("derived", 7)
    edited = copy.deepcopy(manifest)
    edited["root_seed"] = -1
    for segment in edited["segments"]:
        segment["seed"] = 4242
    result = _validate(edited)
    assert [s["seed"] for s in result["segments"]] == [4242, 4242]


# --------------------------------------------------------------------------
# the frontend send path (no TS test runner in this repo: source pins)
# --------------------------------------------------------------------------


def _read(*parts: str) -> str:
    with open(os.path.join(_REPO_ROOT, "frontend", "src", *parts), encoding="utf-8") as handle:
        return handle.read()


def _chain_start(source: str) -> str:
    """`handleVideoChainStart` only -- Choice 1 (`handleVideoChainGenerateAtCap`)
    builds a `cappedParams` of its own and is not a chain."""
    start = source.index("const handleVideoChainStart =")
    return source[start : source.index("\n  };", start)]


def test_video_chain_module_sends_the_manifest_seed():
    source = _read("utils", "videoChain.ts")
    # The helper falls back to the caller's own seed, which is what keeps the
    # legacy (no-manifest) path byte-identical to before.
    assert "export const segmentChainSeed" in source
    assert re.search(
        r"manifest\?\.segments\.find\(\(s\) => s\.index === segmentIndex\)\?\.seed \?\? fallback",
        source,
    )
    # Continuation requests send it; `base.seed` remains the fallback.
    assert "seed: segmentSeed ?? base.seed," in source
    assert "segmentChainSeed(args.manifest, segmentIndex, args.continuationBase.seed)" in source


def test_both_panels_send_the_manifest_seed_for_segment_zero():
    txt2img = _read("components", "generation", "Txt2ImgPanel.tsx")
    assert "segmentChainSeed" in txt2img
    started = _chain_start(txt2img)
    assert "seed: segmentChainSeed(manifest, 0, videoParams.seed)," in started

    img2img = _chain_start(_read("components", "generation", "Img2ImgPanel.tsx"))
    assert "const mainSeed = segmentChainSeed(manifest, 0, base.seed);" in img2img
    # Both segment-0 branches (ref2vid and img2vid) carry it.
    assert img2img.count("seed: mainSeed,") == 2
