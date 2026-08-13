"""The chain-context capability is the ONE source of what a chain can be given.

Design sec.7.1/7.7 of the video chain context design. Three claims:

* the served `chain_context` block matches `openapi.yaml`'s
  `ChainContextCapability` -- required keys, types, nullability, and the mode
  enum -- so a client typed off the spec reads what the backend actually sends;
* it never advertises a continuation mode that is not implemented
  (`video_chain_context.CONTINUATION_MODES` is the ceiling). The wire enum is
  wider on purpose: the Phase-B candidates are named so they can be refused BY
  NAME, and naming one here would be promising a mode no code implements;
* `/video-chain/plan` refuses an unadvertised mode with a 400 derived from THIS
  table, for the architecture-level and the per-variant entry alike. A second
  hardcoded list at the route is what this test exists to prevent.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_chain_capability_test.py -v
"""

from __future__ import annotations

import os
import sys

import pytest
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND_ROOT = os.path.join(_REPO_ROOT, "backend")
for _path in (_REPO_ROOT, _BACKEND_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from api.arch_capabilities import (  # noqa: E402
    chain_context_for, chain_context_payload, video_constraints_payload,
)
from api.param_defaults import VIDEO_CHAIN_DEFAULTS  # noqa: E402
from core.inference.video_chain_context import CONTINUATION_MODES  # noqa: E402

_SPEC_PATH = os.path.join(_REPO_ROOT, "openapi.yaml")
_JSON_TYPES = {"integer": int, "boolean": bool, "string": str, "array": list, "object": dict}


@pytest.fixture(scope="module")
def variant_schema():
    with open(_SPEC_PATH, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    return spec["components"]["schemas"]["ChainContextVariantCapability"]


def _entries(payload):
    """Every (label, entry) pair: architecture-level and per-variant."""
    for arch, entry in payload.items():
        yield arch, entry
        for name, variant in (entry.get("variants") or {}).items():
            yield f"{arch}/{name}", variant


def test_payload_matches_openapi_schema(variant_schema):
    payload = chain_context_payload()
    assert payload, "no architecture declares a chain context"
    properties = variant_schema["properties"]
    for label, entry in _entries(payload):
        for key in variant_schema["required"]:
            assert key in entry, f"{label}: missing {key}"
        extra = set(entry) - set(properties) - {"variants"}
        assert not extra, f"{label}: keys absent from the schema: {sorted(extra)}"
        for key, value in entry.items():
            if key == "variants":
                continue
            spec = properties[key]
            if value is None:
                assert spec.get("nullable") is True, f"{label}.{key}: null but not nullable"
                continue
            expected = _JSON_TYPES[spec["type"]]
            assert isinstance(value, expected) and not (
                expected is int and isinstance(value, bool)
            ), f"{label}.{key}: {value!r} is not {spec['type']}"
            if spec["type"] == "array":
                allowed = spec["items"]["enum"]
                assert all(item in allowed for item in value), f"{label}.{key}: {value} outside enum"
            if "minimum" in spec and isinstance(value, int):
                assert value >= spec["minimum"], f"{label}.{key}: {value} below minimum"


def test_only_implemented_modes_are_advertised():
    for label, entry in _entries(chain_context_payload()):
        modes = entry["chain_continuation_modes"]
        assert modes, f"{label}: advertises no mode at all"
        unimplemented = [m for m in modes if m not in CONTINUATION_MODES]
        assert not unimplemented, f"{label}: advertises unimplemented {unimplemented}"
        assert entry["chain_default_continuation_mode"] in modes, f"{label}: default outside its own list"


def test_default_mode_comes_from_param_defaults():
    """The default is `VIDEO_CHAIN_DEFAULTS`, not a second literal."""
    requested = VIDEO_CHAIN_DEFAULTS["continuation_mode"]
    for label, entry in _entries(chain_context_payload()):
        if requested in entry["chain_continuation_modes"]:
            assert entry["chain_default_continuation_mode"] == requested, label


def test_declared_for_exactly_the_video_architectures():
    assert sorted(chain_context_payload()) == sorted(video_constraints_payload())


def test_context_frames_land_on_vae_group_boundaries():
    """A context length that splits a latent frame cannot be conditioned on.

    The valid lengths are the cumulative sums of the arch's
    `latent_chunk_pattern`; an architecture that declares no pattern has no
    boundaries to check against.
    """
    constraints = video_constraints_payload()
    for label, entry in _entries(chain_context_payload()):
        pattern = constraints[label.split("/")[0]]["latent_chunk_pattern"]
        if not pattern:
            continue
        aligned, cursor = set(), 0
        for index in range(64):
            cursor += pattern[index % len(pattern)]
            aligned.add(cursor)
        for key in ("chain_context_min_frames", "chain_context_max_frames"):
            value = entry[key]
            if value is None:
                continue
            assert value in aligned, f"{label}.{key}={value} is not a VAE group boundary"


def test_variant_lookup_falls_back_to_the_architecture_entry():
    arch = chain_context_payload()["minimax_h3"]
    # fl2va has no entry of its own, so it answers with the arch-level one.
    assert chain_context_for("minimax_h3", "fl2va") == arch
    assert chain_context_for("minimax_h3", "REF2VA") == arch["variants"]["ref2va"]
    # ref2va is the one variant with an entry of its own: the preserved clip's
    # tail becomes an automatic video reference there (build_outpaint_references).
    assert chain_context_for("minimax_h3", "ref2va")["chain_supports_reference_video"] is True
    assert chain_context_for("minimax_h3")["chain_supports_reference_video"] is False
    assert chain_context_for("sdxl") is None


def _app():
    """The two planning routes on a bare app (same harness as the sibling
    video-chain tests): they load no model and touch no database."""
    from fastapi import FastAPI

    import api.routes as routes
    from api.error_handlers import register_error_handlers

    app = FastAPI()
    register_error_handlers(app)
    app.post("/video-chain/plan")(routes.plan_video_chain_route)
    return app


def _post(payload: dict):
    import asyncio

    import httpx

    async def run():
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/video-chain/plan", json=payload)
            return response.status_code, response.json()

    return asyncio.run(run())


def _plan_body(**overrides):
    body = {
        "architecture": "minimax_h3",
        "root_prompt": "a cat walks across a sunlit room",
        "target_frames": 700,
        "fps": 24.0,
        "context_mode": "legacy_repeat",
        "continuation_mode": "boundary_frame",
    }
    body.update(overrides)
    return body


@pytest.mark.parametrize("mode", ["motion_preroll", "tail_reference_video",
                                  "sampler_state", "not_a_mode"])
@pytest.mark.parametrize("variant", [None, "ref2va"])
def test_unadvertised_mode_is_refused(mode, variant):
    body = _plan_body(continuation_mode=mode)
    if variant:
        body["variant"] = variant
    status, payload = _post(body)
    assert status == 400, payload
    # The refusal names the capability it came from, so the client knows where
    # to read the answer instead of guessing another mode.
    assert "chain_context" in payload["detail"]


@pytest.mark.parametrize("architecture,segment_frames", [("minimax_h3", None), ("ltx2", 121)])
def test_advertised_default_is_accepted(architecture, segment_frames):
    default = chain_context_payload()[architecture]["chain_default_continuation_mode"]
    body = _plan_body(architecture=architecture, continuation_mode=default,
                      target_frames=400)
    if segment_frames:
        body["requested_segment_frames"] = segment_frames
    status, payload = _post(body)
    assert status == 200, payload
