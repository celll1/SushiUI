"""Chain provenance on a generated video row (design sec.13).

Three claims:

* the four video routes a chain can run on -- segment 0's txt2vid / img2vid /
  ref2vid and every continuation's outpaint/video -- all ACCEPT the provenance
  fields. A field the request shape does not declare is silently dropped
  (CLAUDE.md's Form-parameter trap), so this asserts against the app's own
  generated schema rather than against the source;
* `resolve_chain_provenance` accepts a complete stamp and refuses a partial or
  malformed one, instead of recording provenance that leads nowhere;
* the recording path carries ids/integers/hashes ONLY. The root prompt and the
  canonical timeline are never copied onto a segment's metadata or gallery row
  -- that is what the two hashes are for.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_chain_provenance_test.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND_ROOT = os.path.join(_REPO_ROOT, "backend")
for _path in (_REPO_ROOT, _BACKEND_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from api.error_handlers import ValidationError  # noqa: E402
from api.generation_utils import resolve_chain_provenance  # noqa: E402
from api.param_defaults import VIDEO_CHAIN_PROVENANCE_DEFAULTS  # noqa: E402
from database.models import GeneratedImage  # noqa: E402
from utils.video_utils import _chain_provenance_tags  # noqa: E402

PROVENANCE_KEYS = tuple(VIDEO_CHAIN_PROVENANCE_DEFAULTS)

COMPLETE = {
    "chain_id": "3f1c2b9e-3c8a-4f0a-9a1c-2b7d5f0e4a11",
    "chain_manifest_version": 1,
    "chain_plan_hash": "9" * 64,
    "chain_segment_index": 1,
    "chain_segment_count": 4,
    "chain_global_frame_start": 362,
    "chain_global_frame_end": 723,
    "chain_context_mode": "timeline",
    "chain_root_prompt_hash": "a" * 64,
}


def _request_body_properties(schema, path: str) -> set:
    content = schema["paths"][path]["post"]["requestBody"]["content"]
    body = list(content.values())[0]["schema"]
    if "$ref" in body:
        body = schema["components"]["schemas"][body["$ref"].rsplit("/", 1)[-1]]
    return set(body.get("properties", {}))


@pytest.mark.parametrize("path", [
    "/api/v1/generate/txt2vid",
    "/api/v1/generate/img2vid",
    "/api/v1/generate/ref2vid",
    "/api/v1/generate/outpaint/video",
])
def test_every_chain_capable_route_accepts_provenance(path):
    from main import app

    properties = _request_body_properties(app.openapi(), path)
    assert not [key for key in PROVENANCE_KEYS if key not in properties]


def test_complete_stamp_round_trips():
    assert resolve_chain_provenance(dict(COMPLETE)) == COMPLETE


def test_no_stamp_is_not_a_chain_segment():
    assert resolve_chain_provenance({"prompt": "p"}) == VIDEO_CHAIN_PROVENANCE_DEFAULTS


@pytest.mark.parametrize("raw", [
    # The identifying triple travels together or not at all.
    {"chain_plan_hash": "9" * 64},
    {"chain_id": "c", "chain_plan_hash": "9" * 64},
    {"chain_id": "c", "chain_segment_index": 0},
    # Malformed members.
    {"chain_id": "c", "chain_plan_hash": "not-a-digest", "chain_segment_index": 0},
    {"chain_id": "c", "chain_plan_hash": "9" * 64, "chain_segment_index": -1},
    {"chain_id": "c", "chain_plan_hash": "9" * 64, "chain_segment_index": 4,
     "chain_segment_count": 4},
    {"chain_id": "c", "chain_plan_hash": "9" * 64, "chain_segment_index": 0,
     "chain_global_frame_start": 10, "chain_global_frame_end": 10},
    {"chain_id": "c", "chain_plan_hash": "9" * 64, "chain_segment_index": 0,
     "chain_context_mode": "whatever"},
])
def test_partial_or_malformed_stamp_is_refused(raw):
    with pytest.raises(ValidationError):
        resolve_chain_provenance(raw)


def test_video_metadata_carries_ids_only():
    params = dict(COMPLETE, prompt="a long root prompt", canonical_timeline={"events": []})
    tags = _chain_provenance_tags(params)
    assert set(tags) == set(PROVENANCE_KEYS)
    assert "prompt" not in tags and "canonical_timeline" not in tags
    assert _chain_provenance_tags({"prompt": "p"}) == {}


def test_gallery_row_exposes_the_stamp_without_the_manifest():
    row = GeneratedImage(filename="chain.mp4", prompt="segment 1 prompt",
                         negative_prompt="",
                         parameters=dict(COMPLETE, is_video=True,
                                         canonical_timeline={"events": []}))
    result = row.to_dict()
    assert result["chain_segment_index"] == "1"
    assert result["chain_plan_hash"] == "9" * 64
    # The compiled segment prompt is the row's prompt; the root prompt and the
    # timeline are reachable only through the manifest the hashes name.
    assert result["prompt"] == "segment 1 prompt"
    assert "root_prompt" not in result and "canonical_timeline" not in result

    plain = GeneratedImage(filename="plain.mp4", prompt="p", negative_prompt="",
                           parameters={"is_video": True}).to_dict()
    assert not [key for key in PROVENANCE_KEYS if key in plain]
