"""A merged MiniMax-H3 DiT's provenance reaches the gallery row.

``record_model_variant`` writes ``model_hybrid_*`` into the row's ``parameters``
JSON, but ``GeneratedImage.to_dict()`` serves a WHITELIST -- a key it does not
name is stored and never returned. This asserts the two agree, and that a
base-only row grows none of the keys.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_hybrid_gallery_row_test.py -v
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND_ROOT = os.path.join(_REPO_ROOT, "backend")
for _path in (_REPO_ROOT, _BACKEND_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from api.generation_utils import record_model_variant  # noqa: E402
from database.models import GeneratedImage  # noqa: E402

HYBRID_KEYS = (
    "model_hybrid_base",
    "model_hybrid_overlay",
    "model_hybrid_preset",
    "model_hybrid_block_range",
    "model_hybrid_final_adaln_from_overlay",
    "model_hybrid_digest",
    "model_hybrid_quantization",
)


class _FakeManager:
    is_minimax_h3_model = True

    def __init__(self, model_info):
        self.current_model_info = model_info


def _recorded_params(model_info):
    params: dict = {}
    record_model_variant(params, _FakeManager(model_info))
    return params


HYBRID_INFO = {
    "variant": "hybrid",
    "hybrid": {
        "variant": "hybrid",
        "base_variant": "fl2va",
        "overlay_variant": "ref2va",
        "base_file": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "overlay_file": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
        "hybrid_recipe": {
            "preset": "block_range_adaln",
            "block_range_start": 25,
            "block_range_end": 49,
            "final_adaln_from_overlay": False,
        },
        "compatibility_digest": "d" * 64,
        "quantization_format": "fp8_scaled",
    },
}


def test_every_recorded_hybrid_key_survives_to_dict():
    params = _recorded_params(HYBRID_INFO)
    # Derived from the PRODUCER, not asserted against the list above: a key
    # added to record_model_variant and not to the whitelist would otherwise be
    # dropped from every gallery row with both tests still green.
    produced = {key for key in params if key.startswith("model_hybrid_")}
    assert produced == set(HYBRID_KEYS)

    row = GeneratedImage(filename="hybrid.mp4", prompt="p", negative_prompt="",
                         parameters=dict(params, is_video=True)).to_dict()
    # The whitelist and the producer, compared as SETS -- this is the assertion
    # that fails when one of them grows a key and the other does not.
    assert {key for key in row if key.startswith("model_hybrid_")} == produced
    assert row["model_variant"] == "hybrid"
    assert row["model_hybrid_base"] == "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
    assert row["model_hybrid_overlay"] == "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
    assert row["model_hybrid_preset"] == "block_range_adaln"
    assert row["model_hybrid_block_range"] == "25..49"
    assert row["model_hybrid_final_adaln_from_overlay"] is False
    assert row["model_hybrid_digest"] == "d" * 64
    assert row["model_hybrid_quantization"] == "fp8_scaled"


def test_a_single_checkpoint_row_carries_none_of_them():
    params = _recorded_params({"variant": "fl2va"})
    assert params == {"model_variant": "fl2va"}

    row = GeneratedImage(filename="plain.mp4", prompt="p", negative_prompt="",
                         parameters=dict(params, is_video=True)).to_dict()
    assert row["model_variant"] == "fl2va"
    assert [key for key in HYBRID_KEYS if key in row] == []
