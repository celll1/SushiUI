"""MiniMax Music 3 API layer: audit-finding fixes (F1, F4, F5, F7).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_api_audit_fixes_test.py -v

Covers the fixes from the independent phase-4 audit that are not already
exercised by the other `minimax_music3_api_*` test files:

- F1 (both legs): the real `num_inference_steps`/`flow_guidance_scale`
  reach BOTH the generic audio metadata sidecar (`utils.audio_utils.
  save_audio_with_metadata`) AND the gallery row
  (`database.models.GeneratedImage.to_dict`), instead of ACE-Step's inert
  8 / 1.0 defaults silently shadowing them.
- F4: `generation_utils.validate_audio_params` clamps an over-ceiling
  `audio_duration` WITH a warning (not silently), and refuses a
  sub-minimum `num_inference_steps` with a 400 before any GPU work.
- F5: the frame-code sidecar is one of the files `DELETE /images/{id}`
  removes, via `routes._generated_image_file_paths`.
- F7: `resolve_audio_defaults` does not alias a mutable SSOT default
  (`AUDIO_GEN_DEFAULTS["loras"]`) into the request `params` dict.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.error_handlers import ValidationError
from api.generation_utils import resolve_audio_defaults, validate_audio_params
from api.param_defaults import AUDIO_GEN_DEFAULTS
from database.models import GeneratedImage


# ---------------------------------------------------------------------------
# F1, leg 1: the generic audio sidecar (utils.audio_utils)
# ---------------------------------------------------------------------------

def test_save_audio_with_metadata_sidecar_prefers_the_music3_step_and_cfg_keys(tmp_path, monkeypatch):
    from config.settings import settings
    from utils.audio_utils import save_audio_with_metadata

    monkeypatch.setattr(settings, "outputs_dir", str(tmp_path))

    waveform = torch.zeros(2, 44100)
    params = {
        "prompt": "ambient",
        "lyrics": "[verse]\nla",
        "seed": 42,
        # ACE-Step's fields, unconditionally present on every Txt2AudRequest
        # (the exact trap F1 describes: a plain `dict.get(k, fallback)`
        # keyed on these never reaches its fallback arm).
        "inference_steps": 8,
        "guidance_scale": 1.0,
        # The REAL values for this (simulated) MiniMax Music 3 generation.
        "num_inference_steps": 50,
        "flow_guidance_scale": 3.0,
    }
    filename = save_audio_with_metadata(waveform, 44100, params, "txt2aud", model_info=None)
    base_name = os.path.splitext(filename)[0]
    sidecar_path = tmp_path / f"{base_name}.json"
    assert sidecar_path.is_file()

    import json
    with open(sidecar_path, encoding="utf-8") as fh:
        sidecar = json.load(fh)

    assert sidecar["inference_steps"] == 50
    assert sidecar["guidance_scale"] == 3.0


def test_save_audio_with_metadata_sidecar_still_uses_acestep_keys_when_music3_keys_absent(tmp_path, monkeypatch):
    """No-op proof for the F1 fix: an ACE-Step request (no `num_inference_steps`/
    `flow_guidance_scale` keys at all) still records ITS OWN inference_steps/
    guidance_scale, unchanged.
    """
    from config.settings import settings
    from utils.audio_utils import save_audio_with_metadata

    monkeypatch.setattr(settings, "outputs_dir", str(tmp_path))

    waveform = torch.zeros(2, 44100)
    params = {
        "prompt": "synthpop", "lyrics": "", "seed": 1,
        "inference_steps": 8, "guidance_scale": 1.0,
    }
    filename = save_audio_with_metadata(waveform, 44100, params, "txt2aud", model_info=None)
    base_name = os.path.splitext(filename)[0]

    import json
    with open(tmp_path / f"{base_name}.json", encoding="utf-8") as fh:
        sidecar = json.load(fh)
    assert sidecar["inference_steps"] == 8
    assert sidecar["guidance_scale"] == 1.0


def test_save_audio_with_metadata_treats_a_real_zero_acestep_value_as_present(tmp_path, monkeypatch):
    """`is not None`, not truthiness: an ACE-Step `guidance_scale` of 0 (an
    edge value, not a real one, but the dict-lookup logic must not treat it
    as "absent" the way a bare `or`/truthiness check would).
    """
    from config.settings import settings
    from utils.audio_utils import save_audio_with_metadata

    monkeypatch.setattr(settings, "outputs_dir", str(tmp_path))

    waveform = torch.zeros(2, 100)
    params = {
        "prompt": "p", "lyrics": "", "seed": 1,
        "inference_steps": 0, "guidance_scale": 0.0,
    }
    filename = save_audio_with_metadata(waveform, 44100, params, "txt2aud", model_info=None)
    base_name = os.path.splitext(filename)[0]

    import json
    with open(tmp_path / f"{base_name}.json", encoding="utf-8") as fh:
        sidecar = json.load(fh)
    assert sidecar["inference_steps"] == 0
    assert sidecar["guidance_scale"] == 0.0


# ---------------------------------------------------------------------------
# F1, leg 2: the gallery row (database.models.GeneratedImage.to_dict)
# ---------------------------------------------------------------------------

def test_generated_image_to_dict_surfaces_music3_step_and_cfg_fields():
    image = GeneratedImage(
        filename="txt2aud_x.flac",
        parameters={
            "is_audio": True,
            "duration": 12.3,
            "sample_rate": 44100,
            "audio_duration": 60.0,
            "inference_steps": 8,        # ACE-Step's own field; irrelevant here
            "guidance_scale": 1.0,       # ACE-Step's own field; irrelevant here
            "num_inference_steps": 45,
            "flow_guidance_scale": 2.5,
        },
    )
    result = image.to_dict()
    assert result["num_inference_steps"] == "45"
    assert result["flow_guidance_scale"] == "2.5"
    # ACE-Step's own fields are STILL surfaced too -- both pairs coexist,
    # neither overwrites the other.
    assert result["inference_steps"] == "8"
    assert result["guidance_scale"] == "1.0"


def test_generated_image_to_dict_is_a_noop_when_music3_fields_absent():
    image = GeneratedImage(
        filename="txt2aud_y.flac",
        parameters={"is_audio": True, "inference_steps": 8, "guidance_scale": 1.0},
    )
    result = image.to_dict()
    assert "num_inference_steps" not in result
    assert "flow_guidance_scale" not in result
    assert result["inference_steps"] == "8"


# ---------------------------------------------------------------------------
# F4: arch-specific audio bounds enforcement
# ---------------------------------------------------------------------------

def test_validate_audio_params_clamps_an_over_ceiling_duration_with_a_warning():
    params = {"audio_duration": 600.0}
    warnings = validate_audio_params(params, "minimax_music3")
    assert params["audio_duration"] == 360.0
    assert len(warnings) == 1
    assert "360" in warnings[0]
    assert "600" in warnings[0]


def test_validate_audio_params_leaves_an_in_range_duration_untouched():
    params = {"audio_duration": 60.0}
    warnings = validate_audio_params(params, "minimax_music3")
    assert params["audio_duration"] == 60.0
    assert warnings == []


def test_validate_audio_params_has_no_declared_ceiling_for_acestep():
    """ACE-Step has no established `audio_duration` ceiling (design doc's
    360s bound is MiniMax Music 3-specific) -- the same 600.0 that gets
    clamped for Music3 above must pass through UNCHANGED here.
    """
    params = {"audio_duration": 600.0}
    warnings = validate_audio_params(params, "acestep")
    assert params["audio_duration"] == 600.0
    assert warnings == []


def test_validate_audio_params_rejects_zero_inference_steps_before_any_gpu_work():
    params = {"num_inference_steps": 0}
    with pytest.raises(ValidationError):
        validate_audio_params(params, "minimax_music3")


def test_validate_audio_params_rejects_negative_inference_steps():
    params = {"num_inference_steps": -3}
    with pytest.raises(ValidationError):
        validate_audio_params(params, "minimax_music3")


def test_validate_audio_params_accepts_the_minimum_step_count():
    params = {"num_inference_steps": 1}
    validate_audio_params(params, "minimax_music3")  # must not raise


def test_validate_audio_params_ignores_a_none_step_count():
    """`None` is the Pydantic sentinel BEFORE `resolve_audio_defaults` runs
    (or an arch, like ACE-Step, that never populates this key) -- must not
    raise; there is nothing to validate yet.
    """
    params = {"num_inference_steps": None}
    validate_audio_params(params, "minimax_music3")  # must not raise


def test_validate_audio_params_is_a_full_noop_for_an_unknown_arch():
    params = {"audio_duration": 999.0, "num_inference_steps": 0}
    warnings = validate_audio_params(params, None)
    assert warnings == []
    assert params["audio_duration"] == 999.0
    assert params["num_inference_steps"] == 0


# ---------------------------------------------------------------------------
# F5: the frame-code sidecar is removed on delete
# ---------------------------------------------------------------------------

class _FakeImageRow:
    """Everything `routes._generated_image_file_paths` reads off a
    `GeneratedImage` row (`.filename`/`.parameters`) -- a real ORM instance
    works too (see the F1-leg-2 tests above), but this keeps this test from
    depending on any column this function does NOT touch.
    """

    def __init__(self, filename, parameters=None):
        self.filename = filename
        self.parameters = parameters or {}


def test_generated_image_file_paths_includes_the_frame_codes_sidecar():
    from api.routes import _generated_image_file_paths
    from core.models.minimax_music3.frame_codes import sidecar_path_for_audio

    image = _FakeImageRow("txt2aud_20260101_000000_1.flac", {"is_audio": True})
    paths = _generated_image_file_paths(image)
    assert "frame_codes_sidecar" in paths
    expected = sidecar_path_for_audio(paths["media"])
    assert paths["frame_codes_sidecar"] == expected


def test_generated_image_file_paths_includes_the_sidecar_key_for_an_acestep_row_too():
    """Added unconditionally (not gated on model type/`is_audio`): the
    DELETE loop only removes a path that `os.path.exists`, so a row that
    never had one (ACE-Step, or a pre-Music3 row) is unaffected -- verified
    here by checking the key exists with a syntactically valid path, not
    that a matching file is ever created for such a row.
    """
    from api.routes import _generated_image_file_paths

    image = _FakeImageRow("txt2aud_acestep_row.flac", {"is_audio": True})
    paths = _generated_image_file_paths(image)
    assert paths["frame_codes_sidecar"].endswith(".mm3frames.json")


# ---------------------------------------------------------------------------
# F7: resolve_audio_defaults must not alias a mutable SSOT default
# ---------------------------------------------------------------------------

def test_resolve_audio_defaults_does_not_alias_the_ssot_loras_list():
    params = {"loras": None}  # Pydantic default for an omitted field, pre-resolution
    resolve_audio_defaults(params, provided_keys=set(), arch="acestep")
    assert params["loras"] == AUDIO_GEN_DEFAULTS["loras"]
    assert params["loras"] is not AUDIO_GEN_DEFAULTS["loras"]

    # Proof by mutation: appending to the resolved list must NOT corrupt the
    # process-lifetime SSOT dict (and therefore every future
    # /schema/generation-defaults response).
    params["loras"].append({"path": "x.safetensors", "strength": 1.0})
    assert AUDIO_GEN_DEFAULTS["loras"] == []
