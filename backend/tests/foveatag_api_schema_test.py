"""Native API defaults and architecture-specific refusals are checked without loading weights."""

import pytest
from pydantic import ValidationError

from api.param_defaults import TAGGER_TRAINING_DEFAULTS
from api.routes import TaggerTrainingRunCreateRequest


def params(**overrides):
    return {**TAGGER_TRAINING_DEFAULTS, "architecture": "foveatag", "training_method": "full",
            "vision_encoder_path": "", "dataset_configs": [{"dataset_id": 1}], **overrides}


def test_native_full_can_start_without_external_weights():
    request = TaggerTrainingRunCreateRequest(**params())
    assert request.architecture == "foveatag"
    assert request.vision_encoder_path == ""
    for key in TAGGER_TRAINING_DEFAULTS:
        if key.startswith("fovea_"):
            assert getattr(request, key) == TAGGER_TRAINING_DEFAULTS[key]


def test_all_shared_request_defaults_match_authoritative_table():
    fields = TaggerTrainingRunCreateRequest.model_fields
    for name, value in TAGGER_TRAINING_DEFAULTS.items():
        if name == "vision_encoder_path":
            assert fields[name].is_required()
        else:
            assert fields[name].default == value, name


def test_documented_shared_defaults_match_authoritative_table():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[2]
    spec = yaml.safe_load((root / "openapi.yaml").read_text(encoding="utf-8"))
    properties = spec["components"]["schemas"]["TaggerTrainingRunCreateRequest"]["properties"]
    assert set(TAGGER_TRAINING_DEFAULTS).issubset(properties)
    for name, value in properties.items():
        if name in TAGGER_TRAINING_DEFAULTS and "default" in value:
            assert value["default"] == TAGGER_TRAINING_DEFAULTS[name], name


@pytest.mark.parametrize("overrides,reason", [
    ({"training_method": "lora"}, "scratch-pretrained"),
    ({"cls_dim": 512}, "pooling dimensions"),
    ({"hidden_proj_dim": 512}, "pooling dimensions"),
    ({"vision_encoder_path": "https://huggingface.co/example/model"}, "local scratch"),
    ({"training_method": "head_only"}, "full or lora"),
    ({"checkpoint_save_mode": "invalid"}, "save mode"),
    ({"mixed_precision": "invalid"}, "mixed precision"),
    ({"train_f1_initial_threshold": -1}, "probability"),
    ({"train_f1_initial_threshold": 2}, "probability"),
])
def test_invalid_native_configs_are_refused_before_catalog_creation(overrides, reason):
    with pytest.raises(ValidationError, match=reason):
        TaggerTrainingRunCreateRequest(**params(**overrides))


def test_native_lora_accepts_local_scratch_path_as_pending_config():
    request = TaggerTrainingRunCreateRequest(**params(
        training_method="lora", vision_encoder_path="D:/scratch/encoder.safetensors"))
    assert request.training_method == "lora"


def test_native_switch_does_not_change_siglip_defaults():
    request = TaggerTrainingRunCreateRequest(**{
        "vision_encoder_path": "example/model", "dataset_configs": []})
    assert request.architecture == TAGGER_TRAINING_DEFAULTS["architecture"] == "siglip2"


@pytest.mark.parametrize("key", ["fovea_overview_size", "fovea_local_size", "fovea_feature_dim", "fovea_max_source_pixels"])
def test_native_positive_dimensions_refuse_zero(key):
    with pytest.raises(ValidationError):
        TaggerTrainingRunCreateRequest(**params(**{key: 0}))


def test_zero_local_budget_and_zero_seed_are_explicitly_supported():
    request = TaggerTrainingRunCreateRequest(**params(fovea_local_views=0, fovea_seed=0))
    assert request.fovea_local_views == request.fovea_seed == 0
