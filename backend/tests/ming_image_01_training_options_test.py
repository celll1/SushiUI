"""Offline API/default/YAML/ownership contract for Ming training groups."""

import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*a, **k):
    raise AssertionError("Ming options test attempted CUDA initialization")
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda

from pydantic import ValidationError
from api.routes import TrainingRunCreateRequest
from api.param_defaults import TRAINING_DEFAULTS
from core.models.ming_image_01.training_options import COMPONENTS, resolve_ownership
from core.training.training_config import _build_train_section, train_section_key_vocabulary


def request(**values):
    return TrainingRunCreateRequest(training_method="full_finetune", base_model_path="offline-Ming", **values)


def test_defaults_inheritance_explicit_freeze_and_zero_rate_preservation():
    assert request().ming_conditioner_groups is TRAINING_DEFAULTS["ming_conditioner_groups"]
    assert request().ming_conditioner_lrs is TRAINING_DEFAULTS["ming_conditioner_lrs"]
    assert resolve_ownership({"train_text_encoder": False}) == (set(), {})
    assert resolve_ownership({"train_text_encoder": True, "lr": .003})[1] == {name: .003 for name in COMPONENTS}
    groups, rates = resolve_ownership({"train_text_encoder": True, "learning_rate": .001})
    assert groups == set(COMPONENTS) and set(rates.values()) == {.001}
    assert resolve_ownership({"train_text_encoder": True, "ming_conditioner_groups": []}) == (set(), {})
    groups, rates = resolve_ownership({"ming_conditioner_groups": ["language", "projections"],
        "ming_conditioner_lrs": {"language": 0}, "text_encoder_lr": .0002, "learning_rate": .001})
    assert groups == {"language", "projections"} and rates == {"language": 0, "projections": .0002}


@pytest.mark.parametrize("values", [
    {"ming_conditioner_groups": ["language", "language"]},
    {"ming_conditioner_groups": ["foreign"]}, {"ming_conditioner_groups": "language"},
    {"ming_conditioner_groups": [True]}, {"ming_conditioner_lrs": {"foreign": .001}},
    {"ming_conditioner_lrs": {"language": True}}, {"ming_conditioner_lrs": {"language": "0.001"}},
    {"ming_conditioner_lrs": {"language": -.001}}, {"ming_conditioner_lrs": {"language": float("nan")}},
])
def test_api_refuses_malformed_or_coerced_ownership(values):
    with pytest.raises(ValidationError): request(**values)


def test_actual_api_to_yaml_and_openapi_frontend_contract():
    req = request(ming_conditioner_groups=["language", "projections"], ming_conditioner_lrs={"language": 0})
    section = _build_train_section(req.model_dump(), arch="ming_image_01", total_steps=2, epochs=None, train_unet=True, train_text_encoder=False)
    restored = yaml.safe_load(yaml.safe_dump(section))
    assert restored["ming_conditioner_groups"] == req.ming_conditioner_groups
    assert restored["ming_conditioner_lrs"] == {"language": 0}
    from api.routes import _extract_request_params_from_yaml
    roundtrip = _extract_request_params_from_yaml({"train": restored}, "full_finetune")
    assert roundtrip["ming_conditioner_groups"] == req.ming_conditioner_groups
    assert roundtrip["ming_conditioner_lrs"] == {"language": 0}
    assert {"ming_conditioner_groups", "ming_conditioner_lrs"} <= train_section_key_vocabulary()
    root = Path(__file__).resolve().parents[2]
    schema = yaml.safe_load((root / "openapi.yaml").read_text(encoding="utf-8"))["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    assert schema["ming_conditioner_groups"]["items"]["enum"] == list(COMPONENTS)
    assert schema["ming_conditioner_groups"]["uniqueItems"]
    assert schema["ming_conditioner_lrs"]["additionalProperties"]["minimum"] == 0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("architecture,method,allowed", [("ming_image_01", "full_finetune", True),
    ("qwen_image_21", "full_finetune", False), ("ming_image_01", "lora", False)])
def test_write_boundary_rejects_options_on_wrong_architecture_or_method(monkeypatch, architecture, method, allowed):
    from fastapi import HTTPException
    from api.routes import _check_ming_conditioner_params
    from core.training import training_config
    monkeypatch.setattr(training_config, "_detect_arch", lambda path: architecture)
    req = TrainingRunCreateRequest(training_method=method, base_model_path="offline", ming_conditioner_groups=[])
    if allowed:
        _check_ming_conditioner_params(req)
    else:
        with pytest.raises(HTTPException) as caught: _check_ming_conditioner_params(req)
        assert caught.value.status_code == 400
