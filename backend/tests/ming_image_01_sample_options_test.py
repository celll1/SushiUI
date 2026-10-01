"""Offline native-profile sample defaults and API/YAML/UI parameter contracts."""
import sys
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*a, **k):
    raise AssertionError("Ming sample option test attempted CUDA initialization")
torch.cuda.get_device_capability=lambda *a, **k: (8, 9)
torch.cuda.is_available=lambda: False
torch.cuda._lazy_init=forbidden_cuda
torch._C._cuda_init=forbidden_cuda
from ming_image_01_artifact_test import package, change
from api import routes
from api.param_defaults import TRAINING_DEFAULTS, TRAINING_SAMPLE_DEFAULTS_BY_ARCH
from core.training.training_config import TrainingConfigGenerator, _sample_defaults_arch, train_section_key_vocabulary
from core.training.train_runner import _resolve_training_sample_config
from pydantic import ValidationError


@pytest.mark.parametrize("layer", [False, True])
def test_profile_defaults_and_layer_count_round_trip(package, layer):
    if layer:
        change(package, "transformer", alignment_padding_mode="learned", multi_frame_output=True)
    req = routes.TrainingRunCreateRequest(training_method="full_finetune", base_model_path=str(package),
        ming_image_01_sample_layers=5, total_steps=1)
    payload = req.model_dump()
    payload["_explicit_fields"] = sorted(req.model_fields_set)
    config = yaml.safe_load(TrainingConfigGenerator.generate_full_finetune_config(payload,
        run_name="offline", base_model_path=str(package), output_dir=str(package / "run")))
    process = config["config"]["process"][0]
    key = "ming_image_01_layer" if layer else "ming_image_01"
    assert _sample_defaults_arch(str(package), "ming_image_01") == key
    defaults = TRAINING_SAMPLE_DEFAULTS_BY_ARCH[key]
    assert process["sample"]["sample_steps"] == defaults["sample_steps"]
    assert process["sample"]["guidance_scale"] == defaults["sample_cfg_scale"]
    assert process["train"]["ming_image_01_sample_layers"] == 5
    restored = routes._extract_request_params_from_yaml(process, "full_finetune")
    assert restored["ming_image_01_sample_layers"] == 5
    minimal = _resolve_training_sample_config({"model": {"name_or_path": str(package)}}, "ming_image_01")
    assert minimal["guidance_scale"] == defaults["sample_cfg_scale"]
    assert minimal["sample_steps"] == defaults["sample_steps"]
    explicit = _resolve_training_sample_config({"model": {"name_or_path": str(package)},
        "sample": {"sample_steps": 2, "guidance_scale": 0.}}, "ming_image_01")
    assert explicit["sample_steps"] == 2 and explicit["guidance_scale"] == 0.
    assert "ming_image_01_sample_layers" in train_section_key_vocabulary()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("value", [0, -1, True, "3", 1.5])
def test_layer_count_refuses_coercion(value):
    with pytest.raises(ValidationError):
        routes.TrainingRunCreateRequest(training_method="full_finetune", base_model_path="offline",
            ming_image_01_sample_layers=value)


def test_openapi_defaults_and_frontend_transport():
    root = Path(__file__).resolve().parents[2]
    schema = yaml.safe_load((root / "openapi.yaml").read_text())
    field = schema["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]["ming_image_01_sample_layers"]
    assert field["default"] == TRAINING_DEFAULTS["ming_image_01_sample_layers"]
    assert field["minimum"] == 1
    for name in ["api.ts"]:
        assert "ming_image_01_sample_layers?: number" in (root / "frontend/src/utils" / name).read_text()
    assert '"ming_image_01_sample_layers"' in (root / "frontend/src/components/training/trainingParams.ts").read_text()


def test_model_listing_reports_header_only_layer_profile(package, monkeypatch):
    import os
    change(package, "transformer", alignment_padding_mode="learned", multi_frame_output=True)
    monkeypatch.setattr(routes, "settings", SimpleNamespace(models_dir=str(package.parent)))
    monkeypatch.setattr(routes, "_models_cache", None)
    monkeypatch.setattr(routes, "_models_cache_timestamp", 0)
    actual = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path: [package.name] if Path(path) == package.parent else actual(path))
    db = SimpleNamespace(query=lambda *a: SimpleNamespace(first=lambda: None))
    result = routes.get_models(db=db, force_rescan=True)
    matching = [m for m in result["models"] if m["path"] == str(package)]
    assert len(matching) == 1 and matching[0]["profile"] == "layer_decompose"
    assert matching[0]["architecture"] == "ming_image_01"
    assert not torch.cuda.is_initialized()
