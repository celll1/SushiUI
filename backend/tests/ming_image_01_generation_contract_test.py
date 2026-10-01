"""CPU-only schema/default/transport invariants for Ming generation settings."""

import ast
import inspect
import sys
from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def forbidden_cuda(*args, **kwargs):
    raise AssertionError("Ming API contract test attempted CUDA initialization")


torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda

from api import routes
from api.generation_utils import resolve_image_defaults
from api.param_defaults import GENERATION_DEFAULTS, OUTPAINT_DEFAULTS, IMAGE_GEN_ARCH_OVERLAYS, image_defaults_for_arch


@pytest.mark.parametrize("profile,key", [("generation_edit", "ming_image_01"), ("layer_decompose", "ming_image_01_layer")])
def test_profile_defaults_match_served_overlay_and_preserve_explicit_fields(profile, key):
    resolved = image_defaults_for_arch("ming_image_01", profile=profile)
    assert all(resolved[name] == value for name, value in IMAGE_GEN_ARCH_OVERLAYS[key].items())
    params = dict(steps=None, cfg_scale=None, ming_image_01_layers=GENERATION_DEFAULTS["ming_image_01_layers"])
    resolve_image_defaults(params, set(), "ming_image_01", profile=profile)
    assert params["steps"] == resolved["steps"] and params["cfg_scale"] == resolved["cfg_scale"]
    params.update(steps=5, cfg_scale=0., ming_image_01_layers=7)
    resolve_image_defaults(params, set(params), "ming_image_01", profile=profile)
    assert params == dict(steps=5, cfg_scale=0., ming_image_01_layers=7)
    assert image_defaults_for_arch("sdxl", profile=profile) == GENERATION_DEFAULTS


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_json_layer_count_refuses_invalid_supervision_geometry(value):
    with pytest.raises(ValidationError):
        routes.GenerationParams(prompt="", ming_image_01_layers=value)


def test_layer_default_and_all_four_multipart_consumers_use_central_value():
    expected = GENERATION_DEFAULTS["ming_image_01_layers"]
    assert routes.GenerationParams(prompt="").ming_image_01_layers == expected
    assert OUTPAINT_DEFAULTS["ming_image_01_layers"] == expected
    module = ast.parse((ROOT / "backend/api/routes.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in module.body if isinstance(node, ast.AsyncFunctionDef)}
    for name in ("generate_txt2img", "generate_img2img", "generate_inpaint", "generate_outpaint"):
        function = getattr(routes, name)
        default = inspect.signature(function).parameters["ming_image_01_layers"].default
        assert default.default == expected
        node = functions[name]
        assert any(isinstance(item, ast.Dict) and any(isinstance(key, ast.Constant) and key.value == "ming_image_01_layers"
                   for key in item.keys) for item in ast.walk(node))
    schema = yaml.safe_load((ROOT / "openapi.yaml").read_text(encoding="utf-8"))
    field = schema["components"]["schemas"]["GenerationParams"]["properties"]["ming_image_01_layers"]
    assert field["default"] == expected and field["minimum"] == 1
    assert not torch.cuda.is_initialized()
