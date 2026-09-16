from pathlib import Path

import yaml

from api.param_defaults import CHIMERA_INITIALIZE_DEFAULTS


REPO = Path(__file__).resolve().parents[2]


def test_initializer_openapi_defaults_match_single_source():
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    assert "/models/sensenova-sdxl-chimera/initialize" in spec["paths"]
    schema = spec["components"]["schemas"]["InitializeSenseNovaSDXLChimeraRequest"]
    for key, value in CHIMERA_INITIALIZE_DEFAULTS.items():
        assert schema["properties"][key]["default"] == value


def test_route_defaults_are_references_not_literals():
    source = (REPO / "backend" / "api" / "routes.py").read_text(encoding="utf-8")
    for key in CHIMERA_INITIALIZE_DEFAULTS:
        assert f'CHIMERA_INITIALIZE_DEFAULTS["{key}"]' in source


def test_initializer_accepts_only_configured_target_directories():
    source = (REPO / "backend" / "api" / "routes.py").read_text(encoding="utf-8")
    request_start = source.index("class InitializeSenseNovaSDXLChimeraRequest")
    route_end = source.index('@router.post("/models/minit2i/create-scratch")')
    initializer = source[request_start:route_end]
    assert "target_dir: Optional[str] = None" in initializer
    assert "model_root = request.target_dir or settings.models_dir" in initializer
    assert "if model_root not in allowed:" in initializer
