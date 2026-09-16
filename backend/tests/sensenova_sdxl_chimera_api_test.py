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
