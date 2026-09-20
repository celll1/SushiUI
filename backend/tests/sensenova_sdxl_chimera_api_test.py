from pathlib import Path
import re

import yaml

from api.param_defaults import CHIMERA_INITIALIZE_DEFAULTS


REPO = Path(__file__).resolve().parents[2]


def test_initializer_openapi_defaults_match_single_source():
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    assert "/models/sensenova-sdxl-chimera/initialize" in spec["paths"]
    schema = spec["components"]["schemas"]["InitializeSenseNovaSDXLChimeraRequest"]
    for key, value in CHIMERA_INITIALIZE_DEFAULTS.items():
        assert schema["properties"][key]["default"] == value
    assert "v3" in schema["properties"]["flow_version"]["enum"]
    assert "v4" in schema["properties"]["flow_version"]["enum"]
    assert "chimera_warmstart_source" in schema["properties"]
    response = spec["components"]["schemas"]["InitializeSenseNovaSDXLChimeraResponse"]
    assert 4 in response["properties"]["format_version"]["enum"]
    assert "chimera_v2_warmstart" in response["properties"]["unet_initialization"]["enum"]


def test_training_schema_advertises_v3_without_loose_path_controls():
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    schema = spec["components"]["schemas"]["TrainingRunCreateRequest"]
    assert "v3" in schema["properties"]["chimera_flow_version"]["enum"]
    assert "v4" in schema["properties"]["chimera_flow_version"]["enum"]
    assert "chimera_v3_angular_endpoint_slope" not in schema["properties"]


def test_route_defaults_are_references_not_literals():
    source = (REPO / "backend" / "api" / "routes.py").read_text(encoding="utf-8")
    for key in CHIMERA_INITIALIZE_DEFAULTS:
        assert re.search(
            rf'CHIMERA_INITIALIZE_DEFAULTS\s*\[\s*"{re.escape(key)}"\s*\]',
            source,
        )


def test_initializer_accepts_only_configured_target_directories():
    source = (REPO / "backend" / "api" / "routes.py").read_text(encoding="utf-8")
    request_start = source.index("class InitializeSenseNovaSDXLChimeraRequest")
    route_end = source.index('@router.post("/models/minit2i/create-scratch")')
    initializer = source[request_start:route_end]
    assert "target_dir: Optional[str] = None" in initializer
    assert "model_root = request.target_dir or settings.models_dir" in initializer
    assert "if model_root not in allowed:" in initializer
