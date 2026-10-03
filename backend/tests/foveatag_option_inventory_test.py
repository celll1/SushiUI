"""The option audit is complete as an inventory, not a blanket behavior claim."""

import ast
import json
from pathlib import Path


def test_inventory_covers_actual_defaults_and_request_fields():
    root = Path(__file__).resolve().parents[2]
    defaults_module = ast.parse((root / "backend/api/param_defaults.py").read_text(encoding="utf-8"))
    defaults = next(node.value for node in defaults_module.body if isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name) and node.target.id == "TAGGER_TRAINING_DEFAULTS")
    default_names = {ast.literal_eval(key) for key in defaults.keys}
    routes = ast.parse((root / "backend/api/routes.py").read_text(encoding="utf-8"))
    request = next(node for node in routes.body if isinstance(node, ast.ClassDef)
                   and node.name == "TaggerTrainingRunCreateRequest")
    request_names = {node.target.id for node in request.body if isinstance(node, ast.AnnAssign)
                     and isinstance(node.target, ast.Name)}
    document = json.loads((root / "docs/audits/FOVEATAG_OPTION_PARITY_2026-10-03.json").read_text())
    entries = {entry["name"]: entry for entry in document["options"]}
    assert len(entries) == len(document["options"])
    assert set(entries) == default_names | request_names
    for name, entry in entries.items():
        assert entry["in_defaults"] == (name in default_names)
        assert entry["in_request"] == (name in request_names)
