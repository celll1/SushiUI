"""Recognize complete native Ming config packages before broad SD heuristics."""

import json
from pathlib import Path

from .artifact import inspect_artifact


def _marker(path, key):
    try:
        with path.open(encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value.get(key) if isinstance(value, dict) else None


def discover_native_package(path):
    path = Path(path)
    manifest = path / "manifest.json" if path.is_dir() else path
    candidate = None
    if manifest.name == "manifest.json" and manifest.is_file():
        if _marker(manifest, "model_type") == "ming_image_01":
            candidate = manifest
    if candidate is None and path.is_dir():
        config = path / "transformer" / "config.json"
        if config.is_file() and (path / "mlp" / "config.json").is_file():
            if _marker(config, "_class_name") == "DiffusionTransformer":
                candidate = path
    if candidate is None:
        return None
    return inspect_artifact(candidate, require_weights=False)
