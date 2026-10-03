"""Frozen component references for standalone ACE-Step DiT artifacts."""
import json
from pathlib import Path

METADATA_KEY = "acestep_components"


def validate_sources(sources):
    keys = {"schema_version", "root", "vae", "text_encoder", "tokenizer"}
    if not isinstance(sources, dict) or set(sources) != keys or type(sources.get("schema_version")) is not int or sources["schema_version"] != 1:
        raise ValueError("ACE-Step component references require schema version 1 and all native sources")
    for key in keys - {"schema_version"}:
        value = sources[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"ACE-Step component source {key} must be a nonempty string")
    for key in ("root", "vae", "text_encoder"):
        path = Path(sources[key])
        valid = path.is_dir() if key == "root" else path.is_file()
        if not path.is_absolute() or not valid:
            raise ValueError(f"ACE-Step declared {key} source is unavailable: {path}")
    tokenizer = Path(sources["tokenizer"])
    if tokenizer.is_absolute() and not tokenizer.is_dir():
        raise ValueError(f"ACE-Step declared tokenizer directory is unavailable: {tokenizer}")
    return dict(sources)


def read_sources(path):
    from safetensors import SafetensorError, safe_open
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except SafetensorError:
        return None
    if METADATA_KEY not in metadata:
        return None
    return validate_sources(json.loads(metadata[METADATA_KEY]))


def export_sources(model):
    sources = getattr(model, "_acestep_component_sources", None)
    if sources is None:
        return {}
    return {METADATA_KEY: json.dumps(validate_sources(sources), allow_nan=False)}
