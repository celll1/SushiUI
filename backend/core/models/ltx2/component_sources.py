"""Frozen companion root references for standalone LTX-2 transformer artifacts."""
import json
from pathlib import Path

METADATA_KEY = "ltx2_components"
COMPONENTS = ("text_encoder", "tokenizer", "connectors", "vae", "audio_vae", "vocoder", "scheduler")


def validate_sources(sources):
    if (not isinstance(sources, dict) or set(sources) != {"schema_version", "root"} or
            type(sources.get("schema_version")) is not int or sources["schema_version"] != 1):
        raise ValueError("LTX-2 companion references require schema version 1 and a native root")
    value = sources["root"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("LTX-2 companion root must be a nonempty absolute directory")
    root = Path(value)
    if not root.is_absolute() or not root.is_dir() or not (root / "model_index.json").is_file():
        raise ValueError(f"LTX-2 declared companion root is unavailable: {root}")
    index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("_class_name") != "LTX2Pipeline":
        raise ValueError("LTX-2 companion root must declare the native LTX2Pipeline")
    for name in COMPONENTS:
        declaration = index.get(name)
        if (not isinstance(declaration, list) or len(declaration) != 2 or
                any(not isinstance(item, str) or not item for item in declaration) or not (root / name).is_dir()):
            raise ValueError(f"LTX-2 declared companion component is unavailable: {name}")
    if not (root / "transformer").is_dir():
        raise ValueError("LTX-2 companion root requires the native transformer component")
    return dict(schema_version=1, root=str(root.resolve()))


def read_sources(path):
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology
    source = Path(path)
    if not source.exists() and Path(str(source)+".index.json").is_file():
        source = Path(str(source)+".index.json")
    read_artifact_topology(source)
    if str(source).endswith(".safetensors.index.json"):
        index = json.loads(source.read_text(encoding="utf-8"))
        metadata = index.get("metadata") or {}
        shards = {source.parent / name for name in index["weight_map"].values()}
    else:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        shards = {source}
    declaration = metadata.get(METADATA_KEY)
    sources = None if declaration is None else json.loads(declaration)
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            declared = (handle.metadata() or {}).get(METADATA_KEY)
        if (None if declared is None else json.loads(declared)) != sources:
            raise ValueError("LTX-2 index and shard companion references disagree")
    return None if sources is None else validate_sources(sources)


def export_sources(model):
    inner = getattr(model, "transformer", model)
    sources = getattr(inner, "_ltx2_component_sources", None)
    return {} if sources is None else {METADATA_KEY: json.dumps(validate_sources(sources), allow_nan=False)}
