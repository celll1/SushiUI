"""Exact companion selections for H3 native Full artifacts."""

import json
from pathlib import Path

KEY = "minimax_h3.components"
FILES = ("source_vae", "audio_vae", "text_encoder", "te_projection", "image_vae")


def capture_sources(layout, projection):
    def absolute(value):
        return str(Path(value).resolve()) if value else None
    return dict(version=1, root=absolute(layout["root"]), official=absolute(layout["official"]),
        variant=layout["variant"], source_vae=absolute(layout["vae"]),
        audio_vae=absolute(layout["audio_vae"]), text_encoder=absolute(layout.get("text_encoder")),
        te_projection=absolute((projection or {}).get("path")), image_vae=absolute(layout.get("image_vae")))


def validate_sources(value):
    keys = {"version", "root", "official", "variant", *FILES}
    if (not isinstance(value, dict) or set(value) != keys or type(value["version"]) is not int
            or value["version"] != 1 or value["variant"] not in ("t2va", "ref2va", "fl2va")):
        raise ValueError("Unsupported H3 companion references")
    result = dict(value)
    for key in ("root", "official", *FILES):
        path = value[key]
        if path is None and key in ("te_projection", "image_vae"):
            continue
        if not isinstance(path, str) or not path or not Path(path).is_absolute():
            raise ValueError(f"H3 companion {key} requires an absolute path")
        source = Path(path)
        if not (source.is_dir() if key in ("root", "official") else source.is_file()):
            raise FileNotFoundError(f"H3 declared companion is unavailable: {key}: {source}")
        result[key] = str(source.resolve())
    official = Path(result["official"])
    index = json.loads((official/"model_index.json").read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("_class_name") != "MiniMaxH3ModularPipeline":
        raise ValueError("H3 companion configs must declare MiniMaxH3ModularPipeline")
    for name in ("vae", "audio_vae", "text_encoder"):
        config = json.loads((official/name/"config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"H3 companion config is invalid: {name}")
    for name in ("tokenizer", "processor", "scheduler", "audio_scheduler"):
        if not (official/name).is_dir():
            raise FileNotFoundError(f"H3 declared companion directory is unavailable: {name}")
    paths = [result[key] for key in FILES if result[key] is not None]
    if len(set(paths)) != len(paths):
        raise ValueError("H3 companion components must use distinct files")
    return result


def read_sources(metadata, *, required=False):
    raw = metadata.get(KEY)
    if raw is None and not required:
        return None
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 65536:
        raise ValueError("H3 native loader requires companion references")
    return validate_sources(json.loads(raw))


def export_sources(model):
    inner = getattr(model, "transformer", model)
    value = getattr(inner, "_h3_component_sources", None)
    return {} if value is None else {KEY: json.dumps(validate_sources(value), sort_keys=True, allow_nan=False)}


def companion_layout(metadata):
    value = read_sources(metadata, required=True)
    return dict(root=value["root"], official=value["official"], variant=value["variant"],
        vae=value["source_vae"], audio_vae=value["audio_vae"], text_encoder=value["text_encoder"],
        image_vae=value["image_vae"], te_projection=value["te_projection"])
