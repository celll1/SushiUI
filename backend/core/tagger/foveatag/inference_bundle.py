"""Publish completed model-only artifacts separately from resumable bundles."""

import json
import os
from pathlib import Path

from .shared_bundle import _sha256


REQUIRED = (".safetensors", "_metadata.json", "_vocabulary.json")
OPTIONAL = ("_tag_metrics.npz", "_ood_ref.npz", "_export.safetensors", "_export_metadata.json")


def publish_inference_bundle(directory, name, *, optional_suffixes):
    directory = Path(directory)
    if Path(name).name != name or not set(optional_suffixes).issubset(OPTIONAL):
        raise ValueError("invalid native inference publication paths")
    metadata = json.loads((directory / f"{name}_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "foveatag_shared_trainer_weights":
        raise ValueError("native inference publication requires native weights")
    export_selected = "_export.safetensors" in optional_suffixes
    if export_selected != ("_export_metadata.json" in optional_suffixes):
        raise ValueError("native inference publication export pair is incomplete")
    if export_selected != bool(metadata.get("inference_export")):
        raise ValueError("native inference export selection differs from metadata")
    if export_selected and metadata["inference_export"] != f"{name}_export.safetensors":
        raise ValueError("invalid native inference export reference")
    if export_selected:
        exported = json.loads((directory / f"{name}_export_metadata.json").read_text(encoding="utf-8"))
        if ("_export_metadata.json" not in optional_suffixes
                or exported.get("kind") != "foveatag_inference_export"
                or exported.get("step") != metadata["step"]
                or exported.get("vocabulary_file") != f"{name}_vocabulary.json"):
            raise ValueError("native inference export metadata differs")
    files = {}
    for suffix in REQUIRED + tuple(optional_suffixes):
        path = directory / f"{name}{suffix}"
        stat = path.stat()
        files[path.name] = {"sha256": _sha256(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("native inference file changed during publication")
    document = {"version": 1, "kind": "foveatag_inference_commit", "step": metadata["step"],
                "files": files}
    path = directory / f"{name}_inference_commit.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def verify_inference_bundle(directory, name, *, verify_hash=True):
    directory = Path(directory)
    document = json.loads((directory / f"{name}_inference_commit.json").read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or document.get("version") != 1
            or document.get("kind") != "foveatag_inference_commit"):
        raise ValueError("unsupported native inference publication")
    files = document.get("files", {})
    required = {f"{name}{suffix}" for suffix in REQUIRED}
    allowed = required | {f"{name}{suffix}" for suffix in OPTIONAL}
    if not isinstance(files, dict) or not required.issubset(files) or not set(files).issubset(allowed):
        raise ValueError("native inference publication files differ")
    if (f"{name}_export.safetensors" in files) != (f"{name}_export_metadata.json" in files):
        raise ValueError("native inference publication export pair is incomplete")
    for filename, record in files.items():
        if (not isinstance(record, dict) or not {"size", "mtime_ns", "sha256"}.issubset(record)
                or type(record["size"]) is not int or type(record["mtime_ns"]) is not int
                or not isinstance(record["sha256"], str) or len(record["sha256"]) != 64):
            raise ValueError("invalid native inference file record")
        path = directory / filename
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError("native inference publication is stale")
        if verify_hash and _sha256(path) != record["sha256"]:
            raise ValueError("native inference publication integrity differs")
    metadata = json.loads((directory / f"{name}_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("step") != document.get("step"):
        raise ValueError("native inference publication step differs")
    return document


def published_inference_paths(directory, name):
    """Cheap generation checks for UI polling; loading verifies the hashes."""
    try:
        document = verify_inference_bundle(directory, name, verify_hash=False)
        directory = Path(directory)
        paths = [{"name": name, "mode": "full_unmerged", "path": str(directory / f"{name}.safetensors"),
                  "vocabulary_path": str(directory / f"{name}_vocabulary.json"), "requires_backbone": False}]
        if f"{name}_export_metadata.json" in document["files"]:
            metadata = json.loads((directory / f"{name}_export_metadata.json").read_text(encoding="utf-8"))
            mode = metadata["export_mode"]
            if mode not in ("lora", "merged"):
                raise ValueError("invalid native inference export mode")
            paths.append({"name": name, "mode": mode, "path": str(directory / f"{name}_export.safetensors"),
                          "vocabulary_path": str(directory / metadata["vocabulary_file"]),
                          "requires_backbone": mode == "lora"})
        return paths
    except (OSError, ValueError, KeyError, TypeError):
        return []
