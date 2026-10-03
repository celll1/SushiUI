"""Commit native multi-file resume bundles only after all artifacts are saved."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


REQUIRED_SUFFIXES = (".safetensors", "_metadata.json", "_state.json", "_optimizer.pt", "_vocabulary.json")
OPTIONAL_SUFFIXES = ("_tag_metrics.npz", "_metrics_runtime.npz", "_f1_runtime.npz", "_ood_ref.npz", "_ood_reservoir.npz",
                    "_export.safetensors", "_export_metadata.json")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit_native_bundle(directory, name, *, optional_suffixes=None):
    directory = Path(directory)
    state = json.loads((directory / f"{name}_state.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / f"{name}_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "foveatag_shared_trainer_weights" or "native_runtime" not in state:
        raise ValueError("native bundle lacks native weights or runtime")
    if metadata.get("step") != state["global_step"]:
        raise ValueError("native bundle model and runtime steps differ")
    files = {}
    selected = OPTIONAL_SUFFIXES if optional_suffixes is None else tuple(optional_suffixes)
    if not set(selected).issubset(OPTIONAL_SUFFIXES):
        raise ValueError("unsupported native optional artifact")
    for suffix in REQUIRED_SUFFIXES + selected:
        path = directory / f"{name}{suffix}"
        if optional_suffixes is not None or suffix in REQUIRED_SUFFIXES or path.is_file():
            files[path.name] = _sha256(path)
    document = {"version": 1, "kind": "foveatag_shared_resume_commit",
                "global_step": state["global_step"], "files": files}
    path = directory / f"{name}_resume_commit.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def verify_native_bundle(directory, name):
    directory = Path(directory)
    path = directory / f"{name}_resume_commit.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("native resume bundle has no readable commit") from exc
    if document.get("version") != 1 or document.get("kind") != "foveatag_shared_resume_commit":
        raise ValueError("unsupported native resume commit")
    files = document.get("files", {})
    if not isinstance(files, dict) or not all(f"{name}{suffix}" in files for suffix in REQUIRED_SUFFIXES):
        raise ValueError("native resume commit lacks required files")
    allowed = {f"{name}{suffix}" for suffix in REQUIRED_SUFFIXES + OPTIONAL_SUFFIXES}
    if not set(files).issubset(allowed):
        raise ValueError("native resume commit contains unexpected paths")
    for filename, expected in files.items():
        try:
            actual = _sha256(directory / filename)
        except OSError as exc:
            raise ValueError("native resume bundle file is missing or unreadable") from exc
        if actual != expected:
            raise ValueError("native resume bundle integrity differs")
    state = json.loads((directory / f"{name}_state.json").read_text(encoding="utf-8"))
    if state.get("global_step") != document.get("global_step"):
        raise ValueError("native resume commit and runtime steps differ")
    return document
