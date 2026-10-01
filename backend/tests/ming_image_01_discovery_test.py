"""Ming native discovery never loads tensors or matches directory names."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.models.ming_image_01.discovery import discover_native_package
from core.models.ming_image_01.artifact import write_manifest
from ming_image_01_artifact_test import package, change, write_json


def test_native_directory_and_manifest_identity(package):
    assert discover_native_package(package).profile == "generation_edit"
    manifest = write_manifest(package)
    assert discover_native_package(manifest).profile == "generation_edit"


def test_unrelated_manifest_and_config_do_not_claim_ming(tmp_path):
    write_json(tmp_path / "manifest.json", ["not a Ming object"])
    assert discover_native_package(tmp_path) is None
    write_json(tmp_path / "manifest.json", {"model_type": "qwen_image_21"})
    assert discover_native_package(tmp_path / "manifest.json") is None
    (tmp_path / "transformer").mkdir()
    (tmp_path / "mlp").mkdir()
    write_json(tmp_path / "transformer" / "config.json", {"_class_name": "OtherTransformer"})
    write_json(tmp_path / "mlp" / "config.json", {})
    assert discover_native_package(tmp_path) is None


def test_recognized_native_package_cannot_fall_through_on_profile_corruption(package):
    change(package, "transformer", multi_frame_output=True)
    with pytest.raises(ValueError, match="profile"):
        discover_native_package(package)
