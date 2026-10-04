"""Fresh model-only publication and training-detail path contracts."""

import json
import os
from types import SimpleNamespace

import pytest

from core.tagger.foveatag.inference_bundle import (
    publish_inference_bundle, published_inference_paths, verify_inference_bundle,
)
from foveatag_inference_manager_test import artifact


def publish(directory, name="best_f1", mode="full"):
    optional = ["_tag_metrics.npz"]
    if mode != "full":
        optional.extend(("_export.safetensors", "_export_metadata.json"))
    publish_inference_bundle(directory, name, optional_suffixes=optional)


@pytest.mark.parametrize("mode", ["full", "lora", "merged"])
def test_model_only_publication_without_optimizer(mode, tmp_path):
    _, _, _ = artifact(tmp_path, "lora", mode)
    # The publication protocol has no optimizer/state requirement.
    meta_path = tmp_path / "best_f1_metadata.json"
    metadata = json.loads(meta_path.read_text())
    metadata["step"] = 19
    meta_path.write_text(json.dumps(metadata))
    if mode != "full":
        export_meta_path = tmp_path / "best_f1_export_metadata.json"
        export_meta = json.loads(export_meta_path.read_text())
        export_meta["step"] = 19
        export_meta_path.write_text(json.dumps(export_meta))
    publish(tmp_path, mode=mode)
    assert verify_inference_bundle(tmp_path, "best_f1")["step"] == 19
    paths = published_inference_paths(tmp_path, "best_f1")
    assert len(paths) == (1 if mode == "full" else 2)
    assert paths[0]["mode"] == "full_unmerged"
    assert paths[0]["requires_backbone"] is False
    if mode != "full":
        assert paths[1]["mode"] == mode
        assert paths[1]["requires_backbone"] == (mode == "lora")


def prepare(tmp_path):
    artifact(tmp_path, "full", "full")
    path = tmp_path / "best_f1_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["step"] = 1
    path.write_text(json.dumps(metadata))
    publish(tmp_path)


def test_unpublished_or_overwritten_files_not_listed(tmp_path):
    assert published_inference_paths(tmp_path, "best_f1") == []
    prepare(tmp_path)
    assert published_inference_paths(tmp_path, "best_f1")
    (tmp_path / "best_f1_vocabulary.json").write_text("{}")
    assert published_inference_paths(tmp_path, "best_f1") == []


def test_hash_verification_catches_same_generation_tampering(tmp_path):
    prepare(tmp_path)
    path = tmp_path / "best_f1_vocabulary.json"
    stat = path.stat()
    value = path.read_bytes().replace(b"tag_0", b"tag_x")
    path.write_bytes(value)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="integrity differs"):
        verify_inference_bundle(tmp_path, "best_f1")


def test_native_detail_returns_only_published_paths(tmp_path):
    from api.routes import get_tagger_training_run
    prepare(tmp_path)
    run = SimpleNamespace(config={"architecture": "foveatag"}, output_dir=str(tmp_path),
                          to_list_dict=lambda: {"run_id": "test"})
    class Query:
        def filter(self, *args):
            return self
        def first(self):
            return run
    db = SimpleNamespace(query=lambda *args: Query())
    response = get_tagger_training_run("test", db)
    assert response["inference_checkpoint_paths"][0]["name"] == "best_f1"
    assert len(response["inference_checkpoint_paths"]) == 1


@pytest.mark.parametrize("document", [[], {"version": 1, "kind": "foveatag_inference_commit", "files": []}])
def test_corrupt_publication_not_exposed(document, tmp_path):
    (tmp_path / "best_f1_inference_commit.json").write_text(json.dumps(document))
    assert published_inference_paths(tmp_path, "best_f1") == []
