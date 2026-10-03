"""Incomplete native checkpoint bundles cannot silently initialize a new run."""

import json

import pytest

from core.tagger.foveatag.shared_bundle import commit_native_bundle, verify_native_bundle


def bundle(directory, name, step):
    (directory / f"{name}.safetensors").write_bytes(b"weights")
    (directory / f"{name}_optimizer.pt").write_bytes(b"optimizer")
    (directory / f"{name}_metadata.json").write_text(json.dumps({
        "kind": "foveatag_shared_trainer_weights", "step": step}))
    (directory / f"{name}_state.json").write_text(json.dumps({
        "global_step": step, "native_runtime": {"version": 1}}))
    (directory / f"{name}_vocabulary.json").write_text(json.dumps({"num_tags": 5}))
    commit_native_bundle(directory, name)


@pytest.mark.parametrize("suffix", [".safetensors", "_optimizer.pt", "_state.json", "_vocabulary.json"])
def test_required_file_corruption_is_detected(tmp_path, suffix):
    bundle(tmp_path, "latest", 2)
    (tmp_path / f"latest{suffix}").write_bytes(b"partial replacement")
    with pytest.raises(ValueError, match="integrity differs"):
        verify_native_bundle(tmp_path, "latest")


def test_falls_back_to_earlier_committed_step_not_new_training(tmp_path):
    from core.tagger.tagger_trainer import _find_resume_checkpoint
    bundle(tmp_path, "step_000001", 1)
    bundle(tmp_path, "latest", 2)
    (tmp_path / "latest_optimizer.pt").write_bytes(b"partial replacement")
    name, state = _find_resume_checkpoint(str(tmp_path))
    assert name == "step_000001"
    assert state["global_step"] == 1
    (tmp_path / "step_000001.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="refusing to start from scratch"):
        _find_resume_checkpoint(str(tmp_path))


def test_uncommitted_native_state_is_not_a_resume_candidate(tmp_path):
    from core.tagger.tagger_trainer import _find_resume_checkpoint
    bundle(tmp_path, "latest", 2)
    (tmp_path / "latest_resume_commit.json").unlink()
    with pytest.raises(ValueError, match="refusing to start from scratch"):
        _find_resume_checkpoint(str(tmp_path))


def test_different_model_and_runtime_steps_cannot_be_committed(tmp_path):
    bundle(tmp_path, "latest", 2)
    (tmp_path / "latest_metadata.json").write_text(json.dumps({
        "kind": "foveatag_shared_trainer_weights", "step": 3}))
    with pytest.raises(ValueError, match="steps differ"):
        commit_native_bundle(tmp_path, "latest")


def test_commit_paths_cannot_escape_checkpoint_bundle(tmp_path):
    bundle(tmp_path, "latest", 2)
    path = tmp_path / "latest_resume_commit.json"
    document = json.loads(path.read_text())
    document["files"]["../outside"] = "ignored"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unexpected paths"):
        verify_native_bundle(tmp_path, "latest")


def test_optional_metrics_and_exports_are_integrity_bound(tmp_path):
    (tmp_path / "latest_ood_reservoir.npz").write_bytes(b"reservoir")
    bundle(tmp_path, "latest", 2)
    (tmp_path / "latest_ood_reservoir.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity differs"):
        verify_native_bundle(tmp_path, "latest")


def test_stale_optional_files_are_not_published_when_disabled(tmp_path):
    bundle(tmp_path, "latest", 2)
    (tmp_path / "latest_export.safetensors").write_bytes(b"old export")
    (tmp_path / "latest_tag_metrics.npz").write_bytes(b"old metrics")
    commit_native_bundle(tmp_path, "latest", optional_suffixes=[])
    files = verify_native_bundle(tmp_path, "latest")["files"]
    assert "latest_export.safetensors" not in files
    assert "latest_tag_metrics.npz" not in files


def test_requested_optional_artifact_must_exist(tmp_path):
    bundle(tmp_path, "latest", 2)
    with pytest.raises(OSError):
        commit_native_bundle(tmp_path, "latest", optional_suffixes=["_tag_metrics.npz"])


def test_native_sidecar_save_errors_cannot_publish_stale_data(tmp_path):
    from core.tagger.tagger_trainer import _save_vocabulary_snapshot, _save_tag_metrics, _save_ood_reference
    class Broken:
        has_data = True
        n_seen = 10
        num_tags = 0
        idx_to_tag = {}
        def to_dict(self):
            raise OSError("unavailable storage")
        def save(self, *args, **kwargs):
            raise OSError("unavailable storage")
        def finalize(self, *args, **kwargs):
            raise OSError("unavailable storage")
    value = Broken()
    for call in (
        lambda: _save_vocabulary_snapshot(value, str(tmp_path), "latest", strict=True),
        lambda: _save_tag_metrics(value, str(tmp_path), "latest", value, False, strict=True),
        lambda: _save_ood_reference(value, str(tmp_path), "latest", strict=True),
    ):
        with pytest.raises(OSError, match="unavailable storage"):
            call()


def test_native_best_reservoir_does_not_overwrite_committed_latest(tmp_path):
    from core.tagger.tagger_trainer import _save_ood_reference
    class Accumulator:
        n_seen = 10
        def finalize(self, path):
            from pathlib import Path
            Path(path).write_bytes(b"reference")
        def save_reservoir(self, path):
            from pathlib import Path
            Path(path).write_bytes(b"new reservoir")
    latest = tmp_path / "latest_ood_reservoir.npz"
    latest.write_bytes(b"committed latest")
    _save_ood_reference(Accumulator(), str(tmp_path), "best_f1", strict=True)
    assert latest.read_bytes() == b"committed latest"
    assert (tmp_path / "best_f1_ood_reservoir.npz").read_bytes() == b"new reservoir"
