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
