"""CPU contracts for Phase-1 benchmark prediction setup."""

import hashlib

from core.tagger.foveatag.control_prediction_runner import (
    _control_artifact_hashes,
    _view_seed,
)


def test_control_prediction_view_seed_is_stable_and_resolution_specific():
    assert _view_seed("sample-1", "native") == _view_seed("sample-1", "native")
    assert _view_seed("sample-1", "native") != _view_seed("sample-1", "384")
    assert _view_seed("sample-1", "native") != _view_seed("sample-2", "native")


def test_control_prediction_hashes_weights_and_metadata(tmp_path):
    weights = tmp_path / "best_f1.safetensors"
    metadata = tmp_path / "best_f1_metadata.json"
    weights.write_bytes(b"weights")
    metadata.write_bytes(b"metadata")

    assert _control_artifact_hashes(weights) == {
        "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
        "checkpoint_metadata_sha256": hashlib.sha256(b"metadata").hexdigest(),
    }
