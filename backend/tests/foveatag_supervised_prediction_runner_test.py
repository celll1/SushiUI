"""CPU contracts for Phase-3 benchmark prediction metadata."""

from core.tagger.foveatag.supervised_prediction_runner import (
    _artifact_input_mode,
    _artifact_vocabulary,
    _view_seed,
)


def test_artifact_input_mode_accepts_only_registered_phase3_branches():
    assert _artifact_input_mode(
        {"provenance": {"input_mode": "source_native"}}
    ) == "source_native"

    try:
        _artifact_input_mode({"provenance": {"input_mode": "unsupported"}})
    except ValueError as error:
        assert "valid Phase-3 input mode" in str(error)
    else:
        raise AssertionError("expected semantic special-case mode to fail")


def test_prediction_view_seed_is_stable_per_sample_and_resolution():
    first = _view_seed("sample-1", "native")

    assert first == _view_seed("sample-1", "native")
    assert first != _view_seed("sample-1", "512")
    assert first != _view_seed("sample-2", "native")


def test_artifact_vocabulary_accepts_serialized_index_mapping():
    vocabulary = {
        "tag_to_idx": {"blue eyes": 0, "text": 1},
        "idx_to_tag": {"0": "blue eyes", "1": "text"},
    }

    assert _artifact_vocabulary({"vocabulary": vocabulary}, 2) is vocabulary
