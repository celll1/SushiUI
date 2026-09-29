"""CPU contracts for Phase-1 benchmark prediction setup."""

from core.tagger.foveatag.control_prediction_runner import _view_seed


def test_control_prediction_view_seed_is_stable_and_resolution_specific():
    assert _view_seed("sample-1", "native") == _view_seed("sample-1", "native")
    assert _view_seed("sample-1", "native") != _view_seed("sample-1", "384")
    assert _view_seed("sample-1", "native") != _view_seed("sample-2", "native")
