"""`DiffusionPipelineManager.generate_txt2aud` dispatch onto MiniMax Music 3
vs. ACE-Step vs. neither (weight-free).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_dispatch_test.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.pipeline import DiffusionPipelineManager


def test_dispatches_to_minimax_music3_backend(monkeypatch):
    manager = DiffusionPipelineManager.__new__(DiffusionPipelineManager)
    manager.is_acestep_model = False
    manager.is_minimax_music3_model = True

    called = {}

    def _fake(params, progress_callback, step_callback):
        called["params"] = params
        called["progress_callback"] = progress_callback
        called["step_callback"] = step_callback
        return "music3-result"

    monkeypatch.setattr(manager, "_generate_txt2aud_minimax_music3", _fake, raising=False)

    result = manager.generate_txt2aud({"prompt": "x"}, progress_callback="cb", step_callback="sb")
    assert result == "music3-result"
    assert called["params"] == {"prompt": "x"}
    assert called["progress_callback"] == "cb"
    assert called["step_callback"] == "sb"


def test_dispatches_to_acestep_backend(monkeypatch):
    manager = DiffusionPipelineManager.__new__(DiffusionPipelineManager)
    manager.is_acestep_model = True
    manager.is_minimax_music3_model = False

    called = {}

    def _fake(params, progress_callback, step_callback):
        called["hit"] = True
        return "acestep-result"

    monkeypatch.setattr(manager, "_generate_txt2aud_acestep", _fake, raising=False)

    result = manager.generate_txt2aud({"prompt": "x"})
    assert result == "acestep-result"
    assert called.get("hit") is True


def test_neither_flag_set_raises_validation_error():
    manager = DiffusionPipelineManager.__new__(DiffusionPipelineManager)
    manager.is_acestep_model = False
    manager.is_minimax_music3_model = False

    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError):
        manager.generate_txt2aud({"prompt": "x"})


def test_acestep_takes_priority_when_both_flags_are_somehow_set(monkeypatch):
    # Defensive: mirrors the existing `if self.is_acestep_model: ...` /
    # `if self.is_minimax_music3_model: ...` ordering in generate_txt2aud --
    # ACE-Step is checked first. Both flags being True simultaneously should
    # never happen (model-load code always resets the sibling flags), but the
    # dispatch order itself is worth pinning.
    manager = DiffusionPipelineManager.__new__(DiffusionPipelineManager)
    manager.is_acestep_model = True
    manager.is_minimax_music3_model = True

    monkeypatch.setattr(manager, "_generate_txt2aud_acestep", lambda *a, **k: "acestep-result", raising=False)
    monkeypatch.setattr(manager, "_generate_txt2aud_minimax_music3", lambda *a, **k: "music3-result", raising=False)

    assert manager.generate_txt2aud({}) == "acestep-result"
