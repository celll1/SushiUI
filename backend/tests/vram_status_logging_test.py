from __future__ import annotations

import os
import sys

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.vram_optimization import log_device_status


class _Pipeline:
    def __getattribute__(self, name):
        if name.startswith("text_encoder") or name in {"unet", "vae"}:
            raise AssertionError("normal status logging inspected model components")
        return super().__getattribute__(name)


def test_normal_status_logging_does_not_walk_models(capsys):
    log_device_status("denoise ready", _Pipeline())

    output = capsys.readouterr().out
    assert "denoise ready" in output


def test_detailed_status_logging_remains_available(capsys):
    log_device_status("diagnostic", None, show_details=True)

    output = capsys.readouterr().out
    assert "diagnostic" in output
