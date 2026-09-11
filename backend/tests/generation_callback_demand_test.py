from __future__ import annotations

import os
import sys

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from api.generation_utils import create_progress_callback_factory
from core.inference.callback_utils import callback_requests


def _callback(*, predicted_x0=True, enabled=True, interval=4):
    return create_progress_callback_factory(
        None,
        None,
        is_sdxl=False,
        preview_predicted_x0=predicted_x0,
        preview_enabled=enabled,
        preview_interval=interval,
    )


def test_factory_declares_only_preview_steps():
    callback = _callback()

    requested = [
        callback_requests(callback, "wants_predicted_x0", step, 10)
        for step in range(10)
    ]
    assert requested == [True, False, False, False, True, False, False, False, True, True]
    assert callback_requests(callback, "wants_cfg_metrics", 4, 10)
    assert not callback_requests(callback, "wants_cfg_metrics", 5, 10)


def test_factory_declines_unused_or_disabled_predicted_x0():
    assert not callback_requests(_callback(predicted_x0=False), "wants_predicted_x0", 0, 10)
    assert not callback_requests(_callback(enabled=False), "wants_predicted_x0", 0, 10)


def test_unknown_callbacks_preserve_previous_behavior():
    assert callback_requests(lambda *args: None, "wants_predicted_x0", 3, 10)
    assert not callback_requests(None, "wants_predicted_x0", 3, 10)

    def broken_predicate(*args):
        raise RuntimeError("callback-owned predicate failed")

    callback = lambda *args: None
    callback.wants_predicted_x0 = broken_predicate
    assert callback_requests(callback, "wants_predicted_x0", 3, 10)
