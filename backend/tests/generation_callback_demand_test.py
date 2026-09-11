from __future__ import annotations

import os
import sys

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from api.generation_utils import create_progress_callback_factory
from core.inference.callback_utils import callback_requests, compose_sampler_callbacks


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


def test_composed_callbacks_keep_both_calling_conventions():
    calls = []

    def progress(step, total, latents, metrics, pred_x0):
        calls.append(("progress", step, total, latents, metrics, pred_x0))

    progress.wants_predicted_x0 = lambda step, total: step == total - 1
    progress.wants_cfg_metrics = lambda step, total: step == 0

    def diffusers_step(pipe, step, timestep, kwargs):
        calls.append(("step", pipe, step, timestep, kwargs))
        return kwargs

    callback = compose_sampler_callbacks(progress, diffusers_step)
    callback(2, 3, "latents", "metrics", "x0")

    assert calls == [
        ("step", None, 2, None, {"latents": "latents"}),
        ("progress", 2, 3, "latents", "metrics", "x0"),
    ]
    assert callback_requests(callback, "wants_predicted_x0", 2, 3)
    assert not callback_requests(callback, "wants_cfg_metrics", 2, 3)


def test_step_only_callback_declines_preview_diagnostics():
    calls = []
    callback = compose_sampler_callbacks(
        None, lambda pipe, step, timestep, kwargs: calls.append((step, kwargs))
    )

    callback(1, 4, "latents")

    assert calls == [(1, {"latents": "latents"})]
    assert not callback_requests(callback, "wants_predicted_x0", 1, 4)
    assert not callback_requests(callback, "wants_cfg_metrics", 1, 4)
