from __future__ import annotations

import os
import sys

import pytest
import torch

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.inference import custom_sampling
from core.models.anima.anima_pipeline_ops import _apply_advanced_cfg
from core.models.ideogram4.ideogram4_pipeline_ops import _blend_guidance as ideogram_blend
from core.models.krea2.krea2_pipeline_ops import _blend_guidance as krea_blend
from core.models.lens.lens_pipeline_ops import _apply_advanced_cfg_lens


@pytest.mark.parametrize(
    "apply_cfg",
    [
        lambda cond, uncond, collect: _apply_advanced_cfg(
            cond, uncond, 4.0, 0.5, 1.0, {"developer_mode": True}, collect
        )[0],
        lambda cond, uncond, collect: _apply_advanced_cfg_lens(
            cond, uncond, 4.0, 0.5, 1.0, {"developer_mode": True}, collect
        )[0],
        lambda cond, uncond, collect: krea_blend(
            cond, uncond, 3.0, 0.5, {"developer_mode": True}, collect
        )[0],
        lambda cond, uncond, collect: ideogram_blend(
            cond, uncond, 4.0, 0.5, {"developer_mode": True}, collect
        )[0],
    ],
)
def test_architecture_cfg_helpers_skip_unrequested_metrics(monkeypatch, apply_cfg):
    calls = []
    original = custom_sampling.calculate_cfg_metrics

    def observe(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(custom_sampling, "calculate_cfg_metrics", observe)
    cond = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    uncond = torch.tensor([[0.1, -0.2], [0.0, 0.3]])

    without_metrics = apply_cfg(cond, uncond, False)
    assert calls == []

    with_metrics = apply_cfg(cond, uncond, True)
    assert len(calls) == 1
    assert torch.equal(without_metrics, with_metrics)
