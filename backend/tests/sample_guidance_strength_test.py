"""Guard: conditioning strength must be readable without costing a training step.

Why this file exists
--------------------
The direct question a conditioning fine-tune has to answer is "is the text
conditioning still doing anything", and the training loss cannot answer it: the
`loss_cond` / `loss_null` split is drawn once per MNT window at a 10% rate, so a
4k-step bin holds ~20 null IMAGES and the gap it reports is dominated by
image-to-image variance.

What CFG actually consumes is `v_cond - v_uncond`. Both branches are already
computed at every step of any cfg_scale > 1 generation -- including the periodic
sample a training run makes anyway -- so the guidance strength comes out for the
price of a norm over tensors already in memory. There is no per-training-
iteration cost, which is the whole constraint this was built under.

So these tests pin: the collector is inert unless armed, arming it does not
change a single value the generation produces, and the numbers actually
discriminate a collapsed conditioning from a healthy one.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inference.custom_sampling import calculate_cfg_metrics
from core.models.sensenova import sensenova_pipeline_ops as ops
from core.training.ops.sensenova_ops import _log_sample_guidance


def _pair(seed=0, mix=0.1):
    torch.manual_seed(seed)
    cond = torch.randn(1, 64, 48)
    uncond = cond * (1 - mix) + mix * torch.randn_like(cond)
    return cond, uncond


def test_collector_is_disarmed_by_default():
    """The cost when nobody armed it is one module-global lookup per denoise
    step, and nothing whatsoever per training iteration."""
    assert ops._GUIDANCE_COLLECTOR is None
    cond, uncond = _pair()
    ops._cfg_combine(cond, uncond, 4.0, "global", 0)  # must not raise or record
    assert ops._GUIDANCE_COLLECTOR is None


def test_arming_does_not_change_the_generation():
    """A diagnostic that perturbs the image it measures is worthless."""
    cond, uncond = _pair()
    plain = ops._cfg_combine(cond, uncond, 4.0, "global", 3)
    with ops.collect_guidance_metrics():
        armed = ops._cfg_combine(cond, uncond, 4.0, "global", 3)
    assert torch.equal(plain, armed)

    for norm in ("none", "channel", "cfg_zero_star"):
        a = ops._cfg_combine(cond, uncond, 4.0, norm, 3)
        with ops.collect_guidance_metrics():
            b = ops._cfg_combine(cond, uncond, 4.0, norm, 3)
        assert torch.equal(a, b), norm


def test_one_row_per_denoise_step():
    cond, uncond = _pair()
    with ops.collect_guidance_metrics() as rows:
        for i in range(5):
            ops._cfg_combine(cond, uncond, 4.0, "global", i)
    assert [r["step"] for r in rows] == [0, 1, 2, 3, 4]
    assert set(rows[0]) == {"step", "relative_diff", "cosine_similarity"}


def test_collapsed_conditioning_reads_as_collapsed():
    """v_cond == v_uncond is exactly what a dead text conditioning looks like:
    guidance magnitude 0, branches perfectly aligned. cfg_scale then does
    nothing to the image no matter what it is set to."""
    cond = torch.randn(1, 32, 16)
    with ops.collect_guidance_metrics() as rows:
        ops._cfg_combine(cond, cond.clone(), 4.0, "none", 0)
    assert rows[0]["relative_diff"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["cosine_similarity"] == pytest.approx(1.0, abs=1e-4)


def test_guidance_strength_is_monotone_in_branch_separation():
    """The series has to move in the right direction to be readable as a trend."""
    seen = []
    for mix in (0.02, 0.1, 0.4):
        cond, uncond = _pair(seed=1, mix=mix)
        with ops.collect_guidance_metrics() as rows:
            ops._cfg_combine(cond, uncond, 4.0, "none", 0)
        seen.append((rows[0]["relative_diff"], rows[0]["cosine_similarity"]))
    rels = [r for r, _ in seen]
    coss = [c for _, c in seen]
    assert rels == sorted(rels), rels
    assert coss == sorted(coss, reverse=True), coss


def test_nesting_restores_the_outer_collector():
    cond, uncond = _pair()
    with ops.collect_guidance_metrics() as outer:
        with ops.collect_guidance_metrics() as inner:
            ops._cfg_combine(cond, uncond, 4.0, "none", 0)
        ops._cfg_combine(cond, uncond, 4.0, "none", 1)
    assert len(inner) == 1 and len(outer) == 1
    assert ops._GUIDANCE_COLLECTOR is None


def test_recording_never_breaks_a_generation():
    """It runs inside the denoise loop of a real generation."""
    with ops.collect_guidance_metrics() as rows:
        ops._record_guidance(torch.randn(4), "not a tensor", 4.0, 0)
    assert rows == []


def test_calculate_cfg_metrics_returns_the_cosine_it_documents():
    """Its docstring listed cosine_similarity as a key metric but the dict never
    carried it; the magnitude metrics alone cannot separate 'the branches agree'
    from 'both predictions shrank'."""
    a = torch.randn(2, 8, 8)
    m = calculate_cfg_metrics(a, a.clone(), 4.0, developer_mode=True)
    assert m["cosine_similarity"] == pytest.approx(1.0, abs=1e-4)
    assert calculate_cfg_metrics(a, -a, 4.0, developer_mode=True)["cosine_similarity"] < -0.99
    assert calculate_cfg_metrics(a, a, 4.0, developer_mode=False) is None


def test_zero_norm_branch_does_not_divide_by_zero():
    z = torch.zeros(2, 4, 4)
    m = calculate_cfg_metrics(z, z, 4.0, developer_mode=True)
    assert m["cosine_similarity"] == 0.0
    assert m["relative_diff"] == 0.0


# ------------------------------------------------------------------ log path

class LogHarness:
    def __init__(self):
        self.logged = {}
        self.log_prefix = "[Test]"

    def log_extra_metric(self, key, value):
        self.logged[key] = value


def test_logged_series_are_the_registered_ones():
    from core.training.metric_registry import EXTRA_METRIC_DEFS

    h = LogHarness()
    _log_sample_guidance(h, [
        {"step": i, "relative_diff": 0.1 * (i + 1), "cosine_similarity": 0.9}
        for i in range(6)
    ])
    assert set(h.logged) == {"cfg_guidance_rel", "cfg_guidance_rel_early", "cfg_guidance_cos"}
    for key in h.logged:
        assert key in EXTRA_METRIC_DEFS, key
    # mean of 0.1..0.6, and of the first third (0.1, 0.2)
    assert h.logged["cfg_guidance_rel"] == pytest.approx(0.35)
    assert h.logged["cfg_guidance_rel_early"] == pytest.approx(0.15)
    assert h.logged["cfg_guidance_cos"] == pytest.approx(0.9)


def test_no_steps_logs_nothing():
    """cfg_scale <= 1 builds no unconditional branch, and a failed sample has no
    trajectory at all. Neither should write a misleading zero."""
    h = LogHarness()
    _log_sample_guidance(h, [])
    _log_sample_guidance(h, None)
    assert h.logged == {}


def test_log_failure_does_not_propagate():
    class Broken(LogHarness):
        def log_extra_metric(self, key, value):
            raise RuntimeError("boom")

    _log_sample_guidance(Broken(), [{"step": 0, "relative_diff": 1.0, "cosine_similarity": 1.0}])
