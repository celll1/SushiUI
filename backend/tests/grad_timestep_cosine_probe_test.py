"""Guard: the sketched noisy-vs-clean gradient cosine must be a real cosine.

Why this file exists
--------------------
The probe answers whether gradients from distant timesteps CONFLICT (negative
cosine) or are merely uncorrelated (~0). That answer decides whether the
negative-transfer literature -- Min-SNR's multi-task framing, timestep
clustering, per-interval experts, PCGrad -- applies to this model at all, or
whether it is a variance problem that stratified sampling already handles. A
biased or mis-scaled estimator would answer it wrongly, so the estimator itself
needs pinning.

The gradients cannot be held twice (2 x 15 GiB per MoT half), so each is
compressed by a bilinear sketch S = L^T G R with L, R fixed Gaussians of
variance 1/k. That gives E<S_A, S_B> = <A, B>. These tests check the estimator
against the exact cosine on tensors small enough to compute both, plus the
operational contracts: the same projection must be used for both buckets, a
one-sided window must report nothing rather than a meaningless number, and a
failure inside the probe must never propagate into the backward pass.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.base_trainer import BaseTrainer as BaseTrainerRef
from core.training.probes.grad_timestep_cosine import GradTimestepCosineProbe


def _param(grad):
    p = torch.nn.Parameter(torch.zeros_like(grad))
    p.grad = grad
    return p


def _run(probe, passes):
    """passes = [(t, {param: grad}), ...]; grads are re-attached per pass."""
    probe.begin_window()
    for t, grads in passes:
        probe.begin_pass(t)
        for p, g in grads.items():
            p.grad = g
            probe.observe(p)
    return probe.finish_window()


def _exact_cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()))


@pytest.mark.parametrize("angle", [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi])
def test_sketched_cosine_tracks_the_true_cosine(angle):
    """The whole point: the sign and rough magnitude must survive the sketch."""
    torch.manual_seed(0)
    base = torch.randn(256, 256)
    orth = torch.randn(256, 256)
    orth -= (orth * base).sum() / (base * base).sum() * base  # make it orthogonal
    orth *= base.norm() / orth.norm()
    other = math.cos(angle) * base + math.sin(angle) * orth

    p = _param(base)
    probe = GradTimestepCosineProbe(t_split=0.5, sketch_dim=32)
    out = _run(probe, [(0.2, {p: base}), (0.8, {p: other})])

    assert out["grad_cos_t_all"] == pytest.approx(_exact_cos(base, other), abs=0.12)


def test_zero_cosine_is_reported_as_zero_not_as_conflict():
    """The decision this probe drives is 'uncorrelated vs conflicting', so an
    orthogonal pair must not read negative."""
    torch.manual_seed(1)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    b -= (b * a).sum() / (a * a).sum() * a
    p = _param(a)
    out = _run(GradTimestepCosineProbe(t_split=0.5, sketch_dim=32), [(0.1, {p: a}), (0.9, {p: b})])
    assert abs(out["grad_cos_t_all"]) < 0.1


def test_opposed_gradients_report_negative():
    torch.manual_seed(2)
    a = torch.randn(256, 256)
    p = _param(a)
    out = _run(GradTimestepCosineProbe(t_split=0.5, sketch_dim=16), [(0.1, {p: a}), (0.9, {p: -a})])
    assert out["grad_cos_t_all"] < -0.85


def test_larger_sketch_is_a_better_estimate():
    torch.manual_seed(3)
    a = torch.randn(256, 256)
    b = 0.5 * a + 0.5 * torch.randn(256, 256)
    truth = _exact_cos(a, b)
    p = _param(a)

    def err(k, trials=12):
        total = 0.0
        for seed in range(trials):
            out = _run(GradTimestepCosineProbe(t_split=0.5, sketch_dim=k, seed=seed),
                       [(0.1, {p: a}), (0.9, {p: b})])
            total += abs(out["grad_cos_t_all"] - truth)
        return total / trials

    assert err(32) < err(2)


def test_passes_accumulate_within_a_bucket():
    """Two passes in one bucket must sum, the way the real window accumulates."""
    torch.manual_seed(4)
    a = torch.randn(128, 128)
    p = _param(a)
    out = _run(GradTimestepCosineProbe(t_split=0.5, sketch_dim=32),
               [(0.1, {p: a}), (0.2, {p: a}), (0.9, {p: a})])
    # low bucket holds 2a, high holds a -> still perfectly aligned
    assert out["grad_cos_t_all"] == pytest.approx(1.0, abs=0.02)
    assert out["grad_cos_t_npass_low"] == 2.0
    assert out["grad_cos_t_npass_high"] == 1.0


def test_one_sided_window_reports_nothing():
    """All draws on one side of the split is normal without stratification; a
    cosine from one bucket is not a number worth charting."""
    torch.manual_seed(5)
    a = torch.randn(64, 64)
    p = _param(a)
    assert _run(GradTimestepCosineProbe(t_split=0.5, sketch_dim=8),
                [(0.1, {p: a}), (0.2, {p: a})]) == {}


def test_per_component_cosines_are_split_by_the_component_map():
    torch.manual_seed(6)
    gen_g, und_g = torch.randn(128, 128), torch.randn(128, 128)
    gen_p, und_p = _param(gen_g), _param(und_g)
    probe = GradTimestepCosineProbe(
        t_split=0.5, sketch_dim=32,
        components={id(gen_p): "unet", id(und_p): "te1"},
    )
    # gen agrees across t, und opposes across t -- the two must not be blended.
    out = _run(probe, [
        (0.1, {gen_p: gen_g, und_p: und_g}),
        (0.9, {gen_p: gen_g, und_p: -und_g}),
    ])
    assert out["grad_cos_t_unet"] > 0.85
    assert out["grad_cos_t_te1"] < -0.85


def test_component_names_match_the_metric_registry():
    """A series the chart has no entry for is invisible, so the names must agree."""
    from core.training.metric_registry import EXTRA_METRIC_DEFS

    for name in ("grad_cos_t_all", "grad_cos_t_unet", "grad_cos_t_te1"):
        assert name in EXTRA_METRIC_DEFS, name


def test_non_2d_parameters_are_skipped_not_crashed():
    """Norms and biases are trainable too and have no bilinear sketch."""
    torch.manual_seed(7)
    mat, vec = torch.randn(64, 64), torch.randn(64)
    mp, vp = _param(mat), _param(vec)
    out = _run(GradTimestepCosineProbe(t_split=0.5, sketch_dim=16),
               [(0.1, {mp: mat, vp: vec}), (0.9, {mp: mat, vp: vec})])
    assert out["grad_cos_t_all"] == pytest.approx(1.0, abs=0.02)


def test_observe_never_raises_into_the_backward_pass():
    """It is called from inside a post-accumulate-grad hook; a diagnostic must
    not be able to take down a training step."""
    probe = GradTimestepCosineProbe(t_split=0.5, sketch_dim=8)
    probe.begin_window()
    probe.begin_pass(0.1)

    class Exploding:
        @property
        def grad(self):
            raise RuntimeError("boom")

    probe.observe(Exploding())  # must not propagate

    p = _param(torch.randn(32, 32))
    p.grad = None
    probe.observe(p)  # no gradient yet is also fine


def test_observe_outside_a_pass_is_a_noop():
    probe = GradTimestepCosineProbe(t_split=0.5, sketch_dim=8)
    probe.begin_window()
    p = _param(torch.randn(32, 32))
    probe.observe(p)  # begin_pass was never called
    assert probe.finish_window() == {}


def test_projection_is_stable_across_buckets_and_windows():
    """The two buckets must be sketched by the SAME projection, or the inner
    product is meaningless. Also across windows, so the series is comparable."""
    probe = GradTimestepCosineProbe(t_split=0.5, sketch_dim=8, seed=11)
    first = probe._projection(64, torch.device("cpu"), torch.float32).clone()
    probe.begin_window()
    again = probe._projection(64, torch.device("cpu"), torch.float32)
    assert torch.equal(first, again)

    fresh = GradTimestepCosineProbe(t_split=0.5, sketch_dim=8, seed=11)
    assert torch.equal(first, fresh._projection(64, torch.device("cpu"), torch.float32))


def test_probe_is_armed_after_the_optimizer_exists():
    """Regression: it was armed next to the timestep sampler, which runs BEFORE
    setup_optimizer. `use_fused_backward` is set while the optimizer is built,
    so the probe's own guard disabled it every time and it silently produced no
    data. The two conditions it needs -- the fused path and the adapter's
    parameter classification -- only exist after setup_optimizer."""
    import inspect

    from core.training.base_trainer import BaseTrainer

    src = inspect.getsource(BaseTrainer.train)
    setup = src.index("self.setup_optimizer(")
    arm = src.index("self._maybe_build_grad_t_cos_probe(")
    assert setup < arm, "probe armed before setup_optimizer; it will disable itself"


def test_probe_refuses_without_the_fused_backward_path():
    class Harness:
        _maybe_build_grad_t_cos_probe = BaseTrainerRef._maybe_build_grad_t_cos_probe
        _grad_t_cos_components = BaseTrainerRef._grad_t_cos_components

        def __init__(self, fused):
            self.config = {"grad_timestep_cosine_probe": True}
            self.log_prefix = "[Test]"
            self.use_fused_backward = fused

        def _full_parameter_grad_components(self):
            return {}

    class _Sampler:
        def icdf(self, u):
            return torch.full_like(u, 0.31)

    h = Harness(False)
    h._maybe_build_grad_t_cos_probe(_Sampler(), 16)
    assert h._grad_t_cos_probe is None

    h = Harness(True)
    h._maybe_build_grad_t_cos_probe(_Sampler(), 16)
    assert h._grad_t_cos_probe is not None
    assert h._grad_t_cos_probe.t_split == pytest.approx(0.31)

    # MNT=1 has no second bucket to compare against.
    h = Harness(True)
    h._maybe_build_grad_t_cos_probe(_Sampler(), 1)
    assert h._grad_t_cos_probe is None


def test_projection_does_not_consume_the_training_rng():
    """A diagnostic consuming the training RNG has bitten this repo before
    (commit 28377024)."""
    torch.manual_seed(99)
    expected = torch.randn(4)

    torch.manual_seed(99)
    probe = GradTimestepCosineProbe(t_split=0.5, sketch_dim=8)
    probe._projection(128, torch.device("cpu"), torch.float32)
    probe._projection(4096, torch.device("cpu"), torch.float32)
    assert torch.equal(torch.randn(4), expected)
