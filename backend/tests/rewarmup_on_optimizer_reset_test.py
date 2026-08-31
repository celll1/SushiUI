"""Guard: a resume that gets a FRESH optimizer must re-apply its warmup.

Why this file exists
--------------------
``BaseTrainer.train()`` fast-forwards every LR scheduler to the resumed step and
only then calls ``load_optimizer_state()``, because ``Optimizer.load_state_dict``
would otherwise reinstate the checkpoint's ``lr``. That order is right while the
state restores. When it does not -- the ``_optimizer.pt`` was pruned or never
written, or the load was rejected because the optimizer type or trainable
parameter set changed -- the run continued with ``exp_avg``/``exp_avg_sq`` at
zero while the schedule sat past its warmup. Adam's first step with a zero second
moment is ``~lr`` on every parameter at once regardless of the gradient, which is
the transient warmup exists to damp.

Run 121 took that path twice, at steps 8400 and 39672 ("No optimizer state file
found" -> "Starting with fresh optimizer state"), each time resuming a zeroed
optimizer straight into lr 1e-6.

The tests drive the real ``BaseTrainer`` helpers against real ``torch.optim``
objects (no model, no dataset, no GPU). What they pin down:

* the ramp is re-applied from the RESUMED step, not from 0;
* the underlying schedule keeps its absolute position (a cosine/plateau segment
  is not restarted);
* every scheduler is re-armed, not just ``self.lr_scheduler`` -- fused optimizer
  groups each carry their own;
* it is inert when the state restores, when the warmup is 0, and when the
  config turns it off.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.base_trainer import BaseTrainer


class RearmHarness:
    """Minimal stand-in exposing only the methods under test."""

    _rearm_warmup_after_optimizer_reset = BaseTrainer._rearm_warmup_after_optimizer_reset
    _fast_forward_lr_schedulers = BaseTrainer._fast_forward_lr_schedulers
    # Both are @staticmethod on BaseTrainer; reading them off the class yields a
    # plain function, so they must be re-wrapped or the harness would rebind
    # them as instance methods and pass `self` as the first argument.
    _compose_warmup_lambda = staticmethod(BaseTrainer._compose_warmup_lambda)
    _fast_forward_one_lr_scheduler = staticmethod(BaseTrainer._fast_forward_one_lr_scheduler)

    def __init__(self, warmup=1000, config=None, n_groups=1, base_lr=1e-6):
        self.log_prefix = "[Test]"
        self.optimizer_warmup_steps = warmup
        self.config = config if config is not None else {}
        self.optimizers = []
        self.lr_schedulers = []
        for _ in range(n_groups):
            param = torch.nn.Parameter(torch.zeros(2))
            optimizer = torch.optim.AdamW([param], lr=base_lr)
            self.optimizers.append(optimizer)
            self.lr_schedulers.append(
                torch.optim.lr_scheduler.LambdaLR(optimizer, _plateau_then_half(warmup))
            )
        self.optimizer = self.optimizers[0]
        self.lr_scheduler = self.lr_schedulers[0]
        # all_lr_schedulers() only consults `lr_schedulers` when this is set;
        # otherwise it returns [lr_scheduler]. Mirror what _setup_fused_
        # optimizer_groups does so the multi-group tests exercise the real
        # fan-out rather than silently testing group 0 twice.
        self.fused_optimizer_groups = self.optimizers if n_groups > 1 else None

    def multiplier(self, step, which=0):
        """The schedule multiplier the live lambda would produce at ``step``."""
        return float(self.lr_schedulers[which].lr_lambdas[0](step))


def _plateau_then_half(warmup):
    """warmup ramp -> 1.0 -> 0.5 after step 50000. Stands in for any real schedule."""

    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return step / float(warmup)
        return 1.0 if step < 50000 else 0.5

    return lr_lambda


def test_rearm_applies_the_ramp_from_the_resumed_step():
    h = RearmHarness(warmup=1000)
    h._fast_forward_lr_schedulers(60000)
    assert h.multiplier(60000) == pytest.approx(0.5), "precondition: past warmup"

    assert h._rearm_warmup_after_optimizer_reset(60000) is True

    # Ramp restarts at the resume point...
    assert h.multiplier(60000) == pytest.approx(0.0)
    assert h.multiplier(60250) == pytest.approx(0.5 * 0.25)
    assert h.multiplier(60500) == pytest.approx(0.5 * 0.50)
    # ...and is fully open again one warmup later.
    assert h.multiplier(61000) == pytest.approx(0.5)
    assert h.multiplier(70000) == pytest.approx(0.5)


def test_underlying_schedule_keeps_its_position():
    """The decay boundary must NOT move: only an attenuation is composed on top."""
    h = RearmHarness(warmup=1000)
    h._fast_forward_lr_schedulers(40000)
    h._rearm_warmup_after_optimizer_reset(40000)

    # Warmup is over by 41000, so the composed value is the bare schedule again.
    assert h.multiplier(49999) == pytest.approx(1.0)
    # The step-50000 boundary of the ORIGINAL schedule still fires at 50000,
    # not at 50000 + 40000. A rewind-the-schedule fix would fail here.
    assert h.multiplier(50000) == pytest.approx(0.5)


def test_every_fused_group_scheduler_is_rearmed():
    h = RearmHarness(warmup=1000, n_groups=4)
    h._fast_forward_lr_schedulers(60000)

    assert h._rearm_warmup_after_optimizer_reset(60000) is True

    for i in range(4):
        assert h.multiplier(60000, which=i) == pytest.approx(0.0), f"group {i}"
        assert h.multiplier(60500, which=i) == pytest.approx(0.25), f"group {i}"


def test_each_wrapper_keeps_its_own_inner_lambda():
    """A loop-body closure would make every scheduler share the last lambda."""
    h = RearmHarness(warmup=100, n_groups=3)
    for i, scheduler in enumerate(h.lr_schedulers):
        scheduler.lr_lambdas = [lambda step, k=i: float(k + 1)]

    h._rearm_warmup_after_optimizer_reset(0)

    for i in range(3):
        assert h.multiplier(100, which=i) == pytest.approx(float(i + 1))


def test_noop_when_warmup_is_zero():
    h = RearmHarness(warmup=0)
    h._fast_forward_lr_schedulers(60000)
    before = h.multiplier(60000)

    assert h._rearm_warmup_after_optimizer_reset(60000) is False
    assert h.multiplier(60000) == pytest.approx(before)


def test_noop_when_disabled_in_config():
    h = RearmHarness(warmup=1000, config={"rewarmup_on_optimizer_reset": False})
    h._fast_forward_lr_schedulers(60000)

    assert h._rearm_warmup_after_optimizer_reset(60000) is False
    assert h.multiplier(60000) == pytest.approx(0.5), "schedule left untouched"


def test_default_is_on_without_an_explicit_config_key():
    """The default comes from param_defaults, not from a literal at the callsite."""
    from api.param_defaults import TRAINING_DEFAULTS

    assert TRAINING_DEFAULTS["rewarmup_on_optimizer_reset"] is True
    h = RearmHarness(warmup=1000, config={})
    assert h._rearm_warmup_after_optimizer_reset(60000) is True


def test_non_lambdalr_scheduler_is_skipped_not_crashed():
    """ReLoRA's restart scheduler has no lr_lambdas to compose against."""

    class NoLambdas:
        pass

    h = RearmHarness(warmup=1000, n_groups=2)
    h.lr_schedulers[1] = NoLambdas()

    assert h._rearm_warmup_after_optimizer_reset(60000) is True
    assert h.multiplier(60500, which=0) == pytest.approx(0.25)


def test_written_lr_is_ramped_before_the_first_post_resume_step():
    """The re-arm must precede _reassert_config_lr_on_resume, which evaluates
    the live lambdas -- the loop steps the optimizer BEFORE the scheduler, so a
    group left at the un-ramped LR would take one full-size step anyway."""
    from core.training.lr_utils import reassert_config_lr

    h = RearmHarness(warmup=1000, base_lr=1e-6)
    h._fast_forward_lr_schedulers(60000)
    h._rearm_warmup_after_optimizer_reset(60000)

    reassert_config_lr(h.optimizer, h.lr_scheduler, 1e-6, verbose=False)

    # base 1e-6 * schedule 0.5 * warmup factor 0.0 at the resume step itself.
    assert h.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    # And 250 steps in it is a quarter of the way up, not the full rate.
    h._rearm_warmup_after_optimizer_reset.__get__(h)  # no-op; keeps ruff quiet
    h.lr_scheduler.last_epoch = 60250
    reassert_config_lr(h.optimizer, h.lr_scheduler, 1e-6, verbose=False)
    assert h.optimizer.param_groups[0]["lr"] == pytest.approx(1e-6 * 0.5 * 0.25)


def test_ramp_shape_matches_the_configured_warmup():
    """Same linear shape diffusers' get_scheduler uses, so re-armed and
    first-run warmups are the same curve."""
    h = RearmHarness(warmup=800)
    h._fast_forward_lr_schedulers(10000)
    h._rearm_warmup_after_optimizer_reset(10000)

    for k in (0, 100, 400, 799, 800, 1600):
        expected = min(1.0, k / 800.0)
        assert h.multiplier(10000 + k) == pytest.approx(expected), k
    assert not math.isnan(h.multiplier(10000))
