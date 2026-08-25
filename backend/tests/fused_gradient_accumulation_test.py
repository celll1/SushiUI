"""Gradient accumulation does not happen under fused backward, and now says so.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/fused_gradient_accumulation_test.py -v

The fused hooks apply each parameter's update and free its gradient as soon as it
exists, so nothing survives to be summed across the window: every backward pass
becomes its own optimizer step. With gradient_accumulation_steps=4 that is four
steps on four single-batch gradients (each scaled by 1/4), not one step on their
mean -- while the LR schedule and the reported step count still move once per
four backwards.

The numeric tests below are the negative control: they run the same four
backwards through a fused and a non-fused optimizer of the same type and show the
results differ, which is what "silently a different training run" means. A plain
SGD arm is the counter-control -- for a linear update rule the two are equal, so
the divergence is the optimizer's non-linearity, not test noise.

CPU-only, no CUDA and no model: the mechanism is the hooks and the optimizer.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest

import torch
from torch import nn

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import core.training.optimizers.fused_optimizer_groups as fog  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402

ACCUM = 4
LR = 1e-2
DIM = 4
# Constant per-micro-batch gradients: d/dw of (w * x).sum() is x, independent of w,
# so "one step on the mean" and "four steps on the quarters" can be compared
# without the weight trajectory feeding back into the gradients.
GRADS = [torch.tensor([1.0, -2.0, 3.0, -4.0]) * (i + 1) for i in range(ACCUM)]


def _weights():
    return nn.Parameter(torch.zeros(DIM))


def _backward(param, grad, accum=ACCUM):
    """One micro-batch backward, with the trainer's 1/accum loss scaling."""
    ((param * grad).sum() / accum).backward()


class _CountingOptimizer:
    """Wraps an optimizer and counts the steps the hooks actually take."""

    def __init__(self, inner):
        self.inner = inner
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self, *args, **kwargs):
        self.steps += 1
        return self.inner.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        return self.inner.zero_grad(*args, **kwargs)


def _fused_run(make_optimizer, reset_per_backward=True):
    """The fused-groups path: hooks step as each backward completes."""
    param = _weights()
    optimizer = _CountingOptimizer(make_optimizer([param]))
    groups = fog.FusedOptimizerGroups([optimizer], max_grad_norm=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        groups.register_hooks()
    for grad in GRADS:
        if reset_per_backward:
            groups.reset_counters()  # base_trainer does this once per batch
        _backward(param, grad)
    return param.detach().clone(), optimizer.steps


def _non_fused_run(make_optimizer):
    """The reference: accumulate over the window, then step once."""
    param = _weights()
    optimizer = make_optimizer([param])
    for grad in GRADS:
        _backward(param, grad)
    optimizer.step()
    optimizer.zero_grad()
    return param.detach().clone()


def _separate_steps(make_optimizer):
    """Four independent steps on the quarter gradients, written out by hand."""
    param = _weights()
    optimizer = make_optimizer([param])
    for grad in GRADS:
        _backward(param, grad)
        optimizer.step()
        optimizer.zero_grad()
    return param.detach().clone()


def _adamw(params):
    return torch.optim.AdamW(params, lr=LR, weight_decay=0.0)


def _sgd(params):
    return torch.optim.SGD(params, lr=LR)


class TheWindowDoesNotSurviveTest(unittest.TestCase):
    def test_the_hooks_step_once_per_backward_not_once_per_window(self):
        _, steps = _fused_run(_adamw)
        self.assertEqual(steps, ACCUM)

    def test_no_gradient_is_left_to_accumulate(self):
        param = _weights()
        optimizer = _CountingOptimizer(_adamw([param]))
        groups = fog.FusedOptimizerGroups([optimizer], max_grad_norm=0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            groups.register_hooks()
        for grad in GRADS:
            groups.reset_counters()
            _backward(param, grad)
            self.assertIsNone(param.grad, "the hook frees the gradient every backward")

    def test_the_non_fused_reference_really_does_accumulate(self):
        param = _weights()
        _adamw([param])
        for i, grad in enumerate(GRADS):
            _backward(param, grad)
            self.assertIsNotNone(param.grad)
            expected = sum(GRADS[: i + 1]) / ACCUM
            self.assertTrue(torch.allclose(param.grad, expected, atol=1e-6))


class TheTwoRunsDifferTest(unittest.TestCase):
    """Negative control: fused and non-fused train differently on the same data."""

    def test_fused_adamw_is_four_separate_steps(self):
        fused, _ = _fused_run(_adamw)
        self.assertTrue(torch.allclose(fused, _separate_steps(_adamw), atol=1e-9))

    def test_fused_adamw_moves_much_further_than_the_accumulated_step(self):
        fused, _ = _fused_run(_adamw)
        reference = _non_fused_run(_adamw)
        fused_distance = fused.norm().item()
        reference_distance = reference.norm().item()
        self.assertGreater(reference_distance, 0.0)
        # AdamW's update is near scale-invariant, so dividing each micro-batch
        # loss by 4 does not shrink its step: four of them move ~4x as far.
        self.assertGreater(fused_distance / reference_distance, 3.0)
        self.assertFalse(torch.allclose(fused, reference, atol=1e-4))

    def test_plain_sgd_is_the_counter_control(self):
        # A linear update rule sums to the same place, so the divergence above is
        # the optimizer's non-linearity and not an artifact of the harness.
        fused, steps = _fused_run(_sgd)
        self.assertEqual(steps, ACCUM)
        self.assertTrue(torch.allclose(fused, _non_fused_run(_sgd), atol=1e-9))


# --------------------------------------------------------------------------
# The warning
# --------------------------------------------------------------------------


class _Trainer:
    log_prefix = "[test]"

    def __init__(self, fused=False, groups=None):
        self.use_fused_backward = fused
        self.fused_optimizer_groups = groups
        self._grad_accum_steps = ACCUM

    _warn_gradient_accumulation_ignored_under_fused = (
        BaseTrainer._warn_gradient_accumulation_ignored_under_fused
    )
    _warn_grad_clipping_ignored_under_fused = (
        BaseTrainer._warn_grad_clipping_ignored_under_fused
    )


def _warn(trainer, accum=ACCUM, batch_size=2, mnt=1, times=1):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for _ in range(times):
            trainer._warn_gradient_accumulation_ignored_under_fused(accum, batch_size, mnt)
    return out.getvalue()


class AccumulationWarningTest(unittest.TestCase):
    def test_it_names_the_setting_and_the_real_effective_batch(self):
        message = _warn(_Trainer(fused=True), accum=4, batch_size=2)
        self.assertIn("gradient_accumulation_steps=4", message)
        self.assertIn("IGNORED", message)
        self.assertIn("ONE batch of 2", message)
        self.assertIn("effective batch of 8", message)

    def test_it_gives_the_reason(self):
        message = _warn(_Trainer(fused=True))
        self.assertIn("free", message)
        self.assertIn("across backward passes", message)

    def test_multi_noise_timesteps_divides_the_window(self):
        message = _warn(_Trainer(fused=True), accum=4, batch_size=2, mnt=2)
        self.assertIn("effective batch of 4", message)

    def test_it_fires_once(self):
        trainer = _Trainer(fused=True)
        # One emission is two stdout lines: the human one and the machine one
        # TrainingProcess lifts off the stream (core/training/training_events.py).
        # Count the human line only.
        from core.training.training_events import TRAINING_EVENT_SENTINEL
        lines = [line for line in _warn(trainer, times=5).splitlines()
                 if not line.startswith(TRAINING_EVENT_SENTINEL)]
        self.assertEqual(sum(line.count("IGNORED") for line in lines), 1)

    def test_the_fused_optimizer_groups_path_warns_too(self):
        self.assertIn("fused optimizer groups", _warn(_Trainer(groups=object())))

    def test_it_is_silent_without_a_fused_path(self):
        self.assertEqual(_warn(_Trainer(fused=False)), "")

    def test_it_is_silent_when_nothing_is_being_accumulated(self):
        self.assertEqual(_warn(_Trainer(fused=True), accum=1), "")

    def test_it_does_not_rewrite_the_setting(self):
        trainer = _Trainer(fused=True)
        _warn(trainer)
        self.assertEqual(trainer._grad_accum_steps, ACCUM)

    def test_it_does_not_duplicate_the_clipping_warning(self):
        trainer = _Trainer(fused=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            trainer._warn_gradient_accumulation_ignored_under_fused(ACCUM, 2, 1)
            trainer._warn_grad_clipping_ignored_under_fused(1.0)
        message = out.getvalue()
        self.assertEqual(message.count("WARNING"), 2)
        accumulation, clipping = message.split("WARNING")[1:]
        self.assertNotIn("max_grad_norm", accumulation)
        self.assertNotIn("gradient_accumulation_steps", clipping)


if __name__ == "__main__":
    unittest.main()
