"""Fused optimizer groups dropped every step after the first backward of a batch.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/fused_optimizer_groups_mnt_test.py -v

The hook counts gradients and steps a group when the count equals that group's
parameter count. Gradients arrive once per parameter per BACKWARD, but
``reset_counters()`` was called once per BATCH (base_trainer, top of the batch
loop). With multi_noise_timesteps > 1 -- and equally with micro-split or an OOM
retry -- one batch runs several backwards, so from the second backward on the
count is already past the group size and ``== num_parameters_per_group`` never
holds again. Three consequences, all real: the step is dropped, the gradient is
never freed (which is the residency fused backward exists to avoid), and the
leftover is summed into whatever the next batch's first step does step on.

The reset now happens immediately before every backward instead
(``BaseTrainer._reset_fused_group_counters``), so the counter counts what one
backward produced. ``reset=PER_BATCH`` runs the same loop with the old reset
position as the negative control: every test in ``TheDefect`` fails under it, and
``TheNegativeControl`` records what it does instead.

Resetting per backward, rather than inside the hook when a group steps, also
keeps the step condition meaning "all of MY parameters got a gradient in THIS
backward" -- see ``AnIncompleteGroupStillDoesNotStep``.

Consistent with 0d843213: under a fused path each backward is its own optimizer
step. That commit says so for gradient accumulation; this makes it true for MNT
instead of silently true for the first backward only.

CPU-only, no CUDA and no model: the mechanism is the hook and the counter.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import re
import sys
import unittest

import torch
from torch import nn

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import core.training.optimizers.fused_optimizer_groups as fog  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402

MNT = 3
LR = 1e-2
DIM = 4
GRAD = torch.tensor([1.0, 1.0, 1.0, 1.0])

PER_BACKWARD = "per_backward"  # fixed
PER_BATCH = "per_batch"  # pre-fix


class _CountingOptimizer:
    """Counts the steps the hooks actually take."""

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


def _snapshot(params):
    # Clone: a live .grad accumulates IN PLACE, so storing the tensor itself
    # would make every recorded step show the last one's value.
    return [None if p.grad is None else p.grad.detach().clone() for p in params]


def _params(n=1):
    return [nn.Parameter(torch.zeros(DIM)) for _ in range(n)]


def _groups(optimizers):
    groups = fog.FusedOptimizerGroups(optimizers, max_grad_norm=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        groups.register_hooks()
    return groups


def _backward(params, grad=GRAD):
    torch.stack([(p * grad).sum() for p in params]).sum().backward()


def _run(n_batches, mnt, reset=PER_BACKWARD, optimizer_factory=None, n_params=1):
    """base_trainer's loop shape: a batch of `mnt` backwards, reset where `reset` says."""
    factory = optimizer_factory or (lambda p: torch.optim.SGD(p, lr=LR))
    params = _params(n_params)
    optimizer = _CountingOptimizer(factory(params))
    groups = _groups([optimizer])
    trajectory = []
    grads_after_backward = []
    grads_at_batch_start = []
    for _ in range(n_batches):
        if reset == PER_BATCH:
            groups.reset_counters()
        grads_at_batch_start.append(_snapshot(params))
        for _ in range(mnt):
            if reset == PER_BACKWARD:
                groups.reset_counters()
            _backward(params)
            grads_after_backward.append(_snapshot(params))
            trajectory.append(params[0].detach().clone())
    return {
        "steps": optimizer.steps,
        "trajectory": trajectory,
        "grads_after_backward": grads_after_backward,
        "grads_at_batch_start": grads_at_batch_start,
        "params": params,
    }


class TheDefect(unittest.TestCase):
    """Each of these fails at ``reset=PER_BATCH`` -- see TheNegativeControl."""

    def test_every_mnt_iteration_steps(self):
        self.assertEqual(_run(n_batches=1, mnt=MNT)["steps"], MNT)

    def test_the_weight_moves_on_every_mnt_iteration(self):
        # SGD lr=1e-2 on a constant gradient of 1: -0.01 per step, so a dropped
        # step shows up as a flat stretch in the trajectory.
        result = _run(n_batches=1, mnt=MNT)
        for i, got in enumerate(result["trajectory"]):
            want = torch.full((DIM,), -LR * (i + 1))
            self.assertTrue(torch.allclose(got, want, atol=1e-7),
                            f"mnt {i}: {got.tolist()} != {want.tolist()}")

    def test_the_gradient_is_freed_after_every_backward(self):
        result = _run(n_batches=1, mnt=MNT)
        for i, grads in enumerate(result["grads_after_backward"]):
            self.assertIsNone(grads[0], f"gradient survived backward {i}")

    def test_nothing_carries_into_the_next_batch(self):
        result = _run(n_batches=2, mnt=MNT)
        for i, grads in enumerate(result["grads_at_batch_start"]):
            self.assertIsNone(grads[0], f"batch {i} started with a live gradient")
        self.assertEqual(result["steps"], 2 * MNT)
        self.assertTrue(torch.allclose(
            result["params"][0].detach(), torch.full((DIM,), -LR * 2 * MNT), atol=1e-7))

    def test_it_is_not_specific_to_one_parameter_per_group(self):
        result = _run(n_batches=2, mnt=MNT, n_params=3)
        self.assertEqual(result["steps"], 2 * MNT)
        for grads in result["grads_after_backward"]:
            self.assertTrue(all(g is None for g in grads))

    def test_an_adaptive_optimizer_reaches_a_different_place(self):
        # Not just a count: AdamW's trajectory differs, so this was a different
        # training run and not merely a bookkeeping difference.
        factory = lambda p: torch.optim.AdamW(p, lr=LR, weight_decay=0.0)  # noqa: E731
        fixed = _run(n_batches=2, mnt=MNT, optimizer_factory=factory)
        pre_fix = _run(n_batches=2, mnt=MNT, reset=PER_BATCH, optimizer_factory=factory)
        self.assertGreater(fixed["params"][0].detach().norm().item(),
                           pre_fix["params"][0].detach().norm().item() * 2.0)


class TheNegativeControl(unittest.TestCase):
    """The old reset position, measured. These record the broken behaviour."""

    def test_pre_fix_drops_every_step_after_the_first_backward(self):
        result = _run(n_batches=1, mnt=MNT, reset=PER_BATCH)
        self.assertEqual(result["steps"], 1)
        for weight in result["trajectory"]:  # flat: mnt 1 and 2 never stepped
            self.assertTrue(torch.allclose(weight, torch.full((DIM,), -LR), atol=1e-7))

    def test_pre_fix_leaks_the_gradient_it_did_not_step_on(self):
        result = _run(n_batches=1, mnt=MNT, reset=PER_BATCH)
        self.assertIsNone(result["grads_after_backward"][0][0])
        for i in range(1, MNT):
            grad = result["grads_after_backward"][i][0]
            self.assertIsNotNone(grad, f"backward {i} freed its gradient after all")
            self.assertTrue(torch.allclose(grad, GRAD * i, atol=1e-7))

    def test_pre_fix_carries_that_gradient_into_the_next_batch(self):
        result = _run(n_batches=2, mnt=MNT, reset=PER_BATCH)
        carried = result["grads_at_batch_start"][1][0]
        self.assertIsNotNone(carried, "nothing carried over")
        self.assertTrue(torch.allclose(carried, GRAD * (MNT - 1), atol=1e-7))
        # The next batch's one surviving step lands on MNT-1 stale gradients plus
        # its own: 2 steps for 6 backwards, and the second is MNT times too big.
        self.assertEqual(result["steps"], 2)
        self.assertTrue(torch.allclose(
            result["params"][0].detach(), torch.full((DIM,), -LR * (1 + MNT)), atol=1e-7))


class MntOneIsUnchanged(unittest.TestCase):
    """One backward per batch never reached the second-backward case."""

    def test_the_two_reset_positions_agree_exactly_at_mnt_1(self):
        fixed = _run(n_batches=5, mnt=1)
        pre_fix = _run(n_batches=5, mnt=1, reset=PER_BATCH)
        self.assertEqual(fixed["steps"], pre_fix["steps"])
        self.assertEqual(fixed["steps"], 5)
        for a, b in zip(fixed["trajectory"], pre_fix["trajectory"]):
            self.assertTrue(torch.equal(a, b))

    def test_that_also_holds_for_an_adaptive_optimizer(self):
        factory = lambda p: torch.optim.AdamW(p, lr=LR, weight_decay=0.0)  # noqa: E731
        fixed = _run(n_batches=5, mnt=1, optimizer_factory=factory, n_params=2)
        pre_fix = _run(n_batches=5, mnt=1, reset=PER_BATCH,
                       optimizer_factory=factory, n_params=2)
        for a, b in zip(fixed["trajectory"], pre_fix["trajectory"]):
            self.assertTrue(torch.equal(a, b))


class TheGroupSplitStillMeansSomething(unittest.TestCase):
    """num_optimizer_groups divides parameters; a group steps on ITS OWN count."""

    def _two_groups(self):
        group_a, group_b = _params(2), _params(3)
        opt_a = _CountingOptimizer(torch.optim.SGD(group_a, lr=LR))
        opt_b = _CountingOptimizer(torch.optim.SGD(group_b, lr=LR))
        return group_a, group_b, opt_a, opt_b, _groups([opt_a, opt_b])

    def test_parameters_are_counted_per_group(self):
        self.assertEqual(self._two_groups()[4].num_parameters_per_group, [2, 3])

    def test_a_group_steps_only_when_all_of_its_own_parameters_are_ready(self):
        group_a, group_b, opt_a, opt_b, groups = self._two_groups()
        groups.reset_counters()
        _backward(group_a)  # only group A gets gradients
        self.assertEqual((opt_a.steps, opt_b.steps), (1, 0))
        self.assertTrue(all(p.grad is None for p in group_a))
        self.assertTrue(all(p.grad is None for p in group_b))
        groups.reset_counters()
        _backward(group_a + group_b)
        self.assertEqual((opt_a.steps, opt_b.steps), (2, 1))

    def test_both_groups_step_on_every_mnt_iteration(self):
        for reset, expected in ((PER_BACKWARD, 2 * MNT), (PER_BATCH, 2)):
            group_a, group_b, opt_a, opt_b, groups = self._two_groups()
            for _ in range(2):  # batches
                if reset == PER_BATCH:
                    groups.reset_counters()
                for _ in range(MNT):
                    if reset == PER_BACKWARD:
                        groups.reset_counters()
                    _backward(group_a + group_b)
            self.assertEqual((opt_a.steps, opt_b.steps), (expected, expected), reset)


class AnIncompleteGroupStillDoesNotStep(unittest.TestCase):
    """Why the reset is per backward and not inside the hook after a step.

    Some parameters get no gradient in some backwards (the Vision Encoder on a
    reference-free batch, for instance), so their group's count never reaches its
    size and the HOOK does not step it. What applies those gradients is the flush
    after the backward returns, which these cases do not call --
    ``fused_group_partial_flush_test.py`` covers it. The point here is only that
    nothing steps DURING the backward on a mixture of two backwards' gradients.

    Carrying counts across backwards is strictly worse than leaving them alone:
    they add up until they happen to land on the group size and the group steps
    mid-backward, on some parameters' fresh gradients and others' stale ones,
    clearing the whole group as it goes. That is what the old per-batch reset did
    at MNT > 1 (measured below), and it is what a reset inside the hook would
    reintroduce.
    """

    def _run_partial(self, reset):
        group_a, group_b = _params(2), _params(3)
        opt_a = _CountingOptimizer(torch.optim.SGD(group_a, lr=LR))
        opt_b = _CountingOptimizer(torch.optim.SGD(group_b, lr=LR))
        groups = _groups([opt_a, opt_b])
        if reset == PER_BATCH:
            groups.reset_counters()
        for _ in range(MNT):
            if reset == PER_BACKWARD:
                groups.reset_counters()
            _backward(group_b[:2])  # 2 of group B's 3
        return opt_a.steps, opt_b.steps

    def test_per_backward_leaves_it_alone(self):
        self.assertEqual(self._run_partial(PER_BACKWARD), (0, 0))

    def test_carrying_counts_makes_it_step_mid_backward(self):
        # 2 gradients per backward against a group of 3: the running count is
        # 2, 3, 4, ... so the third gradient of the batch trips `== 3` in the
        # middle of the second backward.
        self.assertEqual(self._run_partial(PER_BATCH), (0, 1))


class TheTrainerArmsThemPerBackward(unittest.TestCase):
    """Wiring: the reset call sits before the backward, at every backward site."""

    def _source(self, func):
        return inspect.getsource(func)

    def _reset_precedes_backward(self, source):
        reset = source.index("self._reset_fused_group_counters()")
        backward = re.search(r"\.backward\(|torch\.autograd\.backward\(", source)
        self.assertIsNotNone(backward)
        return reset < backward.start()

    def test_the_helper_is_a_no_op_without_fused_groups(self):
        class _Bare:
            _reset_fused_group_counters = BaseTrainer._reset_fused_group_counters
        _Bare()._reset_fused_group_counters()  # no attribute at all
        bare = _Bare()
        bare.fused_optimizer_groups = None
        bare._reset_fused_group_counters()

    def test_the_helper_resets_the_groups_it_is_given(self):
        class _Bare:
            _reset_fused_group_counters = BaseTrainer._reset_fused_group_counters
        groups = _groups([torch.optim.SGD(_params(2), lr=LR)])
        groups.optimizer_hooked_count = {0: 7}
        bare = _Bare()
        bare.fused_optimizer_groups = groups
        bare._reset_fused_group_counters()
        self.assertEqual(groups.optimizer_hooked_count, {0: 0})

    def test_the_main_backward_is_armed(self):
        self.assertTrue(self._reset_precedes_backward(
            self._source(BaseTrainer._execute_forward_backward)))

    def test_the_micro_split_encoder_backward_is_armed(self):
        # Its chunk backwards go through _execute_forward_backward; the two-stage
        # encoder backward at the end is a separate call site.
        self.assertTrue(self._reset_precedes_backward(
            self._source(BaseTrainer._microbatch_two_stage)))


if __name__ == "__main__":
    unittest.main()
