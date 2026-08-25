"""A fused group froze whole when one of its parameters got no gradient.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/fused_group_partial_flush_test.py -v

The hook steps a group when its count reaches the group's parameter count, so a
group containing any parameter that a backward does not reach never steps -- and
the parameters that DID get a gradient and merely share the group are frozen with
it. Which parameters share a group is decided by nothing but index order
(``create_optimizer_groups`` slices a flat list), so the pairing is accidental.

Parameters that get no gradient are not hypothetical. The Vision Encoder gets
none on a reference-free batch (base_trainer.py, the epoch VE-offload note:
"reference-free batches produce no VE grad (set_to_none)") while its parameters
sit at the END of the same flat list the DiT's are sliced from; stochastic depth
(``block_skip_rate``, Anima) drops a different set of blocks every step and
composes with Block Swap, which is the only configuration that creates these
groups at all.

Worse than the freeze: the gradient of a group that does not step is never
freed, so the next backward sums into it -- the same contamination 4e271260
removed from the MNT path, arriving by another route and, unlike that one,
without end.

``BaseTrainer._flush_fused_group_partials`` steps the incomplete groups once the
backward returns, which is the first moment "this parameter got nothing this
time" is decided. ``flush=False`` below runs the identical loop without that call
as the negative control.

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

LR = 1e-2
DIM = 4
GRAD = torch.tensor([1.0, 1.0, 1.0, 1.0])


class _CountingOptimizer:
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


def _params(n):
    return [nn.Parameter(torch.zeros(DIM)) for _ in range(n)]


def _groups(optimizers):
    groups = fog.FusedOptimizerGroups(optimizers, max_grad_norm=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        groups.register_hooks()
    return groups


def _backward(params, grad=GRAD):
    torch.stack([(p * grad).sum() for p in params]).sum().backward()


def _trainer(groups):
    """A bare object carrying only the two methods under test."""
    class _Bare:
        _reset_fused_group_counters = BaseTrainer._reset_fused_group_counters
        _flush_fused_group_partials = BaseTrainer._flush_fused_group_partials
    bare = _Bare()
    bare.fused_optimizer_groups = groups
    return bare


class _Rig:
    """One group of `size` parameters, of which `fed` receive a gradient.

    Mirrors the production shape: a group that mixes parameters a given backward
    reaches (the DiT) with parameters it does not (the Vision Encoder).
    """

    def __init__(self, size=4, fed=3, optimizer_factory=None, extra_groups=()):
        factory = optimizer_factory or (lambda p: torch.optim.SGD(p, lr=LR))
        self.params = _params(size)
        self.fed = self.params[:fed]
        self.starved = self.params[fed:]
        self.optimizer = _CountingOptimizer(factory(self.params))
        optimizers = [self.optimizer]
        self.extra = []
        for n in extra_groups:
            group = _params(n)
            opt = _CountingOptimizer(factory(group))
            self.extra.append((group, opt))
            optimizers.append(opt)
        self.groups = _groups(optimizers)
        self.trainer = _trainer(self.groups)

    def backward(self, params=None, flush=True):
        self.trainer._reset_fused_group_counters()
        _backward(self.params if params is None else params)
        if flush:
            self.trainer._flush_fused_group_partials()


class TheDefect(unittest.TestCase):
    """Each of these fails at ``flush=False`` -- see TheNegativeControl."""

    def test_the_parameters_that_got_a_gradient_are_updated(self):
        rig = _Rig()
        rig.backward(rig.fed)
        for i, p in enumerate(rig.fed):
            self.assertTrue(torch.allclose(p.detach(), -LR * GRAD, atol=1e-7),
                            f"fed parameter {i} did not move: {p.detach().tolist()}")

    def test_the_starved_parameter_is_left_where_it_was(self):
        rig = _Rig()
        rig.backward(rig.fed)
        for p in rig.starved:
            self.assertTrue(torch.equal(p.detach(), torch.zeros(DIM)))

    def test_the_group_steps_once_per_backward(self):
        rig = _Rig()
        for _ in range(5):
            rig.backward(rig.fed)
        self.assertEqual(rig.optimizer.steps, 5)
        self.assertTrue(torch.allclose(
            rig.fed[0].detach(), -LR * 5 * GRAD, atol=1e-7))

    def test_the_gradient_is_freed_and_does_not_reach_the_next_backward(self):
        rig = _Rig()
        rig.backward(rig.fed)
        for p in rig.params:
            self.assertIsNone(p.grad)
        # A second backward therefore steps on ITS OWN gradient, not on two.
        rig.backward(rig.fed)
        self.assertTrue(torch.allclose(
            rig.fed[0].detach(), -LR * 2 * GRAD, atol=1e-7))

    def test_a_complete_group_alongside_still_steps_from_the_hook(self):
        rig = _Rig(extra_groups=(2,))
        complete, complete_opt = rig.extra[0]
        rig.trainer._reset_fused_group_counters()
        _backward(rig.fed + complete)
        # The complete group has already stepped, inside the backward.
        self.assertEqual(complete_opt.steps, 1)
        self.assertEqual(rig.optimizer.steps, 0)
        rig.trainer._flush_fused_group_partials()
        self.assertEqual((complete_opt.steps, rig.optimizer.steps), (1, 1))

    def test_a_group_that_got_nothing_does_not_step(self):
        # The Vision Encoder's own group on a reference-free batch: no gradient
        # anywhere in it, so there is nothing to apply and it must stay put.
        rig = _Rig(extra_groups=(2,))
        idle, idle_opt = rig.extra[0]
        rig.backward(rig.fed)
        self.assertEqual(idle_opt.steps, 0)
        for p in idle:
            self.assertTrue(torch.equal(p.detach(), torch.zeros(DIM)))

    def test_an_adaptive_optimizer_only_states_the_parameters_it_updated(self):
        factory = lambda p: torch.optim.AdamW(p, lr=LR, weight_decay=0.0)  # noqa: E731
        rig = _Rig(optimizer_factory=factory)
        rig.backward(rig.fed)
        state = rig.optimizer.inner.state
        for p in rig.fed:
            self.assertIn(p, state)
        for p in rig.starved:
            self.assertNotIn(p, state)

    def test_it_holds_when_the_starved_parameter_changes_every_backward(self):
        # Stochastic depth drops a different block each step, so the starved
        # member of a group is not the same one twice.
        rig = _Rig(size=4, fed=0)
        for i in range(4):
            fed = [p for j, p in enumerate(rig.params) if j != i]
            rig.backward(fed)
        for i, p in enumerate(rig.params):
            # Each parameter sat out exactly one of the four backwards.
            self.assertTrue(torch.allclose(p.detach(), -LR * 3 * GRAD, atol=1e-7),
                            f"parameter {i}: {p.detach().tolist()}")


class TheNegativeControl(unittest.TestCase):
    """The same loop without the flush. These record the broken behaviour."""

    def test_pre_fix_the_whole_group_freezes(self):
        rig = _Rig()
        for _ in range(5):
            rig.backward(rig.fed, flush=False)
        self.assertEqual(rig.optimizer.steps, 0)
        for p in rig.params:  # including the ones that got a gradient
            self.assertTrue(torch.equal(p.detach(), torch.zeros(DIM)))

    def test_pre_fix_the_gradient_piles_up_across_backwards(self):
        rig = _Rig()
        for i in range(5):
            rig.backward(rig.fed, flush=False)
            self.assertTrue(torch.allclose(rig.fed[0].grad, GRAD * (i + 1), atol=1e-7))

    def test_pre_fix_a_complete_group_in_the_same_run_is_unaffected(self):
        # Which is what makes the freeze silent: the run keeps training, and only
        # the groups that happen to hold a starved parameter stop.
        rig = _Rig(extra_groups=(2,))
        complete, complete_opt = rig.extra[0]
        for _ in range(5):
            rig.trainer._reset_fused_group_counters()
            _backward(rig.fed + complete)
        self.assertEqual(complete_opt.steps, 5)
        self.assertEqual(rig.optimizer.steps, 0)


class TheNormalCaseIsUnchanged(unittest.TestCase):
    """Every parameter gets a gradient: the flush must be a no-op."""

    def _run(self, flush, factory, n_backwards=5):
        rig = _Rig(size=4, fed=4, optimizer_factory=factory, extra_groups=(3,))
        extra, extra_opt = rig.extra[0]
        trajectory = []
        for _ in range(n_backwards):
            rig.trainer._reset_fused_group_counters()
            _backward(rig.params + extra)
            if flush:
                rig.trainer._flush_fused_group_partials()
            trajectory.append([p.detach().clone() for p in rig.params + extra])
        return rig.optimizer.steps, extra_opt.steps, trajectory

    def _compare(self, factory):
        steps, extra_steps, fixed = self._run(True, factory)
        pre_steps, pre_extra_steps, pre_fix = self._run(False, factory)
        self.assertEqual((steps, extra_steps), (5, 5))
        self.assertEqual((steps, extra_steps), (pre_steps, pre_extra_steps))
        for a_row, b_row in zip(fixed, pre_fix):
            for a, b in zip(a_row, b_row):
                self.assertTrue(torch.equal(a, b))

    def test_sgd_is_bit_identical(self):
        self._compare(lambda p: torch.optim.SGD(p, lr=LR))

    def test_adamw_is_bit_identical(self):
        self._compare(lambda p: torch.optim.AdamW(p, lr=LR, weight_decay=0.01))

    def test_the_flush_steps_nothing_when_every_group_is_complete(self):
        rig = _Rig(size=3, fed=3, extra_groups=(2,))
        rig.trainer._reset_fused_group_counters()
        _backward(rig.params + rig.extra[0][0])
        self.assertEqual(rig.groups.step_incomplete_groups(), [])


class MultipleBackwardsPerBatch(unittest.TestCase):
    """MNT > 1 and the micro-split path run several backwards per batch."""

    def test_each_mnt_iteration_applies_its_own_gradient(self):
        MNT = 3
        rig = _Rig()
        for batch in range(2):
            for _ in range(MNT):
                rig.backward(rig.fed)
        self.assertEqual(rig.optimizer.steps, 2 * MNT)
        # 2*MNT steps of exactly one backward's gradient each -- had any gradient
        # survived a backward, the total would exceed this.
        self.assertTrue(torch.allclose(
            rig.fed[0].detach(), -LR * 2 * MNT * GRAD, atol=1e-7))

    def test_a_chunked_batch_plus_an_encoder_backward_behave_the_same(self):
        # _microbatch_two_stage: N chunk backwards through
        # _execute_forward_backward, then one more for the encoder graph.
        rig = _Rig()
        for _ in range(2):  # chunks
            rig.backward(rig.fed)
        rig.backward(rig.fed[:1])  # the encoder-graph backward reaches less
        self.assertEqual(rig.optimizer.steps, 3)
        self.assertTrue(torch.allclose(
            rig.fed[0].detach(), -LR * 3 * GRAD, atol=1e-7))
        self.assertTrue(torch.allclose(
            rig.fed[1].detach(), -LR * 2 * GRAD, atol=1e-7))


class TheAccidentalPairing(unittest.TestCase):
    """create_optimizer_groups slices a flat list, so who shares a group is luck."""

    def test_groups_are_index_order_slices(self):
        params = _params(7)
        with contextlib.redirect_stdout(io.StringIO()):
            optimizers = fog.create_optimizer_groups(
                params=params, optimizer_type="adamw", num_groups=3,
                learning_rate=LR, weight_decay=0.0)
        got = [list(o.param_groups[0]["params"]) for o in optimizers]
        self.assertEqual(got, [params[0:3], params[3:6], params[6:7]])

    def test_a_trailing_vision_encoder_lands_in_a_group_with_dit_parameters(self):
        # setup_optimizer appends the VE group last, so the flat list is
        # [DiT..., VE...] and the slice boundary falls wherever it falls.
        dit, ve = _params(5), _params(2)
        with contextlib.redirect_stdout(io.StringIO()):
            optimizers = fog.create_optimizer_groups(
                params=dit + ve, optimizer_type="adamw", num_groups=2,
                learning_rate=LR, weight_decay=0.0)
        mixed = list(optimizers[1].param_groups[0]["params"])
        self.assertTrue(any(p is d for d in dit for p in mixed))
        self.assertTrue(any(p is v for v in ve for p in mixed))


class TheTrainerFlushesAfterEveryBackward(unittest.TestCase):
    """Wiring: the flush call sits after the backward, at every backward site."""

    def _flush_follows_backward(self, source):
        flush = source.index("self._flush_fused_group_partials()")
        backward = re.search(r"\.backward\(|torch\.autograd\.backward\(", source)
        self.assertIsNotNone(backward)
        return flush > backward.start()

    def test_the_helper_is_a_no_op_without_fused_groups(self):
        class _Bare:
            _flush_fused_group_partials = BaseTrainer._flush_fused_group_partials
        _Bare()._flush_fused_group_partials()  # no attribute at all
        bare = _Bare()
        bare.fused_optimizer_groups = None
        bare._flush_fused_group_partials()

    def test_the_main_backward_is_flushed(self):
        self.assertTrue(self._flush_follows_backward(
            inspect.getsource(BaseTrainer._execute_forward_backward)))

    def test_the_micro_split_encoder_backward_is_flushed(self):
        self.assertTrue(self._flush_follows_backward(
            inspect.getsource(BaseTrainer._microbatch_two_stage)))


if __name__ == "__main__":
    unittest.main()
