"""Guard: a config LR edit must survive a resume, at the schedule's position.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/test_lr_resume_override.py -v
or, without pytest:
    venv/Scripts/python.exe -m unittest discover -s backend/tests -p "test_lr_resume_override.py"

Why this file exists
--------------------
Two separate defects, one shared helper (``core/training/lr_utils.py``).

1. ``VaeTrainer`` builds its optimizer/scheduler from the config (correct, new
   LR) and *then* resumes, which used to silently throw that LR away:

   * ``torch.optim.Optimizer.load_state_dict`` rebuilds each param group taking
     only ``params`` from the live group and EVERY other key -- ``lr`` included
     -- from the SAVED group;
   * ``LRScheduler.load_state_dict`` is ``self.__dict__.update(state_dict)`` and
     ``state_dict()`` carries ``base_lrs``, so the scheduler reverts to the
     checkpoint's base LR and re-writes it into the param groups on every
     subsequent ``step()``.

   Net effect: editing ``train.lr`` and resuming was a no-op with no log line
   saying so. Run 113 trained ~460 recorded steps at 1e-05 while its config said
   2.5e-06.

2. ``BaseTrainer`` re-asserted the config LR **flat**, with no schedule
   multiplier. That is invisible only for as long as something else re-applies
   the multiplier before the next optimizer step -- and nothing does: the
   training loop calls ``optimizer.step()`` BEFORE ``lr_scheduler.step()`` and a
   mid-epoch resume slices the batch list rather than iterating it. So the first
   post-resume step ran at the un-multiplied base LR: 2x too high mid-warmup of
   a 1,000-step warmup at step 500, 159x too high in a diffusers ``cosine`` tail,
   2.29x in a ``plateau_cosine_floor`` tail. This regressed EVERY resume with a
   non-constant schedule, whether or not any LR was edited.

The defects live in torch's own semantics, so everything here is built on a real
``torch.optim.AdamW``, real ``LambdaLR``s (including a real
``diffusers.optimization.get_scheduler("cosine", ...)`` and a copy of
``BaseTrainer``'s ``plateau_cosine_floor`` shape) round-tripped through
``state_dict()`` / ``load_state_dict()``. A mock would reproduce nothing.

CPU-only and hermetic: a handful of 4-element parameters, no model, no dataset,
no GPU.

The invariants asserted:
    * after a resume, the LR IN FORCE is derived from the CONFIGURED base, never
      from the checkpointed one;
    * it is that base times the schedule multiplier at the CURRENT position, per
      param group, so the first post-resume step is on-schedule;
    * per-component LRs (U-Net / TE1 / TE2 / VE) stay distinct -- a scalar
      broadcast that collapsed them would be a worse bug than the one fixed;
    * the optimizer moments (``exp_avg`` / ``exp_avg_sq``) survive. A "fix" that
      rebuilt the optimizer would pass a naive LR assertion while discarding
      exactly the state a resume exists to preserve.

Tolerances here are RELATIVE (``got/expected`` against 1.0). ``assertAlmostEqual``
defaults to an ABSOLUTE 5e-8, which against LRs of order 1e-6 cannot see an
off-by-one in the lambda argument (worth ~2.5e-09 at these magnitudes) and would
pass regardless.

The wiring is covered too, not just the helper: ``test_vae_resume_path_*`` drives
the REAL ``VaeTrainer.load_checkpoint`` against a real temp checkpoint directory,
and ``test_base_trainer_*`` drives the REAL
``BaseTrainer._reassert_config_lr_on_resume`` / ``_build_component_lr_list``.
Deleting either call site turns this file red.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

# `backend` itself must be on sys.path: the modules under test import
# `core.training.*` with backend as the root package dir.
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.lr_utils import reassert_config_lr, resolve_group_lrs

CKPT_LR = 1e-05    # what run 113's checkpoint carried
CFG_LR = 2.5e-06   # what its config asked for after the edit
RESUME_STEP = 178101

WARMUP = 1000
TOTAL = 100000


def _constant_lambda(_step):
    return 1.0


def _warmup_lambda(step):
    """Non-trivial multiplier, so the schedule shape is exercised too."""
    return min(1.0, (step + 1) / 1000.0)


def _make(lr, lr_lambda=_constant_lambda, with_scheduler=True, n_groups=2):
    """A real AdamW (+ optional real LambdaLR) over small CPU tensors.

    TWO param groups by default on purpose: the VAE trainer itself only builds
    one, but the re-assertion iterates every group and ``base_lrs`` is
    per-group, so a one-group fixture would not notice a fix that only touched
    ``param_groups[0]``.
    """
    params = [torch.nn.Parameter(torch.ones(4)) for _ in range(n_groups)]
    optimizer = torch.optim.AdamW([{"params": [p]} for p in params], lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda) if with_scheduler else None
    return params, optimizer, scheduler


def _take_steps(params, optimizer, scheduler, n):
    """Real steps, so exp_avg / exp_avg_sq become non-trivial."""
    for i in range(n):
        for j, p in enumerate(params):
            p.grad = torch.full_like(p, 0.1 * (i + 1) * (j + 1))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)


def _checkpoint_at(lr, lr_lambda=_constant_lambda, with_scheduler=True, steps=5):
    """Produce (optimizer_state, scheduler_state, moments) as a resume source.

    ``last_epoch`` is forced to a realistic mid-run value so the fast-forwarded
    schedule position is visibly distinct from a fresh scheduler's.
    """
    params, optimizer, scheduler = _make(lr, lr_lambda, with_scheduler)
    _take_steps(params, optimizer, scheduler, steps)
    if scheduler is not None:
        scheduler.last_epoch = RESUME_STEP
        scheduler._step_count = RESUME_STEP + 1
    opt_state = copy.deepcopy(optimizer.state_dict())
    sched_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
    moments = [
        (optimizer.state[p]["exp_avg"].clone(), optimizer.state[p]["exp_avg_sq"].clone())
        for p in params
    ]
    return opt_state, sched_state, moments


def _resume_raw(opt_state, sched_state, cfg_lr, lr_lambda=_constant_lambda,
                with_scheduler=True):
    """The PRE-FIX VAE resume: build from config, then load state and stop there."""
    params, optimizer, scheduler = _make(cfg_lr, lr_lambda, with_scheduler)
    optimizer.load_state_dict(opt_state)
    if scheduler is not None and sched_state is not None:
        scheduler.load_state_dict(sched_state)
    return params, optimizer, scheduler


def _plateau_cosine_floor_lambda(W=WARMUP, T=TOTAL, floor_ratio=0.25,
                                 decay_start_ratio=0.85):
    """The multiplier of BaseTrainer._build_plateau_cosine_floor_scheduler.

    Kept as a literal copy rather than imported: the point of the numbers below
    is that they are the real shape run112-class configs use, and a silent
    change to that shape should show up here as a failing expectation rather
    than as a test that follows the code wherever it goes.
    """
    import math
    D = max(W, min(round(decay_start_ratio * T), T))

    def lr_lambda(step):
        if W > 0 and step < W:
            return step / float(W)
        if step < D:
            return 1.0
        if step < T:
            progress = (step - D) / float(max(1, T - D))
            return floor_ratio + 0.5 * (1.0 - floor_ratio) * (1.0 + math.cos(math.pi * progress))
        return floor_ratio

    return lr_lambda


class _LrAssertions(unittest.TestCase):
    """Relative-tolerance LR comparison (see the module docstring)."""

    def assertLrEqual(self, got, expected, msg=""):
        got, expected = float(got), float(expected)
        if expected == 0.0:
            self.assertEqual(got, 0.0, msg)
            return
        self.assertAlmostEqual(
            got / expected, 1.0, places=9,
            msg=f"{msg} (got {got:.6e}, expected {expected:.6e}, "
                f"relative error {abs(got / expected - 1.0):.3e})")

    def assertLrNotEqual(self, got, expected, msg=""):
        got, expected = float(got), float(expected)
        if expected == 0.0:
            self.assertNotEqual(got, 0.0, msg)
            return
        self.assertNotAlmostEqual(
            got / expected, 1.0, places=9,
            msg=f"{msg} (got {got:.6e}, unexpectedly equal to {expected:.6e})")


class LrResumeOverrideTest(_LrAssertions):

    # ---------------------------------------------------------------- helpers
    def _assert_moments_preserved(self, optimizer, params, moments):
        for p, (exp_avg, exp_avg_sq) in zip(params, moments):
            state = optimizer.state[p]
            self.assertIn("exp_avg", state, "optimizer moments were discarded")
            self.assertTrue(torch.allclose(state["exp_avg"], exp_avg),
                            "exp_avg changed across the resume")
            self.assertTrue(torch.allclose(state["exp_avg_sq"], exp_avg_sq),
                            "exp_avg_sq changed across the resume")
            self.assertGreater(float(state["exp_avg"].abs().sum()), 0.0,
                               "moments are all zero -- the fixture stepped nothing")

    # ----------------------------------------------------- the core invariant
    def test_torch_semantics_still_clobber_the_lr(self):
        """Pin the upstream behaviour this fix exists to counter.

        If a future torch stops importing the saved 'lr'/'base_lrs', this case
        turns red and the fix can be reconsidered rather than cargo-culted.
        """
        opt_state, sched_state, _ = _checkpoint_at(CKPT_LR)
        _, optimizer, scheduler = _resume_raw(opt_state, sched_state, CFG_LR)
        self.assertEqual(optimizer.param_groups[0]["lr"], CKPT_LR,
                         "Optimizer.load_state_dict no longer imports the saved lr")
        self.assertEqual(scheduler.base_lrs, [CKPT_LR, CKPT_LR],
                         "LRScheduler.load_state_dict no longer imports base_lrs")

    def test_lambdalr_get_lr_formula_is_what_the_helper_assumes(self):
        """The helper evaluates lr_lambdas at ``last_epoch``; pin that this is
        what LambdaLR itself does (``base_lr * lmbda(self.last_epoch)``).

        An off-by-one here is the single most likely silent regression in the
        helper, and it is why every LR assertion in this file is relative.
        """
        params, optimizer, scheduler = _make(CFG_LR, _warmup_lambda)
        _take_steps(params, optimizer, scheduler, 7)
        self.assertEqual(scheduler.last_epoch, 7)
        self.assertLrEqual(optimizer.param_groups[0]["lr"],
                           CFG_LR * _warmup_lambda(scheduler.last_epoch),
                           "LambdaLR no longer evaluates its lambda at last_epoch")

    def test_config_lr_wins_over_checkpoint_lr(self):
        opt_state, sched_state, moments = _checkpoint_at(CKPT_LR)
        params, optimizer, scheduler = _resume_raw(opt_state, sched_state, CFG_LR)

        prev, bases = reassert_config_lr(optimizer, scheduler, CFG_LR)

        self.assertEqual(prev, [CKPT_LR, CKPT_LR])
        self.assertEqual(bases, [CFG_LR, CFG_LR])
        for group in optimizer.param_groups:
            self.assertLrEqual(group["lr"], CFG_LR)
            self.assertLrEqual(group["initial_lr"], CFG_LR)
        self.assertEqual(scheduler.base_lrs, [CFG_LR, CFG_LR])
        for value in scheduler.get_last_lr():
            self.assertLrEqual(value, CFG_LR)
        self._assert_moments_preserved(optimizer, params, moments)

    def test_config_lr_survives_subsequent_scheduler_steps(self):
        """The scheduler must not re-write the checkpoint's LR on step 1.

        This is the half that bit run 113: even fixing the optimizer alone
        leaves ``base_lrs`` at the checkpoint value, and ``scheduler.step()``
        runs after every ``optimizer.step()``.
        """
        opt_state, sched_state, _ = _checkpoint_at(CKPT_LR)
        params, optimizer, scheduler = _resume_raw(opt_state, sched_state, CFG_LR)
        reassert_config_lr(optimizer, scheduler, CFG_LR)

        _take_steps(params, optimizer, scheduler, 3)

        for group in optimizer.param_groups:
            self.assertLrEqual(group["lr"], CFG_LR,
                               "the scheduler re-imposed the checkpoint's LR")

    def test_schedule_position_and_step_count_are_preserved(self):
        """Only the LR is overridden -- where we are along the schedule is not."""
        opt_state, sched_state, _ = _checkpoint_at(CKPT_LR)
        _, optimizer, scheduler = _resume_raw(opt_state, sched_state, CFG_LR)
        reassert_config_lr(optimizer, scheduler, CFG_LR)

        self.assertEqual(scheduler.last_epoch, RESUME_STEP)
        self.assertEqual(scheduler._step_count, RESUME_STEP + 1)

    def test_non_constant_schedule_keeps_its_multiplier(self):
        """A warmup/cosine schedule stays on-schedule, rescaled to the new base.

        RESUME_STEP is past the warmup, so the multiplier is 1.0 there; the
        mid-warmup case below is what proves the multiplier is applied rather
        than ignored.
        """
        opt_state, sched_state, _ = _checkpoint_at(CKPT_LR, _warmup_lambda)
        _, optimizer, scheduler = _resume_raw(opt_state, sched_state, CFG_LR,
                                              _warmup_lambda)
        reassert_config_lr(optimizer, scheduler, CFG_LR)
        self.assertEqual(scheduler.base_lrs, [CFG_LR, CFG_LR])
        self.assertLrEqual(optimizer.param_groups[0]["lr"], CFG_LR)

        # Mid-warmup resume: LR in force = CFG_LR * lambda(last_epoch),
        # i.e. derived from the CONFIG base, never from the checkpoint's.
        opt_state, sched_state, _ = _checkpoint_at(CKPT_LR, _warmup_lambda)
        _, optimizer, scheduler = _resume_raw(opt_state, sched_state, CFG_LR,
                                              _warmup_lambda)
        scheduler.last_epoch = 499
        reassert_config_lr(optimizer, scheduler, CFG_LR)
        self.assertLrEqual(optimizer.param_groups[0]["lr"],
                           CFG_LR * _warmup_lambda(499))
        self.assertLrNotEqual(optimizer.param_groups[0]["lr"],
                              CKPT_LR * _warmup_lambda(499))
        # An off-by-one in the lambda argument is a 0.2% error here; the
        # relative tolerance above is what makes it visible.
        self.assertLrNotEqual(optimizer.param_groups[0]["lr"],
                              CFG_LR * _warmup_lambda(500),
                              "the lambda was evaluated at the wrong step")

    def test_scheduler_is_none_path(self):
        """build_optimizer legitimately leaves lr_scheduler None (get_scheduler
        raised, e.g. an unsupported name); the param-group write is then the
        whole fix and must not blow up on the missing scheduler."""
        opt_state, sched_state, moments = _checkpoint_at(CKPT_LR, with_scheduler=False)
        self.assertIsNone(sched_state)
        params, optimizer, scheduler = _resume_raw(opt_state, None, CFG_LR,
                                                   with_scheduler=False)
        self.assertIsNone(scheduler)

        prev, bases = reassert_config_lr(optimizer, None, CFG_LR)

        self.assertEqual(prev, [CKPT_LR, CKPT_LR])
        self.assertEqual(bases, [CFG_LR, CFG_LR])
        for group in optimizer.param_groups:
            self.assertLrEqual(group["lr"], CFG_LR)
        self._assert_moments_preserved(optimizer, params, moments)

        # And it stays put: without a scheduler nothing re-writes the LR.
        _take_steps(params, optimizer, None, 2)
        self.assertLrEqual(optimizer.param_groups[0]["lr"], CFG_LR)

    def test_unchanged_lr_resume_is_a_no_op(self):
        """Resuming without editing anything must not perturb anything."""
        opt_state, sched_state, moments = _checkpoint_at(CKPT_LR)
        params, optimizer, scheduler = _resume_raw(opt_state, sched_state, CKPT_LR)

        prev, bases = reassert_config_lr(optimizer, scheduler, CKPT_LR)

        self.assertEqual(prev, [CKPT_LR, CKPT_LR])
        self.assertEqual(bases, [CKPT_LR, CKPT_LR])
        self.assertLrEqual(optimizer.param_groups[0]["lr"], CKPT_LR)
        self.assertEqual(scheduler.base_lrs, [CKPT_LR, CKPT_LR])
        self._assert_moments_preserved(optimizer, params, moments)

    # ------------------------------------------- per-component LRs (F5)
    def test_per_component_lrs_are_not_collapsed(self):
        """A per-group sequence must reach its own group.

        BaseTrainer genuinely runs U-Net / TE1 / TE2 / VE at different LRs. A
        helper that broadcast one scalar over every group would be a worse bug
        than the one it fixes, and nothing else in the suite would notice.
        """
        component_lrs = [1.0e-05, 5.0e-06, 2.0e-06]
        params, optimizer, scheduler = _make(9.9e-09, _warmup_lambda, n_groups=3)
        scheduler.last_epoch = 400
        reassert_config_lr(optimizer, scheduler, component_lrs,
                           component_names=["U-Net", "TE1", "TE2"])

        mult = _warmup_lambda(400)
        for group, base in zip(optimizer.param_groups, component_lrs):
            self.assertLrEqual(group["lr"], base * mult)
            self.assertLrEqual(group["initial_lr"], base)
        self.assertEqual(scheduler.base_lrs, component_lrs)
        # Explicitly: the groups did NOT all end up on the first component's LR.
        self.assertLrNotEqual(optimizer.param_groups[1]["lr"],
                              component_lrs[0] * mult,
                              "TE1 was collapsed onto the U-Net LR")

    def test_short_component_list_falls_back_to_learning_rate(self):
        """Trailing groups the component list does not cover take fallback_lr.

        This reproduces BaseTrainer's pre-existing inline behaviour
        (``component_lrs[i] if i < len(component_lrs) else self.learning_rate``),
        which the REPA projector group relies on.
        """
        _, optimizer, scheduler = _make(9.9e-09, _constant_lambda, n_groups=3)
        reassert_config_lr(optimizer, scheduler, [1.0e-05, 5.0e-06],
                           fallback_lr=7.0e-07)
        self.assertLrEqual(optimizer.param_groups[0]["lr"], 1.0e-05)
        self.assertLrEqual(optimizer.param_groups[1]["lr"], 5.0e-06)
        self.assertLrEqual(optimizer.param_groups[2]["lr"], 7.0e-07)

    def test_short_component_list_without_fallback_is_a_caller_error(self):
        _, optimizer, scheduler = _make(9.9e-09, _constant_lambda, n_groups=3)
        with self.assertRaises(ValueError):
            reassert_config_lr(optimizer, scheduler, [1.0e-05, 5.0e-06])

    def test_more_component_lrs_than_groups_is_not_fatal(self):
        """Fused optimizer groups make this real: self.optimizer is only
        optimizers[0] and holds fewer groups than the component list describes.
        Raising there would turn a cosmetic mismatch into a failed resume."""
        _, optimizer, scheduler = _make(9.9e-09, _constant_lambda, n_groups=1)
        reassert_config_lr(optimizer, scheduler, [1.0e-05, 5.0e-06, 2.0e-06],
                           fallback_lr=1.0e-05)
        self.assertLrEqual(optimizer.param_groups[0]["lr"], 1.0e-05)

    def test_resolve_group_lrs_contract(self):
        self.assertEqual(resolve_group_lrs(3, 1e-5), [1e-5, 1e-5, 1e-5])
        self.assertEqual(resolve_group_lrs(3, [1e-5, 2e-5], fallback_lr=3e-5),
                         [1e-5, 2e-5, 3e-5])
        self.assertEqual(resolve_group_lrs(1, [1e-5, 2e-5]), [1e-5])
        self.assertEqual(resolve_group_lrs(0, 1e-5), [])
        with self.assertRaises(ValueError):
            resolve_group_lrs(3, [1e-5])

    # ------------------------------------- the first post-resume step (F1)
    def test_first_post_resume_step_is_on_schedule(self):
        """The regression F1 fixes, on both schedules that matter.

        A flat re-assertion is invisible only if something re-applies the
        multiplier before the next optimizer step. Nothing does: the loop is
        ``optimizer.step()`` then ``lr_scheduler.step()``, and a mid-epoch
        resume slices batches rather than iterating. So this asserts the value
        sitting in the param groups the instant the resume finishes.
        """
        from diffusers.optimization import get_scheduler

        cases = [
            ("diffusers cosine", lambda opt: get_scheduler(
                "cosine", opt, num_warmup_steps=WARMUP, num_training_steps=TOTAL)),
            ("plateau_cosine_floor", lambda opt: LambdaLR(
                opt, lr_lambda=_plateau_cosine_floor_lambda())),
        ]
        component_lrs = [1.0e-05, 5.0e-06]

        for name, build_scheduler in cases:
            for resume_step in (500, 50000, 95000):
                with self.subTest(schedule=name, resume_step=resume_step):
                    params = [torch.nn.Parameter(torch.ones(4)) for _ in range(2)]
                    optimizer = torch.optim.AdamW(
                        [{"params": [p], "lr": lr}
                         for p, lr in zip(params, component_lrs)])
                    scheduler = build_scheduler(optimizer)

                    # BaseTrainer's resume: fast-forward, then load optimizer
                    # state (which re-imports the checkpoint's lr), then
                    # re-assert.
                    for _ in range(resume_step):
                        scheduler.step()
                    on_schedule = [g["lr"] for g in optimizer.param_groups]
                    for group in optimizer.param_groups:      # load_optimizer_state
                        group["lr"] = 9.876e-06

                    reassert_config_lr(optimizer, scheduler, component_lrs,
                                       component_names=["U-Net", "TE1"],
                                       fallback_lr=component_lrs[0])

                    for i, group in enumerate(optimizer.param_groups):
                        self.assertLrEqual(
                            group["lr"], on_schedule[i],
                            f"{name} @ {resume_step}: first post-resume LR is "
                            f"off-schedule")
                        # And the pre-fix flat write would NOT have matched
                        # (except where the multiplier happens to be 1.0).
                        if abs(on_schedule[i] / component_lrs[i] - 1.0) > 1e-9:
                            self.assertLrNotEqual(
                                component_lrs[i], on_schedule[i],
                                f"{name} @ {resume_step}: fixture is degenerate, "
                                f"the multiplier is 1.0 so it proves nothing")

    def test_flat_write_mutant_is_caught(self):
        """Differentiation for F1: the exact pre-fix statement must fail here."""
        component_lrs = [1.0e-05, 5.0e-06]
        params = [torch.nn.Parameter(torch.ones(4)) for _ in range(2)]
        optimizer = torch.optim.AdamW(
            [{"params": [p], "lr": lr} for p, lr in zip(params, component_lrs)])
        scheduler = LambdaLR(optimizer, lr_lambda=_plateau_cosine_floor_lambda())
        for _ in range(95000):
            scheduler.step()
        on_schedule = [g["lr"] for g in optimizer.param_groups]

        # The pre-fix override: `param_group['lr'] = component_lrs[i]`, flat.
        for group, lr in zip(optimizer.param_groups, component_lrs):
            group["lr"] = lr

        with self.assertRaises(AssertionError):
            self.assertLrEqual(optimizer.param_groups[0]["lr"], on_schedule[0])
        # 1/floor_ratio-ish: the tail multiplier is 0.4375 here.
        self.assertGreater(optimizer.param_groups[0]["lr"] / on_schedule[0], 2.0)

    # --------------------------------------------------- differentiation
    def test_pre_fix_behaviour_is_caught(self):
        """The guard must FAIL against the old semantics, or it guards nothing.

        Mutation 1: no re-assertion at all (the shipped pre-fix VAE code).
        Mutation 2: optimizer param groups fixed but base_lrs left alone (the
                    "obvious" half-fix), which the checkpoint's scheduler undoes
                    on the first step.
        """
        # Mutation 1 -- raw resume, nothing re-asserted.
        opt_state, sched_state, _ = _checkpoint_at(CKPT_LR)
        params, optimizer, scheduler = _resume_raw(opt_state, sched_state, CFG_LR)
        with self.assertRaises(AssertionError):
            self.assertLrEqual(optimizer.param_groups[0]["lr"], CFG_LR)
        with self.assertRaises(AssertionError):
            self.assertEqual(scheduler.base_lrs, [CFG_LR, CFG_LR])

        # Mutation 2 -- param groups only; the scheduler restores the old LR.
        opt_state, sched_state, _ = _checkpoint_at(CKPT_LR)
        params, optimizer, scheduler = _resume_raw(opt_state, sched_state, CFG_LR)
        for group in optimizer.param_groups:
            group["lr"] = CFG_LR
        _take_steps(params, optimizer, scheduler, 1)
        with self.assertRaises(AssertionError):
            self.assertLrEqual(optimizer.param_groups[0]["lr"], CFG_LR)
        self.assertLrEqual(optimizer.param_groups[0]["lr"], CKPT_LR,
                           "expected the checkpoint's base_lrs to win pre-fix")

    def test_moment_reset_would_be_caught(self):
        """Mutation 3: a 'fix' that rebuilds the optimizer passes a naive LR
        assertion; the moment check is what rejects it."""
        opt_state, sched_state, moments = _checkpoint_at(CKPT_LR)
        params, optimizer, _ = _resume_raw(opt_state, sched_state, CFG_LR)
        fresh = torch.optim.AdamW(params, lr=CFG_LR)  # moments discarded
        self.assertEqual(fresh.param_groups[0]["lr"], CFG_LR)  # naive check passes
        with self.assertRaises(AssertionError):
            self._assert_moments_preserved(fresh, params, moments)


class BaseTrainerResumeWiringTest(_LrAssertions):
    """The BaseTrainer half: the real methods, and the real call ordering.

    Everything above exercises the helper. These cases fail if the call site
    disappears or is reordered -- which is exactly what the helper-only suite
    could not see.
    """

    @classmethod
    def setUpClass(cls):
        from core.training.base_trainer import BaseTrainer
        cls.BaseTrainer = BaseTrainer

        # A stand-in carrying only the attributes _build_component_lr_list reads
        # (it is written entirely in terms of getattr defaults). The METHODS are
        # the real ones, taken off the class, so a change to either is seen here.
        class _Stub:
            _build_component_lr_list = BaseTrainer._build_component_lr_list
            _reassert_config_lr_on_resume = BaseTrainer._reassert_config_lr_on_resume
            _configured_component_lr_description = \
                BaseTrainer._configured_component_lr_description
            # No setup_optimizer ran here, so no snapshot exists: these stubs
            # exercise the _build_component_lr_list fallback on purpose.
            _configured_group_lrs = None
            _configured_group_names = None

        cls.Stub = _Stub

    def _stub(self, n_groups, lr_lambda, resume_step, component_lrs):
        stub = self.Stub()
        stub.log_prefix = "[TestTrainer]"
        stub.learning_rate = component_lrs[0]
        # U-Net + TE1 (+ TE2 when a third group is asked for) -> the group order
        # _build_component_lr_list documents.
        stub.train_unet = True
        stub.unet = torch.nn.Linear(1, 1)
        stub.unet_lr = component_lrs[0]
        stub.train_text_encoder = True
        stub.text_encoder = torch.nn.Linear(1, 1)
        stub.text_encoder_1_lr = component_lrs[1]
        stub.is_sdxl = n_groups > 2
        if n_groups > 2:
            stub.text_encoder_2 = torch.nn.Linear(1, 1)
            stub.text_encoder_2_lr = component_lrs[2]

        params = [torch.nn.Parameter(torch.ones(4)) for _ in range(n_groups)]
        stub.optimizer = torch.optim.AdamW(
            [{"params": [p], "lr": lr} for p, lr in zip(params, component_lrs)])
        stub.lr_scheduler = LambdaLR(stub.optimizer, lr_lambda=lr_lambda)
        for _ in range(resume_step):
            stub.lr_scheduler.step()
        return stub

    def test_component_lr_list_matches_the_stub(self):
        stub = self._stub(3, _constant_lambda, 0, [1e-5, 5e-6, 2e-6])
        lrs, names = stub._build_component_lr_list()
        self.assertEqual(names, ["U-Net", "TE1", "TE2"])
        self.assertEqual(lrs, [1e-5, 5e-6, 2e-6])

    def test_resume_applies_the_schedule_multiplier_per_component(self):
        """The F1 invariant, through BaseTrainer's own method.

        Mid-warmup and in a plateau_cosine_floor tail, per component, after a
        load_optimizer_state that re-imported the checkpoint's LR.
        """
        component_lrs = [1e-5, 5e-6, 2e-6]
        cases = [
            ("warmup", _warmup_lambda, 499),
            ("plateau_cosine_floor tail", _plateau_cosine_floor_lambda(), 95000),
        ]
        for name, lr_lambda, resume_step in cases:
            with self.subTest(schedule=name):
                stub = self._stub(3, lr_lambda, resume_step, component_lrs)
                on_schedule = [g["lr"] for g in stub.optimizer.param_groups]
                mult = lr_lambda(resume_step)
                self.assertNotAlmostEqual(mult, 1.0, places=6,
                                          msg="degenerate fixture")

                for group in stub.optimizer.param_groups:   # load_optimizer_state
                    group["lr"] = 9.876e-06

                stub._reassert_config_lr_on_resume()

                for i, group in enumerate(stub.optimizer.param_groups):
                    self.assertLrEqual(group["lr"], on_schedule[i],
                                       f"group {i} is off-schedule after resume")
                    self.assertLrEqual(group["lr"], component_lrs[i] * mult)
                    # per-component: not collapsed onto the U-Net LR
                    if i > 0:
                        self.assertLrNotEqual(group["lr"], component_lrs[0] * mult)
                self.assertEqual(stub.lr_scheduler.base_lrs, component_lrs)

    def test_resume_honours_an_edited_config_lr(self):
        """The original defect, on the BaseTrainer side: an edited LR wins."""
        stub = self._stub(2, _constant_lambda, 1000, [1e-5, 5e-6])
        stub.unet_lr = 2.5e-06          # the user's YAML edit
        stub.text_encoder_1_lr = 1.25e-06
        for group in stub.optimizer.param_groups:
            group["lr"] = 1e-5          # what the checkpoint carried
        stub._reassert_config_lr_on_resume()
        self.assertLrEqual(stub.optimizer.param_groups[0]["lr"], 2.5e-06)
        self.assertLrEqual(stub.optimizer.param_groups[1]["lr"], 1.25e-06)

    def test_no_optimizer_is_a_no_op(self):
        stub = self.Stub()
        stub.optimizer = None
        stub.log_prefix = "[TestTrainer]"
        stub.learning_rate = 1e-5
        stub._reassert_config_lr_on_resume()   # must not raise

    # ------------------------------------------------------ call-site wiring
    def test_train_loads_optimizer_state_before_reasserting_the_lr(self):
        """Both resume branches must re-assert AFTER load_optimizer_state.

        Source-level rather than behavioural: ``BaseTrainer.train()`` is a
        ~3,000-line method that owns dataloaders, the epoch loop and the DB
        session, so driving it in a unit test would prove less than it costs.
        The ordering is the whole content of the change, and it is mechanically
        checkable.
        """
        source = inspect.getsource(self.BaseTrainer.train)
        load_positions = [i for i, line in enumerate(source.splitlines())
                          if "self.load_optimizer_state(checkpoint_step)" in line]
        reassert_positions = [i for i, line in enumerate(source.splitlines())
                              if "self._reassert_config_lr_on_resume()" in line]
        self.assertEqual(len(load_positions), 2,
                         "expected exactly two resume branches")
        self.assertEqual(len(reassert_positions), 2,
                         "a resume branch no longer re-asserts the config LR")
        for load_at, reassert_at in zip(load_positions, reassert_positions):
            self.assertLess(load_at, reassert_at,
                            "load_optimizer_state runs AFTER the LR re-assertion; "
                            "it would reinstate the checkpoint's LR")

    def test_no_flat_lr_override_remains_in_train(self):
        """The old flat write must be gone, not merely bypassed."""
        source = inspect.getsource(self.BaseTrainer.train)
        self.assertNotIn("param_group['lr'] = new_lr", source)
        self.assertNotIn("self.lr_scheduler.base_lrs[i] = new_base_lr", source)


class VaeResumeWiringTest(_LrAssertions):
    """Drive the REAL VaeTrainer.load_checkpoint against a real checkpoint dir.

    No GPU and no VAE: ``load_checkpoint`` only needs ``trainable_names`` /
    ``trainable_params`` to line up with the safetensors file, so two 4-element
    parameters stand in for the decoder. Every other step it performs -- the
    component-set assertion, the weight copy, the optimizer and scheduler
    restores, the RNG restore, the train_state read -- runs for real.

    Deleting the ``reassert_config_lr`` call from ``load_checkpoint`` turns this
    red; the helper-only cases above would not notice.
    """

    def setUp(self):
        from safetensors.torch import save_file
        from core.training.vae.vae_trainer import VaeTrainer

        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = Path(self._tmp.name) / "step_00178101"
        self.ckpt_dir.mkdir(parents=True)

        # ---- write a checkpoint that carries the OLD LR --------------------
        names = ["decoder.a", "decoder.b"]
        ckpt_params = [torch.nn.Parameter(torch.full((4,), 3.0)) for _ in names]
        ckpt_opt = torch.optim.AdamW(
            [{"params": [p]} for p in ckpt_params], lr=CKPT_LR,
            weight_decay=0.2)
        ckpt_sched = LambdaLR(ckpt_opt, lr_lambda=_warmup_lambda)
        _take_steps(ckpt_params, ckpt_opt, ckpt_sched, 5)
        ckpt_sched.last_epoch = 499          # mid-warmup, multiplier = 0.5
        ckpt_sched._step_count = 500

        save_file({n: p.detach().clone() for n, p in zip(names, ckpt_params)},
                  str(self.ckpt_dir / "vae_decoder.safetensors"))
        torch.save(ckpt_opt.state_dict(), self.ckpt_dir / "optimizer.pt")
        torch.save(ckpt_sched.state_dict(), self.ckpt_dir / "lr_scheduler.pt")
        with open(self.ckpt_dir / "train_state.json", "w", encoding="utf-8") as f:
            json.dump({"step": RESUME_STEP,
                       "config": {"train_encoder": False, "train_decoder": True,
                                  "decoder_blocks": "all", "optimizer": "adamw",
                                  "optimizer_weight_decay": 0.2}}, f)

        # ---- a trainer built from the NEW config LR ------------------------
        trainer = VaeTrainer.__new__(VaeTrainer)
        trainer.log_prefix = "[VaeTrainer]"
        trainer.device = "cpu"
        trainer.ema = None
        trainer.train_encoder = False
        trainer.cfg = {"learning_rate": CFG_LR, "optimizer": "adamw",
                       "optimizer_weight_decay": 0.03, "train_decoder": True,
                       "decoder_blocks": "all", "encoder_blocks": "none"}
        trainer.trainable_names = names
        trainer.trainable_params = [torch.nn.Parameter(torch.zeros(4))
                                    for _ in names]
        trainer.optimizer = torch.optim.AdamW(
            [{"params": [p]} for p in trainer.trainable_params], lr=CFG_LR,
            weight_decay=0.03)
        trainer.lr_scheduler = LambdaLR(trainer.optimizer, lr_lambda=_warmup_lambda)
        self.trainer = trainer

    def tearDown(self):
        self._tmp.cleanup()

    def test_vae_resume_path_reasserts_the_config_lr(self):
        self.trainer.load_checkpoint(self.ckpt_dir)

        # The resume itself worked: weights, step and schedule position.
        self.assertEqual(self.trainer.global_step, RESUME_STEP)
        self.assertEqual(self.trainer.lr_scheduler.last_epoch, 499)
        self.assertTrue(torch.allclose(self.trainer.trainable_params[0].detach(),
                                       torch.full((4,), 3.0)),
                        "checkpoint weights were not restored")

        # ...and the LR in force comes from the CONFIG, at the schedule's
        # position (mid-warmup, multiplier 0.5), not from the checkpoint.
        expected = CFG_LR * _warmup_lambda(499)
        for group in self.trainer.optimizer.param_groups:
            self.assertLrEqual(group["lr"], expected)
        self.assertEqual(self.trainer.lr_scheduler.base_lrs, [CFG_LR, CFG_LR])
        self.assertLrNotEqual(self.trainer.optimizer.param_groups[0]["lr"],
                              CKPT_LR * _warmup_lambda(499),
                              "the checkpoint's LR survived the resume")

    def test_vae_resume_keeps_the_config_lr_across_later_steps(self):
        """base_lrs must be rewritten too, or step 1 undoes the fix."""
        self.trainer.load_checkpoint(self.ckpt_dir)
        _take_steps(self.trainer.trainable_params, self.trainer.optimizer,
                    self.trainer.lr_scheduler, 3)
        expected = CFG_LR * _warmup_lambda(self.trainer.lr_scheduler.last_epoch)
        self.assertLrEqual(self.trainer.optimizer.param_groups[0]["lr"], expected)

    def test_vae_resume_reasserts_config_weight_decay(self):
        self.trainer.load_checkpoint(self.ckpt_dir)
        self.assertEqual(
            [group["weight_decay"] for group in self.trainer.optimizer.param_groups],
            [0.03, 0.03],
        )


if __name__ == "__main__":
    unittest.main()
