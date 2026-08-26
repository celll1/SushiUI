"""A fused path under FP16 mixed precision never skipped an overflowing step.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/fused_path_grad_scaler_test.py -v

THE DEFECT
----------
``_execute_forward_backward`` calls ``grad_scaler.scale(loss).backward()``
whenever ``use_grad_scaler`` (mixed_precision + training_dtype=fp16). The only
site that calls ``unscale_()`` / ``step()`` / ``update()`` is guarded by
``if not self.use_fused_backward and self.fused_optimizer_groups is None:``, so
with Block Swap active the per-parameter post-accumulate-grad hooks applied and
freed each gradient with the scale factor still in it.

The MAGNITUDE survives that, and this file says so rather than implying
otherwise: every optimizer ``OptimizerFactory`` can build is Adam-family,
Adafactor or sign-based, where the scale cancels in the update.
``ShippedOptimizersAbsorbTheScaleTest`` measures it (adamw 1.0001, adafactor
exactly 1.0). ``ScaleReachesTheFusedHooksTest`` uses SGD only to isolate the
mechanism -- that the scaled gradient reaches the hook and is freed there --
and SGD is not selectable in this product.

The operative harm is the rest of GradScaler's contract:
``OverflowIsAppliedInsteadOfSkippedTest`` injects one inf under the shipped
optimizers and measures NaN weights that are still NaN five finite steps later,
with the scale pinned at its initial value because ``update()`` never runs.

THE FIX
-------
A refusal. A fused-aware scaler is implementable (unscale by the public
``get_scale()``, skip the individual non-finite parameter, hold the scale fixed)
but is not implemented; what per-parameter hooks cannot reproduce is
GradScaler's whole-step, all-or-nothing skip.
``base_trainer.refuse_grad_scaler_under_fused_path`` refuses in ``__init__``
(before the model load) and again at each fused setup, naming the settings that
conflict.

CPU-only, no CUDA and no model: the mechanism is the hook and the scale factor.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re
import sys
import unittest
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import core.training.optimizers.fused_optimizer_groups as fog  # noqa: E402
from core.training.base_trainer import (  # noqa: E402
    BaseTrainer,
    refuse_grad_scaler_under_fused_path,
)

LR = 1e-2
DIM = 4
INIT_SCALE = 2 ** 20  # base_trainer's GradScaler init_scale
GRAD = torch.ones(DIM)


def _scaler(init_scale=INIT_SCALE):
    return torch.amp.GradScaler("cpu", init_scale=init_scale,
                                growth_factor=2.0, backoff_factor=0.5,
                                growth_interval=2000)


def _param():
    return nn.Parameter(torch.zeros(DIM))


def _loss(param, grad=GRAD):
    return (param * grad).sum()


def _fused_groups(optimizers):
    groups = fog.FusedOptimizerGroups(optimizers, max_grad_norm=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        groups.register_hooks()
    return groups


def _shipped_optimizer(name, params, lr=1e-3):
    from core.training.optimizer_factory import OptimizerFactory
    with contextlib.redirect_stdout(io.StringIO()):
        return OptimizerFactory.create_optimizer(
            optimizer_type=name, params=params, learning_rate=lr)


class ScaleReachesTheFusedHooksTest(unittest.TestCase):
    """SGD, to isolate the mechanism -- NOT to claim a wrong learning rate.

    SGD is scale-linear, so it makes the arithmetic visible: the gradient that
    reaches the hook is the SCALED one, and the hook frees it, so no later
    ``unscale_()`` can correct anything. SGD is not selectable in this product
    (``OptimizerFactory`` offers Adam-family, Adafactor and sign-based
    optimizers only), and every optimizer that IS selectable absorbs the scale
    -- see ``ShippedOptimizersAbsorbTheScaleTest``.
    """

    def test_the_hook_sees_the_scaled_gradient_and_frees_it(self):
        param = _param()
        _fused_groups([torch.optim.SGD([param], lr=LR)])

        _scaler().scale(_loss(param)).backward()

        self.assertIsNone(param.grad)
        self.assertTrue(torch.equal(param.detach(),
                                    torch.full((DIM,), -LR * INIT_SCALE)))

    def test_the_non_fused_path_unscales_before_stepping(self):
        param = _param()
        optimizer = torch.optim.SGD([param], lr=LR)
        scaler = _scaler()

        scaler.scale(_loss(param)).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        self.assertTrue(torch.equal(param.detach(), torch.full((DIM,), -LR)))
        self.assertEqual(scaler.get_scale(), INIT_SCALE)


class ShippedOptimizersAbsorbTheScaleTest(unittest.TestCase):
    """Why the harm is NOT a wrong learning rate.

    Measured on CPU, 10 steps, gradients 1e-4, scale 2**20: the scale cancels
    between numerator and denominator in Adam-family and Adafactor updates.
    """

    STEPS = 10
    GRAD_MAGNITUDE = 1e-4

    def _walk(self, name, scaled):
        param = _param()
        optimizer = _shipped_optimizer(name, [param])
        for _ in range(self.STEPS):
            grad = torch.full((DIM,), self.GRAD_MAGNITUDE)
            param.grad = grad * (INIT_SCALE if scaled else 1.0)
            optimizer.step()
            param.grad = None
        return param.detach().clone()

    def test_adafactor_is_exactly_scale_invariant(self):
        self.assertTrue(torch.equal(self._walk("adafactor", True),
                                    self._walk("adafactor", False)))

    def test_adamw_is_scale_invariant_to_within_its_epsilon(self):
        ratio = (self._walk("adamw", True) / self._walk("adamw", False))[0].item()
        # 1.0001 measured: the residual is eps=1e-8 becoming negligible next to
        # a scaled second moment, not a learning-rate error.
        self.assertLess(abs(ratio - 1.0), 1e-3)


class OverflowIsAppliedInsteadOfSkippedTest(unittest.TestCase):
    """The real defect, under the optimizers a run can actually select."""

    NAMES = ("adamw", "adafactor")

    def test_one_inf_permanently_poisons_the_fused_path(self):
        for name in self.NAMES:
            with self.subTest(optimizer=name):
                param = _param()
                _fused_groups([_shipped_optimizer(name, [param])])
                scaler = _scaler()

                scaler.scale(_loss(param, torch.full((DIM,), float("inf")))).backward()
                self.assertTrue(torch.isnan(param.detach()).all())

                # Five finite steps later it is still NaN: the overflow is in the
                # optimizer state, not just in that one update.
                for _ in range(5):
                    scaler.scale(_loss(param, torch.full((DIM,), 1e-4))).backward()
                self.assertTrue(torch.isnan(param.detach()).all())

                # update() never ran, so the scale cannot back off out of it.
                self.assertEqual(scaler.get_scale(), INIT_SCALE)

    def test_the_non_fused_path_skips_the_step_and_backs_the_scale_off(self):
        for name in self.NAMES:
            with self.subTest(optimizer=name):
                param = _param()
                optimizer = _shipped_optimizer(name, [param])
                scaler = _scaler()

                scaler.scale(_loss(param, torch.full((DIM,), float("inf")))).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()

                self.assertTrue(torch.equal(param.detach(), torch.zeros(DIM)))
                self.assertEqual(scaler.get_scale(), INIT_SCALE / 2)


class TheStepSiteStillSkipsTheScalerUnderFusedTest(unittest.TestCase):
    """Pin: the refusal stays justified only while this guard stays."""

    def test_unscale_step_and_update_are_guarded_by_the_non_fused_branch(self):
        # Indentation-coupled regex: it breaks loudly (assertIsNotNone below) if
        # train() is re-indented, which is the intended failure mode.
        source = inspect.getsource(BaseTrainer.train)
        guard = re.search(
            r"if not self\.use_fused_backward and self\.fused_optimizer_groups is None:"
            r"(.*?)\n {28}else:",
            source, re.S)
        self.assertIsNotNone(guard, "the non-fused guard moved; re-check the refusal")
        body = guard.group(1)
        for call in ("grad_scaler.unscale_(", "grad_scaler.step(", "grad_scaler.update()"):
            self.assertIn(call, body)
        # ... and nowhere else in train().
        self.assertEqual(source.count("self.grad_scaler.update()"), 1)


class _StubModule(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.weight = param


class _StubTrainer:
    """The smallest object ``BaseTrainer.setup_optimizer`` can run against.

    Copied from ``optimizer_refusal_test``, plus the scaler attributes.
    """

    setup_optimizer = BaseTrainer.setup_optimizer
    _report_effective_component_lrs = BaseTrainer._report_effective_component_lrs
    _record_configured_group_lrs = BaseTrainer._record_configured_group_lrs
    _name_configured_groups = BaseTrainer._name_configured_groups
    _build_component_lr_list = BaseTrainer._build_component_lr_list
    _resolved_optimizer_hyperparameters = BaseTrainer._resolved_optimizer_hyperparameters
    _ringbuffer_optimizer_kwargs = BaseTrainer._ringbuffer_optimizer_kwargs
    _announce_host_state_budget = BaseTrainer._announce_host_state_budget
    _assert_ringbuffer_state_host_resident = (
        BaseTrainer._assert_ringbuffer_state_host_resident)
    _RINGBUFFER_HOST_STATE_BYTES_PER_PARAM = (
        BaseTrainer._RINGBUFFER_HOST_STATE_BYTES_PER_PARAM)
    _setup_fused_backward_pass = BaseTrainer._setup_fused_backward_pass
    _setup_fused_optimizer_groups = BaseTrainer._setup_fused_optimizer_groups
    _fused_backward_target_module = BaseTrainer._fused_backward_target_module
    _attach_stochastic_rounding = BaseTrainer._attach_stochastic_rounding
    _RINGBUFFER_ONLY_OPTIONS = BaseTrainer._RINGBUFFER_ONLY_OPTIONS
    _NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS = BaseTrainer._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS
    _BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS = BaseTrainer._BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS

    def __init__(self, **overrides: Any):
        self.log_prefix = "[StubTrainer]"
        self.learning_rate = LR
        self.weight_dtype = torch.bfloat16
        self.blocks_to_swap = 0
        self.num_optimizer_groups = 0
        self.use_ema = False
        self.use_grad_scaler = False
        self.grad_scaler = None
        self.config: Dict[str, Any] = {}
        self.optimizer_cautious = False
        self.optimizer_beta1 = None
        self.optimizer_beta2 = None
        self.optimizer_epsilon = None
        self.optimizer_weight_decay = None
        self.optimizer_schedule_free = False
        self.optimizer_warmup_steps = 0
        self.optimizer_schedule_free_r = 0.0
        self.optimizer_schedule_free_weight_lr_power = 2.0
        self.optimizer_use_radam = False
        self.optimizer_stochastic_rounding = False
        for key, value in overrides.items():
            setattr(self, key, value)
        self.param = nn.Parameter(torch.randn(256) * 0.02)
        self.transformer = _StubModule(self.param)
        self.unet = None

    def setup_trainable_parameters(self):
        return [{"params": [self.param], "lr": self.learning_rate}]

    def _setup_ema(self):
        pass


def _setup(optimizer_type: str, **overrides):
    trainer = _StubTrainer(**overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.setup_optimizer(optimizer_type=optimizer_type, total_steps=10)
    return trainer


class GradScalerFusedRefusalTest(unittest.TestCase):
    """It fires for the reachable configurations and only for those."""

    # (optimizer, num_optimizer_groups) pairs that install a fused path under
    # blocks_to_swap > 0 without needing CUDA.
    FUSED = (("adafactor", 0), ("adamw8bit", 0), ("adamw", 2), ("adamw", 4))

    def test_every_fused_configuration_is_refused_under_fp16_mixed_precision(self):
        for optimizer_type, groups in self.FUSED:
            with self.subTest(optimizer=optimizer_type, num_optimizer_groups=groups):
                with self.assertRaises(ValueError) as ctx:
                    _setup(optimizer_type, blocks_to_swap=8,
                           num_optimizer_groups=groups, use_grad_scaler=True)
                message = str(ctx.exception)
                # The message names the settings that produced the conflict...
                self.assertIn("training_dtype=fp16", message)
                self.assertIn("mixed_precision=True", message)
                self.assertIn("blocks_to_swap=8", message)
                self.assertIn(f"num_optimizer_groups={groups}", message)
                self.assertIn(f"optimizer={optimizer_type}", message)
                # ... and a way out.
                self.assertIn("training_dtype=bf16", message)
                self.assertIn("blocks_to_swap=0", message)

    def test_bf16_reaches_the_same_fused_paths(self):
        """The refusal is about the scaler, not about Block Swap."""
        for optimizer_type, groups in self.FUSED:
            with self.subTest(optimizer=optimizer_type, num_optimizer_groups=groups):
                trainer = _setup(optimizer_type, blocks_to_swap=8,
                                 num_optimizer_groups=groups, use_grad_scaler=False)
                fused = (getattr(trainer, "use_fused_backward", False)
                         or getattr(trainer, "fused_optimizer_groups", None) is not None)
                self.assertTrue(fused)

    def test_fp16_without_block_swap_is_not_refused(self):
        """No fused path -> the guarded unscale_/step/update flow runs correctly."""
        trainer = _setup("adamw", blocks_to_swap=0, num_optimizer_groups=4,
                         use_grad_scaler=True)
        self.assertIsNone(getattr(trainer, "fused_optimizer_groups", None))
        self.assertFalse(getattr(trainer, "use_fused_backward", False))

    def test_fp16_with_block_swap_but_no_fused_optimizer_is_not_refused(self):
        """Plain adamw under Block Swap takes neither fused branch."""
        trainer = _setup("adamw", blocks_to_swap=8, num_optimizer_groups=0,
                         use_grad_scaler=True)
        self.assertIsNone(getattr(trainer, "fused_optimizer_groups", None))
        self.assertFalse(getattr(trainer, "use_fused_backward", False))

    def test_the_guard_is_a_no_op_when_the_attribute_is_absent(self):
        """Why the guard uses getattr: the many trainer stubs that lack it.

        The rename risk that buys is covered by
        ``test_base_trainer_still_defines_use_grad_scaler``.
        """
        class _Bare:
            pass
        self.assertIsNone(refuse_grad_scaler_under_fused_path(
            _Bare(), "adafactor", "fused backward pass"))

    def test_base_trainer_still_defines_use_grad_scaler(self):
        """A rename would otherwise disable the refusal silently."""
        self.assertIn("self.use_grad_scaler = ",
                      inspect.getsource(BaseTrainer.__init__))

    @unittest.skipUnless(torch.cuda.is_available(), "ring-buffer optimizers need CUDA")
    def test_the_ring_buffer_fused_paths_are_refused_too(self):
        for optimizer_type in ("adamw8bit_ringbuffer", "lion8bit_ringbuffer"):
            with self.subTest(optimizer=optimizer_type):
                with self.assertRaises(ValueError) as ctx:
                    _setup(optimizer_type, blocks_to_swap=8, use_grad_scaler=True)
                self.assertIn("training_dtype=bf16", str(ctx.exception))


class TheRefusalRunsBeforeTheModelLoadTest(unittest.TestCase):
    """F4: paying a model load and a caching pass before the error is the bug."""

    def test_init_refuses_ahead_of_load_model_components(self):
        source = inspect.getsource(BaseTrainer.__init__)
        refusal = source.find("refuse_grad_scaler_under_fused_path(")
        load = source.find("_load_model_components(")
        self.assertGreater(refusal, 0)
        self.assertGreater(load, 0)
        self.assertLess(refusal, load)


class UntouchedPathsAreUnchangedTest(unittest.TestCase):
    """Without the scaler, the fused hooks behave exactly as before."""

    def test_a_fused_group_step_is_bit_identical_to_the_plain_optimizer_step(self):
        fused, plain = _param(), _param()
        fused_opt = _shipped_optimizer("adafactor", [fused])
        plain_opt = _shipped_optimizer("adafactor", [plain])
        _fused_groups([fused_opt])

        _loss(fused).backward()
        _loss(plain).backward()
        plain_opt.step()

        self.assertTrue(torch.equal(fused.detach(), plain.detach()))
        self.assertFalse(torch.equal(fused.detach(), torch.zeros(DIM)))


if __name__ == "__main__":
    unittest.main()
