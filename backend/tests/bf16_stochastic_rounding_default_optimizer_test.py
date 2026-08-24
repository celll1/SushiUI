"""Guard: the DEFAULT full-fine-tune optimizer must not freeze BF16 weights either.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/bf16_stochastic_rounding_default_optimizer_test.py -v

THE DEFECT
----------
``bf16_stochastic_rounding_test.py`` pins the underlying numerical failure and
its repair in the two ring-buffer optimizers: BF16 has an 8-bit significand, so
round-to-nearest deterministically discards every optimizer update below half a
ULP, and an Adam-family element therefore only ever moves when
``|w| <= 512*lr`` -- the rest are frozen at their initial bit pattern for the
whole run, not merely slow.

That repair did not reach the optimizer a default run actually uses.
``TRAINING_DEFAULTS["optimizer"]`` is ``adamw8bit`` and ``train_runner``
defaults full fine-tuning to it, and neither ``bitsandbytes`` nor the two
fused-backward patches in this package had any stochastic-rounding path. So a
user who changed nothing got the frozen-weight defect; ticking the checkbox only
produced "not supported by this optimizer".

Measured here on ``transformer_blocks.10.attn.to_gate.weight`` from Krea 2
(BF16, 6144x6144, 65536 elements sampled) driven through the real bitsandbytes
AdamW8bit CUDA kernel for 400 steps at lr 1e-5, against the same optimizer run
on an FP32 copy of the same weights:

    round-to-nearest      8.3% of elements ever move,   6.2% of the drift
    stochastic rounding   100% of elements ever move,  100.2% of the drift

WHAT EACH GROUP PINS
--------------------
* ``FusedAdamW8bitDefectTest``  -- ``adamw8bit_fused.step_param``, the update a
  block-swapped ``adamw8bit`` full fine-tune runs. It delegates to bitsandbytes'
  per-parameter seam, so this is the 8-bit kernel too; the state format and the
  patch contract are pinned in ``adamw8bit_fused_bnb_state_test.py``.
* ``FusedAdafactorDefectTest``  -- ``adafactor_fused.step_param``, covered by
  the generic interposer rather than by an edit to the optimizer.
* ``BitsandbytesKernelDefectTest`` -- the shipped default itself: real
  ``bnb.optim.AdamW8bit``, real 8-bit CUDA kernel. Skipped without CUDA.
* ``SeamCoverageTest``          -- which optimizers can be covered at all, and
  that an optimizer with no per-parameter seam is reported instead of being
  left to look covered.
* ``TrainerAttachmentTest``     -- ``BaseTrainer.setup_optimizer`` attaches it
  for the default optimizer, and only when the flag is set.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.optimizers.stochastic_rounding import (  # noqa: E402
    NATIVE_STEP_PARAM,
    WRAPPED_ATTR,
    attach_stochastic_rounding,
)

LR = 1e-6
STEPS = 200
SEED = 20260805
N = 1 << 14

# An Adam step peaks at lr/(1-beta1) = 10*lr on the first step (bias
# correction), so "cannot move under round-to-nearest" needs 512 * 10 * lr here.
MAX_STEP = 10 * LR


def _weights() -> torch.Tensor:
    """DiT-shaped weights: ~N(0, 0.02), so most of the mass is far above 512*lr."""
    torch.manual_seed(SEED)
    return (torch.randn(N) * 0.02).bfloat16()


class _Bf16UpdateAssertions:
    """Shared assertions: RTN must freeze bitwise, SR must carry the drift."""

    def _assert_defect_then_fix(self, run, label):
        w0 = _weights()

        after_rtn = run(w0, stochastic_rounding=False)
        frozen = w0.abs().float() >= 512 * MAX_STEP
        self.assertGreater(
            frozen.float().mean().item(), 0.5,
            "the test tensor must actually contain weights that RTN cannot move",
        )
        self.assertTrue(
            torch.equal(after_rtn[frozen], w0[frozen]),
            f"{label}: round-to-nearest is expected to leave these weights bitwise "
            f"unchanged; if this fails the ULP analysis is wrong",
        )

        after_sr = run(w0, stochastic_rounding=True)
        drift = (after_sr[frozen].float() - w0[frozen].float()).mean().item()
        self.assertGreater(
            drift, 0.5 * STEPS * LR,
            f"{label}: stochastic rounding did not carry the sub-ULP updates",
        )
        moved = after_sr[frozen].ne(w0[frozen]).float().mean().item()
        self.assertGreater(moved, 0.5, f"{label}: too few weights ever moved")


@unittest.skipUnless(torch.cuda.is_available(), "the 8-bit kernels require CUDA")
class FusedAdamW8bitDefectTest(unittest.TestCase, _Bf16UpdateAssertions):
    """``adamw8bit`` + Block Swap: ``step_param``, the seam the hooks drive.

    It used to be a hand-written Python AdamW writing ``p.addcdiv_(...)`` into
    BF16 storage with dense state. It now delegates to bitsandbytes' own
    per-parameter update, so this drives the real 8-bit kernel; the state format
    is pinned in ``adamw8bit_fused_bnb_state_test.py``.
    """

    def _run(self, w0, stochastic_rounding):
        import bitsandbytes as bnb
        from core.training.optimizers.adamw8bit_fused import patch_adamw8bit_fused

        p = torch.nn.Parameter(w0.clone().cuda())
        optimizer = bnb.optim.AdamW8bit([p], lr=LR, weight_decay=0.0)
        patch_adamw8bit_fused(optimizer, stochastic_rounding)
        group = optimizer.param_groups[0]

        for _ in range(STEPS):
            # BF16 gradient: what autograd produces for a BF16 parameter.
            p.grad = torch.full_like(p, -1.0)
            optimizer.step_param(p, group)
            p.grad = None
        return p.detach().cpu()

    def test_stochastic_rounding_reaches_the_fused_adamw8bit_update(self):
        self._assert_defect_then_fix(self._run, "adamw8bit_fused.step_param")

    def test_the_optimizer_state_stays_8_bit(self):
        """SR must not silently multiply optimizer-state memory.

        The FP32 image of the parameter lives for one call; ``init_state``
        allocates the moments with an explicit uint8 dtype, not from the
        parameter, so the block-swap configuration chosen to save memory keeps
        its 2.03 B/param.
        """
        import bitsandbytes as bnb
        from core.training.optimizers.adamw8bit_fused import patch_adamw8bit_fused

        p = torch.nn.Parameter(_weights().cuda())
        optimizer = bnb.optim.AdamW8bit([p], lr=LR, weight_decay=0.0)
        patch_adamw8bit_fused(optimizer, True)
        p.grad = torch.full_like(p, -1.0)
        optimizer.step_param(p, optimizer.param_groups[0])

        state = optimizer.state[p]
        self.assertEqual(state["state1"].dtype, torch.uint8)
        self.assertEqual(state["state2"].dtype, torch.uint8)

    def test_the_generic_interposer_leaves_this_step_param_alone(self):
        """Wrapping it as well would round twice."""
        import bitsandbytes as bnb
        from core.training.optimizers.adamw8bit_fused import patch_adamw8bit_fused

        p = torch.nn.Parameter(_weights().cuda())
        optimizer = bnb.optim.AdamW8bit([p], lr=LR, weight_decay=0.0)
        patch_adamw8bit_fused(optimizer, True)
        # Reported as covered-by-the-optimizer, and NOT wrapped.
        self.assertEqual(attach_stochastic_rounding(optimizer), (NATIVE_STEP_PARAM,))
        self.assertFalse(getattr(optimizer.step_param, WRAPPED_ATTR, False))


class FusedAdafactorDefectTest(unittest.TestCase, _Bf16UpdateAssertions):
    """``adafactor``: covered by the generic interposer, with no edit to the step.

    ``adafactor_step_param`` already builds ``p_data_fp32 = p.float()`` and then
    ends with ``p.copy_(p_data_fp32)`` -- a round-to-nearest write of the whole
    accumulated update. With the parameter made FP32 for the call, that copy is
    skipped and the interposer rounds stochastically instead.
    """

    @staticmethod
    def _optimizer(p):
        from transformers.optimization import Adafactor

        return Adafactor(
            [p], lr=LR, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8,
            beta1=None, weight_decay=0.0, scale_parameter=False,
            relative_step=False, warmup_init=False,
        )

    def _build(self, w0, stochastic_rounding):
        from core.training.optimizers.adafactor_fused import patch_adafactor_fused

        p = torch.nn.Parameter(w0.clone())
        optimizer = self._optimizer(p)
        patch_adafactor_fused(optimizer)
        if stochastic_rounding:
            self.assertIn("step_param", attach_stochastic_rounding(optimizer))
        return p, optimizer

    def _run_via_step_param(self, w0, stochastic_rounding):
        """How a Block-Swap run updates: the post-accumulate-grad hook."""
        p, optimizer = self._build(w0, stochastic_rounding)
        group = optimizer.param_groups[0]
        for _ in range(STEPS):
            p.grad = torch.full((N,), -1.0, dtype=torch.bfloat16)
            optimizer.step_param(p, group)
        return p.detach()

    def _run_via_step(self, w0, stochastic_rounding):
        """How every other Adafactor run updates: the ordinary ``optimizer.step()``."""
        p, optimizer = self._build(w0, stochastic_rounding)
        for _ in range(STEPS):
            p.grad = torch.full((N,), -1.0, dtype=torch.bfloat16)
            optimizer.step()
        return p.detach()

    def test_stochastic_rounding_reaches_the_fused_adafactor_update(self):
        # Adafactor's update is ~lr per step for a constant gradient, so the
        # same 512*lr threshold applies; MAX_STEP keeps it conservative.
        self._assert_defect_then_fix(self._run_via_step_param, "adafactor_fused.step_param")

    def test_stochastic_rounding_reaches_adafactor_through_optimizer_step(self):
        """The defect this class originally missed entirely.

        ``adafactor_step`` looped over the parameters calling the MODULE-LEVEL
        ``adafactor_step_param(self, p, group)``. Interposition rebinds the
        INSTANCE attribute, so every configuration that updates through
        ``step()`` -- i.e. Adafactor without Block Swap, which is the common one,
        and Adafactor under fused optimizer groups -- bypassed stochastic
        rounding completely while the setup log said it was attached. Three of
        Adafactor's four reachable configurations were inert.

        Driving ``step()`` rather than ``step_param()`` is the whole point of
        this test: the previous suite only ever drove the one path that worked.
        """
        self._assert_defect_then_fix(self._run_via_step, "adafactor_fused.step")

    def test_both_entry_points_reach_the_same_interposed_update(self):
        """Pins the mechanism, so the two paths cannot drift apart again."""
        p, optimizer = self._build(_weights(), True)
        self.assertTrue(getattr(optimizer.step_param, WRAPPED_ATTR, False))

        seen = []
        wrapped = optimizer.step_param
        optimizer.step_param = lambda param, group: (seen.append(param.dtype),
                                                     wrapped(param, group))[1]
        p.grad = torch.full((N,), -1.0, dtype=torch.bfloat16)
        optimizer.step()
        self.assertEqual(seen, [torch.bfloat16],
                         "optimizer.step() did not dispatch through self.step_param")


@unittest.skipUnless(torch.cuda.is_available(), "the 8-bit kernels require CUDA")
class BitsandbytesKernelDefectTest(unittest.TestCase, _Bf16UpdateAssertions):
    """The shipped default, exactly as it runs: bnb AdamW8bit, real 8-bit kernel.

    The kernel dispatches on the GRADIENT dtype and reads the parameter through
    a pointer of that same type, so the interposition has to replace both
    ``p.data`` and ``p.grad`` with FP32 images -- replacing only one silently
    reinterprets memory.
    """

    def _run(self, w0, stochastic_rounding):
        import bitsandbytes as bnb

        p = torch.nn.Parameter(w0.clone().cuda())
        optimizer = bnb.optim.AdamW8bit([p], lr=LR, weight_decay=0.0)
        if stochastic_rounding:
            self.assertIn("update_step", attach_stochastic_rounding(optimizer))
        for _ in range(STEPS):
            p.grad = torch.full_like(p, -1.0)
            optimizer.step()
        return p.detach().cpu()

    def test_stochastic_rounding_reaches_the_default_optimizer(self):
        self._assert_defect_then_fix(self._run, "bitsandbytes AdamW8bit")

    def test_the_state_stays_8_bit(self):
        """Handing the kernel an FP32 parameter must not turn the state FP32."""
        import bitsandbytes as bnb

        p = torch.nn.Parameter(_weights().cuda())
        optimizer = bnb.optim.AdamW8bit([p], lr=LR, weight_decay=0.0)
        attach_stochastic_rounding(optimizer)
        p.grad = torch.full_like(p, -1.0)
        optimizer.step()
        self.assertEqual(optimizer.state[p]["state1"].dtype, torch.uint8)
        self.assertEqual(optimizer.state[p]["state2"].dtype, torch.uint8)

    def test_the_parameter_is_bf16_again_after_the_step(self):
        """The FP32 image lives for one update call only; nothing else sees it."""
        import bitsandbytes as bnb

        p = torch.nn.Parameter(_weights().cuda())
        optimizer = bnb.optim.AdamW8bit([p], lr=LR, weight_decay=0.0)
        attach_stochastic_rounding(optimizer)
        grad = torch.full_like(p, -1.0)
        p.grad = grad
        optimizer.step()
        self.assertEqual(p.dtype, torch.bfloat16)
        self.assertEqual(p.data.dtype, torch.bfloat16)
        self.assertIs(p.grad, grad)
        self.assertEqual(p.grad.dtype, torch.bfloat16)


class SeamCoverageTest(unittest.TestCase):
    """Which optimizers can be covered, and honesty about the ones that cannot."""

    def test_bitsandbytes_optimizers_expose_a_per_parameter_seam(self):
        import bitsandbytes as bnb

        for factory in (bnb.optim.AdamW8bit, bnb.optim.Lion8bit,
                        bnb.optim.PagedAdamW8bit, bnb.optim.PagedLion8bit):
            with self.subTest(optimizer=factory.__name__):
                optimizer = factory([torch.nn.Parameter(torch.zeros(8).bfloat16())], lr=LR)
                self.assertEqual(attach_stochastic_rounding(optimizer), ("update_step",))

    def test_torch_adamw_has_no_seam_and_is_reported_as_uncovered(self):
        """``optimizer: adamw`` updates every parameter inside one opaque call.

        An empty result is the signal BaseTrainer turns into a warning naming
        the optimizer; anything else here would make it claim coverage it does
        not have.
        """
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(8).bfloat16())], lr=LR)
        self.assertEqual(attach_stochastic_rounding(optimizer), ())

    def test_attaching_twice_does_not_nest_two_interpositions(self):
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit([torch.nn.Parameter(torch.zeros(8).bfloat16())], lr=LR)
        self.assertEqual(attach_stochastic_rounding(optimizer), ("update_step",))
        self.assertEqual(attach_stochastic_rounding(optimizer), ())

    def test_a_non_bf16_parameter_passes_through_untouched(self):
        """FP16 and FP32 parameters must reach the optimizer as themselves."""
        from core.training.optimizers.stochastic_rounding import (
            Fp32ScratchPool, fp32_master_update,
        )

        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                p = torch.nn.Parameter(torch.ones(8, dtype=dtype))
                p.grad = torch.ones(8, dtype=dtype)
                storage = p.data_ptr()
                with fp32_master_update(p, Fp32ScratchPool()) as interposed:
                    self.assertFalse(interposed)
                    self.assertEqual(p.data_ptr(), storage)
                    self.assertEqual(p.dtype, dtype)

    def test_a_parameter_without_a_gradient_passes_through_untouched(self):
        from core.training.optimizers.stochastic_rounding import (
            Fp32ScratchPool, fp32_master_update,
        )

        p = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
        with fp32_master_update(p, Fp32ScratchPool()) as interposed:
            self.assertFalse(interposed)

    def test_two_overlapping_updates_on_one_pool_are_refused(self):
        """The pool hands out one buffer per slot, so nesting would corrupt.

        The inner update would be given the same storage as the outer one, and
        the outer parameter's FP32 image -- including its accumulated update --
        would be silently overwritten. Unreachable today (every interposition
        handles one parameter at a time and does not nest), but it loses updates
        without any error, so it is refused rather than left to chance.
        """
        from core.training.optimizers.stochastic_rounding import (
            Fp32ScratchPool, fp32_master_update,
        )

        pool = Fp32ScratchPool()
        a = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
        a.grad = torch.ones(8, dtype=torch.bfloat16)
        b = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
        b.grad = torch.ones(8, dtype=torch.bfloat16)

        with fp32_master_update(a, pool):
            with self.assertRaises(RuntimeError):
                with fp32_master_update(b, pool):
                    pass
        # And the pool is usable again once the outer update has finished.
        with fp32_master_update(b, pool) as interposed:
            self.assertTrue(interposed)

    def test_the_body_sees_a_matching_fp32_parameter_and_gradient(self):
        """The kernel dtype contract, at the interposition boundary itself."""
        from core.training.optimizers.stochastic_rounding import (
            Fp32ScratchPool, fp32_master_update,
        )

        p = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
        grad = torch.full((8,), -1.0, dtype=torch.bfloat16)
        p.grad = grad
        with fp32_master_update(p, Fp32ScratchPool()) as interposed:
            self.assertTrue(interposed)
            self.assertEqual(p.dtype, torch.float32)
            self.assertEqual(p.grad.dtype, torch.float32)
            self.assertEqual(p.dtype, p.grad.dtype)
            p.data.add_(0.25)
        self.assertEqual(p.dtype, torch.bfloat16)
        self.assertIs(p.grad, grad)
        # 1.25 is exactly representable in BF16, so no rounding happens here.
        self.assertTrue(torch.equal(p.data, torch.full((8,), 1.25, dtype=torch.bfloat16)))


class _StubTrainer:
    """The smallest object BaseTrainer's optimizer setup can run against.

    Mirrors the stub in ``optimizer_option_threading_test``: real methods, no
    model, no dataset, no CUDA -- the code under test is optimizer construction
    and what gets attached to the result.
    """

    setup_optimizer = BaseTrainer.setup_optimizer
    _resolved_optimizer_hyperparameters = BaseTrainer._resolved_optimizer_hyperparameters
    _ringbuffer_optimizer_kwargs = BaseTrainer._ringbuffer_optimizer_kwargs
    _setup_fused_backward_pass = BaseTrainer._setup_fused_backward_pass
    _setup_fused_optimizer_groups = BaseTrainer._setup_fused_optimizer_groups
    _attach_stochastic_rounding = BaseTrainer._attach_stochastic_rounding
    _RINGBUFFER_ONLY_OPTIONS = BaseTrainer._RINGBUFFER_ONLY_OPTIONS
    _NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS = BaseTrainer._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS

    def __init__(self, **overrides: Any):
        self.log_prefix = "[StubTrainer]"
        self.learning_rate = LR
        self.weight_dtype = torch.bfloat16
        self.blocks_to_swap = 0
        self.num_optimizer_groups = 0
        self.use_ema = False
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
        # A real DiT-shaped weight, not a placeholder: these trainers are driven
        # end-to-end below and the assertion is about which elements move.
        self.param = torch.nn.Parameter(_weights())

    def setup_trainable_parameters(self):
        return [{"params": [self.param], "lr": self.learning_rate}]

    def _setup_ema(self):
        pass


def _is_wrapped(optimizer, name: str) -> bool:
    return bool(getattr(getattr(optimizer, name, None), WRAPPED_ATTR, False))


class TrainerAttachmentTest(unittest.TestCase):
    """setup_optimizer must attach it for the shipped default, and only on request."""

    DEFAULT = "adamw8bit"  # TRAINING_DEFAULTS["optimizer"], and train_runner's full-FT default

    def test_the_shipped_default_optimizer_is_the_one_that_needs_this(self):
        from api.param_defaults import TRAINING_DEFAULTS

        self.assertEqual(TRAINING_DEFAULTS["optimizer"], self.DEFAULT)
        self.assertNotIn(self.DEFAULT, BaseTrainer._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS)

    def test_the_default_optimizer_gets_stochastic_rounding_when_requested(self):
        trainer = _StubTrainer(optimizer_stochastic_rounding=True)
        trainer.setup_optimizer(optimizer_type=self.DEFAULT, total_steps=10)
        self.assertTrue(
            _is_wrapped(trainer.optimizer, "update_step"),
            "the shipped default full-FT optimizer did not get a stochastic-rounding path",
        )

    def test_nothing_is_attached_when_the_flag_is_off(self):
        trainer = _StubTrainer(optimizer_stochastic_rounding=False)
        trainer.setup_optimizer(optimizer_type=self.DEFAULT, total_steps=10)
        self.assertFalse(_is_wrapped(trainer.optimizer, "update_step"))

    def test_the_ring_buffer_optimizers_are_left_to_their_own_implementation(self):
        trainer = _StubTrainer(optimizer_stochastic_rounding=True)
        trainer.setup_optimizer(optimizer_type="adamw8bit_ringbuffer", total_steps=10)
        self.assertFalse(_is_wrapped(trainer.optimizer, "step_param"))
        self.assertTrue(trainer.optimizer.param_groups[0]["stochastic_rounding"])

    def test_the_flag_is_no_longer_declared_ring_buffer_only(self):
        """It used to be warned about as unsupported for every other optimizer."""
        names = [name for name, _ in BaseTrainer._RINGBUFFER_ONLY_OPTIONS]
        self.assertNotIn("optimizer_stochastic_rounding", names)


class ReachableConfigurationMatrixTest(unittest.TestCase, _Bf16UpdateAssertions):
    """Every (optimizer, blocks_to_swap, num_optimizer_groups) a full FT can reach.

    Built after an audit found that ``adafactor`` was inert in three of its four
    reachable configurations while ``setup_optimizer`` logged that stochastic
    rounding was attached, and that the suite of the day asserted only that a
    method had been rebound -- never that the weights moved. So each case here
    goes through the REAL ``BaseTrainer.setup_optimizer`` and then drives the
    update through whichever entry point that configuration actually uses,
    asserting the numerical outcome:

        round-to-nearest  -> the high-magnitude weights are bitwise unchanged
        stochastic rounding -> they move, and carry the drift

    If stochastic rounding silently stops applying to any row of this matrix,
    the case named in the subTest fails.
    """

    # (optimizer, blocks_to_swap, num_optimizer_groups)
    CASES = (
        ("adamw8bit", 0, 0),    # the shipped default, as shipped
        ("adamw8bit", 0, 6),    # groups are only built when block swap is on
        ("adamw8bit", 22, 0),   # fused backward pass: native step_param
        ("adafactor", 0, 0),    # the common Adafactor run
        ("adafactor", 0, 6),
        ("adafactor", 22, 0),   # fused backward pass
        ("adafactor", 22, 6),   # fused optimizer groups
        ("lion8bit", 0, 0),
    )

    # adamw8bit + block swap + optimizer groups is refused at setup (8-bit
    # optimizers cannot update CPU-resident parameters), and adamw has no seam
    # at all -- both are asserted elsewhere rather than driven here.

    @staticmethod
    def _needs_cuda(optimizer_type, blocks_to_swap) -> bool:
        """The bitsandbytes kernels only run on CUDA, in both paths.

        The Block Swap path used to be exempt because its ``step_param`` was
        plain Python; it now delegates to the same kernel.
        """
        return optimizer_type in ("adamw8bit", "lion8bit")

    def _drive(self, optimizer_type, blocks_to_swap, groups, stochastic_rounding, w0):
        trainer = _StubTrainer(
            optimizer_stochastic_rounding=stochastic_rounding,
            blocks_to_swap=blocks_to_swap,
            num_optimizer_groups=groups,
        )
        device = "cuda" if self._needs_cuda(optimizer_type, blocks_to_swap) else "cpu"
        trainer.param = torch.nn.Parameter(w0.clone().to(device))
        trainer.setup_optimizer(optimizer_type=optimizer_type, total_steps=STEPS)

        p = trainer.param
        fused = getattr(trainer, "fused_optimizer_groups", None)
        optimizers = list(fused.optimizers) if fused is not None else [trainer.optimizer]

        for _ in range(STEPS):
            p.grad = torch.full_like(p, -1.0)
            if getattr(trainer, "use_fused_backward", False):
                # Block Swap: the post-accumulate-grad hook calls step_param.
                trainer.optimizer.step_param(p, trainer.optimizer.param_groups[0])
            else:
                # Everything else -- including fused optimizer groups, whose
                # hooks call optimizer.step() on the group that owns the param.
                for optimizer in optimizers:
                    optimizer.step()
            p.grad = None
        return p.detach().cpu()

    def test_every_reachable_configuration_applies_stochastic_rounding(self):
        w0 = _weights()
        for optimizer_type, blocks_to_swap, groups in self.CASES:
            label = f"{optimizer_type} blocks_to_swap={blocks_to_swap} groups={groups}"
            with self.subTest(case=label):
                if (self._needs_cuda(optimizer_type, blocks_to_swap)
                        and not torch.cuda.is_available()):
                    self.skipTest("bitsandbytes kernels require CUDA")
                self._assert_defect_then_fix(
                    lambda w, stochastic_rounding, o=optimizer_type,
                    b=blocks_to_swap, g=groups:
                    self._drive(o, b, g, stochastic_rounding, w),
                    label,
                )

    def test_adamw_is_refused_rather_than_reported_as_covered(self):
        """The one selectable optimizer with no per-parameter seam."""
        trainer = _StubTrainer(optimizer_stochastic_rounding=True)
        trainer.setup_optimizer(optimizer_type="adamw", total_steps=STEPS)
        self.assertEqual(attach_stochastic_rounding(trainer.optimizer), ())

    def test_the_matrix_covers_every_optimizer_the_panel_offers(self):
        """A configuration absent from CASES is a configuration nothing pins."""
        offered = {"adamw", "adamw8bit", "adamw8bit_ringbuffer",
                   "lion8bit", "lion8bit_ringbuffer", "adafactor"}
        driven = {optimizer for optimizer, _, _ in self.CASES}
        driven |= {"adamw"}  # covered by the refusal test above
        driven |= set(BaseTrainer._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS)  # own suite
        self.assertEqual(offered - driven, set())


if __name__ == "__main__":
    unittest.main()
