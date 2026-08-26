"""Guard: BF16 optimizer updates below half a ULP must not be silently discarded.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/bf16_stochastic_rounding_test.py -v

THE DEFECT
----------
Full fine-tuning writes optimizer updates straight into BF16 storage.
``train_runner`` forces ``weight_dtype=bf16`` for Z-Image, Anima, Ideogram 4,
MiniT2I, Krea 2 and the bf16-native models (Lens / LTX-2.3 / ACE-Step), and no
full-FT configuration keeps an FP32 master weight. Flux2, SD1.5 and SDXL are NOT
in that list: they keep the configured weight dtype (fp16 by default), where the
same failure exists with a threshold of ``|w| <= 2048*lr`` -- stochastic rounding
as implemented covers bf16 parameters only, so fp16 runs are not protected.

BF16 has an 8-bit significand: for a weight in [2^e, 2^(e+1)) one ULP is
2^(e-7), so round-to-nearest discards any update smaller than 2^(e-8). An
Adam-family step has magnitude ~lr, which gives the closed form pinned by
``test_round_to_nearest_freezes_every_weight_above_512_lr``:

    a weight only ever moves when |w| <= 512 * lr.

At the shipped default lr of 1e-5 that is |w| <= 5.12e-3, which excludes most of
a DiT's weights, and because round-to-nearest is deterministic those weights are
frozen at their initial bit pattern for the entire run -- not "slow to train",
frozen. Measured on real checkpoints against an FP32 reference: Krea 2 realized
4.9% of the intended drift with 8.7% of elements ever moving.

Stochastic rounding is the fix: rounding up with probability equal to the
fractional part makes ``E[round(x)] == x``, so a sub-ULP update survives in
expectation instead of vanishing. The flag existed at every layer of the stack
(``param_defaults`` -> ``routes`` -> YAML -> optimizer constructor) except the
two that would have made it work, so ticking the checkbox changed nothing.

WHAT EACH GROUP PINS
--------------------
* ``Bf16RoundingBehaviourTest``  -- the numerical defect itself, in the
  production dtype (BF16 params, BF16 grads): RTN freezes, SR moves.
* ``KernelDtypeContractTest``    -- the FP32 image of the parameter and the
  gradient handed to the 8-bit CUDA kernels must have the SAME dtype. Passing an
  FP32 master with the BF16 gradient autograd actually produces is what made the
  stochastic-rounding path raise at step 1
  (``TORCH_CHECK(param.dtype() == grad.dtype())``, adamw8bit_cuda.cpp:129).
* ``OptimizerStepTest``          -- both ring-buffer optimizers, driven through
  their real ``step()``, with BF16 params and BF16 grads.
* ``FlagWiringTest``             -- the flag reaches the optimizer from the
  config, at every trainer call site.
"""

from __future__ import annotations

import ast
import inspect
import sys
import unittest
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.optimizers.stochastic_rounding import (  # noqa: E402
    Fp32ScratchPool,
    copy_stochastic_bf16,
    prepare_master_and_grad,
    should_use_stochastic_rounding,
)

# The shipped full fine-tune default (param_defaults TRAINING_DEFAULTS["lr"] is
# 1e-4 for LoRA; full FT runs at 1e-5). One Adam step moves a weight by ~lr.
LR = 1e-5
STEPS = 400
SEED = 20260805


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Distance to the next representable BF16 value above |x| (8-bit significand)."""
    exponent = torch.floor(torch.log2(x.abs().float()))
    return torch.pow(2.0, exponent - 7.0)


def _rtn_steps(w: torch.Tensor, update: float, steps: int) -> torch.Tensor:
    """Apply ``update`` per step with round-to-nearest into BF16 (what ships today)."""
    w = w.clone()
    for _ in range(steps):
        w.copy_((w.float() + update).bfloat16())
    return w


def _sr_steps(w: torch.Tensor, update: float, steps: int) -> torch.Tensor:
    """Apply ``update`` per step with stochastic rounding into BF16 (the fix)."""
    w = w.clone()
    pool = Fp32ScratchPool()
    for _ in range(steps):
        master = pool.copy_of("master", w)
        master.add_(update)
        copy_stochastic_bf16(w, master)
    return w


class Bf16RoundingBehaviourTest(unittest.TestCase):
    """The defect, measured on a DiT-shaped weight tensor."""

    def setUp(self):
        torch.manual_seed(SEED)
        # DiT weights are ~N(0, 0.02): most of the mass sits far above 512*lr.
        self.w0 = (torch.randn(1 << 16) * 0.02).bfloat16()
        self.frozen_mask = self.w0.abs().float() >= 512 * LR
        # The tensor has to actually contain such weights for the test to mean
        # anything.
        self.assertGreater(self.frozen_mask.float().mean().item(), 0.5)

    def test_round_to_nearest_freezes_every_weight_above_512_lr(self):
        """|w| > 512*lr => half a ULP exceeds the step => bitwise unchanged, forever."""
        after = _rtn_steps(self.w0, LR, STEPS)

        frozen = after[self.frozen_mask]
        original = self.w0[self.frozen_mask]
        self.assertTrue(
            torch.equal(frozen, original),
            "round-to-nearest moved a weight it cannot move; the ULP analysis is wrong",
        )

        # And the intended drift is fully lost for those elements.
        intended = STEPS * LR
        realized = (frozen.float() - original.float()).mean().item()
        self.assertEqual(realized, 0.0)
        self.assertGreater(intended, 0.0)

    def test_round_to_nearest_moves_only_the_low_magnitude_tail(self):
        """The elements that do move are the small ones, and they move in ULP jumps."""
        after = _rtn_steps(self.w0, LR, STEPS)
        moved = after.ne(self.w0)

        # Nothing above the threshold moved; everything that moved is below it.
        self.assertEqual(moved[self.frozen_mask].sum().item(), 0)
        moved_frac = moved.float().mean().item()
        self.assertGreater(moved_frac, 0.0)
        self.assertLess(moved_frac, 0.5)

    def test_stochastic_rounding_realizes_the_intended_drift(self):
        """SR is unbiased, so the same weights drift by steps*lr in expectation."""
        after = _sr_steps(self.w0, LR, STEPS)

        frozen_before = self.w0[self.frozen_mask].float()
        after_frozen = after[self.frozen_mask].float()

        intended = STEPS * LR
        realized = (after_frozen - frozen_before).mean().item()
        self.assertAlmostEqual(realized / intended, 1.0, delta=0.02)

        moved_frac = after_frozen.ne(frozen_before).float().mean().item()
        self.assertGreater(moved_frac, 0.95)

    def test_a_single_copy_is_unbiased(self):
        """E[stochastic_round(x)] == x, including for values between two BF16 neighbours."""
        torch.manual_seed(SEED)
        base = torch.full((1 << 16,), 1.0, dtype=torch.bfloat16)
        ulp = _bf16_ulp(base)[0].item()

        for fraction in (0.1, 0.5, 0.9):
            source = base.float() + fraction * ulp
            target = torch.empty_like(base)
            copy_stochastic_bf16(target, source)

            # Only the two neighbouring BF16 values may appear.
            distinct = torch.unique(target.float())
            self.assertLessEqual(len(distinct), 2)
            self.assertAlmostEqual(
                target.float().mean().item(), source[0].item(), delta=0.02 * ulp
            )

    def test_round_to_nearest_would_have_been_biased_for_the_same_value(self):
        """The contrast: RTN sends every element the same way, SR does not."""
        base = torch.full((1024,), 1.0, dtype=torch.bfloat16)
        ulp = _bf16_ulp(base)[0].item()
        source = base.float() + 0.1 * ulp

        rtn = source.bfloat16()
        self.assertTrue(torch.equal(rtn, base))  # the whole 0.1 ULP is discarded

        torch.manual_seed(SEED)
        sr = torch.empty_like(base)
        copy_stochastic_bf16(sr, source)
        self.assertFalse(torch.equal(sr, base))

    def test_rejects_a_non_bf16_target_or_non_fp32_source(self):
        with self.assertRaises(ValueError):
            copy_stochastic_bf16(torch.zeros(4), torch.zeros(4))
        with self.assertRaises(ValueError):
            copy_stochastic_bf16(
                torch.zeros(4, dtype=torch.bfloat16), torch.zeros(4, dtype=torch.bfloat16)
            )


class KernelDtypeContractTest(unittest.TestCase):
    """The 8-bit kernels require param.dtype == grad.dtype. Autograd gives BF16 grads."""

    def test_master_and_grad_come_back_as_a_matching_fp32_pair(self):
        param = torch.randn(256).bfloat16()
        grad = torch.randn(256).bfloat16()  # what autograd produces for a BF16 param
        pool = Fp32ScratchPool()

        master, grad_fp32 = prepare_master_and_grad(param, grad, pool)

        self.assertEqual(master.dtype, torch.float32)
        self.assertEqual(grad_fp32.dtype, torch.float32)
        self.assertEqual(master.dtype, grad_fp32.dtype)  # the TORCH_CHECK
        self.assertEqual(master.shape, param.shape)
        self.assertTrue(torch.equal(master, param.float()))
        self.assertTrue(torch.equal(grad_fp32, grad.float()))
        self.assertTrue(grad_fp32.is_contiguous())

    def test_the_previous_shape_violated_the_contract(self):
        """Documents the crash: an FP32 master paired with the raw BF16 grad."""
        param = torch.randn(256).bfloat16()
        grad = torch.randn(256).bfloat16()
        old_master = param.detach().clone().to(dtype=torch.float32)
        self.assertNotEqual(old_master.dtype, grad.dtype)

    def test_an_fp32_grad_is_passed_through_without_a_copy(self):
        param = torch.randn(256).bfloat16()
        grad = torch.randn(256)
        pool = Fp32ScratchPool()
        _, grad_fp32 = prepare_master_and_grad(param, grad, pool)
        self.assertIs(grad_fp32, grad)

    def test_a_non_contiguous_grad_is_made_contiguous(self):
        param = torch.randn(128).bfloat16()
        grad = torch.randn(256)[::2]
        self.assertFalse(grad.is_contiguous())
        pool = Fp32ScratchPool()
        _, grad_fp32 = prepare_master_and_grad(param, grad, pool)
        self.assertTrue(grad_fp32.is_contiguous())
        self.assertTrue(torch.equal(grad_fp32, grad))

    def test_the_scratch_pool_reuses_storage_across_parameters(self):
        """Per-step allocation would be a real cost: this runs per parameter per step."""
        pool = Fp32ScratchPool()
        a = torch.randn(1024).bfloat16()
        b = torch.randn(1024).bfloat16()

        first = pool.copy_of("master", a).data_ptr()
        second = pool.copy_of("master", b).data_ptr()
        self.assertEqual(first, second)

        # It grows for a larger parameter and then holds that size.
        big = torch.randn(4096).bfloat16()
        grown = pool.copy_of("master", big).data_ptr()
        again = pool.copy_of("master", a).data_ptr()
        self.assertEqual(grown, again)

    def test_stochastic_rounding_applies_to_bf16_parameters_only(self):
        self.assertTrue(should_use_stochastic_rounding(True, torch.zeros(2, dtype=torch.bfloat16)))
        self.assertFalse(should_use_stochastic_rounding(False, torch.zeros(2, dtype=torch.bfloat16)))
        self.assertFalse(should_use_stochastic_rounding(True, torch.zeros(2)))
        self.assertFalse(should_use_stochastic_rounding(True, torch.zeros(2, dtype=torch.float16)))


class _FakeCudaParameter(torch.nn.Parameter):
    """A CPU parameter that reports ``is_cuda``.

    Both optimizers (and both fused-backward hooks) skip parameters that are not
    on CUDA, because Block Swap offloads them; their CUDA extensions take
    minutes to compile. So the update path is exercised here on CPU tensors that
    answer the residency check, with the extension replaced by a stand-in.
    """

    @property
    def is_cuda(self) -> bool:  # noqa: D401
        return True


def _fake_param(tensor: torch.Tensor) -> _FakeCudaParameter:
    return torch.Tensor._make_subclass(_FakeCudaParameter, tensor, True)


class _RecordingExtension:
    """Stands in for the compiled CUDA extension.

    Enforces the same dtype contract as the real kernels and applies a
    deterministic step of magnitude ``lr``, which is all these tests need.
    """

    def __init__(self):
        self.calls = []

    def init_quantization_maps(self, *args, **kwargs):
        pass

    def _update(self, param, grad, lr):
        if param.dtype != grad.dtype:
            raise RuntimeError("Param and Grad must have same dtype")
        self.calls.append((param.dtype, grad.dtype))
        param.add_(torch.sign(grad.to(param.dtype)), alpha=-lr)

    def adamw_8bit_update(self, param, grad, state1, state2, absmax1, absmax2,
                          beta1, beta2, eps, lr, weight_decay, gnorm_scale,
                          step, cautious):
        self._update(param, grad, lr)

    def lion_8bit_update(self, param, grad, exp_avg, absmax, beta1, beta2, eps,
                         lr, weight_decay, gnorm_scale, step, cautious):
        self._update(param, grad, lr)


class OptimizerStepTest(unittest.TestCase):
    """Both ring-buffer optimizers, through their real ``step()``.

    BF16 parameters with BF16 gradients -- the dtype pair a DiT full fine-tune
    actually produces, not the FP32 pair the stochastic-rounding path had only
    ever been exercised with.
    """

    N = 1 << 14
    STEPS = 200
    # AdamW's own step size peaks at lr / (1 - beta1) = 10*lr on the first step
    # (bias correction), so the "cannot move under round-to-nearest" threshold
    # for this loop is 512 * 10 * lr rather than 512 * lr.
    LR = 1e-6
    MAX_STEP = 10 * LR

    def _params(self):
        torch.manual_seed(SEED)
        weights = (torch.randn(self.N) * 0.02).bfloat16()
        p = _fake_param(weights.clone())
        # A constant negative gradient: every optimizer here then moves the
        # weight by about +lr per step, well below half a ULP.
        p.grad = torch.full((self.N,), -1.0, dtype=torch.bfloat16)
        return p, weights

    def _run_adamw(self, stochastic_rounding):
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        original = mod.get_extension
        mod.get_extension = lambda: _RecordingExtension()
        try:
            p, w0 = self._params()
            opt = mod.AdamW8bit_RingBuffer(
                [p], lr=self.LR, weight_decay=0.0, use_8bit=False,
                stochastic_rounding=stochastic_rounding,
            )
            for _ in range(self.STEPS):
                opt.step()
            return p, w0
        finally:
            mod.get_extension = original

    def _run_lion(self, stochastic_rounding):
        import core.training.optimizers.lion8bit_ringbuffer as mod

        original = mod.get_extension
        mod.get_extension = lambda: _RecordingExtension()
        try:
            p, w0 = self._params()
            opt = mod.Lion8bit_RingBuffer(
                [p], lr=self.LR, weight_decay=0.0, use_8bit=False,
                stochastic_rounding=stochastic_rounding,
            )
            for _ in range(self.STEPS):
                opt.step()
            return p, w0
        finally:
            mod.get_extension = original

    def _assert_defect_then_fix(self, runner, label):
        off_p, w0 = runner(False)
        frozen = w0.abs().float() >= 512 * self.MAX_STEP
        self.assertGreater(frozen.float().mean().item(), 0.5)

        self.assertTrue(
            torch.equal(off_p.data[frozen], w0[frozen]),
            f"{label}: round-to-nearest is expected to freeze these weights",
        )

        on_p, w0b = runner(True)
        self.assertTrue(torch.equal(w0, w0b))
        drift = (on_p.data[frozen].float() - w0[frozen].float()).mean().item()
        self.assertGreater(
            drift, 0.5 * self.STEPS * self.LR,
            f"{label}: stochastic rounding did not carry the sub-ULP updates",
        )
        moved = on_p.data[frozen].ne(w0[frozen]).float().mean().item()
        self.assertGreater(moved, 0.5, f"{label}: too few weights ever moved")

    def test_adamw8bit_ringbuffer_honours_stochastic_rounding(self):
        self._assert_defect_then_fix(self._run_adamw, "AdamW8bit_RingBuffer")

    def test_lion8bit_ringbuffer_honours_stochastic_rounding(self):
        """F4: Lion accepted the flag and did nothing with it (byte-identical runs)."""
        self._assert_defect_then_fix(self._run_lion, "Lion8bit_RingBuffer")


class FusedBackwardHookTest(unittest.TestCase):
    """The fused-backward hooks, driven by a real ``loss.backward()``.

    ``optimizer.step()`` is NOT involved here. When Block Swap is on, the
    trainer registers ``patch_adamw8bit_ringbuffer`` /
    ``register_lion8bit_fused_backward`` and every update happens inside a
    ``post_accumulate_grad`` hook instead -- so a block-swapped full fine-tune,
    the configuration this defect hurts most, never executes the ``step()``
    code the tests above cover. Both hooks ignored ``stochastic_rounding``
    entirely; without these cases, reverting either hook to
    ``ext.<opt>_update(param, param.grad, ...)`` leaves the whole suite green
    while block-swapped full FT silently returns to frozen weights.

    The 8-bit optimizer state is seeded by hand so that no CUDA allocation is
    needed for the absmax metadata; the hooks then take their real
    ``state['is_8bit']`` path.
    """

    IN = 64
    OUT = 64
    BACKWARDS = 200
    LR = 1e-6

    def _model(self):
        torch.manual_seed(SEED)
        model = torch.nn.Sequential(
            torch.nn.Linear(self.IN, self.OUT, bias=False),
            torch.nn.Linear(self.OUT, self.OUT, bias=False),
        )
        for layer in model:
            layer.weight = _fake_param(layer.weight.detach().bfloat16())
        initial = [p.detach().clone() for p in model.parameters()]
        return model, initial

    @staticmethod
    def _seed_adamw_state(optimizer):
        for p in optimizer.param_groups[0]["params"]:
            state = optimizer.state[p]
            state["exp_avg"] = torch.zeros(p.numel(), dtype=torch.uint8)
            state["exp_avg_sq"] = torch.zeros(p.numel(), dtype=torch.uint8)
            state["absmax1"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
            state["absmax2"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
            state["is_8bit"] = True

    @staticmethod
    def _seed_lion_state(optimizer):
        for p in optimizer.param_groups[0]["params"]:
            state = optimizer.state[p]
            state["exp_avg"] = torch.zeros(p.numel(), dtype=torch.uint8)
            state["absmax"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
            state["is_8bit"] = True

    def _drive(self, model):
        """Real forward/backward passes; every update happens inside the hooks."""
        torch.manual_seed(SEED + 1)
        for _ in range(self.BACKWARDS):
            x = torch.randn(8, self.IN, dtype=torch.bfloat16)
            model(x).float().pow(2).mean().backward()

    def _run_adamw(self, stochastic_rounding):
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        ext = _RecordingExtension()
        original = mod.get_extension
        mod.get_extension = lambda: ext
        try:
            model, initial = self._model()
            opt = mod.AdamW8bit_RingBuffer(
                list(model.parameters()), lr=self.LR, weight_decay=0.0,
                use_8bit=True, stochastic_rounding=stochastic_rounding,
            )
            self._seed_adamw_state(opt)
            mod.patch_adamw8bit_ringbuffer(model, opt)
            self._drive(model)
        finally:
            mod.get_extension = original
        return model, initial, ext

    def _run_lion(self, stochastic_rounding):
        import core.training.optimizers.lion8bit_ringbuffer as mod

        ext = _RecordingExtension()
        original = mod.get_extension
        mod.get_extension = lambda: ext
        try:
            model, initial = self._model()
            opt = mod.Lion8bit_RingBuffer(
                list(model.parameters()), lr=self.LR, weight_decay=0.0,
                use_8bit=True, stochastic_rounding=stochastic_rounding,
            )
            self._seed_lion_state(opt)
            mod.register_lion8bit_fused_backward(opt, model)
            self._drive(model)
        finally:
            mod.get_extension = original
        return model, initial, ext

    def _assert_hook_path(self, runner, label):
        # ---- stochastic rounding OFF: the shipped behaviour ----
        model, initial, ext = runner(False)
        expected_calls = self.BACKWARDS * len(initial)
        self.assertEqual(len(ext.calls), expected_calls, f"{label}: hooks did not fire")
        self.assertEqual(
            set(ext.calls), {(torch.bfloat16, torch.bfloat16)},
            f"{label}: the kernel should see the BF16 pair when SR is off",
        )
        # The elements that cannot move under round-to-nearest: |w| >= 512*lr
        # (half a ULP exceeds the step). The rest of the tensor is the
        # low-magnitude tail, which does move -- and overshoots in ULP jumps.
        masks = [w0.abs().float() >= 512 * self.LR for w0 in initial]
        self.assertGreater(
            torch.cat([m.flatten() for m in masks]).float().mean().item(), 0.9,
            f"{label}: the fixture must contain weights above the threshold",
        )
        for p, w0, mask in zip(model.parameters(), initial, masks):
            self.assertTrue(
                torch.equal(p.detach()[mask], w0[mask]),
                f"{label}: round-to-nearest is expected to freeze these weights",
            )

        # ---- stochastic rounding ON: the fix, through the same hook ----
        model, initial, ext = runner(True)
        self.assertEqual(len(ext.calls), expected_calls, f"{label}: hooks did not fire")
        self.assertEqual(
            set(ext.calls), {(torch.float32, torch.float32)},
            f"{label}: the hook must hand the kernel a matching FP32 pair",
        )

        # Measured on the SAME elements round-to-nearest froze above. The
        # gradient sign differs per element, so movement is measured as a
        # magnitude rather than a signed drift.
        for p in model.parameters():
            self.assertEqual(p.dtype, torch.bfloat16, f"{label}: param dtype changed")

        moved = torch.cat([
            p.detach()[mask].ne(w0[mask]).flatten()
            for p, w0, mask in zip(model.parameters(), initial, masks)
        ])
        self.assertGreater(
            moved.float().mean().item(), 0.1,
            f"{label}: stochastic rounding did not reach the fused-backward hook",
        )

    def test_adamw_fused_backward_hook_honours_stochastic_rounding(self):
        self._assert_hook_path(self._run_adamw, "patch_adamw8bit_ringbuffer")

    def test_lion_fused_backward_hook_honours_stochastic_rounding(self):
        self._assert_hook_path(self._run_lion, "register_lion8bit_fused_backward")

    def test_the_hook_clears_the_gradient_it_applied(self):
        """Pins that the update really happened in the hook, not in step()."""
        model, _, _ = self._run_adamw(True)
        for p in model.parameters():
            self.assertIsNone(p.grad)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class QuantizedPathDtypeTest(unittest.TestCase):
    """The 8-bit state path, with the kernel's dtype check enforced by a stand-in.

    Tiny tensors on the real device (the 8-bit state allocator requires CUDA for
    its absmax metadata); the compiled extension is replaced, so no kernel is
    built and no model is loaded.
    """

    N = 1024

    def _param(self):
        torch.manual_seed(SEED)
        w = (torch.randn(self.N, device="cuda") * 0.02).bfloat16()
        p = _fake_param(w)
        p.grad = torch.full((self.N,), -1.0, dtype=torch.bfloat16, device="cuda")
        return p

    def test_adamw_8bit_update_receives_matching_dtypes(self):
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        ext = _RecordingExtension()
        original = mod.get_extension
        mod.get_extension = lambda: ext
        try:
            p = self._param()
            opt = mod.AdamW8bit_RingBuffer(
                [p], lr=LR, weight_decay=0.0, use_8bit=True, stochastic_rounding=True
            )
            opt.step()  # used to raise "Param and Grad must have same dtype"
        finally:
            mod.get_extension = original

        self.assertEqual(ext.calls, [(torch.float32, torch.float32)])

    def test_lion_8bit_update_receives_matching_dtypes(self):
        import core.training.optimizers.lion8bit_ringbuffer as mod

        ext = _RecordingExtension()
        original = mod.get_extension
        mod.get_extension = lambda: ext
        try:
            p = self._param()
            opt = mod.Lion8bit_RingBuffer(
                [p], lr=LR, weight_decay=0.0, use_8bit=True, stochastic_rounding=True
            )
            opt.step()
        finally:
            mod.get_extension = original

        self.assertEqual(ext.calls, [(torch.float32, torch.float32)])

    def test_without_stochastic_rounding_the_kernel_still_sees_bf16(self):
        """The default path is unchanged: BF16 param, BF16 grad, no FP32 image."""
        import core.training.optimizers.adamw8bit_ringbuffer as mod

        ext = _RecordingExtension()
        original = mod.get_extension
        mod.get_extension = lambda: ext
        try:
            p = self._param()
            opt = mod.AdamW8bit_RingBuffer(
                [p], lr=LR, weight_decay=0.0, use_8bit=True, stochastic_rounding=False
            )
            opt.step()
        finally:
            mod.get_extension = original

        self.assertEqual(ext.calls, [(torch.bfloat16, torch.bfloat16)])


class FlagWiringTest(unittest.TestCase):
    """The flag has to survive every hop from the request to the optimizer."""

    TRAINER_CALLS = (
        "LoRATrainer",
        "ReLoRATrainer",
        "FullParameterTrainer",
        "ControlNetTrainer",
    )

    def test_api_default_comes_from_param_defaults(self):
        from api.param_defaults import TRAINING_DEFAULTS
        from api.routes import TrainingRunCreateRequest

        field = TrainingRunCreateRequest.model_fields["optimizer_stochastic_rounding"]
        self.assertEqual(field.default, TRAINING_DEFAULTS["optimizer_stochastic_rounding"])
        # Tri-state now (see sensenova_full_finetune_stochastic_rounding_test's
        # TransportIsTriStateTest): None means "not specified", not "off". A
        # request that never touches this field still trains with
        # round-to-nearest on every architecture except the ones in
        # FULL_FINETUNE_FORCED_STOCHASTIC_ROUNDING_BY_ARCH, unchanged from
        # before this fix -- only the explicit-False path changed.
        self.assertIsNone(field.default)

    def test_full_finetune_config_writes_the_key(self):
        import yaml
        from core.training.training_config import TrainingConfigGenerator

        def _train_section(params):
            text = TrainingConfigGenerator.generate_full_finetune_config(
                params,
                run_name="sr_wiring",
                base_model_path="model.safetensors",
                output_dir="out",
                dataset_path="data",
            )
            return yaml.safe_load(text)["config"]["process"][0]["train"]

        on = _train_section({"total_steps": 10, "optimizer_stochastic_rounding": True})
        self.assertTrue(on["optimizer_stochastic_rounding"])

        off = _train_section({"total_steps": 10})
        self.assertFalse(off.get("optimizer_stochastic_rounding", False))

    def test_base_trainer_passes_it_to_ringbuffer_optimizers(self):
        from core.training.base_trainer import BaseTrainer

        class _Stub:
            _ringbuffer_optimizer_kwargs = BaseTrainer._ringbuffer_optimizer_kwargs

        stub = _Stub()
        stub.optimizer_cautious = False
        stub.optimizer_schedule_free = False
        stub.optimizer_warmup_steps = 0
        stub.optimizer_schedule_free_r = 0.0
        stub.optimizer_schedule_free_weight_lr_power = 2.0
        stub.optimizer_use_radam = False
        stub.optimizer_stochastic_rounding = True

        kwargs = stub._ringbuffer_optimizer_kwargs()
        self.assertTrue(kwargs["stochastic_rounding"])

        stub.optimizer_stochastic_rounding = False
        self.assertFalse(stub._ringbuffer_optimizer_kwargs()["stochastic_rounding"])

    def test_base_trainer_accepts_the_constructor_argument(self):
        from core.training.base_trainer import BaseTrainer

        signature = inspect.signature(BaseTrainer.__init__)
        self.assertIn("optimizer_stochastic_rounding", signature.parameters)
        self.assertIs(signature.parameters["optimizer_stochastic_rounding"].default, False)

    def test_factory_forwards_it_to_both_ringbuffer_optimizers(self):
        from core.training.optimizer_factory import OptimizerFactory
        import core.training.optimizers.adamw8bit_ringbuffer as adamw_mod
        import core.training.optimizers.lion8bit_ringbuffer as lion_mod

        recorded = {}

        def _recorder(name):
            def _factory(params, **kwargs):
                recorded[name] = kwargs
                return object()
            return _factory

        originals = (adamw_mod.AdamW8bit_RingBuffer, lion_mod.Lion8bit_RingBuffer)
        adamw_mod.AdamW8bit_RingBuffer = _recorder("adamw")
        lion_mod.Lion8bit_RingBuffer = _recorder("lion")
        try:
            for optimizer_type, key in (
                ("adamw8bit_ringbuffer", "adamw"),
                ("lion8bit_ringbuffer", "lion"),
            ):
                OptimizerFactory.create_optimizer(
                    optimizer_type=optimizer_type,
                    params=[torch.nn.Parameter(torch.zeros(2))],
                    learning_rate=LR,
                    stochastic_rounding=True,
                )
                self.assertTrue(recorded[key]["stochastic_rounding"], key)
        finally:
            adamw_mod.AdamW8bit_RingBuffer, lion_mod.Lion8bit_RingBuffer = originals

    def test_every_trainer_call_site_passes_the_flag(self):
        """train_runner builds four trainers; each one has to receive it."""
        import core.training.train_runner as train_runner

        source = Path(inspect.getsourcefile(train_runner)).read_text(encoding="utf-8")
        tree = ast.parse(source)

        seen = {name: False for name in self.TRAINER_CALLS}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in seen:
                continue
            keywords = {kw.arg for kw in node.keywords}
            seen[node.func.id] = "optimizer_stochastic_rounding" in keywords

        for name, ok in seen.items():
            self.assertTrue(ok, f"{name}(...) does not pass optimizer_stochastic_rounding")

    def test_train_runner_reads_the_config_key(self):
        import core.training.train_runner as train_runner

        source = Path(inspect.getsourcefile(train_runner)).read_text(encoding="utf-8")
        self.assertIn("train_config.get('optimizer_stochastic_rounding'", source)


if __name__ == "__main__":
    unittest.main()
