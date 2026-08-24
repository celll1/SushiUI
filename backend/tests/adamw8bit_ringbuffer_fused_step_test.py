"""Guard: the ring-buffer fused-backward hook must advance Adam's step counter.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adamw8bit_ringbuffer_fused_step_test.py -v

THE DEFECT
----------
``patch_adamw8bit_ringbuffer``'s hook passed ``optimizer.step_count + 1`` to the
kernel, "because the hook runs before step()". It does not: with
``use_fused_backward`` the trainer never calls ``optimizer.step()``
(base_trainer.py, ``if not self.use_fused_backward and ...``), and ``step()`` is
the only place ``step_count`` is incremented. So every hook call passed step=1
and Adam's bias correction was pinned to its first-step value forever. Measured
by ``core/training/probes/optimizer_bf16_and_vram.py`` over 20 steps with a fixed
gradient: 2.586e-4 of drift against the 2.0e-4 the ``step()`` path produces
(129%). This is the Block-Swap-recommended configuration.

The fix is a per-parameter ``state['step']``, as ``adamw8bit_fused`` and
``adafactor_fused`` already keep -- the hook fires once per PARAMETER, so a
global counter incremented there would advance P times per optimizer step.

The tests run on CPU: the parameters answer ``is_cuda`` (Block Swap residency
check) and the compiled CUDA extension is replaced by a stand-in that does the
real AdamW arithmetic in FP32, so the step number it is handed is observable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import core.training.optimizers.adamw8bit_ringbuffer as rb  # noqa: E402

SEED = 20260824
LR = 1e-3
IN = 8
OUT = 8
BACKWARDS = 6


class _FakeCudaParameter(torch.nn.Parameter):
    """A CPU parameter that reports ``is_cuda`` (see bf16_stochastic_rounding_test)."""

    @property
    def is_cuda(self) -> bool:  # noqa: D401
        return True


def _fake_param(tensor: torch.Tensor) -> _FakeCudaParameter:
    return torch.Tensor._make_subclass(_FakeCudaParameter, tensor, True)


class _AdamWMathExtension:
    """Stand-in for the compiled kernel that honours the ``step`` it is given.

    Keeps its own FP32 moments, so the update is a real bias-corrected AdamW step
    and a wrong step number shows up in the parameter, not only in a log.
    """

    def __init__(self):
        self.steps: list[int] = []
        self._moments: dict = {}

    def init_quantization_maps(self, *args, **kwargs):
        pass

    def adamw_8bit_update(self, param, grad, state1, state2, absmax1, absmax2,
                          beta1, beta2, eps, lr, weight_decay, gnorm_scale,
                          step, cautious):
        self.steps.append(int(step))
        moments = self._moments.get(id(param))
        if moments is None:
            moments = (torch.zeros(param.shape, dtype=torch.float32),
                       torch.zeros(param.shape, dtype=torch.float32))
            self._moments[id(param)] = moments
        exp_avg, exp_avg_sq = moments

        g = grad.float() * gnorm_scale
        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
        if weight_decay:
            param.mul_(1 - lr * weight_decay)
        param.addcdiv_(exp_avg / bias_correction1, denom, value=-lr)


def _model():
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(IN, OUT, bias=False),
        torch.nn.Linear(OUT, OUT, bias=False),
    )
    for layer in model:
        layer.weight = _fake_param(layer.weight.detach().float())
    return model


def _seed_8bit_state(optimizer):
    """Pre-seed the 8-bit state so ``_init_param_state`` (CUDA-only) is skipped."""
    for p in optimizer.param_groups[0]["params"]:
        state = optimizer.state[p]
        state["exp_avg"] = torch.zeros(p.numel(), dtype=torch.uint8)
        state["exp_avg_sq"] = torch.zeros(p.numel(), dtype=torch.uint8)
        state["absmax1"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
        state["absmax2"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
        state["is_8bit"] = True


def _optimizer(model, ext):
    original = rb.get_extension
    rb.get_extension = lambda: ext
    try:
        opt = rb.AdamW8bit_RingBuffer(
            list(model.parameters()), lr=LR, weight_decay=0.0, use_8bit=True,
        )
    finally:
        rb.get_extension = original
    _seed_8bit_state(opt)
    return opt


def _batches(n=BACKWARDS):
    torch.manual_seed(SEED + 1)
    return [torch.randn(4, IN) for _ in range(n)]


def _run_hook_path(n=BACKWARDS):
    ext = _AdamWMathExtension()
    model = _model()
    opt = _optimizer(model, ext)
    rb.patch_adamw8bit_ringbuffer(model, opt)
    for x in _batches(n):
        model(x).pow(2).mean().backward()
    return model, opt, ext


def _run_step_path(n=BACKWARDS):
    ext = _AdamWMathExtension()
    model = _model()
    opt = _optimizer(model, ext)
    for x in _batches(n):
        model(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    return model, opt, ext


class FusedHookStepCounterTest(unittest.TestCase):
    def test_the_hook_advances_a_per_parameter_step(self):
        model, opt, ext = _run_hook_path()

        params = list(model.parameters())
        self.assertEqual(len(ext.steps), BACKWARDS * len(params), "hooks did not fire")
        for p in params:
            self.assertEqual(
                opt.state[p]["step"], BACKWARDS,
                "each parameter's step must count optimizer steps, not hook calls",
            )
        # Every parameter walks 1..BACKWARDS: never pinned at 1, never advanced
        # once per hook call (which would reach BACKWARDS * len(params)).
        self.assertEqual(sorted(ext.steps), sorted(list(range(1, BACKWARDS + 1)) * len(params)))
        self.assertEqual(
            opt.step_count, 0,
            "step() is never called on the fused path -- the global counter cannot drive it",
        )

    def test_the_hook_and_step_agree_on_the_same_gradients(self):
        """The real check: identical trajectories through both paths."""
        hook_model, _, hook_ext = _run_hook_path()
        step_model, step_opt, step_ext = _run_step_path()

        self.assertEqual(sorted(hook_ext.steps), sorted(step_ext.steps))
        for hook_p, step_p in zip(hook_model.parameters(), step_model.parameters()):
            self.assertTrue(
                torch.equal(hook_p.detach(), step_p.detach()),
                "fused hook and step() diverged on the same gradient sequence",
            )
        self.assertEqual(step_opt.step_count, BACKWARDS)

    def test_step_still_drives_bias_correction_from_the_global_counter(self):
        _, opt, ext = _run_step_path()
        params = len(opt.param_groups[0]["params"])
        expected = [s for s in range(1, BACKWARDS + 1) for _ in range(params)]
        self.assertEqual(ext.steps, expected)
        self.assertEqual(opt.step_count, BACKWARDS)


class ResumeTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "load_state_dict pins absmax to CUDA")
    def test_state_dict_round_trip_preserves_the_per_parameter_step(self):
        _, opt, _ = _run_hook_path()
        saved = opt.state_dict()
        self.assertTrue(
            all(st["step"] == BACKWARDS for st in saved["state"].values()),
            "the per-parameter step must be serialized",
        )

        ext = _AdamWMathExtension()
        model = _model()
        resumed = _optimizer(model, ext)
        resumed.load_state_dict(saved)
        rb.patch_adamw8bit_ringbuffer(model, resumed)

        for p in model.parameters():
            self.assertEqual(resumed.state[p]["step"], BACKWARDS)

        model(_batches(1)[0]).pow(2).mean().backward()
        self.assertEqual(
            set(ext.steps), {BACKWARDS + 1},
            "a resumed run must continue bias correction, not restart at step 1",
        )

    def test_a_run_that_switches_from_step_to_the_hook_continues(self):
        """Realistic mixing: the same process/checkpoint changes path (Block Swap
        toggled), so the hook must not restart bias correction at 1."""
        model, opt, ext = _run_step_path()
        rb.patch_adamw8bit_ringbuffer(model, opt)
        model(_batches(1)[0]).pow(2).mean().backward()
        self.assertEqual(set(ext.steps[-2:]), {BACKWARDS + 1})

    def test_converted_state_without_a_step_key_falls_back_to_step_count(self):
        """``optimizer_state_convert`` emits no ``step`` for a ring-buffer target;
        BaseTrainer carries ``step_count`` instead."""
        ext = _AdamWMathExtension()
        model = _model()
        opt = _optimizer(model, ext)
        opt.step_count = 137  # what BaseTrainer sets after a cross-impl conversion
        rb.patch_adamw8bit_ringbuffer(model, opt)
        model(_batches(1)[0]).pow(2).mean().backward()
        self.assertEqual(set(ext.steps), {138})


if __name__ == "__main__":
    unittest.main()
