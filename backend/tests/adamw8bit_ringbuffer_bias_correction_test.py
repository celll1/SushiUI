"""AdamW8bit_RingBuffer's unquantized path applied bias correction 1 twice.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adamw8bit_ringbuffer_bias_correction_test.py -v

THE DEFECT
----------
``step()``'s ``use_8bit=False`` branch computed

    corrected_exp_avg = exp_avg / bias_correction1     # once
    step_size         = scheduled_lr / bias_correction1 # and again
    p.addcdiv_(corrected_exp_avg, denom, value=-step_size)

so every update was oversized by ``1 / (1 - beta1**t)``: 10.0x at step 1, 2.44x
at step 5, decaying to 1.0 only asymptotically. The loss still falls; the step is
simply the wrong size, which is the failure mode this route keeps producing.

It is isolated to that branch. The 8-bit CUDA kernel the optimizer normally runs
divides by bias_correction1 exactly once (``adamw8bit_kernel.cu:230-241``), so
the two paths of the SAME optimizer disagreed -- which is what settles this as a
bug rather than a deliberate variant. ``Lion8bit_RingBuffer``'s unquantized path
is unaffected (Lion has no bias correction).

Found while routing CPU parameters into this branch (U-2-6's fail-loud change
stopped ``step()`` skipping them). Pre-existing, not a regression of that change.

NEGATIVE CONTROL
----------------
``DoubleCorrectionIsWhatShippedTest`` pins the old arithmetic as a closed form
and shows the current code no longer matches it, so the fix cannot silently
revert.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training.optimizers.adamw8bit_ringbuffer import AdamW8bit_RingBuffer  # noqa: E402
from core.training.optimizers.lion8bit_ringbuffer import Lion8bit_RingBuffer  # noqa: E402

LR = 1e-2
BETAS = (0.9, 0.999)
EPS = 1e-8
STEPS = 5
N = 64


def _p0():
    return torch.randn(N, generator=torch.Generator().manual_seed(7))


def _grads():
    gen = torch.Generator().manual_seed(11)
    return [torch.randn(N, generator=gen) for _ in range(STEPS)]


def _run(optimizer_cls, **kwargs):
    p = nn.Parameter(_p0().clone())
    opt = optimizer_cls([p], lr=LR, weight_decay=0.0, use_8bit=False, **kwargs)
    for g in _grads():
        p.grad = g.clone()
        opt.step()
    return p.detach()


def _adam_closed_form(double_bc1: bool):
    beta1, beta2 = BETAS
    p = _p0().clone()
    m = torch.zeros(N)
    v = torch.zeros(N)
    for t, g in enumerate(_grads(), start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t
        denom = (v / bc2).sqrt() + EPS
        step = (LR / bc1) if double_bc1 else LR
        p = p - step * (m / bc1) / denom
    return p


class MatchesReferenceAdamWTest(unittest.TestCase):
    def test_agrees_with_torch_optim_adamw(self):
        got = _run(AdamW8bit_RingBuffer, betas=BETAS, eps=EPS)
        p = nn.Parameter(_p0().clone())
        ref = torch.optim.AdamW([p], lr=LR, betas=BETAS, eps=EPS, weight_decay=0.0)
        for g in _grads():
            p.grad = g.clone()
            ref.step()
        self.assertLess(float((got - p.detach()).abs().max()), 1e-6)

    def test_agrees_with_the_single_correction_closed_form(self):
        got = _run(AdamW8bit_RingBuffer, betas=BETAS, eps=EPS)
        self.assertLess(float((got - _adam_closed_form(False)).abs().max()), 1e-6)

    def test_lion_unquantized_path_is_unaffected(self):
        """Lion has no bias correction; pinned so the fix stays scoped."""
        beta1, beta2 = 0.9, 0.99
        got = _run(Lion8bit_RingBuffer, betas=(beta1, beta2))
        p = _p0().clone()
        m = torch.zeros(N)
        for g in _grads():
            c = beta1 * m + (1 - beta1) * g
            p = p - LR * torch.sign(c)
            m = beta2 * m + (1 - beta2) * g
        self.assertEqual(float((got - p).abs().max()), 0.0)


class DoubleCorrectionIsWhatShippedTest(unittest.TestCase):
    """Negative control: the arithmetic that used to run."""

    def test_current_code_no_longer_matches_the_double_correction(self):
        got = _run(AdamW8bit_RingBuffer, betas=BETAS, eps=EPS)
        drift = float((got - _adam_closed_form(True)).abs().max())
        # Measured at 1.874e-01 against a parameter scale of 2.97 -- the size of
        # the error the shipped code carried, not a tolerance.
        self.assertGreater(drift, 1e-2)

    def test_the_oversize_factor_is_one_over_bias_correction_one(self):
        beta1 = BETAS[0]
        factors = [1.0 / (1.0 - beta1 ** t) for t in range(1, STEPS + 1)]
        self.assertAlmostEqual(factors[0], 10.0, places=4)
        self.assertAlmostEqual(factors[4], 2.4419, places=4)

    def test_the_cuda_kernel_always_divided_once(self):
        """The source of the disagreement, pinned so it stays the reference."""
        kernel = (BACKEND_ROOT / "core/training/optimizers/cuda/adamw8bit_kernel.cu"
                  ).read_text(encoding="utf-8")
        self.assertIn("float corrected_exp_avg = exp_avg / bias_correction1;", kernel)
        self.assertIn("float update = corrected_exp_avg / denom;", kernel)

    def test_the_python_path_no_longer_divides_step_size(self):
        source = (BACKEND_ROOT / "core/training/optimizers/adamw8bit_ringbuffer.py"
                  ).read_text(encoding="utf-8")
        self.assertNotIn("step_size = scheduled_lr / bias_correction1", source)


if __name__ == "__main__":
    unittest.main()
