"""``step()``'s silent CPU-skip in the ring-buffer optimizers.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/ringbuffer_cpu_skip_failloud_test.py -v

THE DEFECT
----------
Both optimizers' ``step()`` began each parameter with

    # Skip parameters on CPU (offloaded by Block Swap)
    # Optimizer updates will be applied when layer returns to GPU
    if not p.is_cuda:
        continue

``step()`` keeps no record of what it skipped and never revisits it, so the
comment's promise is not kept by anything: a parameter that is CPU-resident when
step() runs is never updated, on that step or any later one, while the loss
falls normally. It is the same shape as the fused-hook skip 3a7c9560 already
made loud, on the other update path.

The 8-bit update IS a CUDA kernel, so a CPU parameter genuinely cannot be
updated there -- that case raises. The unquantized (``use_8bit=False``) path is
plain torch and updates a CPU parameter correctly, so it is no longer skipped
either; it now runs.

NEGATIVE CONTROL
----------------
``ShippedSkipBehaviourTest`` reproduces the old branch on a copy of the loop and
records what it did: the parameter is returned unchanged, with no error and no
optimizer state.
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

SOURCES = {
    "adamw8bit_ringbuffer": BACKEND_ROOT / "core/training/optimizers/adamw8bit_ringbuffer.py",
    "lion8bit_ringbuffer": BACKEND_ROOT / "core/training/optimizers/lion8bit_ringbuffer.py",
}


class NoSilentSkipRemainsTest(unittest.TestCase):
    """The old branch must not come back on either path."""

    def test_no_bare_continue_on_a_cpu_parameter(self):
        for name, path in SOURCES.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(optimizer=name):
                self.assertNotIn("Skip parameters on CPU", text)
                # The exact two-line shape the defect had.
                self.assertNotIn("if not p.is_cuda:\n                    continue", text)

    def test_both_paths_raise_rather_than_skip(self):
        for name, path in SOURCES.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(optimizer=name):
                # hook path (3a7c9560) and step() path (this change)
                self.assertIn("fused-backward hook fired for a parameter", text)
                self.assertIn("step() reached a parameter", text)


class StepRaisesOnCpuEightBitTest(unittest.TestCase):
    """The 8-bit branch cannot update a CPU parameter, and says so."""

    def _cpu_param(self):
        p = nn.Parameter(torch.randn(256, dtype=torch.float32))
        p.grad = torch.randn_like(p)
        return p

    def test_adamw_raises(self):
        p = self._cpu_param()
        optimizer = AdamW8bit_RingBuffer([p], lr=1e-5, use_8bit=True)
        with self.assertRaises(RuntimeError) as ctx:
            optimizer.step()
        message = str(ctx.exception)
        self.assertIn("untrained for the whole run", message)
        self.assertIn("patch_adamw8bit_ringbuffer", message)
        self.assertIn("blocks_to_swap=0", message)

    def test_lion_raises(self):
        p = self._cpu_param()
        optimizer = Lion8bit_RingBuffer([p], lr=1e-5, use_8bit=True)
        with self.assertRaises(RuntimeError) as ctx:
            optimizer.step()
        message = str(ctx.exception)
        self.assertIn("untrained for the whole run", message)
        self.assertIn("register_lion8bit_fused_backward", message)

    def test_the_parameter_is_reported_by_shape(self):
        p = nn.Parameter(torch.randn(4, 8))
        p.grad = torch.randn_like(p)
        optimizer = AdamW8bit_RingBuffer([p], lr=1e-5, use_8bit=True)
        with self.assertRaises(RuntimeError) as ctx:
            optimizer.step()
        self.assertIn("(4, 8)", str(ctx.exception))


class CpuUnquantizedIsUpdatedNotSkippedTest(unittest.TestCase):
    """``use_8bit=False`` on CPU is plain torch: it now runs instead of vanishing."""

    def test_adamw_moves_a_cpu_parameter(self):
        p = nn.Parameter(torch.full((64,), 0.5))
        p.grad = torch.ones_like(p)
        before = p.detach().clone()
        optimizer = AdamW8bit_RingBuffer([p], lr=1e-2, use_8bit=False, weight_decay=0.0)
        optimizer.step()
        self.assertFalse(torch.equal(p.detach(), before))

    def test_lion_moves_a_cpu_parameter(self):
        p = nn.Parameter(torch.full((64,), 0.5))
        p.grad = torch.ones_like(p)
        before = p.detach().clone()
        optimizer = Lion8bit_RingBuffer([p], lr=1e-2, use_8bit=False, weight_decay=0.0)
        optimizer.step()
        self.assertFalse(torch.equal(p.detach(), before))


class ShippedSkipBehaviourTest(unittest.TestCase):
    """Negative control: what the removed branch did."""

    def test_the_old_branch_returned_the_parameter_untouched(self):
        p = nn.Parameter(torch.full((64,), 0.5))
        p.grad = torch.ones_like(p)
        before = p.detach().clone()

        optimizer = AdamW8bit_RingBuffer([p], lr=1e-2, use_8bit=True)
        # The removed two lines, in isolation.
        skipped = []
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if not param.is_cuda:
                    skipped.append(param)
                    continue

        self.assertEqual(len(skipped), 1)
        self.assertTrue(torch.equal(p.detach(), before))  # never moved
        self.assertEqual(len(optimizer.state[p]), 0)      # no state, no trace
        # And nothing raised: a whole run of this is indistinguishable from a
        # working one until the weights are compared.


if __name__ == "__main__":
    unittest.main()
