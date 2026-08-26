"""The stochastic-rounding assertion must not reject the ring-buffer optimizers.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sensenova_ringbuffer_assertion_trap_test.py -v

THE TRAP (registered in SENSENOVA_TRAINING_DESIGN.md 13.4, U-2-6)
-----------------------------------------------------------------
``assert_full_finetune_stochastic_rounding_attached`` verifies the MECHANISM
rather than the flag, by checking that ``optimizer.step_param`` exists and is
interposed. Neither ring-buffer optimizer defines ``step_param``: they register
their own post-accumulate-grad hooks and ``_setup_fused_backward_pass`` returns
early for them. Yet they are the only two optimizers that apply stochastic
rounding inside their own update -- which is precisely why
``_attach_stochastic_rounding`` skips them
(``_NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS``).

Checked against step_param they fail on a CORRECT configuration, and both halves
of the failure message are false for them: there is no step_param to find, and
the updates are not round-to-nearest. So the assertion recognises native
coverage. Without that, whoever widens
``SENSENOVA_FULL_FINETUNE_OPTIMIZERS`` gets a crash that reads like a real
defect.

NEGATIVE CONTROL
----------------
``TrapStillClosesOnEveryoneElseTest`` pins that this did not become a blanket
exemption: an optimizer outside the native list with a missing or unwrapped
step_param is still refused, and the two lists are the same list.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.ops.sensenova_ops import (  # noqa: E402
    SENSENOVA_FULL_FINETUNE_OPTIMIZERS,
    assert_full_finetune_stochastic_rounding_attached,
)


class _BareOptimizer:
    """No step_param, exactly like the ring-buffer optimizers."""


class _Trainer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.fused_optimizer_groups = None
        self.optimizer_stochastic_rounding = True


class NativeRoundingCountsAsCoverageTest(unittest.TestCase):
    def test_ring_buffer_optimizers_are_accepted_without_a_step_param(self):
        trainer = _Trainer(_BareOptimizer())
        for name in BaseTrainer._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS:
            with self.subTest(optimizer=name):
                # Must not raise: the seam that applies the update is the
                # optimizer's own hook, and the rounding is inside it.
                assert_full_finetune_stochastic_rounding_attached(trainer, name)

    def test_the_native_list_is_the_one_attach_uses(self):
        # If these ever diverge, the assertion would exempt an optimizer that
        # _attach_stochastic_rounding still tries (and fails) to wrap, or refuse
        # one it deliberately skips.
        self.assertEqual(
            BaseTrainer._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS,
            ("adamw8bit_ringbuffer", "lion8bit_ringbuffer"),
        )


class TrapStillClosesOnEveryoneElseTest(unittest.TestCase):
    """Negative control: the exemption is exactly two names wide."""

    def test_an_optimizer_without_step_param_is_still_refused(self):
        trainer = _Trainer(_BareOptimizer())
        with self.assertRaises(RuntimeError) as caught:
            assert_full_finetune_stochastic_rounding_attached(trainer, "adafactor")
        self.assertIn("step_param", str(caught.exception))

    def test_an_unwrapped_step_param_is_still_refused(self):
        class Unwrapped:
            def step_param(self, p, group):
                return None

        trainer = _Trainer(Unwrapped())
        with self.assertRaises(RuntimeError) as caught:
            assert_full_finetune_stochastic_rounding_attached(trainer, "adamw8bit")
        self.assertIn("nothing is interposed", str(caught.exception))

    def test_an_unnamed_optimizer_is_still_refused(self):
        """The exemption keys on the name the run was started with."""
        trainer = _Trainer(_BareOptimizer())
        with self.assertRaises(RuntimeError):
            assert_full_finetune_stochastic_rounding_attached(trainer, None)


class AllowlistTest(unittest.TestCase):
    """What the trap was guarding: the ring buffers are admitted now.

    G-RB2/G-RB3 are discharged (U-2-6) and ``optimizer_state_host_resident``
    has a setting, so both were added -- which is exactly the act this file's
    assertion trap would have broken. Pinned so widening it further, or letting
    it drift, is a deliberate act.
    """

    def test_allowlist_is_adafactor_plus_the_ring_buffer_pair(self):
        self.assertEqual(
            SENSENOVA_FULL_FINETUNE_OPTIMIZERS,
            ("adafactor", "adamw8bit_ringbuffer", "lion8bit_ringbuffer"))


if __name__ == "__main__":
    unittest.main()
