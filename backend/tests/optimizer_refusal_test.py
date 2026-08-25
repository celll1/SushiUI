"""Guard: optimizer configurations that cannot work must be refused, not attempted.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/optimizer_refusal_test.py -v

Two refusals live here:

* **Block Swap + an optimizer with no per-parameter update** -- described below.
* **lion8bit_ringbuffer + optimizer_schedule_free** -- its Schedule-Free CUDA
  kernel writes Lion's momentum EMA into the parameter instead of the
  Schedule-Free position sequence, replacing the weights with the momentum
  buffer within a few steps (corr(p, z) = 0.994 by step 5). See
  ``schedulefree_z_sequence_test.LionScheduleFreeRefusalTest`` for the
  measurement and for the pin that keeps this refusal honest.

THE BLOCK SWAP DEFECT
---------------------
With ``blocks_to_swap > 0`` and ``num_optimizer_groups = 0``, ``setup_optimizer``
installed a fused backward pass for ``adafactor``, ``adamw8bit`` and the two
ring-buffer optimizers, and did nothing at all for anything else. ``lion8bit``
was in neither list, so that configuration set up neither fused path: the plain
``optimizer.step()`` then ran at the end of the step, by which time Block Swap's
backward hook has moved every swapped block back to the CPU
(``LayerOffloadConductor.offload_layer_to_cpu`` -> ``layer.to('cpu')``, which
moves ``.grad`` with the parameter).

Every bitsandbytes optimizer refuses a CPU-resident parameter -- see
``BitsandbytesCpuParameterTest``, which measures it rather than assuming it. So
the configuration is not "slower" or "silently skipping parameters": it raises,
inside bitsandbytes' own error formatting, with a message
("AttributeError: 'NoneType' object has no attribute 'shape'") that names
neither Block Swap nor the optimizer.

The same hole existed for ``paged_adamw``, ``paged_adamw8bit`` and
``paged_lion8bit`` -- paging moves the optimizer STATE to host memory and does
nothing about the parameter, exactly the point 30f088fb made when it added those
names to the fused-optimizer-groups refusal.

WHY A REFUSAL AND NOT A FUSED PATH
----------------------------------
``patch_adamw8bit_fused`` gives ``adamw8bit`` a per-parameter ``step_param``
that is a plain-Python AdamW with state allocated by ``zeros_like(p)`` -- i.e.
it silently stops being an 8-bit optimizer and doubles its state to the
parameter dtype. Writing the Lion equivalent would silently do the same, while
``lion8bit_ringbuffer`` already implements a REAL 8-bit fused-backward path
(``register_lion8bit_fused_backward``) that keeps the 8-bit state and applies
stochastic rounding. So the remedy is to name that, not to add a second,
quietly-different Lion.
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

LR = 1e-5


class BitsandbytesCpuParameterTest(unittest.TestCase):
    """The premise of the refusal, measured against the installed bitsandbytes.

    If a future bitsandbytes gains a CPU path, this fails and the refusal below
    can be revisited -- instead of the refusal quietly outliving its reason.
    """

    NAMES = ("Lion8bit", "AdamW8bit", "PagedAdamW", "PagedAdamW8bit", "PagedLion8bit")

    def test_every_bitsandbytes_optimizer_refuses_a_cpu_parameter(self):
        bnb = pytest_importorskip()
        for name in self.NAMES:
            with self.subTest(optimizer=name):
                p = torch.nn.Parameter(torch.randn(256, dtype=torch.bfloat16))  # CPU
                p.grad = torch.randn_like(p) * 1e-3
                optimizer = getattr(bnb.optim, name)([p], lr=LR)
                with self.assertRaises(Exception):
                    optimizer.step()

    def test_torch_adamw_does_update_a_cpu_parameter(self):
        """Why plain ``adamw`` is not in the refusal list."""
        p = torch.nn.Parameter(torch.randn(256, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p) * 1e-3
        before = p.detach().clone()
        torch.optim.AdamW([p], lr=1e-2).step()
        self.assertFalse(torch.equal(before, p.detach()))


def pytest_importorskip():
    try:
        import bitsandbytes as bnb
    except ImportError:  # pragma: no cover - bitsandbytes is a hard dep of these optimizers
        raise unittest.SkipTest("bitsandbytes is not installed")
    return bnb


class _StubModule(torch.nn.Module):
    def __init__(self, param: torch.nn.Parameter):
        super().__init__()
        self.weight = param


class _StubTrainer:
    """The smallest object ``BaseTrainer.setup_optimizer`` can run against.

    Mirrors the stub in ``bf16_stochastic_rounding_default_optimizer_test``: real
    methods, no model, no dataset.
    """

    setup_optimizer = BaseTrainer.setup_optimizer
    _report_effective_component_lrs = BaseTrainer._report_effective_component_lrs
    _build_component_lr_list = BaseTrainer._build_component_lr_list
    _resolved_optimizer_hyperparameters = BaseTrainer._resolved_optimizer_hyperparameters
    _ringbuffer_optimizer_kwargs = BaseTrainer._ringbuffer_optimizer_kwargs
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.param = torch.nn.Parameter(
            (torch.randn(256, device=device) * 0.02).to(torch.bfloat16)
        )
        self.transformer = _StubModule(self.param)
        self.unet = None

    def setup_trainable_parameters(self):
        return [{"params": [self.param], "lr": self.learning_rate}]

    def _setup_ema(self):
        pass


class BlockSwapRefusalTest(unittest.TestCase):
    """Block Swap + an optimizer with no per-parameter update must be refused."""

    REFUSED = ("lion8bit", "paged_adamw", "paged_adamw8bit", "paged_lion8bit")

    def _setup(self, optimizer_type: str, blocks_to_swap: int):
        trainer = _StubTrainer(blocks_to_swap=blocks_to_swap)
        trainer.setup_optimizer(optimizer_type=optimizer_type, total_steps=10)
        return trainer

    def test_each_unsupported_optimizer_is_refused_under_block_swap(self):
        for name in self.REFUSED:
            with self.subTest(optimizer=name):
                with self.assertRaises(ValueError) as ctx:
                    self._setup(name, blocks_to_swap=8)
                message = str(ctx.exception)
                self.assertIn(name, message)
                self.assertIn("Block Swap", message)
                # The message has to name a way out, like the other refusals do.
                self.assertIn("ringbuffer", message)
                self.assertIn("blocks_to_swap=0", message)

    def test_the_same_optimizers_are_accepted_without_block_swap(self):
        """The refusal is about Block Swap, not about the optimizer."""
        for name in self.REFUSED:
            with self.subTest(optimizer=name):
                trainer = self._setup(name, blocks_to_swap=0)
                self.assertIsNotNone(trainer.optimizer)

    def test_adamw8bit_still_gets_its_fused_backward_pass(self):
        """It escapes the list only because a per-parameter step_param is installed."""
        trainer = self._setup("adamw8bit", blocks_to_swap=8)
        self.assertTrue(callable(getattr(trainer.optimizer, "step_param", None)))
        self.assertNotIn("adamw8bit", BaseTrainer._BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS)

    def test_adafactor_still_gets_its_fused_backward_pass(self):
        trainer = self._setup("adafactor", blocks_to_swap=8)
        self.assertTrue(callable(getattr(trainer.optimizer, "step_param", None)))

    def test_plain_adamw_is_not_refused(self):
        """torch's AdamW updates CPU parameters correctly, so it needs no fused path."""
        trainer = self._setup("adamw", blocks_to_swap=8)
        self.assertIsNotNone(trainer.optimizer)

    @unittest.skipUnless(torch.cuda.is_available(), "ring-buffer optimizers need CUDA")
    def test_the_ring_buffer_optimizers_are_not_refused(self):
        """They register their own per-parameter hooks -- the remedy the message names."""
        for name in ("adamw8bit_ringbuffer", "lion8bit_ringbuffer"):
            with self.subTest(optimizer=name):
                trainer = self._setup(name, blocks_to_swap=8)
                self.assertTrue(getattr(trainer, "use_fused_backward", False))

    def test_the_refusal_list_and_the_fused_list_do_not_overlap(self):
        """An optimizer in both would be refused after its fused path was installed."""
        fused = {"adafactor", "adamw8bit", "adamw8bit_ringbuffer", "lion8bit_ringbuffer"}
        self.assertFalse(fused & set(BaseTrainer._BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS))

    def test_every_factory_optimizer_is_classified(self):
        """A new optimizer name must land in the fused list or the refusal list."""
        from core.training.optimizer_factory import OptimizerFactory

        fused = {"adafactor", "adamw8bit", "adamw8bit_ringbuffer", "lion8bit_ringbuffer"}
        # 'adamw' is torch's own, which handles CPU parameters.
        classified = fused | set(BaseTrainer._BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS) | {"adamw"}
        unclassified = set(OptimizerFactory.get_available_optimizers()) - classified
        self.assertEqual(
            unclassified, set(),
            "these optimizers reach Block Swap with neither a fused path nor a refusal",
        )


class ScheduleFreeLionRefusalTest(unittest.TestCase):
    """lion8bit_ringbuffer + optimizer_schedule_free, refused at the config layer.

    The constructor refuses too (with RuntimeError, so BaseTrainer's
    ``except (ValueError, ImportError)`` fallback cannot swallow it), but the
    check has to also exist HERE, before the factory call, so the user gets a
    message naming the alternative instead of a raw optimizer error.
    """

    def test_the_trainer_refuses_the_configuration(self):
        trainer = _StubTrainer(optimizer_schedule_free=True)
        with self.assertRaises(ValueError) as ctx:
            trainer.setup_optimizer(optimizer_type="lion8bit_ringbuffer", total_steps=10)
        message = str(ctx.exception)
        self.assertIn("optimizer_schedule_free", message)
        self.assertIn("momentum", message)
        self.assertIn("adamw8bit_ringbuffer", message)

    def test_plain_lion_ringbuffer_is_not_refused(self):
        if not torch.cuda.is_available():
            self.skipTest("ring-buffer optimizers need CUDA")
        trainer = _StubTrainer(optimizer_schedule_free=False)
        trainer.setup_optimizer(optimizer_type="lion8bit_ringbuffer", total_steps=10)
        self.assertIsNotNone(trainer.optimizer)

    def test_adamw_ringbuffer_schedule_free_is_still_allowed(self):
        """The refusal is about Lion, not about Schedule-Free."""
        if not torch.cuda.is_available():
            self.skipTest("ring-buffer optimizers need CUDA")
        trainer = _StubTrainer(optimizer_schedule_free=True)
        trainer.setup_optimizer(optimizer_type="adamw8bit_ringbuffer", total_steps=10)
        self.assertTrue(trainer.optimizer.param_groups[0]["schedule_free"])


if __name__ == "__main__":
    unittest.main()
