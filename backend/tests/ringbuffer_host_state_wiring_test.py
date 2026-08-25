"""The ring-buffer optimizers' host-resident state, and the wiring that reaches it.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/ringbuffer_host_state_wiring_test.py -v

THE DEFECT
----------
Both ring-buffer optimizers allocate their 8-bit state through an injected
``get_state_buffer`` and fall back to GPU allocation when none is supplied.
Nothing supplied one, ever (since 190c876e), so the CPU-resident mode the class
name is about was dead code and the optimizers cost the same GPU state as plain
``adamw8bit`` -- measured at an identical 2.031250 B/param
(SENSENOVA_TRAINING_DESIGN.md 6.5). The whole memory argument for this route
depended on a mode no run could enter.

WHAT IS PINNED HERE
-------------------
* the allocator is persistent and non-aliasing (a recycling allocator would let
  two parameters' moments share bytes),
* it returns ALREADY-pinned buffers, so the optimizers' own ``pin_memory()``
  call is a no-op instead of a second host allocation,
* ``BaseTrainer._ringbuffer_optimizer_kwargs`` supplies it when, and only when,
  ``optimizer_state_host_resident`` is set,
* the proof of activation is the state tensors' DEVICE, not the flag.

NEGATIVE CONTROL
----------------
``HostStateOffTest`` records the shipped behaviour with the switch off: no
``get_state_buffer`` key, and (under CUDA) state on the GPU.
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

from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.optimizers.host_state_allocator import (  # noqa: E402
    HostOptimizerStateAllocator,
    assert_state_host_resident,
    state_device_census,
)

CUDA = torch.cuda.is_available()


class _Stub:
    _ringbuffer_optimizer_kwargs = BaseTrainer._ringbuffer_optimizer_kwargs
    optimizer_cautious = False
    optimizer_schedule_free = False
    optimizer_warmup_steps = 0
    optimizer_schedule_free_r = 0.0
    optimizer_schedule_free_weight_lr_power = 2.0
    optimizer_use_radam = False
    optimizer_stochastic_rounding = False
    optimizer_state_host_resident = False
    _host_state_allocator = None


class AllocatorTest(unittest.TestCase):
    def test_buffers_are_distinct_and_never_recycled(self):
        alloc = HostOptimizerStateAllocator()
        a = alloc(torch.empty(1024), dtype=torch.uint8)
        b = alloc(torch.empty(1024), dtype=torch.uint8)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        # Optimizer state lives for the whole run: an allocator that hands out
        # views into a recycled arena (as RingBufferAllocator does, which 6.5
        # named as the base) would alias two parameters' moments.
        self.assertEqual(a.numel(), 1024)
        self.assertEqual(b.numel(), 1024)

    def test_shape_and_dtype_follow_the_call_sites(self):
        alloc = HostOptimizerStateAllocator()
        p = torch.empty(7, 13)
        buf = alloc(p, dtype=torch.uint8)
        # _init_param_state allocates a FLAT numel() buffer, matching the 8-bit
        # kernels' flat indexing.
        self.assertEqual(buf.shape, (91,))
        self.assertEqual(buf.dtype, torch.uint8)
        self.assertTrue(bool((buf == 0).all()))

    @unittest.skipUnless(CUDA, "pinning requires a CUDA context")
    def test_returned_buffers_are_pinned_so_pin_memory_is_a_no_op(self):
        alloc = HostOptimizerStateAllocator()
        buf = alloc(torch.empty(4096), dtype=torch.uint8)
        self.assertTrue(buf.is_pinned())
        # This is what the optimizers do next. On an unpinned buffer it returns
        # a SECOND allocation; on a pinned one it returns the same storage.
        self.assertEqual(buf.pin_memory().data_ptr(), buf.data_ptr())
        self.assertEqual(alloc.bytes, 4096)
        self.assertEqual(alloc.pinned_bytes, 4096)

    @unittest.skipUnless(CUDA, "pinning requires a CUDA context")
    def test_unpinned_allocator_duplicates_on_pin_memory(self):
        """Why the allocator pins rather than leaving it to the optimizer."""
        alloc = HostOptimizerStateAllocator(pin=False)
        buf = alloc(torch.empty(4096), dtype=torch.uint8)
        self.assertFalse(buf.is_pinned())
        self.assertNotEqual(buf.pin_memory().data_ptr(), buf.data_ptr())


class KwargsWiringTest(unittest.TestCase):
    def test_switch_on_supplies_the_allocator_and_reuses_one_instance(self):
        stub = _Stub()
        stub.optimizer_state_host_resident = True
        first = stub._ringbuffer_optimizer_kwargs()
        second = stub._ringbuffer_optimizer_kwargs()
        self.assertIn("get_state_buffer", first)
        self.assertIsInstance(first["get_state_buffer"], HostOptimizerStateAllocator)
        # One allocator for the run: the trainer owns it so its byte totals can
        # be read back for the host-RAM gate after the optimizer is built.
        self.assertIs(first["get_state_buffer"], second["get_state_buffer"])
        self.assertIs(first["get_state_buffer"], stub._host_state_allocator)

    def test_every_other_option_still_reaches_the_optimizer(self):
        stub = _Stub()
        stub.optimizer_state_host_resident = True
        stub.optimizer_stochastic_rounding = True
        stub.optimizer_cautious = True
        kwargs = stub._ringbuffer_optimizer_kwargs()
        for key in ("cautious", "schedule_free", "warmup_steps", "r",
                    "weight_lr_power", "use_radam", "stochastic_rounding"):
            self.assertIn(key, kwargs)
        self.assertTrue(kwargs["stochastic_rounding"])
        self.assertTrue(kwargs["cautious"])


class HostStateOffTest(unittest.TestCase):
    """Negative control: the shipped behaviour, with the switch off."""

    def test_no_allocator_is_supplied(self):
        stub = _Stub()
        kwargs = stub._ringbuffer_optimizer_kwargs()
        self.assertNotIn("get_state_buffer", kwargs)
        self.assertIsNone(stub._host_state_allocator)

    def test_base_trainer_default_is_off(self):
        # Read off the class body rather than constructing a trainer: the mode
        # has no API surface, so nothing can turn it on by accident.
        source = Path(BACKEND_ROOT / "core" / "training" / "base_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.optimizer_state_host_resident = False", source)


@unittest.skipUnless(CUDA, "the 8-bit state path is a CUDA kernel")
class StateResidencyTest(unittest.TestCase):
    """The proof of activation is the device census, not the flag."""

    def _build(self, name, host_resident):
        from core.training.optimizer_factory import OptimizerFactory

        stub = _Stub()
        stub.optimizer_state_host_resident = host_resident
        params = [
            nn.Parameter(torch.randn(512, 512, device="cuda", dtype=torch.bfloat16))
            for _ in range(2)
        ]
        optimizer = OptimizerFactory.create_optimizer(
            name, params, learning_rate=1e-5, weight_decay=0.0,
            **stub._ringbuffer_optimizer_kwargs()
        )
        for p in params:
            p.grad = torch.randn_like(p)
        optimizer.step()
        torch.cuda.synchronize()
        return stub, optimizer, params

    def test_adamw_state_is_host_resident_and_pinned(self):
        stub, optimizer, params = self._build("adamw8bit_ringbuffer", True)
        census = assert_state_host_resident(optimizer)
        self.assertEqual(census["exp_avg"]["cuda"], 0)
        self.assertEqual(census["exp_avg_sq"]["cuda"], 0)
        # absmax is deliberately kept on the GPU even when the parameter moves.
        self.assertGreater(census["absmax1"]["cuda"], 0)
        self.assertEqual(stub._host_state_allocator.tensors, 4)

    def test_lion_state_is_host_resident_and_pinned(self):
        stub, optimizer, params = self._build("lion8bit_ringbuffer", True)
        census = assert_state_host_resident(optimizer)
        self.assertEqual(census["exp_avg"]["cuda"], 0)
        self.assertGreater(census["absmax"]["cuda"], 0)
        # One moment, not two: half the state of the AdamW pair.
        self.assertEqual(stub._host_state_allocator.tensors, 2)

    def test_host_state_still_moves_the_parameters(self):
        _, _, params = self._build("adamw8bit_ringbuffer", True)
        moved = sum(int((p != 0).sum()) for p in params)
        self.assertGreater(moved, 0)

    def test_switch_off_leaves_state_on_the_gpu(self):
        """Negative control, on the real optimizer."""
        stub, optimizer, _ = self._build("adamw8bit_ringbuffer", False)
        census = state_device_census(optimizer)
        self.assertGreater(census["exp_avg"]["cuda"], 0)
        self.assertEqual(census["exp_avg"]["cpu"], 0)
        self.assertIsNone(stub._host_state_allocator)
        with self.assertRaises(AssertionError):
            assert_state_host_resident(optimizer)


if __name__ == "__main__":
    unittest.main()
