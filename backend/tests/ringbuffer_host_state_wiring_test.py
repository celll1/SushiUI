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

import contextlib
import io
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
    _announce_host_state_budget = BaseTrainer._announce_host_state_budget
    _assert_ringbuffer_state_host_resident = \
        BaseTrainer._assert_ringbuffer_state_host_resident
    _RINGBUFFER_HOST_STATE_BYTES_PER_PARAM = \
        BaseTrainer._RINGBUFFER_HOST_STATE_BYTES_PER_PARAM
    log_prefix = "[StubTrainer]"
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
        # Read off __init__ rather than constructing a trainer: what is pinned
        # is that the key is read from train_config and defaults to off, so
        # nothing turns it on by accident. It now ALSO has an API/UI surface
        # (optimizer_diagnostic_switch_config_test.py records why), and the YAML
        # key stays the channel underneath it.
        source = Path(BACKEND_ROOT / "core" / "training" / "base_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('_tc.get("optimizer_state_host_resident", False)', source)

    def test_no_announce_when_the_switch_is_off(self):
        stub = _Stub()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stub._announce_host_state_budget(
                "adamw8bit_ringbuffer", [{"params": [torch.zeros(1024)]}])
        self.assertEqual(buffer.getvalue(), "")


class HostRamAnnounceTest(unittest.TestCase):
    """The pinned budget is stated BEFORE the allocation is taken.

    MUTANT: delete the call in setup_optimizer and a run commits an unpageable
    30.19 GiB (SenseNova both halves, AdamW) with nothing said in advance.
    """

    def _announce(self, name: str, numel: int) -> str:
        stub = _Stub()
        stub.optimizer_state_host_resident = True
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stub._announce_host_state_budget(
                name, [{"params": [torch.zeros(numel)]}])
        return buffer.getvalue()

    def test_adamw_announces_two_bytes_per_parameter(self):
        line = self._announce("adamw8bit_ringbuffer", 1024 ** 3)
        self.assertIn("HOST RAM announce", line)
        self.assertIn("2 B/param", line)
        self.assertIn("2.00 GiB of host memory", line)
        self.assertIn("unpageable", line)

    def test_lion_announces_half_of_that(self):
        line = self._announce("lion8bit_ringbuffer", 1024 ** 3)
        self.assertIn("1 B/param", line)
        self.assertIn("1.00 GiB", line)

    def test_the_current_working_set_is_reported_next_to_it(self):
        line = self._announce("adamw8bit_ringbuffer", 1024)
        self.assertIn("working set", line)

    def test_a_non_ringbuffer_name_announces_nothing(self):
        self.assertEqual(self._announce("adamw8bit", 1024), "")


class StateResidencyAssertionTest(unittest.TestCase):
    """The trainer checks the census, not the flag.

    MUTANT: replace the call with ``if self.optimizer_state_host_resident:
    pass`` and a ``get_state_buffer`` that handed back CUDA tensors leaves the
    flag true and 32.9 GB on the GPU -- the misbudget this route cannot absorb.
    """

    class _FakeOptimizer:
        def __init__(self, buffers):
            self.param = torch.zeros(64)
            self.param_groups = [{"params": [self.param]}]
            self.state = {self.param: {}}
            self._buffers = buffers
            self.inits = 0

        def _init_param_state(self, p):
            self.state[p] = {"exp_avg": self._buffers(p)}
            self.inits += 1

    def _trainer(self, buffers):
        stub = _Stub()
        stub.optimizer = self._FakeOptimizer(buffers)
        return stub

    def test_lazy_state_is_forced_so_the_census_is_not_vacuous(self):
        # _init_param_state runs on the first BACKWARD, so an unforced census
        # would inspect an empty state dict and pass on any configuration.
        stub = self._trainer(
            lambda p: torch.zeros(p.numel(), dtype=torch.uint8))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(AssertionError):
                stub._assert_ringbuffer_state_host_resident("adamw8bit_ringbuffer")
        self.assertEqual(stub.optimizer.inits, 1)

    @unittest.skipUnless(CUDA, "the failing case allocates a CUDA tensor")
    def test_cuda_resident_state_fails_loudly(self):
        stub = self._trainer(
            lambda p: torch.zeros(p.numel(), dtype=torch.uint8, device="cuda"))
        with self.assertRaises(AssertionError) as raised:
            stub._assert_ringbuffer_state_host_resident("adamw8bit_ringbuffer")
        self.assertIn("bytes on CUDA", str(raised.exception))

    @unittest.skipUnless(CUDA, "pinning requires a CUDA context")
    def test_pinned_host_state_passes_and_is_reported(self):
        allocator = HostOptimizerStateAllocator()
        stub = self._trainer(lambda p: allocator(p, dtype=torch.uint8))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stub._assert_ringbuffer_state_host_resident("adamw8bit_ringbuffer")
        self.assertIn("state census", buffer.getvalue())

    def test_a_non_ringbuffer_optimizer_is_not_touched(self):
        stub = self._trainer(lambda p: torch.zeros(p.numel(), dtype=torch.uint8))
        stub._assert_ringbuffer_state_host_resident("adafactor")
        self.assertEqual(stub.optimizer.inits, 0)


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
