"""Resuming a run whose 8-bit optimizer state is pinned on the HOST.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/optimizer_state_host_resident_resume_test.py -v

THE DEFECT (run 121, SenseNova both-branch full FT, 16.21B params, 48 GB card)
-----------------------------------------------------------------------------
``_load_one_optimizer_state`` moved EVERY tensor of the saved state to CUDA --
its stated reason (absmax* are GPU-only) covers 0.47 GiB of it, not the 30.19
GiB of quantized moments that ``optimizer_state_host_resident`` deliberately
keeps pinned on the host. With the weights already resident that OOMs, and the
OOM was then CAUGHT and reported as "the optimizer type or trainable parameters
changed", so the run continued on a dead CUDA context and died later at a
few-KB allocation with its weights unsaveable.

WHAT IS PINNED HERE
-------------------
* a host-resident resume leaves the bulk state in the allocator's pinned buffers
  (asserted on the tensors' identity/device, not on a flag),
* it does not allocate a second host copy,
* ``absmax*`` is the only key allowed onto CUDA,
* a CUDA OOM during the load RAISES instead of falling back to fresh state,
* the legitimate param-group-count partial load still works and stays non-fatal,
* the residency census re-runs AFTER the load, and fails when the loaded state
  came back on CUDA.

No CUDA is required: "on the device" is exercised with ``device=cpu`` (where the
pre-fix ``.to(device)`` is a no-op that still REPLACES the allocator's buffer
with the tensor read off disk, which is exactly what this asserts against) plus
a tensor that reports ``is_cuda`` for the census.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.optimizers.adamw8bit_ringbuffer import AdamW8bit_RingBuffer  # noqa: E402
from core.training.optimizers.host_state_allocator import (  # noqa: E402
    HostOptimizerStateAllocator,
    state_device_census,
)

CUDA = torch.cuda.is_available()

OOM_TYPE = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def _oom_error_class():
    from core.training.base_trainer import OptimizerStateLoadOOM
    return OptimizerStateLoadOOM


class _PretendCuda(torch.Tensor):
    """A host tensor that the census must count as GPU-resident."""

    @property
    def is_cuda(self):
        return True


class _FakeRingBuffer(torch.optim.Optimizer):
    """A ring-buffer optimizer's state SHAPE, without its CUDA kernels.

    The load path under test is the real one: ``_load_state_dict_uint8`` and
    ``load_state_dict`` are bound straight off ``AdamW8bit_RingBuffer``.
    """

    _load_state_dict_uint8 = AdamW8bit_RingBuffer._load_state_dict_uint8
    load_state_dict = AdamW8bit_RingBuffer.load_state_dict
    _repair_degenerate_schedule_free_state = (
        AdamW8bit_RingBuffer._repair_degenerate_schedule_free_state)

    def __init__(self, params, get_state_buffer=None):
        super().__init__(params, {"lr": 1e-4})
        self.get_state_buffer = get_state_buffer
        self.non_castable_tensor_keys = {
            "exp_avg", "exp_avg_sq", "absmax1", "absmax2", "z", "absmax_z"}
        self.schedule_free = False
        self.use_radam = False
        self.step_count = 0

    def state_dict(self):
        # AdamW8bit_RingBuffer.state_dict() uses a zero-arg super() that cannot
        # be borrowed by another class; its one addition is step_count.
        state_dict = torch.optim.Optimizer.state_dict(self)
        state_dict["step_count"] = self.step_count
        return state_dict

    def init_state(self, fill: int = 0):
        """What ``_init_param_state`` does in host-resident mode (minus absmax,
        which needs a CUDA allocation)."""
        for group in self.param_groups:
            for p in group["params"]:
                for key in ("exp_avg", "exp_avg_sq"):
                    buffer = self.get_state_buffer(p, dtype=torch.uint8)
                    if CUDA and buffer.is_cpu and not buffer.is_pinned():
                        buffer = buffer.pin_memory()
                    buffer.fill_(fill)
                    self.state[p][key] = buffer
                self.state[p]["is_8bit"] = True
                self.state[p]["step"] = 0


class _Trainer:
    """Exactly the surface ``_load_one_optimizer_state`` reads."""

    _load_one_optimizer_state = BaseTrainer._load_one_optimizer_state
    log_prefix = "[test]"

    def __init__(self, host_resident=True):
        self.device = torch.device("cpu")
        self.optimizer_state_host_resident = host_resident


def _params(n=2, numel=512):
    return [nn.Parameter(torch.zeros(numel)) for _ in range(n)]


def _saved_state(params, fill):
    """The payload a previous host-resident run wrote out."""
    allocator = HostOptimizerStateAllocator(pin=False)
    previous = _FakeRingBuffer(params, get_state_buffer=allocator)
    previous.init_state(fill=fill)
    previous.step_count = 29332
    return previous.state_dict()


def _quiet(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


class HostResidentResumeTest(unittest.TestCase):
    def setUp(self):
        self.params = _params()
        self.allocator = HostOptimizerStateAllocator(pin=CUDA)
        self.optimizer = _FakeRingBuffer(
            self.params, get_state_buffer=self.allocator)
        self.optimizer.init_state(fill=0)
        self.buffers = {
            id(p): dict(self.optimizer.state[p]) for p in self.params}
        self.saved = _saved_state(_params(), fill=7)

    def test_bulk_state_lands_in_the_existing_pinned_host_buffers(self):
        trainer = _Trainer()
        ok, _ = _quiet(trainer._load_one_optimizer_state,
                       self.optimizer, self.saved, "run121_optimizer.pt")
        self.assertTrue(ok)
        for p in self.params:
            for key in ("exp_avg", "exp_avg_sq"):
                loaded = self.optimizer.state[p][key]
                self.assertFalse(loaded.is_cuda)
                self.assertEqual(loaded.dtype, torch.uint8)
                # The allocator's own buffer, not a tensor read off disk.
                self.assertEqual(loaded.data_ptr(),
                                 self.buffers[id(p)][key].data_ptr())
                # ... carrying the checkpoint's values.
                self.assertTrue(bool((loaded == 7).all()))
                if CUDA:
                    self.assertTrue(loaded.is_pinned())

    def test_no_second_host_allocation_is_made(self):
        before = self.allocator.summary()
        trainer = _Trainer()
        _quiet(trainer._load_one_optimizer_state,
               self.optimizer, self.saved, "run121_optimizer.pt")
        self.assertEqual(self.allocator.summary(), before)
        # Not merely "the allocator was not called again": the loaded values are
        # IN the buffers it handed out, so no other host copy exists either.
        owned = {self.buffers[id(p)][key].data_ptr()
                 for p in self.params for key in ("exp_avg", "exp_avg_sq")}
        live = {self.optimizer.state[p][key].data_ptr()
                for p in self.params for key in ("exp_avg", "exp_avg_sq")}
        self.assertEqual(live, owned)
        census = state_device_census(self.optimizer)
        self.assertEqual(census["exp_avg"]["cuda"], 0)
        self.assertEqual(census["exp_avg"]["cpu"], 2 * 512)

    def test_the_bulk_state_is_never_offered_to_the_device(self):
        """Pre-fix, a 30 GiB ``.to(cuda)`` ran before load_state_dict."""
        seen = []

        class _Watched(torch.Tensor):
            def to(self, *args, **kwargs):  # noqa: A003
                seen.append((self.numel(), args))
                return super().to(*args, **kwargs)

        for entry in self.saved["state"].values():
            for key in ("exp_avg", "exp_avg_sq"):
                entry[key] = entry[key].as_subclass(_Watched)
        trainer = _Trainer()
        _quiet(trainer._load_one_optimizer_state,
               self.optimizer, self.saved, "run121_optimizer.pt")
        self.assertEqual(seen, [])

    def test_step_counters_still_resume(self):
        trainer = _Trainer()
        _quiet(trainer._load_one_optimizer_state,
               self.optimizer, self.saved, "run121_optimizer.pt")
        self.assertEqual(self.optimizer.step_count, 29332)


class AbsmaxPlacementTest(unittest.TestCase):
    """``absmax*`` is the only key the load may put on CUDA."""

    def test_absmax_goes_to_cuda_and_bulk_state_does_not(self):
        from core.training.optimizers.host_state_allocator import (
            place_loaded_state_tensor,
        )

        class _Recording:
            dtype = torch.float32

            def __init__(self):
                self.moved_to = None

            def to(self, device):  # noqa: A003
                self.moved_to = device
                return self

        param = nn.Parameter(torch.zeros(64))
        allocator = HostOptimizerStateAllocator(pin=False)
        optimizer = _FakeRingBuffer([param], get_state_buffer=allocator)
        optimizer.init_state()

        for key in ("absmax1", "absmax2", "absmax_z"):
            recorder = _Recording()
            placed = place_loaded_state_tensor(optimizer, param, key, recorder)
            self.assertIs(placed, recorder)
            self.assertEqual(recorder.moved_to, torch.device("cuda:0"))

        bulk = torch.full((64,), 3, dtype=torch.uint8)
        placed = place_loaded_state_tensor(optimizer, param, "exp_avg", bulk)
        self.assertEqual(placed.data_ptr(),
                         optimizer.state[param]["exp_avg"].data_ptr())
        self.assertTrue(bool((placed == 3).all()))


class CudaOomIsFatalTest(unittest.TestCase):
    """A CUDA OOM must not be reported as an optimizer/parameter change."""

    class _OomOnLoad(torch.optim.Optimizer):
        def __init__(self, params, exc):
            super().__init__(params, {"lr": 1e-4})
            self._exc = exc
            self.get_state_buffer = None

        def load_state_dict(self, state_dict):
            raise self._exc

    def _run(self, exc):
        params = _params(1)
        optimizer = self._OomOnLoad(params, exc)
        trainer = _Trainer(host_resident=True)
        saved = {"state": {}, "param_groups": [{"params": [0], "lr": 1e-4}]}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(_oom_error_class()) as raised:
                trainer._load_one_optimizer_state(
                    optimizer, saved, "run121_optimizer.pt")
        return str(raised.exception), buffer.getvalue()

    def test_accelerator_style_message_is_fatal(self):
        message, printed = self._run(RuntimeError("CUDA error: out of memory"))
        self.assertIn("ran out of memory", message)
        self.assertIn("run121_optimizer.pt", message)
        self.assertNotIn("Continuing with fresh optimizer state", printed)
        self.assertNotIn("optimizer type or trainable", printed)

    def test_torch_oom_type_is_fatal(self):
        exc = OOM_TYPE("CUDA out of memory. Tried to allocate 30.19 GiB")
        message, _ = self._run(exc)
        self.assertIn("must not fall back to fresh state", message)

    def test_a_bulk_move_that_ooms_is_fatal_too(self):
        """The pre-fix code OOMed in ``move_tensors_to_device``, not in load."""

        class _OomOnMove(torch.Tensor):
            def to(self, *args, **kwargs):  # noqa: A003
                raise RuntimeError("CUDA error: out of memory")

        params = _params(1)
        optimizer = torch.optim.AdamW(params, lr=1e-4)  # no host residency
        saved = {
            "state": {0: {"exp_avg": torch.zeros(4).as_subclass(_OomOnMove)}},
            "param_groups": [{"params": [0], "lr": 1e-4}],
        }
        trainer = _Trainer(host_resident=False)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(_oom_error_class()):
                trainer._load_one_optimizer_state(
                    optimizer, saved, "some_optimizer.pt")

    def test_a_genuine_mismatch_is_still_only_a_warning(self):
        """Negative control: non-OOM failures keep their fallback."""
        params = _params(1)
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        # Saved run had two parameters in its single group; this one has one, so
        # not even a prefix is salvageable.
        saved = {"state": {}, "param_groups": [{"params": [0, 1], "lr": 1e-4}]}
        trainer = _Trainer(host_resident=False)
        ok, printed = _quiet(trainer._load_one_optimizer_state,
                             optimizer, saved, "mismatch_optimizer.pt")
        self.assertFalse(ok)
        self.assertIn("fresh optimizer state", printed)


class PartialLoadStillWorksTest(unittest.TestCase):
    """The REPA-projector prefix-preserving partial load is not collateral."""

    def test_added_trailing_group_keeps_the_model_groups(self):
        model = _params(2, numel=8)
        previous = torch.optim.AdamW(
            [{"params": [model[0]], "lr": 1e-4},
             {"params": [model[1]], "lr": 1e-4}])
        for p in model:
            p.grad = torch.ones_like(p)
        previous.step()
        saved = previous.state_dict()
        expected = previous.state[model[0]]["exp_avg"].clone()

        projector = _params(1, numel=8)[0]
        live = torch.optim.AdamW(
            [{"params": [model[0]], "lr": 1e-4},
             {"params": [model[1]], "lr": 1e-4},
             {"params": [projector], "lr": 1e-4}])
        trainer = _Trainer(host_resident=False)
        ok, printed = _quiet(trainer._load_one_optimizer_state,
                             live, saved, "repa_optimizer.pt")
        self.assertTrue(ok)
        self.assertIn("Partial optimizer state load OK", printed)
        self.assertTrue(torch.equal(live.state[model[0]]["exp_avg"], expected))
        self.assertNotIn(projector, live.state)


class ResidencyIsRecheckedAfterLoadTest(unittest.TestCase):
    """Defect 3: the setup-time census predates ``load_state_dict``."""

    class _LoadsOntoCuda(torch.optim.Optimizer):
        def __init__(self, params, allocator):
            super().__init__(params, {"lr": 1e-4})
            self.get_state_buffer = allocator
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]["exp_avg"] = allocator(p, dtype=torch.uint8)

        def load_state_dict(self, state_dict):
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]["exp_avg"] = (
                        torch.zeros(p.numel(), dtype=torch.uint8)
                        .as_subclass(_PretendCuda))

    def test_state_that_came_back_on_cuda_fails_the_resume(self):
        params = _params(1)
        optimizer = self._LoadsOntoCuda(
            params, HostOptimizerStateAllocator(pin=False))
        trainer = _Trainer(host_resident=True)
        saved = {"state": {}, "param_groups": [{"params": [0], "lr": 1e-4}]}
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(AssertionError) as raised:
                trainer._load_one_optimizer_state(
                    optimizer, saved, "run121_optimizer.pt")
        self.assertIn("bytes on CUDA", str(raised.exception))

    def test_a_healthy_resume_reports_the_census(self):
        params = _params(1)
        allocator = HostOptimizerStateAllocator(pin=CUDA)
        optimizer = _FakeRingBuffer(params, get_state_buffer=allocator)
        optimizer.init_state()
        trainer = _Trainer(host_resident=True)
        ok, printed = _quiet(trainer._load_one_optimizer_state,
                             optimizer, _saved_state(_params(1), fill=3),
                             "run121_optimizer.pt")
        self.assertTrue(ok)
        self.assertIn("Post-resume optimizer state census", printed)

    def test_the_check_is_skipped_when_the_run_is_not_host_resident(self):
        params = _params(1)
        optimizer = self._LoadsOntoCuda(
            params, HostOptimizerStateAllocator(pin=False))
        trainer = _Trainer(host_resident=False)
        saved = {"state": {}, "param_groups": [{"params": [0], "lr": 1e-4}]}
        ok, printed = _quiet(trainer._load_one_optimizer_state,
                             optimizer, saved, "run121_optimizer.pt")
        self.assertTrue(ok)
        self.assertNotIn("Post-resume", printed)


class FromAFileTest(unittest.TestCase):
    """End to end from a ``*_optimizer.pt`` written by a host-resident run."""

    class _Probe:
        load_optimizer_state = BaseTrainer.load_optimizer_state
        _load_one_optimizer_state = BaseTrainer._load_one_optimizer_state
        _split_saved_optimizer_states = staticmethod(
            BaseTrainer._split_saved_optimizer_states)
        _optimizer_state_param_count = staticmethod(
            BaseTrainer._optimizer_state_param_count)
        log_prefix = "[test]"

        def __init__(self, tmp_path, optimizer):
            self.device = torch.device("cpu")
            self.output_dir = Path(tmp_path)
            self.run_name = "20260101_000000_abcdef"
            self.optimizer = optimizer
            self.optimizer_state_host_resident = True
            self.fused_optimizer_groups = None
            self.lr_scheduler = None

    def _resume(self, tmp_path):
        payload = _saved_state(_params(1), fill=11)
        payload["_sushi_opt_class"] = "AdamW8bit_RingBuffer"
        path = Path(tmp_path) / "20260101_000000_abcdef_step_029332_optimizer.pt"
        torch.save(payload, path)

        params = _params(1)
        allocator = HostOptimizerStateAllocator(pin=CUDA)
        optimizer = _FakeRingBuffer(params, get_state_buffer=allocator)
        optimizer.init_state()
        buffers = {key: optimizer.state[params[0]][key].data_ptr()
                   for key in ("exp_avg", "exp_avg_sq")}
        probe = self._Probe(tmp_path, optimizer)
        ok, printed = _quiet(probe.load_optimizer_state, 29332)
        return ok, printed, optimizer, params[0], buffers, allocator

    def test_a_host_resident_file_resumes_into_its_own_buffers(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            ok, printed, optimizer, param, buffers, allocator = self._resume(tmp)
        self.assertTrue(ok)
        for key, pointer in buffers.items():
            loaded = optimizer.state[param][key]
            self.assertEqual(loaded.data_ptr(), pointer)
            self.assertTrue(bool((loaded == 11).all()))
        self.assertEqual(allocator.tensors, 2)
        self.assertNotIn("not mappable", printed)


if __name__ == "__main__":
    unittest.main()
