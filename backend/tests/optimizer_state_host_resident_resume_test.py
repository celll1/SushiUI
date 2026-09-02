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

The audit follow-ups live here too: Lion's Schedule-Free keys (``absmax_z`` cast
to the parameter dtype, ``state_z`` uncensused), the placement guard that read as
a refusal without being one, the census whitelist that let a new bulk key pass,
the unreadable-file fresh start, and the ring-buffer (not ``torch.optim.AdamW``)
negative control for the non-host-resident branch.
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
from core.training.optimizers.lion8bit_ringbuffer import Lion8bit_RingBuffer  # noqa: E402
from core.training.optimizers.host_state_allocator import (  # noqa: E402
    HostOptimizerStateAllocator,
    HostStateLoadMismatch,
    HostStateResidencyError,
    assert_state_host_resident,
    copy_containers_only,
    place_loaded_state_tensor,
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
    _remap_optimizer_state_by_group_prefix = (
        BaseTrainer._remap_optimizer_state_by_group_prefix)
    _optimizer_state_entry_fits_param = staticmethod(
        BaseTrainer._optimizer_state_entry_fits_param)
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
        _remap_optimizer_state_by_group_prefix = (
            BaseTrainer._remap_optimizer_state_by_group_prefix)
        _optimizer_state_entry_fits_param = staticmethod(
            BaseTrainer._optimizer_state_entry_fits_param)
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


class _PretendMoved(torch.Tensor):
    """Records ``.to(device)`` and stays put, so a test never allocates on CUDA."""

    moves = []

    def to(self, *args, **kwargs):  # noqa: A003
        if args and isinstance(args[0], (torch.device, str)):
            device = torch.device(args[0])
            self.moved_to = device
            _PretendMoved.moves.append((self.numel(), device))
            return self
        return super().to(*args, **kwargs)

    def __deepcopy__(self, memo):
        # Identity, so what the load path records is recorded on the test's own
        # reference (the non-host-resident branch deepcopies the state dict).
        return self


class _FakeLionRingBuffer(torch.optim.Optimizer):
    """Lion's Schedule-Free state SHAPE with Lion's real load path."""

    _load_state_dict_uint8 = Lion8bit_RingBuffer._load_state_dict_uint8
    load_state_dict = Lion8bit_RingBuffer.load_state_dict

    def __init__(self, params, get_state_buffer=None):
        super().__init__(params, {"lr": 1e-4})
        self.get_state_buffer = get_state_buffer
        # Verbatim from Lion8bit_RingBuffer.__init__ -- absmax_z is NOT in it.
        self.non_castable_tensor_keys = {"exp_avg", "absmax"}
        self.schedule_free = True
        self.use_radam = False
        self.k = 0

    def init_state(self, fill: int = 0):
        for group in self.param_groups:
            for p in group["params"]:
                buffer = self.get_state_buffer(p, dtype=torch.uint8)
                if CUDA and buffer.is_cpu and not buffer.is_pinned():
                    buffer = buffer.pin_memory()
                buffer.fill_(fill)
                blocks = (p.numel() + 255) // 256
                self.state[p]["state_z"] = buffer
                self.state[p]["absmax_z"] = (
                    torch.ones(blocks, dtype=torch.float32).as_subclass(_PretendMoved))
                self.state[p]["is_8bit"] = True


class LionScheduleFreeResumeTest(unittest.TestCase):
    """Lion's Schedule-Free keys are in neither of the hand-maintained sets.

    ``absmax_z`` (FP32, kernel argument) missed ``non_castable_tensor_keys`` and
    was cast to the PARAMETER dtype; ``state_z`` (the whole bulk budget) missed
    the census's key list.
    """

    def _resume(self):
        params = [nn.Parameter(torch.zeros(512, dtype=torch.bfloat16))]
        allocator = HostOptimizerStateAllocator(pin=CUDA)
        optimizer = _FakeLionRingBuffer(params, get_state_buffer=allocator)
        optimizer.init_state(fill=0)
        buffer_ptr = optimizer.state[params[0]]["state_z"].data_ptr()

        previous = _FakeLionRingBuffer(
            [nn.Parameter(torch.zeros(512, dtype=torch.bfloat16))],
            get_state_buffer=HostOptimizerStateAllocator(pin=False))
        previous.init_state(fill=7)
        previous.k = 4211
        saved = previous.state_dict()

        _quiet(optimizer.load_state_dict, saved)
        return optimizer, params[0], buffer_ptr, allocator

    def test_absmax_z_is_not_cast_to_the_parameter_dtype(self):
        optimizer, param, _, _ = self._resume()
        absmax_z = optimizer.state[param]["absmax_z"]
        self.assertEqual(absmax_z.dtype, torch.float32)
        self.assertEqual(getattr(absmax_z, "moved_to", None), torch.device("cuda:0"))

    def test_state_z_stays_in_the_pinned_host_buffer(self):
        optimizer, param, buffer_ptr, allocator = self._resume()
        state_z = optimizer.state[param]["state_z"]
        self.assertEqual(state_z.dtype, torch.uint8)
        self.assertEqual(state_z.data_ptr(), buffer_ptr)
        self.assertTrue(bool((state_z == 7).all()))
        self.assertEqual(allocator.tensors, 1)  # no second host copy

    def test_the_census_sees_the_bulk_key(self):
        optimizer, param, _, _ = self._resume()
        census = state_device_census(optimizer)
        self.assertEqual(census["state_z"]["cuda"], 0)
        self.assertEqual(census["state_z"]["cpu"], 512)
        optimizer.state[param]["state_z"] = (
            torch.zeros(512, dtype=torch.uint8).as_subclass(_PretendCuda))
        with self.assertRaises(HostStateResidencyError) as raised:
            assert_state_host_resident(optimizer)
        self.assertIn("state_z", str(raised.exception))


class CensusFailsClosedTest(unittest.TestCase):
    """A bulk key the census has never heard of must fail, not be skipped."""

    def test_an_unknown_bulk_key_on_cuda_is_a_failure(self):
        param = nn.Parameter(torch.zeros(64))
        optimizer = _FakeRingBuffer(
            [param], get_state_buffer=HostOptimizerStateAllocator(pin=False))
        optimizer.state[param]["fourth_moment"] = (
            torch.zeros(64, dtype=torch.uint8).as_subclass(_PretendCuda))
        with self.assertRaises(HostStateResidencyError) as raised:
            assert_state_host_resident(optimizer)
        self.assertIn("fourth_moment", str(raised.exception))

    def test_absmax_is_still_the_one_allowed_exception(self):
        param = nn.Parameter(torch.zeros(64))
        allocator = HostOptimizerStateAllocator(pin=CUDA)
        optimizer = _FakeRingBuffer([param], get_state_buffer=allocator)
        optimizer.init_state()
        optimizer.state[param]["absmax1"] = (
            torch.zeros(1, dtype=torch.float32).as_subclass(_PretendCuda))
        assert_state_host_resident(optimizer)


class PlacementRefusesAMismatchTest(unittest.TestCase):
    """The guard read as a refusal and was a reroute to ``param.device``."""

    def _optimizer(self, param, allocator):
        optimizer = _FakeRingBuffer([param], get_state_buffer=allocator)
        optimizer.init_state()
        return optimizer

    def test_a_dtype_disagreement_is_refused(self):
        param = nn.Parameter(torch.zeros(64))
        optimizer = self._optimizer(param, HostOptimizerStateAllocator(pin=False))
        fp32_moment = torch.zeros(64, dtype=torch.float32)
        with self.assertRaises(HostStateLoadMismatch) as raised:
            place_loaded_state_tensor(optimizer, param, "exp_avg", fp32_moment)
        message = str(raised.exception)
        self.assertIn("exp_avg", message)
        self.assertIn("torch.float32", message)
        self.assertIn("torch.uint8", message)

    def test_the_resume_aborts_rather_than_starting_fresh(self):
        """Pre-c8307e40 this reached the GPU and OOMed; it must not now reach
        "Continuing with fresh optimizer state" either."""
        params = _params(1)
        optimizer = _FakeRingBuffer(
            params, get_state_buffer=HostOptimizerStateAllocator(pin=CUDA))
        optimizer.init_state()
        saved = _saved_state(_params(1), fill=7)
        for entry in saved["state"].values():
            entry["exp_avg"] = entry["exp_avg"].to(torch.float32)

        trainer = _Trainer()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            with self.assertRaises(HostStateLoadMismatch):
                trainer._load_one_optimizer_state(
                    optimizer, saved, "run121_optimizer.pt")
        self.assertNotIn("fresh optimizer state", printed.getvalue())

    def test_a_numel_mismatch_leaks_no_pinned_buffer(self):
        param = nn.Parameter(torch.zeros(64))
        allocator = HostOptimizerStateAllocator(pin=False)
        optimizer = _FakeRingBuffer([param], get_state_buffer=allocator)  # no state yet
        before = allocator.tensors
        with self.assertRaises(HostStateLoadMismatch):
            place_loaded_state_tensor(
                optimizer, param, "exp_avg", torch.zeros(32, dtype=torch.uint8))
        self.assertEqual(allocator.tensors, before)


class CallersStateDictIsNotMutatedTest(unittest.TestCase):
    """``copy_containers_only`` copies the CONTAINERS; only tensors are shared.

    Returning ``obj`` for a dict would make ``cast`` rewrite the caller's own
    ``state_dict`` in place -- its moments replaced by this optimizer's buffers.
    """

    def test_containers_are_copied_and_tensors_are_not(self):
        tensor = torch.zeros(4)
        source = {"state": {0: {"exp_avg": tensor}}, "param_groups": [{"params": [0]}]}
        copied = copy_containers_only(source)
        self.assertIsNot(copied, source)
        self.assertIsNot(copied["state"][0], source["state"][0])
        self.assertIs(copied["state"][0]["exp_avg"], tensor)

    def test_a_resume_leaves_the_saved_payload_alone(self):
        params = _params(1)
        allocator = HostOptimizerStateAllocator(pin=CUDA)
        optimizer = _FakeRingBuffer(params, get_state_buffer=allocator)
        optimizer.init_state()
        saved = _saved_state(_params(1), fill=5)
        originals = {index: dict(entry) for index, entry in saved["state"].items()}

        trainer = _Trainer()
        _quiet(trainer._load_one_optimizer_state, optimizer, saved, "run121_optimizer.pt")

        for index, entry in saved["state"].items():
            for key, value in originals[index].items():
                self.assertIs(entry[key], value, f"saved['state'][{index}]['{key}']")


class NonHostResidentRingBufferTest(unittest.TestCase):
    """The ``else`` arm: without host residency the state still goes to the device.

    The other negative controls use ``torch.optim.AdamW``, which never reaches
    ``_load_state_dict_uint8`` and so cannot notice the bulk move disappearing.
    """

    def test_the_saved_state_is_offered_to_the_device_then_absmax_to_cuda(self):
        param = nn.Parameter(torch.zeros(64))  # CPU parameter, CUDA trainer
        optimizer = _FakeRingBuffer([param], get_state_buffer=None)
        saved = {
            "state": {0: {
                "exp_avg": torch.full((64,), 9, dtype=torch.uint8).as_subclass(_PretendMoved),
                "absmax1": torch.ones(1).as_subclass(_PretendMoved),
                "is_8bit": True,
            }},
            "param_groups": [{"params": [0], "lr": 1e-4}],
        }
        absmax1 = saved["state"][0]["absmax1"]
        trainer = _Trainer(host_resident=False)
        trainer.device = torch.device("cuda:0")
        _PretendMoved.moves = []
        ok, _ = _quiet(trainer._load_one_optimizer_state, optimizer, saved, "opt.pt")

        self.assertTrue(ok)
        # The FIRST thing that happens to the bulk state is the move to the
        # trainer's device -- not the optimizer's own per-key placement, which
        # would send it to the (CPU) parameter instead.
        self.assertEqual(_PretendMoved.moves[0], (64, trainer.device))
        # absmax still ends up on CUDA, where the kernel indexes it.
        self.assertEqual(getattr(absmax1, "moved_to", None), torch.device("cuda:0"))


class UnreadableOptimizerFileTest(unittest.TestCase):
    """A file that EXISTS and cannot be read is not a fresh start."""

    class _Probe(FromAFileTest._Probe):
        pass

    def _probe(self, tmp, host_resident=True):
        params = _params(1)
        optimizer = _FakeRingBuffer(
            params, get_state_buffer=HostOptimizerStateAllocator(pin=CUDA))
        optimizer.init_state()
        probe = self._Probe(tmp, optimizer)
        probe.optimizer_state_host_resident = host_resident
        return probe

    def _write_garbage(self, tmp):
        path = Path(tmp) / "20260101_000000_abcdef_step_029332_optimizer.pt"
        path.write_bytes(b"not a torch checkpoint")
        return path

    def test_a_host_resident_run_refuses_to_start_fresh(self):
        import tempfile

        from core.training.base_trainer import OptimizerStateFileUnreadable

        with tempfile.TemporaryDirectory() as tmp:
            self._write_garbage(tmp)
            probe = self._probe(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(OptimizerStateFileUnreadable) as raised:
                    probe.load_optimizer_state(29332)
        self.assertIn("could not be read", str(raised.exception))

    def test_an_absent_file_is_still_a_clean_fresh_start(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            probe = self._probe(tmp)
            ok, printed = _quiet(probe.load_optimizer_state, 29332)
        self.assertFalse(ok)
        self.assertIn("fresh optimizer state", printed)

    def test_a_non_host_resident_run_warns_instead_of_only_printing(self):
        import tempfile

        seen = []
        import core.training.base_trainer as bt

        original = bt.emit_training_warning
        bt.emit_training_warning = lambda message, **kwargs: seen.append(
            (message, kwargs.get("code")))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                self._write_garbage(tmp)
                probe = self._probe(tmp, host_resident=False)
                ok, _ = _quiet(probe.load_optimizer_state, 29332)
        finally:
            bt.emit_training_warning = original
        self.assertFalse(ok)
        self.assertEqual([code for _, code in seen], ["optimizer_state_file_unreadable"])


class PinnedHostAllocIsNotAVramProblemTest(unittest.TestCase):
    """``cudaHostAlloc`` failing is host RAM; the remedy must not say VRAM."""

    def test_the_diagnosis_names_host_memory(self):
        from core.training.base_trainer import OptimizerStateLoadHostAllocFailure

        params = _params(1)
        optimizer = CudaOomIsFatalTest._OomOnLoad(
            params,
            RuntimeError("CUDA error: out of memory (cudaHostAlloc at ..\\..\\aten\\src)"),
        )
        trainer = _Trainer(host_resident=True)
        saved = {"state": {}, "param_groups": [{"params": [0], "lr": 1e-4}]}
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OptimizerStateLoadHostAllocFailure) as raised:
                trainer._load_one_optimizer_state(optimizer, saved, "run121_optimizer.pt")
        message = str(raised.exception)
        self.assertIn("host RAM, not VRAM", message)
        self.assertNotIn("Free VRAM", message)

    def test_a_device_oom_still_reads_as_one(self):
        params = _params(1)
        optimizer = CudaOomIsFatalTest._OomOnLoad(
            params, RuntimeError("CUDA out of memory. Tried to allocate 30.19 GiB"))
        trainer = _Trainer(host_resident=False)
        saved = {"state": {}, "param_groups": [{"params": [0], "lr": 1e-4}]}
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(_oom_error_class()) as raised:
                trainer._load_one_optimizer_state(optimizer, saved, "run121_optimizer.pt")
        self.assertIn("Free VRAM", str(raised.exception))


class UnrelatedAssertionKeepsThePartialLoadTest(unittest.TestCase):
    """Only the residency assertion is fatal, not every ``AssertionError``."""

    class _AssertsOnce(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, {"lr": 1e-4})
            self.get_state_buffer = None
            self.calls = 0

        def load_state_dict(self, state_dict):
            self.calls += 1
            if self.calls == 1:
                raise AssertionError("some third-party invariant")
            torch.optim.Optimizer.load_state_dict(self, state_dict)

    def test_a_third_party_assertion_falls_through_to_the_partial_load(self):
        model = _params(2, numel=8)
        previous = torch.optim.AdamW(
            [{"params": [model[0]], "lr": 1e-4}, {"params": [model[1]], "lr": 1e-4}])
        for p in model:
            p.grad = torch.ones_like(p)
        previous.step()
        saved = previous.state_dict()

        live = self._AssertsOnce(
            [{"params": [model[0]], "lr": 1e-4}, {"params": [model[1]], "lr": 1e-4}])
        trainer = _Trainer(host_resident=False)
        ok, printed = _quiet(trainer._load_one_optimizer_state, live, saved, "opt.pt")
        self.assertTrue(ok)
        self.assertIn("Partial optimizer state load OK", printed)
        self.assertEqual(live.calls, 2)


class AdamWNonScheduleFreeIsUnchangedTest(unittest.TestCase):
    """The live configuration: adamw8bit_ringbuffer, 8-bit, NOT Schedule-Free,
    host-resident, fused backward.

    Every helper the audit fixes touch is shared with it, so each fix is checked
    here against the rule it replaced -- differentially, on the keys that
    configuration actually writes.
    """

    LIVE_KEYS = ("exp_avg", "exp_avg_sq", "absmax1", "absmax2")

    @staticmethod
    def _legacy_placement(optimizer, param, key, tensor):
        """``place_loaded_state_tensor`` as of 297f0a69."""
        if key.startswith("absmax"):
            device = (param.device if param.device.type == "cuda"
                      else torch.device("cuda:0"))
            return tensor.to(device)
        get_buffer = getattr(optimizer, "get_state_buffer", None)
        if get_buffer is None or tensor.dtype != torch.uint8:
            return tensor.to(param.device)
        existing = optimizer.state.get(param)
        buffer = existing.get(key) if isinstance(existing, dict) else None
        if not (isinstance(buffer, torch.Tensor)
                and buffer.dtype == tensor.dtype
                and buffer.numel() == tensor.numel()):
            buffer = get_buffer(param, dtype=tensor.dtype)
            if buffer.is_cpu and not buffer.is_pinned():
                buffer = buffer.pin_memory()
        buffer.copy_(tensor.reshape(buffer.shape))
        return buffer

    @staticmethod
    def _legacy_census_problems(optimizer):
        """``assert_state_host_resident``'s whitelist as of c8307e40."""
        census = state_device_census(optimizer)
        problems = []
        for key in ("exp_avg", "exp_avg_sq", "z"):
            bucket = census.get(key)
            if bucket is None:
                continue
            if bucket["cuda"]:
                problems.append(key)
            if bucket["cpu"] and bucket["cpu_pinned"] != bucket["cpu"]:
                problems.append(key)
        return problems

    def _live_optimizer(self):
        param = nn.Parameter(torch.zeros(512))
        allocator = HostOptimizerStateAllocator(pin=CUDA)
        optimizer = _FakeRingBuffer([param], get_state_buffer=allocator)
        optimizer.init_state()
        optimizer.state[param]["absmax1"] = (
            torch.ones(2, dtype=torch.float32).as_subclass(_PretendCuda))
        optimizer.state[param]["absmax2"] = (
            torch.ones(2, dtype=torch.float32).as_subclass(_PretendCuda))
        return optimizer, param, allocator

    def test_placement_decides_identically_for_every_live_key(self):
        for key in self.LIVE_KEYS:
            with self.subTest(key=key):
                loaded = (torch.full((512,), 5, dtype=torch.uint8)
                          if key.startswith("exp_avg")
                          else torch.full((2,), 0.5).as_subclass(_PretendMoved))
                new_opt, new_param, _ = self._live_optimizer()
                old_opt, old_param, _ = self._live_optimizer()
                new = place_loaded_state_tensor(new_opt, new_param, key, loaded)
                old = self._legacy_placement(old_opt, old_param, key, loaded)
                self.assertEqual(new.dtype, old.dtype)
                self.assertEqual(new.numel(), old.numel())
                self.assertTrue(bool((new.float() == old.float()).all()))
                # ... and the buffer it landed in is the optimizer's own, both ways.
                if key.startswith("exp_avg"):
                    self.assertEqual(new.data_ptr(),
                                     new_opt.state[new_param][key].data_ptr())
                    self.assertEqual(old.data_ptr(),
                                     old_opt.state[old_param][key].data_ptr())

    def test_the_census_verdict_is_identical_healthy_and_broken(self):
        optimizer, param, _ = self._live_optimizer()
        self.assertEqual(self._legacy_census_problems(optimizer), [])
        assert_state_host_resident(optimizer)  # new rule agrees: no problem

        optimizer.state[param]["exp_avg"] = (
            torch.zeros(512, dtype=torch.uint8).as_subclass(_PretendCuda))
        self.assertEqual(self._legacy_census_problems(optimizer), ["exp_avg"])
        with self.assertRaises(HostStateResidencyError):
            assert_state_host_resident(optimizer)

    def test_the_routing_predicate_gains_nothing_for_these_keys(self):
        import inspect

        from core.training.optimizers.host_state_allocator import is_absmax_key

        source = inspect.getsource(AdamW8bit_RingBuffer.__init__)
        for key in self.LIVE_KEYS:
            with self.subTest(key=key):
                # Still spelled out in non_castable_tensor_keys, so the added
                # ``or is_absmax_key(k)`` never decides anything for this run.
                self.assertIn(f'"{key}"', source)
                self.assertTrue(is_absmax_key(key) or key.startswith("exp_avg"))

    def test_a_full_resume_still_lands_in_the_allocator_buffers(self):
        params = _params(1)
        allocator = HostOptimizerStateAllocator(pin=CUDA)
        optimizer = _FakeRingBuffer(params, get_state_buffer=allocator)
        optimizer.init_state()
        pointers = {key: optimizer.state[params[0]][key].data_ptr()
                    for key in ("exp_avg", "exp_avg_sq")}
        trainer = _Trainer()
        ok, _ = _quiet(trainer._load_one_optimizer_state,
                       optimizer, _saved_state(_params(1), fill=7),
                       "run121_optimizer.pt")
        self.assertTrue(ok)
        self.assertEqual(allocator.tensors, 2)
        for key, pointer in pointers.items():
            loaded = optimizer.state[params[0]][key]
            self.assertEqual(loaded.data_ptr(), pointer)
            self.assertEqual(loaded.dtype, torch.uint8)
            self.assertTrue(bool((loaded == 7).all()))


if __name__ == "__main__":
    unittest.main()
