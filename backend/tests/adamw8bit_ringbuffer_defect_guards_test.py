"""Guards for three defects that neighboured the fused-hook step counter (4adb0359).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adamw8bit_ringbuffer_defect_guards_test.py -v

(1) ``step_count`` was not serialized, so ``step()`` -- the path taken whenever
    Block Swap is off -- restarted Adam's bias correction at step 1 on every
    ordinary resume.

(2) The fused hook reads ``exp_avg`` / ``absmax1``, which ``_init_param_state``
    does not allocate under ``schedule_free`` (it allocates ``z`` / ``absmax_z``).
    BaseTrainer refuses the combination; the registration function now refuses it
    too, so a direct caller gets a sentence instead of ``KeyError('exp_avg')``
    raised from inside the autograd engine.

(3) The hook returned silently for a parameter on CPU and for a parameter in no
    param_group. Under fused backward the trainer never calls ``optimizer.step()``
    (base_trainer: ``if not self.use_fused_backward and ...``), so neither case
    was ever made up for later: those parameters went untrained for the whole run
    while the loss kept falling. Both are now refused -- the param_group one at
    registration, before the run starts.

CPU-only. Where an 8-bit path is exercised, the compiled CUDA extension is
replaced by a stand-in and the parameters answer ``is_cuda`` (the Block Swap
residency check), as in adamw8bit_ringbuffer_fused_step_test.
"""

from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import core.training.optimizers.adamw8bit_ringbuffer as rb  # noqa: E402
import core.training.optimizers.lion8bit_ringbuffer as lb  # noqa: E402

SEED = 20260825
LR = 1e-3
IN = 8
OUT = 8


class _FakeCudaParameter(torch.nn.Parameter):
    """A CPU parameter that reports ``is_cuda``."""

    @property
    def is_cuda(self) -> bool:  # noqa: D401
        return True


def _fake_param(tensor: torch.Tensor) -> _FakeCudaParameter:
    return torch.Tensor._make_subclass(_FakeCudaParameter, tensor, True)


class _RecordingExtension:
    """Records the step each 8-bit update is handed; performs no arithmetic."""

    def __init__(self):
        self.steps: list[int] = []

    def init_quantization_maps(self, *args, **kwargs):
        pass

    def adamw_8bit_update(self, param, grad, state1, state2, absmax1, absmax2,
                          beta1, beta2, eps, lr, weight_decay, gnorm_scale,
                          step, cautious):
        self.steps.append(int(step))

    def lion_8bit_update(self, param, grad, state, absmax, beta1, beta2, eps,
                         lr, weight_decay, gnorm_scale, step, cautious):
        self.steps.append(int(step))


def _with_extension(module, ext, factory):
    original = module.get_extension
    module.get_extension = lambda: ext
    try:
        return factory()
    finally:
        module.get_extension = original


def _linear_model(fake_cuda: bool = True):
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(IN, OUT, bias=False),
        torch.nn.Linear(OUT, OUT, bias=False),
    )
    for layer in model:
        w = layer.weight.detach().float()
        layer.weight = _fake_param(w) if fake_cuda else torch.nn.Parameter(w)
    return model


def _seed_adamw_8bit_state(optimizer):
    for p in optimizer.param_groups[0]["params"]:
        state = optimizer.state[p]
        state["exp_avg"] = torch.zeros(p.numel(), dtype=torch.uint8)
        state["exp_avg_sq"] = torch.zeros(p.numel(), dtype=torch.uint8)
        state["absmax1"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
        state["absmax2"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
        state["is_8bit"] = True


def _seed_lion_8bit_state(optimizer):
    for p in optimizer.param_groups[0]["params"]:
        state = optimizer.state[p]
        state["exp_avg"] = torch.zeros(p.numel(), dtype=torch.uint8)
        state["absmax"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
        state["is_8bit"] = True


# ---------------------------------------------------------------------------
# (1) step_count survives a resume
# ---------------------------------------------------------------------------

class StepCountPersistenceTest(unittest.TestCase):
    """The FP32 (use_8bit=False) path is used here: its bias correction reads
    ``self.step_count`` in plain Python (``1 - beta1 ** self.step_count``), so a
    resume that restarts at 1 is visible in the parameter itself, and the whole
    test runs on CPU without the CUDA extension or CUDA-pinned absmax."""

    STEPS = 5

    def _optimizer(self, model):
        ext = _RecordingExtension()
        return _with_extension(rb, ext, lambda: rb.AdamW8bit_RingBuffer(
            list(model.parameters()), lr=LR, weight_decay=0.0, use_8bit=False,
        ))

    @staticmethod
    def _fixed_grads(model):
        torch.manual_seed(SEED + 7)
        return [torch.randn_like(p) for p in model.parameters()]

    def _run(self, model, opt, grads, n):
        for _ in range(n):
            for p, g in zip(model.parameters(), grads):
                p.grad = g.clone()
            opt.step()
            opt.zero_grad()

    def test_step_count_is_serialized(self):
        model = _linear_model()
        opt = self._optimizer(model)
        self._run(model, opt, self._fixed_grads(model), self.STEPS)

        saved = opt.state_dict()
        self.assertIn(
            'step_count', saved,
            "step() drives bias correction from step_count; it must be in state_dict",
        )
        self.assertEqual(saved['step_count'], self.STEPS)

    def test_a_resumed_run_continues_bias_correction(self):
        # Uninterrupted reference: STEPS + 1 steps in one go.
        reference_model = _linear_model()
        reference_opt = self._optimizer(reference_model)
        grads = self._fixed_grads(reference_model)
        self._run(reference_model, reference_opt, grads, self.STEPS)
        # state_dict() hands back the live tensors; the following step mutates
        # them in place. torch.save writes a snapshot, so take one here too.
        saved = deepcopy(reference_opt.state_dict())
        state_after_save = [p.detach().clone() for p in reference_model.parameters()]
        self._run(reference_model, reference_opt, grads, 1)
        reference = [p.detach().clone() for p in reference_model.parameters()]

        # Resumed: reload the checkpoint into a fresh optimizer, take step 6.
        resumed_model = _linear_model()
        for p, saved_p in zip(resumed_model.parameters(), state_after_save):
            p.data.copy_(saved_p)
        resumed_opt = self._optimizer(resumed_model)
        resumed_opt.load_state_dict(saved)
        self.assertEqual(resumed_opt.step_count, self.STEPS)
        self._run(resumed_model, resumed_opt, grads, 1)

        for resumed_p, reference_p in zip(resumed_model.parameters(), reference):
            self.assertTrue(
                torch.allclose(resumed_p, reference_p, atol=0, rtol=0),
                "a resumed run must continue bias correction, not restart at step 1",
            )

    def test_a_state_dict_without_step_count_keeps_the_current_counter(self):
        """BaseTrainer's prefix-preserving partial load rebuilds the dict from
        ``state`` + ``param_groups`` only, and pre-fix checkpoints carry no
        step_count. Neither may reset the counter to 0."""
        model = _linear_model()
        opt = self._optimizer(model)
        self._run(model, opt, self._fixed_grads(model), self.STEPS)

        partial = {"state": {}, "param_groups": opt.state_dict()["param_groups"]}
        opt.load_state_dict(partial)
        self.assertEqual(opt.step_count, self.STEPS)

    def test_the_fused_hook_fallback_reads_the_restored_counter(self):
        """4adb0359's fallback is ``state.get('step', self.step_count)``. With
        step_count now restored, a checkpoint carrying no per-parameter step --
        converted from another implementation -- still resumes at the right step
        through the hook."""
        model = _linear_model()
        ext = _RecordingExtension()
        opt = _with_extension(rb, ext, lambda: rb.AdamW8bit_RingBuffer(
            list(model.parameters()), lr=LR, weight_decay=0.0, use_8bit=True,
        ))
        _seed_adamw_8bit_state(opt)
        opt.step_count = 137  # what load_state_dict now restores
        opt.ext = ext
        rb.patch_adamw8bit_ringbuffer(model, opt)

        torch.manual_seed(SEED + 1)
        model(torch.randn(4, IN)).pow(2).mean().backward()
        self.assertEqual(set(ext.steps), {138})


# ---------------------------------------------------------------------------
# (2) Schedule-Free + fused hooks
# ---------------------------------------------------------------------------

class ScheduleFreeFusedBackwardTest(unittest.TestCase):
    def test_patching_a_schedule_free_optimizer_is_refused(self):
        model = _linear_model()
        ext = _RecordingExtension()
        opt = _with_extension(rb, ext, lambda: rb.AdamW8bit_RingBuffer(
            list(model.parameters()), lr=LR, weight_decay=0.0,
            use_8bit=False, schedule_free=True,
        ))
        with self.assertRaises(RuntimeError) as caught:
            rb.patch_adamw8bit_ringbuffer(model, opt)
        self.assertIn("schedule", str(caught.exception).lower())

    def test_lion_refuses_schedule_free_in_the_constructor(self):
        model = _linear_model()
        ext = _RecordingExtension()
        with self.assertRaises(RuntimeError):
            _with_extension(lb, ext, lambda: lb.Lion8bit_RingBuffer(
                list(model.parameters()), lr=LR, weight_decay=0.0,
                use_8bit=False, schedule_free=True,
            ))


# ---------------------------------------------------------------------------
# (3) Silent skips in the fused hook
# ---------------------------------------------------------------------------

class SilentSkipTest(unittest.TestCase):
    def _adamw(self, model):
        ext = _RecordingExtension()
        opt = _with_extension(rb, ext, lambda: rb.AdamW8bit_RingBuffer(
            list(model.parameters()), lr=LR, weight_decay=0.0, use_8bit=True,
        ))
        opt.ext = ext
        _seed_adamw_8bit_state(opt)
        return opt, ext

    def _lion(self, model):
        ext = _RecordingExtension()
        opt = _with_extension(lb, ext, lambda: lb.Lion8bit_RingBuffer(
            list(model.parameters()), lr=LR, weight_decay=0.0, use_8bit=True,
        ))
        opt.ext = ext
        _seed_lion_8bit_state(opt)
        return opt, ext

    def test_adamw_hook_raises_for_a_cpu_parameter(self):
        model = _linear_model(fake_cuda=False)  # honest CPU parameters
        opt, ext = self._adamw(model)
        rb.patch_adamw8bit_ringbuffer(model, opt)

        torch.manual_seed(SEED + 1)
        with self.assertRaises(RuntimeError) as caught:
            model(torch.randn(4, IN)).pow(2).mean().backward()
        self.assertIn("cpu", str(caught.exception).lower())
        self.assertEqual(ext.steps, [], "no update may be claimed for a skipped parameter")

    def test_lion_hook_raises_for_a_cpu_parameter(self):
        model = _linear_model(fake_cuda=False)
        opt, ext = self._lion(model)
        lb.register_lion8bit_fused_backward(opt, model)

        torch.manual_seed(SEED + 1)
        with self.assertRaises(RuntimeError) as caught:
            model(torch.randn(4, IN)).pow(2).mean().backward()
        self.assertIn("cpu", str(caught.exception).lower())
        self.assertEqual(ext.steps, [])

    def test_adamw_registration_refuses_a_parameter_in_no_param_group(self):
        model = _linear_model()
        opt, _ = self._adamw(model)
        # A trainable parameter the optimizer never saw.
        model.add_module("extra", torch.nn.Linear(OUT, OUT, bias=False))
        model.extra.weight = _fake_param(model.extra.weight.detach().float())

        with self.assertRaises(RuntimeError) as caught:
            rb.patch_adamw8bit_ringbuffer(model, opt)
        self.assertIn("extra.weight", str(caught.exception))

    def test_lion_registration_refuses_a_parameter_in_no_param_group(self):
        model = _linear_model()
        opt, _ = self._lion(model)
        model.add_module("extra", torch.nn.Linear(OUT, OUT, bias=False))
        model.extra.weight = _fake_param(model.extra.weight.detach().float())

        with self.assertRaises(RuntimeError) as caught:
            lb.register_lion8bit_fused_backward(opt, model)
        self.assertIn("extra.weight", str(caught.exception))

    def test_a_fully_covered_model_still_registers_and_updates(self):
        """The guards must not fire on the shipped configuration."""
        model = _linear_model()
        opt, ext = self._adamw(model)
        rb.patch_adamw8bit_ringbuffer(model, opt)

        torch.manual_seed(SEED + 1)
        model(torch.randn(4, IN)).pow(2).mean().backward()
        self.assertEqual(len(ext.steps), len(list(model.parameters())))
        self.assertEqual(set(ext.steps), {1})


if __name__ == "__main__":
    unittest.main()
