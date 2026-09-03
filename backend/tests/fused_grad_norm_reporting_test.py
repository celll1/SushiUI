"""Gradient norms, and the clipping that cannot happen, under fused backward.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/fused_grad_norm_reporting_test.py -v

Two defects:

(1) The fused hooks apply each update and clear ``param.grad`` immediately, so
    ``_calculate_grad_norms`` -- which the trainer calls after the whole backward
    -- found no gradients at all and reported 0.0 for every component of every
    step. The hooks now record each gradient's squared norm before clearing it.
    ``_without_recording`` splices the recording back out (the pre-fix hook body)
    as the negative control: the reporting tests must fail with it.

(2) ``max_grad_norm`` was silently ignored: the fused branch never clipped and
    the ring-buffer hooks pass ``gnorm_scale=1.0``. It still is -- clipping by
    global norm cannot be applied when each parameter is updated before the next
    one's gradient exists -- but it now says so, once.

CPU-only. The compiled CUDA extension is replaced by a stand-in and parameters
answer ``is_cuda``, as in fused_backward_param_coverage_test, whose fixtures this
file reuses.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest

import torch
from torch import nn

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import core.training.optimizers.adamw8bit_ringbuffer as rb  # noqa: E402
import core.training.optimizers.fused_optimizer_groups as fog  # noqa: E402
import core.training.optimizers.lion8bit_ringbuffer as lb  # noqa: E402
from core.training.adapters.base_adapter import (  # noqa: E402
    LORA_COMPONENT_TEXT_ENCODER,
    LORA_COMPONENT_TEXT_ENCODER_1,
    LORA_COMPONENT_TEXT_ENCODER_2,
    LORA_COMPONENT_UNET,
    LORA_COMPONENT_VISION_ENCODER,
)
from core.adapters import LoRALinearLayer  # noqa: E402
from core.training.adapters.sd15_adapter import SD15LoRAAdapter  # noqa: E402
from core.training import base_trainer as bt  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
import core.training.optimizers.fused_grad_norm as fgn  # noqa: E402
from core.training.optimizers.fused_grad_norm import FusedGradNormAccumulator  # noqa: E402

from fused_backward_param_coverage_test import (  # noqa: E402
    DIM,
    LR_MAIN,
    LR_TE,
    LR_VE,
    SEED,
    _UpdatingExtension,
    _fake_param,
    _seed_adamw_state,
    _seed_lion_state,
    _with_extension,
)

PLACES = 6


class _StaticExtension(_UpdatingExtension):
    """Records the update without applying it, so repeated backwards match."""

    def _apply(self, param, lr):
        self.updates.append((id(param), float(lr)))


@contextlib.contextmanager
def _without_recording(*modules):
    """The pre-fix hook body: clear the gradient, record nothing.

    Pass the module the hook resolves the name through: ``rb`` / ``lb`` / ``fog``
    bind it at import, base_trainer imports it at registration (so ``fgn``).
    """
    originals = {m: m.record_fused_grad_norm for m in modules}
    for module in originals:
        module.record_fused_grad_norm = lambda *args, **kwargs: None
    try:
        yield
    finally:
        for module, fn in originals.items():
            module.record_fused_grad_norm = fn


class _Trainee(nn.Module):
    """The three modules ``_calculate_grad_norms``'s full-FT branch looks for."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.unet = nn.Linear(DIM, DIM, bias=False)
        self.text_encoder = nn.Linear(DIM, DIM, bias=False)
        self.vision_encoder = nn.Linear(DIM, DIM, bias=False)
        for module in (self.unet, self.text_encoder, self.vision_encoder):
            module.weight = _fake_param(module.weight.detach().float())

    def forward(self, x):
        return self.unet(x) + self.text_encoder(x) + self.vision_encoder(x)

    def param_groups(self):
        return [
            {"params": list(self.unet.parameters()), "lr": LR_MAIN},
            {"params": list(self.text_encoder.parameters()), "lr": LR_TE},
            {"params": list(self.vision_encoder.parameters()), "lr": LR_VE},
        ]


def _backward(model, scale=1.0):
    torch.manual_seed(SEED + 1)
    model(torch.randn(4, DIM) * scale).pow(2).mean().backward()


class _Trainer:
    """Enough of BaseTrainer for the grad-norm reporting and the clip warning."""

    log_prefix = "[test]"
    text_encoder_2 = None
    transformer_original = None
    controlnet = None
    optimizer_schedule_free = False

    def __init__(self, model=None, fused=False, accumulator=None, groups=None):
        if model is not None:
            self.unet = model.unet
            self.text_encoder = model.text_encoder
            self.vision_encoder = model.vision_encoder
        self._train_vision_encoder = True
        self.use_fused_backward = fused
        self.fused_optimizer_groups = groups
        self._fused_grad_norm = accumulator

    _calculate_grad_norms = BaseTrainer._calculate_grad_norms
    # The full-FT component override hook _calculate_grad_norms consults. No
    # adapter here, so it resolves to {} and every bucket stays module-derived.
    _full_parameter_grad_components = BaseTrainer._full_parameter_grad_components
    _warn_grad_clipping_ignored_under_fused = (
        BaseTrainer._warn_grad_clipping_ignored_under_fused
    )
    _setup_fused_backward_pass = BaseTrainer._setup_fused_backward_pass


class _StubStepParamOptimizer:
    """A ``step_param`` optimizer, as the adafactor / adamw8bit fused path has."""

    def __init__(self, param_groups):
        self.param_groups = param_groups
        self.stepped = []

    def step_param(self, param, group):
        self.stepped.append(id(param))


def _norms(trainer):
    with contextlib.redirect_stdout(io.StringIO()):
        return trainer._calculate_grad_norms()


def _reference_norms():
    """Non-fused: gradients are still there when the norms are taken."""
    model = _Trainee()
    _backward(model)
    return _norms(_Trainer(model, fused=False))


# --------------------------------------------------------------------------
# The fused paths
# --------------------------------------------------------------------------


class _FusedPathMixin:
    """Each fused path must report what the non-fused path would have."""

    modules_that_record = ()

    def _build(self, static=False):
        """Return ``(model, register)`` where ``register(model)`` installs the
        hooks and returns the accumulator they record into.

        ``static``: make the updates no-ops, so two backwards over the same input
        see the same gradients."""
        raise NotImplementedError

    def _trainer(self, model, accumulator):
        return _Trainer(model, fused=True, accumulator=accumulator)

    def _begin_step(self):
        """What the trainer does at the start of a step, besides the gate."""

    def _run(self, gate=True, scale=1.0):
        model, register = self._build()
        with contextlib.redirect_stdout(io.StringIO()):
            accumulator = register(model)
        accumulator.begin_step(gate)
        _backward(model, scale)
        for name, param in model.named_parameters():
            self.assertIsNone(param.grad, f"{name}: the hooks must clear the gradient")
        return accumulator, _norms(self._trainer(model, accumulator))

    def test_the_reported_norms_are_the_non_fused_ones(self):
        _, fused = self._run()
        reference = _reference_norms()
        self.assertGreater(reference[0], 0.0)
        for value, expected, name in zip(
            fused, reference,
            ("total", "text_encoder", "te1", "te2", "unet", "vision_encoder"),
        ):
            self.assertAlmostEqual(value, expected, places=PLACES, msg=name)

    def test_every_component_is_attributed(self):
        _, (total, te, te1, te2, unet, ve) = self._run()
        self.assertGreater(unet, 0.0)
        self.assertGreater(te, 0.0)
        self.assertAlmostEqual(te1, te, places=PLACES)  # self.text_encoder is TE1
        self.assertEqual(te2, 0.0)
        self.assertGreater(ve, 0.0)
        self.assertAlmostEqual(
            total, (unet ** 2 + te ** 2 + ve ** 2) ** 0.5, places=PLACES
        )

    def test_nothing_is_recorded_on_a_step_that_does_not_report(self):
        accumulator, norms = self._run(gate=False)
        self.assertEqual(accumulator.squared_norms(), {})
        self.assertEqual(norms, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_the_squares_are_not_read_back_per_parameter(self):
        accumulator, _ = self._run()
        recorded = list(accumulator._squares.values())
        self.assertTrue(recorded)
        for square in recorded:
            self.assertIsInstance(square, torch.Tensor)
            self.assertEqual(square.shape, torch.Size([]))

    def test_the_accumulation_does_not_carry_across_steps(self):
        model, register = self._build(static=True)
        with contextlib.redirect_stdout(io.StringIO()):
            accumulator = register(model)

        self._begin_step()
        accumulator.begin_step(True)
        _backward(model)
        first = _norms(self._trainer(model, accumulator))

        self._begin_step()
        accumulator.begin_step(True)
        _backward(model)  # same input, same (unchanged) weights -> same gradients
        second = _norms(self._trainer(model, accumulator))

        self.assertGreater(first[0], 0.0)
        for a, b in zip(first, second):
            self.assertAlmostEqual(a, b, places=PLACES)

    def test_without_the_recording_every_norm_is_zero(self):
        """Negative control: the pre-fix hooks reported 0.0 for every step."""
        with _without_recording(*self.modules_that_record):
            _, norms = self._run()
        self.assertEqual(norms, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertGreater(_reference_norms()[0], 0.0)


class AdamWRingBufferTest(_FusedPathMixin, unittest.TestCase):
    modules_that_record = (rb,)

    def _build(self, static=False):
        model = _Trainee()
        ext = _StaticExtension() if static else _UpdatingExtension()

        def register(model):
            with contextlib.redirect_stdout(io.StringIO()):
                opt = _with_extension(rb, ext, lambda: rb.AdamW8bit_RingBuffer(
                    model.param_groups(), lr=LR_MAIN, weight_decay=0.0, use_8bit=True,
                ))
            opt.ext = ext
            _seed_adamw_state(opt)
            accumulator = bt.setup_fused_grad_norm(_Trainer(), [opt])
            rb.patch_adamw8bit_ringbuffer(model.unet, opt)
            return accumulator

        return model, register


class LionRingBufferTest(_FusedPathMixin, unittest.TestCase):
    modules_that_record = (lb,)

    def _build(self, static=False):
        model = _Trainee()
        ext = _StaticExtension() if static else _UpdatingExtension()

        def register(model):
            with contextlib.redirect_stdout(io.StringIO()):
                opt = _with_extension(lb, ext, lambda: lb.Lion8bit_RingBuffer(
                    model.param_groups(), lr=LR_MAIN, weight_decay=0.0, use_8bit=True,
                ))
            opt.ext = ext
            _seed_lion_state(opt)
            accumulator = bt.setup_fused_grad_norm(_Trainer(), [opt])
            lb.register_lion8bit_fused_backward(opt, model.unet)
            return accumulator

        return model, register


class StepParamFusedBackwardTest(_FusedPathMixin, unittest.TestCase):
    """``_setup_fused_backward_pass``'s own hooks (adafactor / adamw8bit)."""

    modules_that_record = (fgn,)

    def _build(self, static=False):
        model = _Trainee()

        def register(model):
            trainer = _Trainer(model)
            trainer.optimizer = _StubStepParamOptimizer(model.param_groups())
            trainer._setup_fused_backward_pass("some_other_optimizer")
            self.registered_trainer = trainer
            assert trainer.use_fused_backward
            return trainer._fused_grad_norm

        return model, register

    def test_the_hook_still_updates_the_parameter(self):
        model, register = self._build()
        with contextlib.redirect_stdout(io.StringIO()):
            register(model)
        _backward(model)
        self.assertEqual(
            sorted(self.registered_trainer.optimizer.stepped),
            sorted(id(p) for p in model.parameters()),
        )


class FusedOptimizerGroupsTest(_FusedPathMixin, unittest.TestCase):
    modules_that_record = (fog,)

    def _build(self, static=False):
        model = _Trainee()

        def register(model):
            optimizers = [
                torch.optim.SGD(group["params"], lr=0.0 if static else group["lr"])
                for group in model.param_groups()
            ]
            accumulator = bt.setup_fused_grad_norm(_Trainer(), optimizers)
            self.groups = fog.FusedOptimizerGroups(optimizers, max_grad_norm=0.0)
            self.groups.register_hooks()
            return accumulator

        return model, register

    def _trainer(self, model, accumulator):
        return _Trainer(model, fused=False, accumulator=accumulator, groups=self.groups)

    def _begin_step(self):
        self.groups.reset_counters()  # base_trainer does this per step


# --------------------------------------------------------------------------
# LoRA component attribution (dd0b10c7) under fused backward
# --------------------------------------------------------------------------


class _LoraTrainee(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.adapter = SD15LoRAAdapter(object(), 2, 4)
        self.lora_layers = {}
        for name, component in (
            ("dit", LORA_COMPONENT_UNET),
            ("te1", LORA_COMPONENT_TEXT_ENCODER_1),
            ("te2", LORA_COMPONENT_TEXT_ENCODER_2),
            ("te", LORA_COMPONENT_TEXT_ENCODER),
            ("ve", LORA_COMPONENT_VISION_ENCODER),
        ):
            layer = LoRALinearLayer(nn.Linear(DIM, DIM, bias=False), rank=2,
                                    alpha=2, lora_name=name)
            # lora_up is zero-initialised, which would zero lora_down's gradient.
            nn.init.normal_(layer.lora_up.weight)
            self.adapter.register_lora_layer(self.lora_layers, name, layer, component)
        self.layers = nn.ModuleList(self.lora_layers.values())

    def forward(self, x):
        return sum(layer(x) for layer in self.layers)

    def param_groups(self):
        return [{"params": [p for p in self.parameters() if p.requires_grad],
                 "lr": LR_MAIN}]


class _LoraTrainer(_Trainer):
    def __init__(self, model, fused, accumulator=None):
        super().__init__(None, fused=fused, accumulator=accumulator)
        self.lora_layers = model.lora_layers
        self.adapter = model.adapter


class LoraComponentsUnderFusedTest(unittest.TestCase):
    def _fused(self):
        model = _LoraTrainee()
        trainer = _LoraTrainer(model, fused=True)
        trainer.optimizer = _StubStepParamOptimizer(model.param_groups())
        with contextlib.redirect_stdout(io.StringIO()):
            trainer._setup_fused_backward_pass("some_other_optimizer")
        trainer._fused_grad_norm.begin_step(True)
        _backward(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNone(param.grad, name)
        return _norms(trainer)

    def _reference(self):
        model = _LoraTrainee()
        _backward(model)
        return _norms(_LoraTrainer(model, fused=False))

    def test_the_adapter_components_still_split_the_norms(self):
        total, te, te1, te2, unet, ve = self._fused()
        for value in (total, te, te1, te2, unet, ve):
            self.assertGreater(value, 0.0)
        # Five layers, one per component: te1 + te2 + the plain-TE one make te,
        # and the DiT and VE ones are outside it.
        self.assertGreater(te ** 2, te1 ** 2 + te2 ** 2)
        self.assertAlmostEqual(total, (te ** 2 + unet ** 2 + ve ** 2) ** 0.5, places=PLACES)

    def test_the_split_matches_the_non_fused_split(self):
        for value, expected in zip(self._fused(), self._reference()):
            self.assertAlmostEqual(value, expected, places=PLACES)

    def test_without_the_recording_every_norm_is_zero(self):
        with _without_recording(fgn):
            self.assertEqual(self._fused(), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertGreater(self._reference()[0], 0.0)


# --------------------------------------------------------------------------
# max_grad_norm under fused backward
# --------------------------------------------------------------------------


class ClipWarningTest(unittest.TestCase):
    def _warn(self, trainer, max_grad_norm, times=1):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for _ in range(times):
                trainer._warn_grad_clipping_ignored_under_fused(max_grad_norm)
        return out.getvalue()

    def test_it_says_the_setting_is_ignored_and_why(self):
        message = self._warn(_Trainer(fused=True), 1.0)
        self.assertIn("max_grad_norm=1.0", message)
        self.assertIn("IGNORED", message)
        self.assertIn("global norm", message)
        self.assertIn("No clipping of any kind is applied", message)

    def test_it_fires_once(self):
        trainer = _Trainer(fused=True)
        # One emission is two stdout lines: the human one and the machine one
        # TrainingProcess lifts off the stream (core/training/training_events.py).
        # Count the human line only.
        from core.training.training_events import TRAINING_EVENT_SENTINEL
        lines = [line for line in self._warn(trainer, 1.0, times=5).splitlines()
                 if not line.startswith(TRAINING_EVENT_SENTINEL)]
        self.assertEqual(sum(line.count("IGNORED") for line in lines), 1)

    def test_the_fused_optimizer_groups_path_warns_too(self):
        trainer = _Trainer(fused=False, groups=object())
        self.assertIn("fused optimizer groups", self._warn(trainer, 1.0))

    def test_it_is_silent_without_a_fused_path(self):
        self.assertEqual(self._warn(_Trainer(fused=False), 1.0), "")

    def test_it_is_silent_when_no_clipping_was_asked_for(self):
        self.assertEqual(self._warn(_Trainer(fused=True), 0.0), "")


# --------------------------------------------------------------------------
# The accumulator itself
# --------------------------------------------------------------------------


class AccumulatorTest(unittest.TestCase):
    def _param(self, value):
        param = nn.Parameter(torch.zeros(4))
        param.grad = torch.full((4,), value)
        return param

    def test_begin_step_drops_the_previous_step(self):
        accumulator = FusedGradNormAccumulator()
        param = self._param(1.0)
        accumulator.begin_step(True)
        accumulator.record(param)
        accumulator.begin_step(True)
        accumulator.record(param)
        self.assertAlmostEqual(accumulator.squared_norms()[id(param)], 4.0, places=PLACES)

    def test_repeated_records_within_a_step_accumulate(self):
        accumulator = FusedGradNormAccumulator()
        param = self._param(1.0)
        accumulator.begin_step(True)
        accumulator.record(param)
        accumulator.record(param)
        self.assertAlmostEqual(accumulator.squared_norms()[id(param)], 8.0, places=PLACES)

    def test_a_bf16_gradient_is_squared_in_fp32(self):
        param = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        param.grad = torch.full((4,), 1e-3, dtype=torch.bfloat16)
        accumulator = FusedGradNormAccumulator()
        accumulator.begin_step(True)
        accumulator.record(param)
        self.assertAlmostEqual(
            accumulator.squared_norms()[id(param)],
            4 * float(param.grad[0]) ** 2,
            places=12,
        )

    def test_the_gate_is_the_callers_to_set(self):
        accumulator = FusedGradNormAccumulator()
        accumulator.begin_step(False)
        self.assertFalse(accumulator.enabled)
        accumulator.begin_step(True)
        self.assertTrue(accumulator.enabled)


if __name__ == "__main__":
    unittest.main()
