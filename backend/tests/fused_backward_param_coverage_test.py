"""Every parameter the ring-buffer optimizer owns must actually be updated.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/fused_backward_param_coverage_test.py -v

Registration used to walk the module BaseTrainer passes
(``_fused_backward_target_module()``: the transformer or the U-Net) and hook its
trainable parameters. ``setup_optimizer`` also appends text-encoder and
vision-encoder groups to the SAME optimizer, and under the fused backward pass
the trainer never calls ``optimizer.step()`` (base_trainer:
``if not self.use_fused_backward and ...``) -- so with ``train_text_encoder`` on
and Block Swap on, those parameters had no hook and no step: they never moved for
the whole run while the loss kept falling.

Registration is now driven by ``optimizer.param_groups``. ``_LEGACY_REGISTRATION``
below is the previous module-walk, spliced back in through the same seam and with
the same hook bodies, as the negative control: the coverage tests must fail with
it and pass without it.

CPU-only. The compiled CUDA extension is replaced by a stand-in that performs a
visible update, and parameters answer ``is_cuda`` (the Block Swap residency
check), as in adamw8bit_ringbuffer_defect_guards_test.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import core.training.optimizers.adamw8bit_ringbuffer as rb  # noqa: E402
import core.training.optimizers.lion8bit_ringbuffer as lb  # noqa: E402

SEED = 20260825
DIM = 8
LR_MAIN = 1e-3
LR_TE = 2e-3
LR_VE = 4e-3


class _FakeCudaParameter(torch.nn.Parameter):
    """A CPU parameter that reports ``is_cuda``."""

    @property
    def is_cuda(self) -> bool:  # noqa: D401
        return True


def _fake_param(tensor: torch.Tensor) -> _FakeCudaParameter:
    return torch.Tensor._make_subclass(_FakeCudaParameter, tensor, True)


class _UpdatingExtension:
    """Applies a visible, lr-dependent update and records what it was handed."""

    def __init__(self):
        self.updates: list[tuple[int, float]] = []

    def init_quantization_maps(self, *args, **kwargs):
        pass

    def _apply(self, param, lr):
        self.updates.append((id(param), float(lr)))
        param.data.add_(-float(lr))

    def adamw_8bit_update(self, param, grad, state1, state2, absmax1, absmax2,
                          beta1, beta2, eps, lr, weight_decay, gnorm_scale,
                          step, cautious):
        self._apply(param, lr)

    def lion_8bit_update(self, param, grad, state, absmax, beta1, beta2, eps,
                         lr, weight_decay, gnorm_scale, step, cautious):
        self._apply(param, lr)


def _with_extension(module, ext, factory):
    original = module.get_extension
    module.get_extension = lambda: ext
    try:
        return factory()
    finally:
        module.get_extension = original


class _ThreeModuleTrainee(torch.nn.Module):
    """Transformer + text encoder + vision encoder, all feeding one loss.

    Mirrors the shape of the reachable configuration: BaseTrainer hands the
    ring-buffer registration ``self.transformer`` only, while the optimizer holds
    all three groups.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.transformer = torch.nn.Linear(DIM, DIM, bias=False)
        self.text_encoder = torch.nn.Linear(DIM, DIM, bias=False)
        self.vision_encoder = torch.nn.Linear(DIM, DIM, bias=False)
        for module in (self.transformer, self.text_encoder, self.vision_encoder):
            module.weight = _fake_param(module.weight.detach().float())

    def forward(self, x):
        return self.transformer(x) + self.text_encoder(x) + self.vision_encoder(x)

    def param_groups(self):
        return [
            {"params": list(self.transformer.parameters()), "lr": LR_MAIN},
            {"params": list(self.text_encoder.parameters()), "lr": LR_TE},
            {"params": list(self.vision_encoder.parameters()), "lr": LR_VE},
        ]


def _seed_adamw_state(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            state["exp_avg"] = torch.zeros(p.numel(), dtype=torch.uint8)
            state["exp_avg_sq"] = torch.zeros(p.numel(), dtype=torch.uint8)
            state["absmax1"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
            state["absmax2"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
            state["is_8bit"] = True


def _seed_lion_state(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            state["exp_avg"] = torch.zeros(p.numel(), dtype=torch.uint8)
            state["absmax"] = torch.zeros((p.numel() + 255) // 256, dtype=torch.float32)
            state["is_8bit"] = True


def _LEGACY_REGISTRATION(optimizer, module, function_name, make_hook):
    """The pre-fix registration: hook the trainable parameters of ``module``."""
    param_to_group = {id(gp): g for g in optimizer.param_groups for gp in g["params"]}
    hooked = 0
    for p in module.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(make_hook(p, param_to_group[id(p)]))
            hooked += 1
    return hooked, []


@contextlib.contextmanager
def _legacy_registration():
    originals = {mod: mod.register_fused_backward_hooks for mod in (rb, lb)}
    for mod in originals:
        mod.register_fused_backward_hooks = _LEGACY_REGISTRATION
    try:
        yield
    finally:
        for mod, fn in originals.items():
            mod.register_fused_backward_hooks = fn


class _CoverageMixin:
    """The optimizer's parameters, all of them, must move on one backward."""

    def _build(self):
        raise NotImplementedError

    def _run_backward(self, model):
        torch.manual_seed(SEED + 1)
        model(torch.randn(4, DIM)).pow(2).mean().backward()

    def _assert_all_optimizer_params_updated(self):
        model, opt, ext = self._build()
        before = {id(p): p.detach().clone()
                  for g in opt.param_groups for p in g["params"]}
        with contextlib.redirect_stdout(io.StringIO()):
            self._register(model, opt)
        self._run_backward(model)

        updated = {pid for pid, _ in ext.updates}
        missing = [name for name, p in model.named_parameters()
                   if id(p) not in updated]
        self.assertEqual(
            missing, [],
            "every parameter in optimizer.param_groups must be updated; under fused "
            "backward there is no optimizer.step() to catch the rest",
        )
        for name, p in model.named_parameters():
            self.assertFalse(
                torch.equal(p.detach(), before[id(p)]),
                f"{name} did not move",
            )

    def test_all_optimizer_params_are_updated(self):
        self._assert_all_optimizer_params_updated()

    def test_the_module_walk_leaves_the_encoders_untrained(self):
        """Negative control: with the pre-fix registration this must fail."""
        with _legacy_registration():
            with self.assertRaises(AssertionError):
                self._assert_all_optimizer_params_updated()

    def test_each_hook_uses_its_own_group_lr(self):
        """Coverage alone is not enough: a hook must carry ITS group's lr, not
        the transformer group's."""
        model, opt, ext = self._build()
        with contextlib.redirect_stdout(io.StringIO()):
            self._register(model, opt)
        self._run_backward(model)

        by_param = dict(ext.updates)
        self.assertEqual(by_param[id(model.transformer.weight)], LR_MAIN)
        self.assertEqual(by_param[id(model.text_encoder.weight)], LR_TE)
        self.assertEqual(by_param[id(model.vision_encoder.weight)], LR_VE)


class AdamWCoverageTest(_CoverageMixin, unittest.TestCase):
    def _build(self):
        model = _ThreeModuleTrainee()
        ext = _UpdatingExtension()
        with contextlib.redirect_stdout(io.StringIO()):
            opt = _with_extension(rb, ext, lambda: rb.AdamW8bit_RingBuffer(
                model.param_groups(), lr=LR_MAIN, weight_decay=0.0, use_8bit=True,
            ))
        opt.ext = ext
        _seed_adamw_state(opt)
        return model, opt, ext

    def _register(self, model, opt):
        # BaseTrainer passes _fused_backward_target_module(), i.e. the transformer.
        rb.patch_adamw8bit_ringbuffer(model.transformer, opt)


class LionCoverageTest(_CoverageMixin, unittest.TestCase):
    def _build(self):
        model = _ThreeModuleTrainee()
        ext = _UpdatingExtension()
        with contextlib.redirect_stdout(io.StringIO()):
            opt = _with_extension(lb, ext, lambda: lb.Lion8bit_RingBuffer(
                model.param_groups(), lr=LR_MAIN, weight_decay=0.0, use_8bit=True,
            ))
        opt.ext = ext
        _seed_lion_state(opt)
        return model, opt, ext

    def _register(self, model, opt):
        lb.register_lion8bit_fused_backward(opt, model.transformer)


class Fp32GroupTest(unittest.TestCase):
    """Residual (B): the hook dropped non-8-bit parameters with the same false
    promise ("updated in optimizer.step()"), which fused backward never keeps."""

    def _model_and_ext(self):
        torch.manual_seed(SEED)
        layer = torch.nn.Linear(DIM, DIM, bias=False)
        layer.weight = _fake_param(layer.weight.detach().float())
        return layer, _UpdatingExtension()

    def test_adamw_registration_refuses_an_fp32_group(self):
        layer, ext = self._model_and_ext()
        with contextlib.redirect_stdout(io.StringIO()):
            opt = _with_extension(rb, ext, lambda: rb.AdamW8bit_RingBuffer(
                list(layer.parameters()), lr=LR_MAIN, weight_decay=0.0, use_8bit=False,
            ))
        with self.assertRaises(RuntimeError) as caught:
            rb.patch_adamw8bit_ringbuffer(layer, opt)
        self.assertIn("use_8bit", str(caught.exception))

    def test_lion_registration_refuses_an_fp32_group(self):
        layer, ext = self._model_and_ext()
        with contextlib.redirect_stdout(io.StringIO()):
            opt = _with_extension(lb, ext, lambda: lb.Lion8bit_RingBuffer(
                list(layer.parameters()), lr=LR_MAIN, weight_decay=0.0, use_8bit=False,
            ))
        with self.assertRaises(RuntimeError) as caught:
            lb.register_lion8bit_fused_backward(opt, layer)
        self.assertIn("use_8bit", str(caught.exception))

    def test_the_legacy_hook_silently_left_an_fp32_parameter_untrained(self):
        """Negative control for the refusal above: the state that reaches the
        hook is what used to be skipped, and nothing updated it afterwards."""
        layer, ext = self._model_and_ext()
        with contextlib.redirect_stdout(io.StringIO()):
            opt = _with_extension(rb, ext, lambda: rb.AdamW8bit_RingBuffer(
                list(layer.parameters()), lr=LR_MAIN, weight_decay=0.0, use_8bit=True,
            ))
        opt.ext = ext
        _seed_adamw_state(opt)
        opt.state[layer.weight]["is_8bit"] = False  # what an FP32 group produces

        with contextlib.redirect_stdout(io.StringIO()):
            rb.patch_adamw8bit_ringbuffer(layer, opt)
        before = layer.weight.detach().clone()
        torch.manual_seed(SEED + 1)
        with self.assertRaises(RuntimeError) as caught:
            layer(torch.randn(4, DIM)).pow(2).mean().backward()
        self.assertIn("8-bit", str(caught.exception))
        self.assertEqual(ext.updates, [])
        self.assertTrue(torch.equal(layer.weight.detach(), before),
                        "the skipped parameter really does stay where it was")


class ShippedConfigurationTest(unittest.TestCase):
    """The shipped configuration -- LoRA adapters only, no text encoder in the
    optimizer -- must register without tripping any of the refusals."""

    def _lora_model(self):
        torch.manual_seed(SEED)
        model = torch.nn.Module()
        model.base = torch.nn.Linear(DIM, DIM, bias=False)
        model.lora_down = torch.nn.Linear(DIM, DIM, bias=False)
        for layer in (model.base, model.lora_down):
            layer.weight = _fake_param(layer.weight.detach().float())
        model.base.weight.requires_grad_(False)  # frozen base weights
        model.forward = lambda x: model.lora_down(model.base(x))
        return model

    def test_frozen_base_weights_do_not_trip_the_guards(self):
        model = self._lora_model()
        ext = _UpdatingExtension()
        with contextlib.redirect_stdout(io.StringIO()):
            opt = _with_extension(rb, ext, lambda: rb.AdamW8bit_RingBuffer(
                [p for p in model.parameters() if p.requires_grad],
                lr=LR_MAIN, weight_decay=0.0, use_8bit=True,
            ))
        opt.ext = ext
        _seed_adamw_state(opt)

        with contextlib.redirect_stdout(io.StringIO()):
            rb.patch_adamw8bit_ringbuffer(model, opt)

        torch.manual_seed(SEED + 1)
        model(torch.randn(4, DIM)).pow(2).mean().backward()
        self.assertEqual([pid for pid, _ in ext.updates], [id(model.lora_down.weight)])

    def test_a_frozen_parameter_inside_a_group_is_reported_not_refused(self):
        """``setup_optimizer`` appends whole encoders; a frozen tensor among them
        receives no gradient, so it is neither hooked nor a silent skip."""
        model = self._lora_model()
        ext = _UpdatingExtension()
        with contextlib.redirect_stdout(io.StringIO()):
            opt = _with_extension(rb, ext, lambda: rb.AdamW8bit_RingBuffer(
                list(model.parameters()),  # includes the frozen base weight
                lr=LR_MAIN, weight_decay=0.0, use_8bit=True,
            ))
        opt.ext = ext
        _seed_adamw_state(opt)

        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            rb.patch_adamw8bit_ringbuffer(model, opt)
        self.assertIn("requires_grad=False", log.getvalue())


if __name__ == "__main__":
    unittest.main()
