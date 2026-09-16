"""Fused-backward hooks must read the LIVE param group, not the one they were registered with.

Loading optimizer state (torch ``Optimizer.load_state_dict`` and the ring-buffer
uint8 loaders alike) replaces ``optimizer.param_groups`` with new dicts, and
``_split_fresh_param_groups_for_warmup`` rewrites them too. The scheduler and the
resume LR re-assertion write the new dicts.

CPU only: the 8-bit CUDA kernels are replaced by a recording stand-in and the
parameters answer ``is_cuda``, as in adamw8bit_ringbuffer_defect_guards_test.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/fused_hook_live_param_group_test.py -v
"""

from __future__ import annotations

import contextlib
import io
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from transformers import Adafactor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.training.optimizers.adamw8bit_ringbuffer as rb  # noqa: E402
import core.training.optimizers.host_state_allocator as hsa  # noqa: E402
import core.training.optimizers.lion8bit_ringbuffer as lb  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.lr_utils import reassert_config_lr  # noqa: E402

IN = 8
BASE_LRS = (1e-3, 2e-3)
KINDS = ("adamw8bit_ringbuffer", "lion8bit_ringbuffer", "adafactor")


class _FakeCudaParameter(torch.nn.Parameter):
    @property
    def is_cuda(self) -> bool:
        return True


class _RecordingExtension:
    def __init__(self, calls):
        self.calls = calls

    def init_quantization_maps(self, *args, **kwargs):
        pass

    def adamw_8bit_update(self, param, grad, state1, state2, absmax1, absmax2,
                          beta1, beta2, eps, lr, weight_decay, gnorm_scale,
                          step, cautious):
        self.calls.append((id(param), {"lr": float(lr), "weight_decay": weight_decay,
                                       "betas": (beta1, beta2), "eps": eps}))

    def lion_8bit_update(self, param, grad, state, absmax, beta1, beta2, eps,
                         lr, weight_decay, gnorm_scale, step, cautious):
        self.calls.append((id(param), {"lr": float(lr), "weight_decay": weight_decay,
                                       "betas": (beta1, beta2)}))


class _Harness:
    _load_one_optimizer_state = BaseTrainer._load_one_optimizer_state
    _load_optimizer_state_by_parameter_name = BaseTrainer._load_optimizer_state_by_parameter_name
    _remap_optimizer_state_by_group_prefix = BaseTrainer._remap_optimizer_state_by_group_prefix
    _optimizer_state_entry_fits_param = staticmethod(BaseTrainer._optimizer_state_entry_fits_param)
    _rearm_warmup_after_optimizer_reset = BaseTrainer._rearm_warmup_after_optimizer_reset
    _split_fresh_param_groups_for_warmup = BaseTrainer._split_fresh_param_groups_for_warmup
    _compose_warmup_lambda = staticmethod(BaseTrainer._compose_warmup_lambda)
    _fast_forward_lr_schedulers = BaseTrainer._fast_forward_lr_schedulers
    _fast_forward_one_lr_scheduler = staticmethod(BaseTrainer._fast_forward_one_lr_scheduler)
    _setup_fused_backward_pass = BaseTrainer._setup_fused_backward_pass

    def __init__(self, model):
        self.model = model
        self.log_prefix = "[test]"
        self.device = torch.device("cpu")
        self.config = {"fused_grad_clip_factor": 0}
        self.optimizer_state_host_resident = False
        self._optimizer_state_partially_fresh = False
        self.optimizer_warmup_steps = 100
        self._grad_accum_steps = 1
        self.fused_optimizer_groups = None
        self.use_grad_scaler = False
        self.optimizer_schedule_free = False
        self.optimizer_stochastic_rounding = False
        self.optimizer = None
        self.lr_scheduler = None

    def _fused_backward_target_module(self):
        return self.model

    def _build_ema_param_name_map(self):
        return {id(p): name for name, p in self.model.named_parameters()}


@pytest.fixture(autouse=True)
def _cpu_state_placement(monkeypatch):
    # absmax* is sent to cuda:0 on load; keep it on the (CPU) parameter's device.
    monkeypatch.setattr(hsa, "place_loaded_state_tensor",
                        lambda optimizer, param, key, tensor: tensor.to(param.device))
    yield
    assert not torch.cuda.is_initialized()


def _model():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(IN, IN, bias=False),
                                torch.nn.Linear(IN, IN, bias=False))
    for layer in model:
        layer.weight = torch.Tensor._make_subclass(
            _FakeCudaParameter, layer.weight.detach().float(), True)
    return model


def _optimizer(kind, groups, calls):
    groups = [{"params": list(params), "lr": lr} for params, lr in groups]
    with contextlib.redirect_stdout(io.StringIO()):
        if kind == "adafactor":
            return Adafactor(groups, relative_step=False, scale_parameter=False,
                             warmup_init=False, weight_decay=0.0)
        module, cls = ((rb, rb.AdamW8bit_RingBuffer) if kind == "adamw8bit_ringbuffer"
                       else (lb, lb.Lion8bit_RingBuffer))
        ext = _RecordingExtension(calls)
        original = module.get_extension
        module.get_extension = lambda: ext
        try:
            optimizer = cls(groups, weight_decay=0.0, use_8bit=True)
        finally:
            module.get_extension = original
    optimizer.ext = ext
    return optimizer


def _give_state(kind, optimizer):
    """Real state on every parameter, so every loader path has something to restore."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            if kind == "adafactor":
                from core.training.optimizers.adafactor_fused import adafactor_step_param
                p.grad = torch.ones_like(p)
                adafactor_step_param(optimizer, p, group)
                p.grad = None
                continue
            state = optimizer.state[p]
            state["exp_avg"] = torch.zeros(p.numel(), dtype=torch.uint8)
            blocks = (p.numel() + 255) // 256
            if kind == "adamw8bit_ringbuffer":
                state["exp_avg_sq"] = torch.zeros(p.numel(), dtype=torch.uint8)
                state["absmax1"] = torch.zeros(blocks, dtype=torch.float32)
                state["absmax2"] = torch.zeros(blocks, dtype=torch.float32)
            else:
                state["absmax"] = torch.zeros(blocks, dtype=torch.float32)
            state["is_8bit"] = True


def _live(kind, model, groups):
    """Optimizer + fused hooks registered the way BaseTrainer registers them."""
    calls = []
    trainer = _Harness(model)
    optimizer = _optimizer(kind, groups, calls)
    _give_state(kind, optimizer)
    trainer.optimizer = optimizer
    with contextlib.redirect_stdout(io.StringIO()):
        trainer._setup_fused_backward_pass(kind)
    if kind == "adafactor":
        inner = optimizer.step_param

        def recording_step_param(p, group):
            calls.append((id(p), {"lr": float(group["lr"]),
                                  "weight_decay": group["weight_decay"],
                                  "decay_rate": group["decay_rate"]}))
            return inner(p, group)

        optimizer.step_param = recording_step_param
    return trainer, optimizer, calls


def _saved(kind, model, groups, named):
    source = _optimizer(kind, groups, [])
    _give_state(kind, source)
    saved = deepcopy(source.state_dict())
    if named:
        names = {id(p): n for n, p in model.named_parameters()}
        saved["_sushi_param_names"] = [[names[id(p)] for p in g["params"]]
                                       for g in source.param_groups]
    return saved


def _backward(model, calls):
    calls.clear()
    torch.manual_seed(1)
    model(torch.randn(4, IN)).pow(2).mean().backward()
    return dict(calls)


def _distinct_hyperparameters(kind, index):
    values = {"lr": 3e-4 * (index + 1), "weight_decay": 0.01 * (index + 1)}
    if kind == "adafactor":
        values["decay_rate"] = -0.5 - 0.1 * index
    else:
        values["betas"] = (0.8 - 0.05 * index, 0.95 - 0.01 * index)
    if kind == "adamw8bit_ringbuffer":
        values["eps"] = 1e-7 * (index + 1)
    return values


def _load(trainer, optimizer, saved, path):
    with contextlib.redirect_stdout(io.StringIO()):
        if path == "optimizer.load_state_dict":
            optimizer.load_state_dict(saved)
            return True
        return trainer._load_one_optimizer_state(optimizer, saved, "opt.pt")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("path", [
    "optimizer.load_state_dict", "trainer_direct", "trainer_named", "trainer_prefix_remap",
])
def test_kernel_sees_hyperparameters_written_to_the_live_group_after_load(kind, path):
    model = _model()
    layout = [([model[0].weight], BASE_LRS[0]), ([model[1].weight], BASE_LRS[1])]
    trainer, optimizer, calls = _live(kind, model, layout)
    saved = _saved(kind, model, layout[:1] if path == "trainer_prefix_remap" else layout,
                   named=(path == "trainer_named"))

    assert _load(trainer, optimizer, saved, path) is True
    for index, group in enumerate(optimizer.param_groups):
        group.update(_distinct_hyperparameters(kind, index))

    seen = _backward(model, calls)
    for group in optimizer.param_groups:
        (p,) = group["params"]
        for key, value in seen[id(p)].items():
            assert value == pytest.approx(group[key]), (key, path)


@pytest.mark.parametrize("kind", KINDS)
def test_kernel_follows_a_param_group_split_made_after_registration(kind):
    model = _model()
    old, new = model[0].weight, model[1].weight
    trainer, optimizer, calls = _live(kind, model, [([old, new], BASE_LRS[0])])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    trainer._split_fresh_param_groups_for_warmup(optimizer, scheduler, {id(new)}, {0})
    assert [g["params"] for g in optimizer.param_groups] == [[old], [new]]
    for index, group in enumerate(optimizer.param_groups):
        group.update(_distinct_hyperparameters(kind, index))

    seen = _backward(model, calls)
    for group in optimizer.param_groups:
        (p,) = group["params"]
        for key, value in seen[id(p)].items():
            assert value == pytest.approx(group[key]), key


@pytest.mark.parametrize("kind", KINDS)
def test_wholly_fresh_group_warms_up_from_zero_in_the_kernel(kind):
    resume, warmup = 6000, 100
    model = _model()
    layout = [([model[0].weight], BASE_LRS[0]), ([model[1].weight], BASE_LRS[1])]
    trainer, optimizer, calls = _live(kind, model, layout)
    schedule = lambda step: 0.5 if step >= 5000 else 1.0  # noqa: E731
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [schedule, schedule])
    with contextlib.redirect_stdout(io.StringIO()):
        trainer._fast_forward_lr_schedulers(resume)

    assert _load(trainer, optimizer, _saved(kind, model, layout[:1], named=False),
                 "trainer_prefix_remap") is True
    assert trainer._optimizer_fresh_param_group_indices[id(optimizer)] == {1}
    with contextlib.redirect_stdout(io.StringIO()):
        assert trainer._rearm_warmup_after_optimizer_reset(resume) is True
        reassert_config_lr(optimizer, trainer.lr_scheduler, list(BASE_LRS), verbose=False)

    restored, fresh = model[0].weight, model[1].weight
    for elapsed, factor in ((0, 0.0), (25, 0.25), (warmup, 1.0), (warmup + 10, 1.0)):
        while trainer.lr_scheduler.last_epoch < resume + elapsed:
            trainer.lr_scheduler.step()
        seen = _backward(model, calls)
        assert seen[id(restored)]["lr"] == pytest.approx(BASE_LRS[0] * 0.5), elapsed
        assert seen[id(fresh)]["lr"] == pytest.approx(BASE_LRS[1] * 0.5 * factor), elapsed


def test_adamw8bit_fused_index_follows_a_param_group_rewrite():
    from core.training.optimizers.adamw8bit_fused import _param_index

    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([a, b], lr=1e-3)
    assert _param_index(optimizer, b) == (0, 1)

    optimizer.param_groups[:] = [dict(optimizer.param_groups[0], params=[a]),
                                 dict(optimizer.param_groups[0], params=[b])]
    assert _param_index(optimizer, b) == (1, 0)
