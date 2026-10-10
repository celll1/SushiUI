"""Cancelled or failed calibration must not publish partial initialization."""
from copy import deepcopy

import pytest
import torch

from core.rewiring.fitting import (FitExample, FitPhase, InitializationCancelled,
                                  fit_teacher_modules, weights_digest)
from hybrid_initialization_test import TinyModel
from hybrid_provenance_test import cpu_only_imports


def run(model, *, progress=None, cancelled=None, examples=None, factory=None):
    value = torch.arange(640, dtype=torch.float32).reshape(1, 8, 8, 10)/640
    target = model.base(value).detach() + 0.1
    phases = [FitPhase("data_noising", 1, 0.03, [FitExample((value,), {})]),
              FitPhase("native_trajectory", 2, 0.02,
                       examples if examples is not None else [FitExample((value,), {}, target)]*2)]
    return fit_teacher_modules(model, phases, lambda network, args, kwargs: network(*args, **kwargs),
        factory or (lambda parameters, lr: torch.optim.SGD(parameters, lr=lr)),
        source_sha256="1"*64, calibration_sha256="2"*64, sample_count=32, seed=718,
        clip_norm=1., progress=progress, cancelled=cancelled)


def test_success_fits_added_parameters_and_records_actual_weights():
    model = TinyModel()
    model.base.bias.requires_grad_(False)
    native = deepcopy(model.base.state_dict())
    before = weights_digest(model.rewiring_modules)
    rng = torch.get_rng_state().clone()
    events = []
    result = run(model, progress=events.append)
    assert result == model._hybrid_initialization
    assert result["initialized_weights_sha256"] == weights_digest(model.rewiring_modules) != before
    assert [row["completed"] for row in events] == [1, 2, 3]
    assert all(row["total"] == 3 for row in events)
    assert all(torch.equal(value, model.base.state_dict()[name]) for name, value in native.items())
    assert not model.base.bias.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)
    result["seed"] = 1
    assert model._hybrid_initialization["seed"] == 718


@pytest.mark.parametrize("reason", ["cancel", "progress", "interrupt", "short", "nan", "shape", "ownership"])
def test_failure_restores_added_weights_flags_rng_and_no_success_record(reason):
    model = TinyModel()
    model.base.eval()
    model.rewiring_modules.requires_grad_(False)
    before = deepcopy(model.state_dict())
    flags = [p.requires_grad for p in model.parameters()]
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    options = {}
    if reason == "cancel":
        calls = []
        def cancel():
            calls.append(1)
            if len(calls) == 4:
                assert any(not torch.equal(value, model.state_dict()[name]) for name, value in before.items())
            return len(calls) == 4
        options["cancelled"] = cancel
        error = InitializationCancelled
    elif reason in ("progress", "interrupt"):
        error = RuntimeError if reason == "progress" else KeyboardInterrupt
        def progress(event):
            if event["completed"] == 2:
                assert any(not torch.equal(value, model.state_dict()[name]) for name, value in before.items())
                torch.rand(3)
                raise error("observer failed")
        options["progress"] = progress
    elif reason == "ownership":
        options["factory"] = lambda parameters, lr: torch.optim.SGD(model.parameters(), lr=lr)
        error = ValueError
    else:
        value = torch.arange(640, dtype=torch.float32).reshape(1, 8, 8, 10)/640
        target = model.base(value).detach()+0.1
        broken = target.clone()
        if reason == "nan":
            broken.fill_(float("nan"))
        elif reason == "shape":
            broken = broken[:,:,:1,:]
        options["examples"] = ([FitExample((value,), {}, target)] if reason == "short" else
                               [FitExample((value,), {}, target), FitExample((value,), {}, broken)])
        error = ValueError
    with pytest.raises(error):
        run(model, **options)
    assert not hasattr(model, "_hybrid_initialization")
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert [p.requires_grad for p in model.parameters()] == flags
    assert [module.training for module in model.modules()] == modes
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)


def test_fitted_artifact_refuses_second_initialization():
    model = TinyModel()
    run(model)
    before = deepcopy(model.state_dict())
    provenance = deepcopy(model._hybrid_initialization)
    with pytest.raises(ValueError, match="must not be fitted again"):
        run(model)
    assert model._hybrid_initialization == provenance
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("multiplier", [1., .25])
def test_skip_multiplier_scales_only_skip_update_and_roundtrips(multiplier):
    class SkipModel(TinyModel):
        def forward(self, value):
            output = super().forward(value)
            if self._residual_unet_spec is not None:
                for skip in self.rewiring_modules.skips.values():
                    output = output + skip.restore_grid(skip.capture_grid(value))
            return output

    from core.rewiring.provenance import checkpoint_metadata, read_checkpoint_metadata
    uniform = SkipModel()
    reduced = deepcopy(uniform)
    before = deepcopy(uniform.state_dict())
    value = torch.arange(640, dtype=torch.float32).reshape(1, 8, 8, 10)/640
    target = uniform.base(value).detach() + .1
    records = []
    for model, rate_multiplier in ((uniform, None), (reduced, multiplier)):
        phase = FitPhase("native_trajectory", 1, .03, [FitExample((value,), {}, target)], rate_multiplier)
        records.append(fit_teacher_modules(model, [phase], lambda net, a, k: net(*a, **k),
            lambda parameters, lr: torch.optim.SGD(parameters, lr=lr), source_sha256="1"*64,
            calibration_sha256="2"*64, sample_count=32, seed=718, clip_norm=1.))
    assert "skip_learning_rate_multiplier" not in records[0]["phases"][0]
    assert records[1]["phases"][0]["skip_learning_rate_multiplier"] == multiplier
    metadata = checkpoint_metadata(records[1], reduced._residual_unet_spec)
    assert read_checkpoint_metadata(metadata, reduced._residual_unet_spec) == records[1]
    nonzero = False
    for name, value in uniform.state_dict().items():
        if name.startswith("rewiring_modules.skips."):
            delta = value - before[name]
            nonzero |= bool(delta.abs().sum())
            torch.testing.assert_close(reduced.state_dict()[name]-before[name], delta*multiplier)
        else:
            torch.testing.assert_close(reduced.state_dict()[name], value, rtol=0, atol=0)
    assert nonzero


def test_skip_multiplier_rejects_optimizer_override_and_rolls_back():
    model = TinyModel()
    before = deepcopy(model.state_dict())
    def factory(parameters, lr):
        optimizer = torch.optim.SGD(parameters, lr=lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return optimizer
    phase = FitPhase("data_noising", 1, .03, [], .25)
    with pytest.raises(ValueError, match="declared parameter learning rates"):
        fit_teacher_modules(model, [phase], lambda net, a, k: net(*a, **k), factory,
            source_sha256="1"*64, calibration_sha256="2"*64, sample_count=32, seed=718, clip_norm=1.)
    assert not hasattr(model, "_hybrid_initialization")
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
