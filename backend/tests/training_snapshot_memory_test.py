"""Snapshot independence and unchanged streaming parameter-change metrics."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.parameter_change_tracker import ParameterChangeTracker


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_streamed_metrics_match_full_snapshots_without_parameter_aliases(dtype):
    model = torch.nn.Sequential(torch.nn.Linear(7, 5), torch.nn.Linear(5, 3)).to(dtype)
    tracker = ParameterChangeTracker({"unet": model}, interval=2)
    reference = [p.detach().to(dtype=torch.float16, copy=True) for p in model.parameters()]
    previous = [p.clone() for p in reference]
    for saved, shared, parameter in zip(tracker._reference["unet"], tracker._prev["unet"], model.parameters()):
        assert saved.data_ptr() == shared.data_ptr()
        assert saved.data_ptr() != parameter.data_ptr()
    assert tracker.compute(1) is None
    for step in (2, 4, 6):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.03125)
        current = [p.detach().to(dtype=torch.float16, copy=True) for p in model.parameters()]
        expected = {"update_norm": {"unet": tracker._delta_norm_sq(current, previous)**.5},
                    "cumulative_drift": {"unet": tracker._delta_norm_sq(current, reference)**.5 /
                                          tracker._norm_sq(reference)**.5}}
        assert tracker.compute(step) == expected
        for saved, initial in zip(tracker._reference["unet"], reference):
            torch.testing.assert_close(saved, initial, rtol=0, atol=0)
        previous = current


def test_ema_snapshot_is_independent_for_cpu_fp32_and_bf16():
    from core.training.base_trainer import BaseTrainer
    for dtype in (torch.float32, torch.bfloat16):
        parameter = torch.nn.Parameter(torch.arange(12, dtype=dtype).reshape(3, 4).t())
        trainer = SimpleNamespace(config={"use_ema": True}, optimizer=torch.optim.AdamW([parameter]),
                                  log_prefix="test", _build_ema_param_name_map=lambda: {id(parameter): "weight"})
        BaseTrainer._setup_ema(trainer)
        saved = trainer.ema_shadow["weight"]
        expected = parameter.detach().float().clone()
        with torch.no_grad():
            parameter.add_(1)
        assert saved.data_ptr() != parameter.data_ptr()
        torch.testing.assert_close(saved, expected, rtol=0, atol=0)


def test_tracker_preserves_dynamic_trainability_metrics():
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.full((8,), float(i + 1))) for i in range(3)])
    tracker = ParameterChangeTracker({"unet": model}, interval=1)
    reference = [p.detach().half().clone() for p in model]
    previous = [p.clone() for p in reference]
    for disabled in (True, False):
        model[0].requires_grad_(not disabled)
        current = [p.detach().half().clone() for p in model if p.requires_grad]
        expected = {"update_norm": {"unet": tracker._delta_norm_sq(current, previous)**.5},
                    "cumulative_drift": {"unet": tracker._delta_norm_sq(current, reference)**.5 /
                                          tracker._norm_sq(reference)**.5}}
        assert tracker.compute(1) == expected
        previous = current


def test_ming_pending_gradients_remain_independent_after_capture_and_restore():
    from core.models.ming_image_01.training_resume import capture_runtime, restore_runtime
    parameter = torch.nn.Parameter(torch.ones(16))
    parameter.grad = torch.arange(16, dtype=torch.float32)
    trainer = SimpleNamespace(device="cpu", optimizer=torch.optim.AdamW([parameter]),
        fused_optimizer_groups=None, run_seed=13, _grad_accum_steps=2, use_grad_scaler=False,
        _build_ema_param_name_map=lambda: {id(parameter): "weight"})
    runtime = capture_runtime(trainer)
    expected = parameter.grad.clone()
    parameter.grad.add_(2)
    torch.testing.assert_close(runtime["gradients"]["weight"], expected, rtol=0, atol=0)
    trainer._ming_pending_runtime = runtime
    restore_runtime(trainer)
    torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    parameter.grad.add_(3)
    torch.testing.assert_close(runtime["gradients"]["weight"], expected, rtol=0, atol=0)


@pytest.mark.parametrize("use_ema", [False, True])
def test_vae_checkpoint_snapshot_does_not_alias_live_weights(use_ema):
    from core.training.vae.vae_trainer import VaeTrainer
    parameter = torch.nn.Parameter(torch.arange(9, dtype=torch.float32).reshape(3, 3).t())
    source = parameter.detach().clone() if use_ema else parameter
    trainer = SimpleNamespace(ema={"weight": source} if use_ema else None,
                              trainable_names=["weight"], trainable_params=[parameter])
    saved = VaeTrainer._trainable_state_dict(trainer, use_ema=use_ema)
    expected = source.detach().clone()
    with torch.no_grad():
        source.add_(1)
    torch.testing.assert_close(saved["weight"], expected, rtol=0, atol=0)
    assert saved["weight"].data_ptr() != source.data_ptr()
