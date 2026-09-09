"""REPA's chart scalar must not synchronize CUDA before backward is enqueued."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training import repa as repa_module  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402


def test_apply_repa_loss_hands_the_scalar_to_the_trainer(monkeypatch):
    rloss = torch.tensor(0.25, requires_grad=True)
    captured = []
    trainer = SimpleNamespace(
        repa_encoder=object(),
        repa_projector=object(),
        repa_weight=0.5,
        repa_size=16,
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        _defer_repa_loss_metric=captured.append,
        log_extra_metric=lambda *_: (_ for _ in ()).throw(
            AssertionError("the production path logged REPA before backward")),
    )
    monkeypatch.setattr(repa_module, "encode_repa_targets",
                        lambda *_args, **_kwargs: torch.zeros(1))
    monkeypatch.setattr(repa_module, "repa_loss",
                        lambda *_args, **_kwargs: rloss)

    result = repa_module.apply_repa_loss(
        trainer, torch.tensor(1.0), torch.zeros(1), torch.zeros(1), 1, 1)

    assert result.item() == 1.125
    assert captured == [rloss]


def test_deferred_scalar_is_logged_and_released():
    logged = []
    trainer = SimpleNamespace(
        log_extra_metric=lambda name, value: logged.append((name, value)))
    scalar = torch.tensor(0.375, requires_grad=True)

    BaseTrainer._defer_repa_loss_metric(trainer, scalar)

    assert logged == []
    assert trainer._pending_repa_loss_metric.grad_fn is None
    wait_s = BaseTrainer._flush_repa_loss_metric_after_backward(trainer)
    assert logged == [("repa_loss", 0.375)]
    assert trainer._pending_repa_loss_metric is None
    assert wait_s >= 0.0


def test_generic_path_reads_repa_after_its_existing_loss_sync():
    source = inspect.getsource(BaseTrainer._execute_forward_backward)

    loss_sync = source.index("loss_value = loss.item()")
    controlnet_read = source.index("self._flush_deferred_controlnet_metrics()")
    generic_read = source.index("self._flush_deferred_extra_metrics()")
    repa_read = source.index("self._flush_repa_loss_metric_after_backward(")
    assert loss_sync < controlnet_read < generic_read < repa_read


def test_generic_deferred_scalar_is_detached_then_logged():
    trainer = SimpleNamespace(
        _pending_extra_metrics={},
        _extra_metrics={},
        log_extra_metric=None,
    )
    trainer.log_extra_metric = lambda name, value: BaseTrainer.log_extra_metric(
        trainer, name, value)
    scalar = torch.tensor(0.625, requires_grad=True)

    BaseTrainer.defer_extra_metric(trainer, "component_loss", scalar)

    assert trainer._extra_metrics == {}
    assert trainer._pending_extra_metrics["component_loss"].grad_fn is None
    BaseTrainer._flush_deferred_extra_metrics(trainer)
    assert trainer._pending_extra_metrics == {}
    assert trainer._extra_metrics == {"component_loss": 0.625}


def test_controlnet_diagnostics_materialize_after_backward():
    trainer = SimpleNamespace(
        _last_gen_region_loss=torch.tensor(0.75),
        _last_seam_ring_loss=(torch.tensor(0.5), torch.tensor(2.0)),
        _last_loss_vs_t={
            "t": torch.tensor([0.1, 0.2]),
            "eps_known": torch.tensor([1.0, float("nan")]),
        },
    )

    BaseTrainer._flush_deferred_controlnet_metrics(trainer)

    assert trainer._last_gen_region_loss == 0.75
    assert trainer._last_seam_ring_loss == 0.5
    assert trainer._last_loss_vs_t["t"] == pytest.approx([0.1, 0.2])
    assert trainer._last_loss_vs_t["eps_known"] == [1.0, None]


def test_controlnet_step_has_no_pre_backward_scalar_read():
    source = inspect.getsource(BaseTrainer.train_step_controlnet)

    assert ".item()" not in source


def test_controlnet_empty_seam_ring_stays_unreported():
    trainer = SimpleNamespace(
        _last_gen_region_loss=None,
        _last_seam_ring_loss=(torch.tensor(0.0), torch.tensor(0.0)),
        _last_loss_vs_t=None,
    )

    BaseTrainer._flush_deferred_controlnet_metrics(trainer)

    assert trainer._last_seam_ring_loss is None
