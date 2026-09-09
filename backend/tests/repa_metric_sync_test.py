"""REPA's chart scalar must not synchronize CUDA before backward is enqueued."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

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
    BaseTrainer._flush_repa_loss_metric_after_backward(trainer)
    assert logged == [("repa_loss", 0.375)]
    assert trainer._pending_repa_loss_metric is None


def test_generic_path_reads_repa_after_its_existing_loss_sync():
    source = inspect.getsource(BaseTrainer._execute_forward_backward)

    loss_sync = source.index("loss_value = loss.item()")
    repa_read = source.index("self._flush_repa_loss_metric_after_backward()")
    assert loss_sync < repa_read
