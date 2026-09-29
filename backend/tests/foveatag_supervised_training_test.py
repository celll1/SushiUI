"""CPU contracts for the scratch FoveaTag supervised training loop."""

import torch
from torch import nn

from core.tagger.foveatag.batch import FoveaViewTensorBatch
from core.tagger.foveatag.supervised_data import FoveaSupervisedBatch
from core.tagger.foveatag.supervised_training import FoveaSupervisedTrainer
from core.tagger.foveatag.training import FoveaTagTrainingLoss


class _Fovea(nn.Module):
    def __init__(self):
        super().__init__()
        self.overview_projection = nn.Linear(2, 2)
        self.local_projection = nn.Linear(2, 2)

    def forward(
        self,
        overview_images,
        overview_geometry,
        local_images,
        local_geometry,
        *,
        local_view_mask,
    ):
        logits = self.overview_projection(overview_images)
        if local_images.shape[1]:
            local = self.local_projection(local_images).mean(dim=1)
            logits = logits + local
        return logits


class _MaskedBCE(nn.Module):
    def forward(self, logits, labels, loss_mask):
        values = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        return (values * loss_mask).sum() / loss_mask.sum().clamp_min(1)


def _batch(value: float, *, local_views: int = 1) -> FoveaSupervisedBatch:
    return FoveaSupervisedBatch(
        views=FoveaViewTensorBatch(
            overview_images=torch.tensor([[value, 1.0]]),
            overview_geometry=torch.zeros(1, 10),
            local_images=torch.full((1, local_views, 2), value),
            local_geometry=torch.zeros(1, local_views, 10),
            local_view_mask=torch.ones(1, local_views, dtype=torch.bool),
        ),
        labels=torch.tensor([[1.0, 0.0]]),
        loss_masks=torch.ones(1, 2),
    )


def test_supervised_trainer_accumulates_and_updates_both_paths():
    model = _Fovea()
    overview_before = model.overview_projection.weight.detach().clone()
    local_before = model.local_projection.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = FoveaSupervisedTrainer(
        model,
        _MaskedBCE(),
        optimizer,
        device="cpu",
        mixed_precision=None,
        gradient_accumulation_steps=2,
    )

    result = trainer.train_epoch([_batch(1.0), None, _batch(2.0), _batch(3.0)])

    assert result.batches == 3
    assert result.optimizer_steps == 2
    assert result.samples == 3
    assert result.mean_loss > 0
    assert not torch.equal(overview_before, model.overview_projection.weight)
    assert not torch.equal(local_before, model.local_projection.weight)


def test_supervised_trainer_supports_overview_only_and_composite_loss():
    model = _Fovea()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = FoveaTagTrainingLoss(_MaskedBCE())
    trainer = FoveaSupervisedTrainer(
        model,
        criterion,
        optimizer,
        device="cpu",
        mixed_precision=None,
    )

    result = trainer.train_epoch([_batch(1.0, local_views=0)])

    assert result.optimizer_steps == 1
    assert result.mean_loss > 0


def test_supervised_trainer_validates_masks_and_restores_mode():
    model = _Fovea()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = FoveaSupervisedTrainer(
        model,
        _MaskedBCE(),
        optimizer,
        device="cpu",
        mixed_precision=None,
    )
    trainer.train_epoch([_batch(1.0)])

    result = trainer.validate_epoch(
        [_batch(1.0), None, _batch(2.0)],
        thresholds=(0.25, 0.5, 0.75),
    )

    assert result.batches == 2
    assert result.samples == 2
    assert result.known_labels == 4
    assert result.active_tags == 1
    assert result.mean_loss > 0
    assert 0 <= result.macro_f1 <= 1
    assert result.threshold in (0.25, 0.5, 0.75)
    assert model.training


def test_supervised_trainer_steps_scheduler_with_optimizer():
    model = _Fovea()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = FoveaSupervisedTrainer(
        model,
        _MaskedBCE(),
        optimizer,
        device="cpu",
        mixed_precision=None,
        scheduler=scheduler,
    )

    trainer.train_epoch([_batch(1.0), _batch(2.0)])

    assert scheduler.last_epoch == 2
    assert optimizer.param_groups[0]["lr"] == 0.025
