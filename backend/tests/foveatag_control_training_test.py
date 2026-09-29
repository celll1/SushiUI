"""CPU contracts for the Phase-1 control training loop."""

import torch
from torch import nn

from core.tagger.foveatag.control_training import SigLIP2ControlTrainer
from core.tagger.foveatag.siglip2_control import SigLIP2ViewInputs
from core.tagger.foveatag.siglip2_control_data import SigLIP2ControlBatch


class _Control(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Linear(2, 2)
        self.vision_encoder.requires_grad_(False)
        self.head = nn.Linear(2, 2)

    def forward(self, overview, local, *, local_view_mask):
        return self.head(overview.pixel_values)


class _MaskedBCE(nn.Module):
    def forward(self, logits, labels, loss_mask):
        values = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        return (values * loss_mask).sum() / loss_mask.sum().clamp_min(1)


def _batch(value: float) -> SigLIP2ControlBatch:
    return SigLIP2ControlBatch(
        overview=SigLIP2ViewInputs(pixel_values=torch.tensor([[value, 1.0]])),
        local=SigLIP2ViewInputs(pixel_values=torch.empty(1, 0, 2)),
        local_view_mask=torch.empty(1, 0, dtype=torch.bool),
        labels=torch.tensor([[1.0, 0.0]]),
        loss_masks=torch.ones(1, 2),
    )


def test_control_trainer_accumulates_and_keeps_frozen_encoder_in_eval_mode():
    model = _Control()
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    trainer = SigLIP2ControlTrainer(
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
    assert not model.vision_encoder.training
    assert not torch.equal(before, model.head.weight)


def test_control_trainer_validates_masks_and_restores_module_modes():
    model = _Control()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    trainer = SigLIP2ControlTrainer(
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
    assert not model.vision_encoder.training


def test_control_trainer_steps_scheduler_with_optimizer():
    model = _Control()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = SigLIP2ControlTrainer(
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
