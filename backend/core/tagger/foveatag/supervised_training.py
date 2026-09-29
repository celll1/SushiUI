"""Production training loop for scratch FoveaTag supervised stages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from torch import nn

from .supervised_data import FoveaSupervisedBatch


@dataclass(frozen=True)
class FoveaEpochResult:
    batches: int
    optimizer_steps: int
    samples: int
    mean_loss: float


@dataclass(frozen=True)
class FoveaValidationResult:
    batches: int
    samples: int
    known_labels: int
    active_tags: int
    mean_loss: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    threshold: float


class FoveaSupervisedTrainer:
    """Optimize all scratch-model components from coordinate-aware view batches."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device | str,
        mixed_precision: Optional[torch.dtype] = torch.bfloat16,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        scheduler: Optional[object] = None,
    ) -> None:
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when supplied")
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and mixed_precision == torch.float16,
        )

    def train_epoch(
        self,
        batches: Iterable[Optional[FoveaSupervisedBatch]],
    ) -> FoveaEpochResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch_count = optimizer_steps = sample_count = 0
        loss_sum = 0.0
        pending = 0

        for batch in batches:
            if batch is None:
                continue
            batch = batch.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self._forward(batch)
                loss = _loss_tensor(self.criterion(logits, batch.labels, batch.loss_masks))
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                pending = 0
                continue

            self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
            pending += 1
            batch_count += 1
            samples = int(batch.labels.shape[0])
            sample_count += samples
            loss_sum += float(loss.detach()) * samples
            if pending == self.gradient_accumulation_steps:
                self._optimizer_step()
                optimizer_steps += 1
                pending = 0

        if pending:
            correction = self.gradient_accumulation_steps / pending
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            self._optimizer_step()
            optimizer_steps += 1

        mean_loss = loss_sum / sample_count if sample_count else math.nan
        return FoveaEpochResult(batch_count, optimizer_steps, sample_count, mean_loss)

    def _forward(self, batch: FoveaSupervisedBatch) -> torch.Tensor:
        views = batch.views
        return self.model(
            views.overview_images,
            views.overview_geometry,
            views.local_images,
            views.local_geometry,
            local_view_mask=views.local_view_mask,
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision,
            enabled=self.device.type == "cuda" and self.mixed_precision is not None,
        )

    def _optimizer_step(self) -> None:
        if self.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                self.max_grad_norm,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.inference_mode()
    def validate_epoch(
        self,
        batches: Iterable[Optional[FoveaSupervisedBatch]],
        *,
        thresholds: Sequence[float] = tuple(value / 20 for value in range(1, 20)),
    ) -> FoveaValidationResult:
        if not thresholds or any(not 0 < value < 1 for value in thresholds):
            raise ValueError("validation thresholds must be within (0, 1)")
        threshold_values = tuple(float(value) for value in thresholds)
        was_training = self.model.training
        self.model.eval()
        batch_count = sample_count = known_count = 0
        loss_total = 0.0
        true_positive = false_positive = false_negative = positive_support = None
        try:
            for batch in batches:
                if batch is None:
                    continue
                batch = batch.to(self.device, non_blocking=True)
                with self._autocast():
                    logits = self._forward(batch)
                    loss = _loss_tensor(self.criterion(logits, batch.labels, batch.loss_masks))
                if not torch.isfinite(loss):
                    continue

                known = batch.loss_masks > 0
                labels = batch.labels >= 0.5
                probabilities = logits.sigmoid()
                threshold_tensor = probabilities.new_tensor(threshold_values).view(-1, 1, 1)
                predicted = probabilities.unsqueeze(0) >= threshold_tensor
                known_expanded = known.unsqueeze(0)
                labels_expanded = labels.unsqueeze(0)
                batch_tp = (predicted & labels_expanded & known_expanded).sum(dim=1).cpu()
                batch_fp = (predicted & ~labels_expanded & known_expanded).sum(dim=1).cpu()
                batch_fn = (~predicted & labels_expanded & known_expanded).sum(dim=1).cpu()
                batch_support = (labels & known).sum(dim=0).cpu()
                if true_positive is None:
                    true_positive = batch_tp
                    false_positive = batch_fp
                    false_negative = batch_fn
                    positive_support = batch_support
                else:
                    true_positive += batch_tp
                    false_positive += batch_fp
                    false_negative += batch_fn
                    positive_support += batch_support

                batch_count += 1
                samples = int(batch.labels.shape[0])
                sample_count += samples
                known_count += int(known.sum().item())
                loss_total += float(loss) * samples
        finally:
            self.model.train(was_training)

        if true_positive is None or known_count == 0:
            raise RuntimeError("validation produced no finite labeled batches")
        metrics = _threshold_metrics(
            true_positive,
            false_positive,
            false_negative,
            positive_support,
        )
        best_index = max(
            range(len(threshold_values)),
            key=lambda index: (metrics["f1"][index], -threshold_values[index]),
        )
        return FoveaValidationResult(
            batches=batch_count,
            samples=sample_count,
            known_labels=known_count,
            active_tags=int((positive_support > 0).sum().item()),
            mean_loss=loss_total / sample_count,
            macro_f1=float(metrics["f1"][best_index]),
            macro_precision=float(metrics["precision"][best_index]),
            macro_recall=float(metrics["recall"][best_index]),
            threshold=threshold_values[best_index],
        )


def _loss_tensor(value: object) -> torch.Tensor:
    loss = value if torch.is_tensor(value) else getattr(value, "loss", None)
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise TypeError("criterion must return a scalar tensor or an object with scalar .loss")
    return loss


def _threshold_metrics(
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
    positive_support: torch.Tensor,
) -> dict[str, torch.Tensor]:
    active = positive_support > 0
    if not active.any():
        zeros = torch.zeros(true_positive.shape[0], dtype=torch.float64)
        return {"f1": zeros, "precision": zeros.clone(), "recall": zeros.clone()}
    tp = true_positive[:, active].to(torch.float64)
    fp = false_positive[:, active].to(torch.float64)
    fn = false_negative[:, active].to(torch.float64)
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float64).eps)
    return {
        "f1": f1.mean(dim=1),
        "precision": precision.mean(dim=1),
        "recall": recall.mean(dim=1),
    }
