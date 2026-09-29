"""Training loop for the Phase-1 SigLIP2 encoder-swap control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

from .siglip2_control_data import SigLIP2ControlBatch


@dataclass(frozen=True)
class ControlEpochResult:
    batches: int
    optimizer_steps: int
    samples: int
    mean_loss: float


class SigLIP2ControlTrainer:
    """Optimize the control head while preserving existing masked tag labels."""

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
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and mixed_precision == torch.float16,
        )

    def train_epoch(
        self,
        batches: Iterable[Optional[SigLIP2ControlBatch]],
    ) -> ControlEpochResult:
        self.model.train()
        vision_encoder = getattr(self.model, "vision_encoder", None)
        if vision_encoder is not None and not any(
            parameter.requires_grad for parameter in vision_encoder.parameters()
        ):
            vision_encoder.eval()
        self.optimizer.zero_grad(set_to_none=True)
        batch_count = optimizer_steps = sample_count = 0
        loss_sum = 0.0
        pending = 0

        for batch in batches:
            if batch is None:
                continue
            batch = batch.to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision,
                enabled=self.device.type == "cuda" and self.mixed_precision is not None,
            ):
                logits = self.model(
                    batch.overview,
                    batch.local,
                    local_view_mask=batch.local_view_mask,
                )
                loss = self.criterion(logits, batch.labels, batch.loss_masks)
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
        return ControlEpochResult(batch_count, optimizer_steps, sample_count, mean_loss)

    def _optimizer_step(self) -> None:
        if self.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                self.max_grad_norm,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
