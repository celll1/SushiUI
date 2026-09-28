"""Composite supervised and router objectives for FoveaTag training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .router import router_diversity_loss, router_ranking_loss


def prediction_consistency_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    first_evidence_density: torch.Tensor,
    second_evidence_density: torch.Tensor,
    required_density: torch.Tensor,
) -> torch.Tensor:
    """Match predictions only where both view sets resolve the tag's evidence."""
    if first_logits.shape != second_logits.shape or first_logits.ndim != 2:
        raise ValueError("logits must have equal [batch, tags] shapes")
    for name, density in (
        ("first_evidence_density", first_evidence_density),
        ("second_evidence_density", second_evidence_density),
    ):
        if density.shape != first_logits.shape:
            raise ValueError(f"{name} must match logits")
    if required_density.shape not in ((first_logits.shape[1],), first_logits.shape):
        raise ValueError("required_density must have shape [tags] or [batch, tags]")
    required = required_density.to(device=first_logits.device, dtype=first_evidence_density.dtype)
    valid = (first_evidence_density >= required) & (second_evidence_density >= required)
    if not valid.any():
        return (first_logits.sum() + second_logits.sum()) * 0.0
    first_probability = first_logits.sigmoid()
    second_probability = second_logits.sigmoid()
    midpoint = (first_probability + second_probability) / 2.0
    first_kl = _bernoulli_kl(first_probability, midpoint)
    second_kl = _bernoulli_kl(second_probability, midpoint)
    return ((first_kl + second_kl) * 0.5)[valid].mean()


def _bernoulli_kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(first.dtype).eps
    first = first.clamp(epsilon, 1.0 - epsilon)
    second = second.clamp(epsilon, 1.0 - epsilon)
    return first * (first / second).log() + (1.0 - first) * ((1.0 - first) / (1.0 - second)).log()


@dataclass(frozen=True)
class TagTrainingLossOutput:
    loss: torch.Tensor
    classification: torch.Tensor
    consistency: torch.Tensor
    router_ranking: torch.Tensor
    router_diversity: torch.Tensor


class FoveaTagTrainingLoss(nn.Module):
    """Compose the existing masked tag criterion with optional Phase 4 losses."""

    def __init__(
        self,
        classification_criterion: nn.Module,
        *,
        consistency_weight: float = 0.1,
        router_weight: float = 1.0,
        diversity_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if min(consistency_weight, router_weight, diversity_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        self.classification_criterion = classification_criterion
        self.consistency_weight = consistency_weight
        self.router_weight = router_weight
        self.diversity_weight = diversity_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        *,
        second_logits: Optional[torch.Tensor] = None,
        first_evidence_density: Optional[torch.Tensor] = None,
        second_evidence_density: Optional[torch.Tensor] = None,
        required_density: Optional[torch.Tensor] = None,
        router_scores: Optional[torch.Tensor] = None,
        oracle_targets: Optional[torch.Tensor] = None,
        router_slot_scores: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> TagTrainingLossOutput:
        classification = self.classification_criterion(logits, labels, loss_mask)
        zero = logits.sum() * 0.0
        consistency = zero
        consistency_values = (
            second_logits,
            first_evidence_density,
            second_evidence_density,
            required_density,
        )
        if any(value is not None for value in consistency_values):
            if not all(value is not None for value in consistency_values):
                raise ValueError("all consistency inputs must be supplied together")
            consistency = prediction_consistency_loss(
                logits,
                second_logits,
                first_evidence_density,
                second_evidence_density,
                required_density,
            )

        ranking = zero
        if router_scores is not None or oracle_targets is not None:
            if router_scores is None or oracle_targets is None:
                raise ValueError("router_scores and oracle_targets must be supplied together")
            ranking = router_ranking_loss(router_scores, oracle_targets, candidate_mask)
        diversity = zero
        if router_slot_scores is not None:
            diversity = router_diversity_loss(router_slot_scores)
        loss = (
            classification
            + consistency * self.consistency_weight
            + ranking * self.router_weight
            + diversity * self.diversity_weight
        )
        return TagTrainingLossOutput(loss, classification, consistency, ranking, diversity)
