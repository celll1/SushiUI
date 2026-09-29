"""Bounded dense-oracle evaluation and generic router optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .encoder import FoveaEncoderOutput
from .model import FoveaTagger
from .router import (
    EvidenceRouter,
    dense_oracle_targets,
    router_diversity_loss,
    router_ranking_loss,
)


@dataclass(frozen=True)
class DenseOracleBatch:
    overview: FoveaEncoderOutput
    overview_logits: torch.Tensor
    candidate_logits: torch.Tensor


@dataclass(frozen=True)
class RouterTrainResult:
    loss: float
    ranking: float
    diversity: float
    mean_positive_utility: float


@torch.no_grad()
def evaluate_dense_candidates(
    model: FoveaTagger,
    overview_images: torch.Tensor,
    overview_geometry: torch.Tensor,
    candidate_images: torch.Tensor,
    candidate_geometry: torch.Tensor,
    *,
    candidate_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
) -> DenseOracleBatch:
    """Evaluate candidates independently while encoding the overview only once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if candidate_images.ndim != 5 or candidate_images.shape[2] != 3:
        raise ValueError("candidate_images must have shape [batch, candidates, 3, height, width]")
    batch, candidates = candidate_images.shape[:2]
    if candidates == 0:
        raise ValueError("at least one dense candidate is required")
    geometry_dim = model.encoder.config.geometry_dim
    if overview_images.ndim != 4 or overview_images.shape[0] != batch:
        raise ValueError("overview_images must have shape [batch, 3, height, width]")
    if overview_geometry.shape != (batch, geometry_dim):
        raise ValueError(f"overview_geometry must have shape [batch, {geometry_dim}]")
    if candidate_geometry.shape != (batch, candidates, geometry_dim):
        raise ValueError(
            f"candidate_geometry must have shape [batch, candidates, {geometry_dim}]"
        )
    valid = _candidate_mask(candidate_mask, batch, candidates, candidate_images.device)
    was_training = model.training
    model.eval()
    try:
        overview = model.encoder(overview_images, overview_geometry)
        empty_images = candidate_images.new_empty(
            (batch, 0, 3, candidate_images.shape[-2], candidate_images.shape[-1])
        )
        empty_geometry = candidate_geometry.new_empty((batch, 0, geometry_dim))
        overview_logits = model.classify_encoded_overview(
            overview,
            empty_images,
            empty_geometry,
            local_view_mask=valid[:, :0],
        )
        chunks = []
        for start in range(0, candidates, chunk_size):
            stop = min(start + chunk_size, candidates)
            count = stop - start
            repeated_overview = _repeat_overview(overview, count)
            images = candidate_images[:, start:stop].reshape(
                batch * count, 1, *candidate_images.shape[2:]
            )
            geometry = candidate_geometry[:, start:stop].reshape(
                batch * count, 1, geometry_dim
            )
            mask = valid[:, start:stop].reshape(batch * count, 1)
            logits = model.classify_encoded_overview(
                repeated_overview,
                images,
                geometry,
                local_view_mask=mask,
            )
            chunks.append(logits.reshape(batch, count, -1))
        candidate_logits = (
            torch.cat(chunks, dim=1)
            if chunks else overview_logits.new_empty((batch, 0, overview_logits.shape[1]))
        )
        return DenseOracleBatch(
            overview=_detach_overview(overview),
            overview_logits=overview_logits.detach(),
            candidate_logits=candidate_logits.detach(),
        )
    finally:
        model.train(was_training)


def train_router_batch(
    router: EvidenceRouter,
    optimizer: torch.optim.Optimizer,
    oracle: DenseOracleBatch,
    candidate_boxes: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_mask: Optional[torch.Tensor] = None,
    candidate_mask: Optional[torch.Tensor] = None,
    diversity_weight: float = 0.05,
) -> RouterTrainResult:
    """Optimize vocabulary-independent proposal scores against detached utility."""
    if diversity_weight < 0:
        raise ValueError("diversity_weight must be non-negative")
    batch, candidates, tags = oracle.candidate_logits.shape
    if candidate_boxes.shape != (batch, candidates, 4):
        raise ValueError("candidate_boxes must have shape [batch, candidates, 4]")
    if labels.shape != (batch, tags):
        raise ValueError("labels must match dense-oracle tag logits")
    device = oracle.candidate_logits.device
    if (
        oracle.overview_logits.device != device
        or oracle.overview.local_map.device != device
        or candidate_boxes.device != device
        or labels.device != device
    ):
        raise ValueError("oracle, candidates, and labels must be on the same device")
    valid = _candidate_mask(candidate_mask, batch, candidates, candidate_boxes.device)
    targets = dense_oracle_targets(
        oracle.candidate_logits,
        oracle.overview_logits,
        labels > 0,
        loss_mask=loss_mask,
        candidate_mask=valid,
    )
    optimizer.zero_grad(set_to_none=True)
    routing = router(
        oracle.overview.local_map.detach(),
        oracle.overview.local_coordinates.detach(),
        candidate_boxes,
        valid,
    )
    ranking = router_ranking_loss(routing.scores, targets, valid)
    diversity = router_diversity_loss(routing.slot_scores)
    loss = ranking + diversity * diversity_weight
    loss.backward()
    optimizer.step()
    positive = targets[targets > 0]
    return RouterTrainResult(
        loss=float(loss.detach()),
        ranking=float(ranking.detach()),
        diversity=float(diversity.detach()),
        mean_positive_utility=float(positive.mean()) if positive.numel() else 0.0,
    )


def _candidate_mask(
    mask: Optional[torch.Tensor], batch: int, candidates: int, device: torch.device
) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch, candidates), dtype=torch.bool, device=device)
    if mask.shape != (batch, candidates):
        raise ValueError(f"candidate_mask must have shape [{batch}, {candidates}]")
    return mask.to(device=device, dtype=torch.bool)


def _repeat_overview(overview: FoveaEncoderOutput, count: int) -> FoveaEncoderOutput:
    return FoveaEncoderOutput(**{
        name: getattr(overview, name).repeat_interleave(count, dim=0)
        for name in overview.__dataclass_fields__
    })


def _detach_overview(overview: FoveaEncoderOutput) -> FoveaEncoderOutput:
    return FoveaEncoderOutput(**{
        name: getattr(overview, name).detach()
        for name in overview.__dataclass_fields__
    })
