"""Bounded dense-oracle evaluation and generic router optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch import nn

from .benchmark import evaluate_dense_oracle_recovery
from .encoder import FoveaEncoderOutput
from .model import FoveaTagger
from .router_data import FoveaRouterBatch
from .router import (
    EvidenceRouter,
    RouterOutput,
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


@dataclass(frozen=True)
class RouterObjectiveOutput:
    loss: torch.Tensor
    ranking: torch.Tensor
    diversity: torch.Tensor
    targets: torch.Tensor
    routing: RouterOutput


@dataclass(frozen=True)
class RouterValidationResult:
    samples: int
    candidates_per_sample: int
    budget: int
    mean_utility_recovery: float
    mean_oracle_topk_recall: float
    random_mean_utility_recovery: float
    random_mean_oracle_topk_recall: float


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
    device = overview_images.device
    valid = _candidate_mask(candidate_mask, batch, candidates, device)
    was_training = model.training
    model.eval()
    try:
        overview = model.encoder(overview_images, overview_geometry)
        empty_images = overview_images.new_empty(
            (batch, 0, 3, candidate_images.shape[-2], candidate_images.shape[-1])
        )
        empty_geometry = overview_geometry.new_empty((batch, 0, geometry_dim))
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
            images = candidate_images[:, start:stop].to(
                device=device,
                dtype=overview_images.dtype,
                non_blocking=True,
            ).reshape(
                batch * count, 1, *candidate_images.shape[2:]
            )
            geometry = candidate_geometry[:, start:stop].to(
                device=device,
                non_blocking=True,
            ).reshape(
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
    objective = router_objective(
        router,
        oracle,
        candidate_boxes,
        labels,
        loss_mask=loss_mask,
        candidate_mask=candidate_mask,
        diversity_weight=diversity_weight,
    )
    optimizer.zero_grad(set_to_none=True)
    objective.loss.backward()
    optimizer.step()
    positive = objective.targets[objective.targets > 0]
    return RouterTrainResult(
        loss=float(objective.loss.detach()),
        ranking=float(objective.ranking.detach()),
        diversity=float(objective.diversity.detach()),
        mean_positive_utility=float(positive.mean()) if positive.numel() else 0.0,
    )


def router_objective(
    router: EvidenceRouter,
    oracle: DenseOracleBatch,
    candidate_boxes: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_mask: Optional[torch.Tensor] = None,
    candidate_mask: Optional[torch.Tensor] = None,
    diversity_weight: float = 0.05,
) -> RouterObjectiveOutput:
    """Build detached dense-oracle targets and differentiable router losses."""
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
    routing = router(
        oracle.overview.local_map.detach(),
        oracle.overview.local_coordinates.detach(),
        candidate_boxes,
        valid,
    )
    ranking = router_ranking_loss(routing.scores, targets, valid)
    diversity = router_diversity_loss(routing.slot_scores)
    loss = ranking + diversity * diversity_weight
    return RouterObjectiveOutput(
        loss=loss,
        ranking=ranking,
        diversity=diversity,
        targets=targets,
        routing=routing,
    )


class FoveaRouterTrainer:
    """Optimize and validate a generic router against a frozen stage-2 tagger."""

    def __init__(
        self,
        model: FoveaTagger,
        router: EvidenceRouter,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device | str,
        mixed_precision: Optional[torch.dtype] = torch.bfloat16,
        oracle_chunk_size: int = 16,
        diversity_weight: float = 0.05,
        max_grad_norm: Optional[float] = 1.0,
        scheduler: Optional[Any] = None,
    ) -> None:
        if oracle_chunk_size <= 0:
            raise ValueError("oracle_chunk_size must be positive")
        if diversity_weight < 0:
            raise ValueError("diversity_weight must be non-negative")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when supplied")
        self.model = model
        self.router = router
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.mixed_precision = mixed_precision
        self.oracle_chunk_size = oracle_chunk_size
        self.diversity_weight = diversity_weight
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and mixed_precision == torch.float16,
        )

    def train_batch(self, batch: FoveaRouterBatch) -> Optional[RouterTrainResult]:
        self.router.train()
        batch = batch.stage_for_routing(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            oracle = self._oracle(batch)
            objective = self._objective(batch, oracle)
        if not torch.isfinite(objective.loss):
            self.optimizer.zero_grad(set_to_none=True)
            return None
        self.scaler.scale(objective.loss).backward()
        if self.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.router.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        positive = objective.targets[objective.targets > 0]
        return RouterTrainResult(
            loss=float(objective.loss.detach()),
            ranking=float(objective.ranking.detach()),
            diversity=float(objective.diversity.detach()),
            mean_positive_utility=(
                float(positive.mean()) if positive.numel() else 0.0
            ),
        )

    @torch.inference_mode()
    def validate(
        self,
        batches: Iterable[Optional[FoveaRouterBatch]],
        *,
        budget: int,
        random_seed: int = 0,
    ) -> RouterValidationResult:
        if budget <= 0:
            raise ValueError("budget must be positive")
        was_training = self.router.training
        self.router.eval()
        scores = []
        targets = []
        masks = []
        try:
            for batch in batches:
                if batch is None:
                    continue
                batch = batch.stage_for_routing(self.device, non_blocking=True)
                with self._autocast():
                    oracle = self._oracle(batch)
                    objective = self._objective(batch, oracle)
                scores.append(objective.routing.scores.detach().float().cpu())
                targets.append(objective.targets.detach().float().cpu())
                masks.append(batch.views.local_view_mask.detach().cpu())
        finally:
            self.router.train(was_training)
        if not scores:
            raise ValueError("router validation received no samples")
        learned_scores = torch.cat(scores)
        oracle_targets = torch.cat(targets)
        candidate_mask = torch.cat(masks)
        if budget > learned_scores.shape[1]:
            raise ValueError("budget exceeds validation candidate count")
        learned = evaluate_dense_oracle_recovery(
            learned_scores,
            oracle_targets,
            k=budget,
            candidate_mask=candidate_mask,
        )
        generator = torch.Generator().manual_seed(random_seed)
        random_scores = torch.rand(learned_scores.shape, generator=generator)
        random_baseline = evaluate_dense_oracle_recovery(
            random_scores,
            oracle_targets,
            k=budget,
            candidate_mask=candidate_mask,
        )
        return RouterValidationResult(
            samples=learned_scores.shape[0],
            candidates_per_sample=learned_scores.shape[1],
            budget=budget,
            mean_utility_recovery=float(learned["mean_utility_recovery"] or 0.0),
            mean_oracle_topk_recall=float(learned["mean_oracle_topk_recall"] or 0.0),
            random_mean_utility_recovery=float(
                random_baseline["mean_utility_recovery"] or 0.0
            ),
            random_mean_oracle_topk_recall=float(
                random_baseline["mean_oracle_topk_recall"] or 0.0
            ),
        )

    def _oracle(self, batch: FoveaRouterBatch) -> DenseOracleBatch:
        views = batch.views
        return evaluate_dense_candidates(
            self.model,
            views.overview_images,
            views.overview_geometry,
            views.local_images,
            views.local_geometry,
            candidate_mask=views.local_view_mask,
            chunk_size=self.oracle_chunk_size,
        )

    def _objective(
        self, batch: FoveaRouterBatch, oracle: DenseOracleBatch
    ) -> RouterObjectiveOutput:
        return router_objective(
            self.router,
            oracle,
            batch.candidate_boxes,
            batch.labels,
            loss_mask=batch.loss_masks,
            candidate_mask=batch.views.local_view_mask,
            diversity_weight=self.diversity_weight,
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision,
            enabled=self.device.type == "cuda" and self.mixed_precision is not None,
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
