"""Stage-4 routed tagger fine-tuning with retained random coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .batch import FoveaViewTensorBatch
from .model import FoveaTagger
from .router import EvidenceRouter, select_training_proposals
from .router_data import FoveaRouterBatch
from .router_training import evaluate_dense_candidates, router_objective
from .supervised_training import _loss_tensor


@dataclass(frozen=True)
class FoveaJointTrainResult:
    loss: float
    classification: float
    router_ranking: float
    router_diversity: float
    oracle_used: bool
    selected_views: int


class FoveaJointTrainer:
    """Fine-tune the tagger on routed views and periodically refresh the router."""

    def __init__(
        self,
        model: FoveaTagger,
        router: EvidenceRouter,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device | str,
        selection_budget: int,
        random_count: int,
        oracle_interval: int,
        oracle_chunk_size: int = 16,
        router_weight: float = 1.0,
        diversity_weight: float = 0.05,
        nms_iou: float = 0.5,
        mixed_precision: Optional[torch.dtype] = torch.bfloat16,
        max_grad_norm: Optional[float] = 1.0,
        scheduler: Optional[object] = None,
    ) -> None:
        if selection_budget <= 0 or not 0 <= random_count <= selection_budget:
            raise ValueError("selection budget and random coverage are invalid")
        if oracle_interval <= 0 or oracle_chunk_size <= 0:
            raise ValueError("oracle interval and chunk size must be positive")
        if min(router_weight, diversity_weight) < 0 or not 0 <= nms_iou <= 1:
            raise ValueError("router loss weights or NMS threshold are invalid")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when supplied")
        self.model = model
        self.router = router
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.selection_budget = selection_budget
        self.random_count = random_count
        self.oracle_interval = oracle_interval
        self.oracle_chunk_size = oracle_chunk_size
        self.router_weight = router_weight
        self.diversity_weight = diversity_weight
        self.nms_iou = nms_iou
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and mixed_precision == torch.float16,
        )

    def train_batch(
        self,
        batch: FoveaRouterBatch,
        *,
        global_step: int,
        random_seed: int,
    ) -> Optional[FoveaJointTrainResult]:
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        self.model.train()
        self.router.train()
        batch = batch.to(self.device, non_blocking=True)
        use_oracle = global_step % self.oracle_interval == 0
        oracle = None
        if use_oracle:
            with self._autocast():
                oracle = evaluate_dense_candidates(
                    self.model,
                    batch.views.overview_images,
                    batch.views.overview_geometry,
                    batch.views.local_images,
                    batch.views.local_geometry,
                    candidate_mask=batch.views.local_view_mask,
                    chunk_size=self.oracle_chunk_size,
                )

        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            overview = self.model.encoder(
                batch.views.overview_images,
                batch.views.overview_geometry,
            )
            routing = self.router(
                overview.local_map.detach(),
                overview.local_coordinates.detach(),
                batch.candidate_boxes,
                batch.views.local_view_mask,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(random_seed))
            selected = select_training_proposals(
                routing.scores.detach(),
                batch.candidate_boxes,
                budget=self.selection_budget,
                random_count=self.random_count,
                nms_iou=self.nms_iou,
                candidate_mask=batch.views.local_view_mask,
                generator=generator,
            )
            selected_views = gather_selected_views(batch.views, selected)
            logits = self.model.classify_encoded_overview(
                overview,
                selected_views.local_images,
                selected_views.local_geometry,
                local_view_mask=selected_views.local_view_mask,
            )
            classification = _loss_tensor(
                self.criterion(logits, batch.labels, batch.loss_masks)
            )
            zero = classification * 0.0
            ranking = diversity = zero
            if oracle is not None:
                router_loss = router_objective(
                    self.router,
                    oracle,
                    batch.candidate_boxes,
                    batch.labels,
                    loss_mask=batch.loss_masks,
                    candidate_mask=batch.views.local_view_mask,
                    diversity_weight=0.0,
                )
                ranking = router_loss.ranking
                diversity = router_loss.diversity
            loss = (
                classification
                + ranking * self.router_weight
                + diversity * self.diversity_weight
            )
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            return None
        self.scaler.scale(loss).backward()
        if self.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                (
                    parameter
                    for group in self.optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.requires_grad
                ),
                self.max_grad_norm,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return FoveaJointTrainResult(
            loss=float(loss.detach()),
            classification=float(classification.detach()),
            router_ranking=float(ranking.detach()),
            router_diversity=float(diversity.detach()),
            oracle_used=use_oracle,
            selected_views=int(selected_views.local_view_mask.sum().item()),
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision,
            enabled=self.device.type == "cuda" and self.mixed_precision is not None,
        )


def gather_selected_views(
    views: FoveaViewTensorBatch,
    selected: torch.Tensor,
) -> FoveaViewTensorBatch:
    """Gather fixed-budget local candidates while preserving invalid slots."""
    batch, candidates = views.local_view_mask.shape
    if selected.ndim != 2 or selected.shape[0] != batch:
        raise ValueError("selected must have shape [batch, budget]")
    if candidates == 0:
        raise ValueError("cannot gather from an empty candidate set")
    if (selected >= candidates).any():
        raise ValueError("selected candidate index is out of range")
    valid = selected >= 0
    safe = selected.clamp_min(0)
    image_indices = safe[..., None, None, None].expand(
        -1, -1, *views.local_images.shape[2:]
    )
    geometry_indices = safe[..., None].expand(-1, -1, views.local_geometry.shape[-1])
    local_images = torch.gather(views.local_images, 1, image_indices)
    local_geometry = torch.gather(views.local_geometry, 1, geometry_indices)
    source_valid = torch.gather(views.local_view_mask, 1, safe)
    local_view_mask = valid & source_valid
    return FoveaViewTensorBatch(
        overview_images=views.overview_images,
        overview_geometry=views.overview_geometry,
        local_images=local_images,
        local_geometry=local_geometry,
        local_view_mask=local_view_mask,
    )
