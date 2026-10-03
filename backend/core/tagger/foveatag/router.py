"""Budgeted, vocabulary-independent evidence router for FoveaTag."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import RMSNorm


@dataclass(frozen=True)
class RouterOutput:
    scores: torch.Tensor
    slot_scores: torch.Tensor


class EvidenceRouter(nn.Module):
    """Score generic source-space candidate boxes from an overview feature map."""

    def __init__(
        self,
        feature_dim: int,
        *,
        routing_dim: int = 256,
        routing_slots: int = 4,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        if min(feature_dim, routing_dim, routing_slots) <= 0 or temperature <= 0:
            raise ValueError("router dimensions and temperature must be positive")
        self.routing_slots = routing_slots
        self.temperature = temperature
        self.feature_projection = nn.Linear(feature_dim, routing_dim)
        self.geometry_projection = nn.Sequential(
            nn.Linear(5, routing_dim),
            nn.SiLU(),
            nn.Linear(routing_dim, routing_dim),
        )
        self.norm = RMSNorm(routing_dim)
        self.slots = nn.Parameter(torch.empty(routing_slots, routing_dim))
        nn.init.normal_(self.slots, std=routing_dim ** -0.5)

    def forward(
        self,
        overview_map: torch.Tensor,
        overview_coordinates: torch.Tensor,
        candidate_boxes: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> RouterOutput:
        if overview_map.ndim != 4:
            raise ValueError("overview_map must have shape [batch, channels, height, width]")
        batch, _, height, width = overview_map.shape
        if overview_coordinates.shape != (batch, height, width, 2):
            raise ValueError("overview_coordinates must align with overview_map")
        if candidate_boxes.ndim != 3 or candidate_boxes.shape[0] != batch or candidate_boxes.shape[2] != 4:
            raise ValueError("candidate_boxes must have shape [batch, candidates, 4]")
        if (candidate_boxes[..., 2:4] <= 0).any():
            raise ValueError("candidate widths and heights must be positive")
        candidates = candidate_boxes.shape[1]
        valid = _candidate_mask(candidate_mask, batch, candidates, overview_map.device)
        sampled = nearest_source_features(overview_map, overview_coordinates, candidate_boxes[..., :2])
        area = candidate_boxes[..., 2] * candidate_boxes[..., 3]
        geometry = torch.cat((candidate_boxes, area.clamp_min(1e-8).log2().unsqueeze(-1)), dim=-1)
        features = self.norm(
            self.feature_projection(sampled) + self.geometry_projection(geometry.to(sampled.dtype))
        )
        slot_scores = torch.einsum("bcd,rd->brc", features, self.slots) / math.sqrt(features.shape[-1])
        slot_scores = slot_scores.masked_fill(~valid[:, None], -torch.inf)
        scores = self.temperature * torch.logsumexp(slot_scores / self.temperature, dim=1)
        scores = scores.masked_fill(~valid, -torch.inf)
        return RouterOutput(scores=scores, slot_scores=slot_scores)


def build_scale_space_candidates(
    overview_coordinates: torch.Tensor,
    field_sizes: Sequence[Tuple[float, float]],
) -> torch.Tensor:
    """Attach each configured normalized field size to every overview cell."""
    if overview_coordinates.ndim != 4 or overview_coordinates.shape[-1] != 2:
        raise ValueError("overview_coordinates must have shape [batch, height, width, 2]")
    if not field_sizes or any(width <= 0 or height <= 0 for width, height in field_sizes):
        raise ValueError("field_sizes must contain positive width/height pairs")
    centers = overview_coordinates.flatten(1, 2)
    levels = centers.new_tensor(field_sizes)
    centers = centers[:, :, None, :].expand(-1, -1, len(field_sizes), -1)
    sizes = levels[None, None].expand(centers.shape[0], centers.shape[1], -1, -1)
    return torch.cat((centers, sizes), dim=-1).reshape(centers.shape[0], -1, 4)


def fit_candidate_boxes(boxes: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    if min(image_width, image_height) <= 0:
        raise ValueError("source dimensions must be positive")
    result = boxes.clone()
    side = torch.minimum(boxes[..., 2] * image_width, boxes[..., 3] * image_height)
    side = side.clamp_max(min(image_width, image_height))
    for axis, size in ((0, image_width), (1, image_height)):
        center = boxes[..., axis] * size
        result[..., axis] = torch.minimum(torch.maximum(center, side / 2), size - side / 2) / size
        result[..., axis + 2] = side / size
    return result


def source_candidate_mask(candidate_boxes: torch.Tensor) -> torch.Tensor:
    """Reject candidates centered in overview padding outside the source image."""
    if candidate_boxes.ndim != 3 or candidate_boxes.shape[-1] != 4:
        raise ValueError("candidate_boxes must have shape [batch, candidates, 4]")
    centers = candidate_boxes[..., :2]
    return ((centers >= 0.0) & (centers <= 1.0)).all(dim=-1)


def nearest_source_features(
    feature_map: torch.Tensor,
    feature_coordinates: torch.Tensor,
    candidate_centers: torch.Tensor,
) -> torch.Tensor:
    """Gather the nearest overview cell for each source-space proposal center."""
    features = feature_map.flatten(2).transpose(1, 2)
    coordinates = feature_coordinates.flatten(1, 2).to(torch.float32)
    distances = torch.cdist(candidate_centers.to(torch.float32), coordinates)
    indices = distances.argmin(dim=-1)
    return torch.gather(features, 1, indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))


def dense_oracle_targets(
    candidate_tag_scores: torch.Tensor,
    overview_tag_scores: torch.Tensor,
    positive_labels: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert vocabulary-agnostic positive-margin improvements to [0, 1]."""
    if candidate_tag_scores.ndim != 3:
        raise ValueError("candidate_tag_scores must have shape [batch, candidates, tags]")
    batch, candidates, tags = candidate_tag_scores.shape
    if overview_tag_scores.shape != (batch, tags):
        raise ValueError("overview_tag_scores must have shape [batch, tags]")
    if positive_labels.shape != (batch, tags):
        raise ValueError("positive_labels must have shape [batch, tags]")
    if loss_mask is not None and loss_mask.shape != (batch, tags):
        raise ValueError("loss_mask must have shape [batch, tags]")
    valid = _candidate_mask(candidate_mask, batch, candidates, candidate_tag_scores.device)
    positives = positive_labels.to(device=candidate_tag_scores.device, dtype=torch.bool)
    if loss_mask is not None:
        positives &= loss_mask.to(device=candidate_tag_scores.device) > 0
    improvements = (
        candidate_tag_scores.detach()
        - overview_tag_scores.detach().to(candidate_tag_scores.device)[:, None]
    )
    improvements = improvements.masked_fill(~positives[:, None], -torch.inf)
    values = improvements.max(dim=-1).values
    has_positive = positives.any(dim=-1)
    values = torch.where(has_positive[:, None] & valid, values, torch.zeros_like(values))
    values = F.relu(values)
    maximum = values.max(dim=1, keepdim=True).values
    return torch.where(maximum > 0, values / maximum.clamp_min(1e-8), values)


def router_ranking_loss(
    router_scores: torch.Tensor,
    oracle_targets: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if router_scores.shape != oracle_targets.shape or router_scores.ndim != 2:
        raise ValueError("router_scores and oracle_targets must have equal [batch, candidates] shapes")
    valid = _candidate_mask(
        candidate_mask,
        router_scores.shape[0],
        router_scores.shape[1],
        router_scores.device,
    )
    if not valid.any():
        return router_scores[valid].sum() * 0.0
    return F.binary_cross_entropy_with_logits(router_scores[valid], oracle_targets.detach()[valid])


def router_diversity_loss(slot_scores: torch.Tensor) -> torch.Tensor:
    """Penalize routing slots whose candidate distributions collapse together."""
    if slot_scores.ndim != 3:
        raise ValueError("slot_scores must have shape [batch, slots, candidates]")
    slots = slot_scores.shape[1]
    if slots < 2:
        return slot_scores.sum() * 0.0
    distributions = torch.nan_to_num(slot_scores.softmax(dim=-1))
    normalized = F.normalize(distributions, dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    off_diagonal = ~torch.eye(slots, dtype=torch.bool, device=slot_scores.device)
    return similarity[:, off_diagonal].mean()


def select_proposals(
    scores: torch.Tensor,
    candidate_boxes: torch.Tensor,
    *,
    budget: int,
    nms_iou: float = 0.5,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Greedy scale-space NMS, returning -1 where a batch has fewer proposals."""
    if scores.ndim != 2 or candidate_boxes.shape != (*scores.shape, 4):
        raise ValueError("scores and candidate_boxes shapes do not align")
    if budget < 0 or not 0 <= nms_iou <= 1:
        raise ValueError("budget and nms_iou are invalid")
    valid = _candidate_mask(candidate_mask, *scores.shape, scores.device)
    selected = torch.full((scores.shape[0], budget), -1, dtype=torch.long, device=scores.device)
    for batch_index in range(scores.shape[0]):
        order = scores[batch_index].masked_fill(~valid[batch_index], -torch.inf).argsort(descending=True)
        kept = []
        for index in order.tolist():
            if not valid[batch_index, index] or not torch.isfinite(scores[batch_index, index]):
                continue
            if all(_box_iou(candidate_boxes[batch_index, index], candidate_boxes[batch_index, old]) <= nms_iou for old in kept):
                kept.append(index)
                if len(kept) == budget:
                    break
        if kept:
            selected[batch_index, : len(kept)] = torch.tensor(kept, device=scores.device)
    return selected


def select_training_proposals(
    scores: torch.Tensor,
    candidate_boxes: torch.Tensor,
    *,
    budget: int,
    random_count: int,
    nms_iou: float = 0.5,
    candidate_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Keep a non-annealed random fraction of the training view budget."""
    if not 0 <= random_count <= budget:
        raise ValueError("random_count must be between zero and budget")
    routed_budget = budget - random_count
    routed = select_proposals(
        scores,
        candidate_boxes,
        budget=routed_budget,
        nms_iou=nms_iou,
        candidate_mask=candidate_mask,
    )
    valid = _candidate_mask(candidate_mask, *scores.shape, scores.device)
    result = torch.full((scores.shape[0], budget), -1, dtype=torch.long, device=scores.device)
    if routed_budget:
        result[:, :routed_budget] = routed
    for batch_index in range(scores.shape[0]):
        excluded = set(routed[batch_index][routed[batch_index] >= 0].tolist())
        choices = [index for index in valid[batch_index].nonzero().flatten().tolist() if index not in excluded]
        if choices and random_count:
            order = torch.randperm(len(choices), generator=generator)[:random_count].tolist()
            chosen = [choices[index] for index in order]
            result[batch_index, routed_budget : routed_budget + len(chosen)] = torch.tensor(
                chosen,
                device=scores.device,
            )
    return result


def _box_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    first_min = first[:2] - first[2:] / 2
    first_max = first[:2] + first[2:] / 2
    second_min = second[:2] - second[2:] / 2
    second_max = second[:2] + second[2:] / 2
    intersection = (torch.minimum(first_max, second_max) - torch.maximum(first_min, second_min)).clamp_min(0)
    intersection_area = intersection.prod()
    union = first[2:].prod() + second[2:].prod() - intersection_area
    return float((intersection_area / union.clamp_min(1e-8)).item())


def _candidate_mask(
    mask: Optional[torch.Tensor],
    batch: int,
    candidates: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch, candidates), dtype=torch.bool, device=device)
    if mask.shape != (batch, candidates):
        raise ValueError(f"candidate_mask must have shape [{batch}, {candidates}]")
    return mask.to(device=device, dtype=torch.bool)
