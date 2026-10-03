"""Style encoder losses (design §8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


def positive_weights(artists: Sequence[int], content_ids: Sequence[Sequence[int]],
                     view_positive_weight: float, positive_weight_min: float) -> torch.Tensor:
    """[2N, 2N] SupCon positive weights for N images; embedding i and i + N are the two views of image i."""
    n = len(artists)
    if len(content_ids) != n:
        raise ValueError(f"{len(content_ids)} content id sets for {n} images")
    if not view_positive_weight > 0.0:
        raise ValueError(f"view_positive_weight must be > 0, got {view_positive_weight!r}")
    if not 0.0 <= positive_weight_min <= 1.0:
        raise ValueError(f"positive_weight_min must be in [0, 1], got {positive_weight_min!r}")
    columns: dict[int, int] = {}
    entries = [(i, columns.setdefault(int(c), len(columns))) for i, ids in enumerate(content_ids)
               for c in set(int(c) for c in ids)]
    incidence = torch.zeros(n, len(columns), dtype=torch.float32)
    if entries:
        rows, cols = zip(*entries)
        incidence[list(rows), list(cols)] = 1.0
    inter = incidence @ incidence.T
    size = incidence.sum(dim=1)
    union = size[:, None] + size[None, :] - inter
    jaccard = torch.where(union > 0, inter / union.clamp_min(1.0), torch.zeros_like(inter))
    artist = torch.as_tensor(list(artists), dtype=torch.int64)
    same_artist = artist[:, None] == artist[None, :]
    weights = torch.where(same_artist, (1.0 - jaccard).clamp_min(positive_weight_min), torch.zeros_like(inter))
    weights.fill_diagonal_(float(view_positive_weight))
    full = weights.repeat(2, 2)
    full.fill_diagonal_(0.0)
    return full


@dataclass(frozen=True)
class SupConResult:
    loss: torch.Tensor              # fp32 scalar, mean over anchors with positive weight
    anchors_used: torch.Tensor      # int64 scalar
    anchors_excluded: torch.Tensor  # int64 scalar, anchors whose positive weights sum to 0


def weighted_supcon_loss(embeddings: torch.Tensor, weights: torch.Tensor, temperature: float) -> SupConResult:
    """Weighted multi-positive SupCon (design §8.1). `weights` is [M, M], e.g. from positive_weights;
    its diagonal is ignored. Computed in fp32 whatever the input dtype."""
    if not temperature > 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    if embeddings.dim() != 2 or embeddings.shape[0] < 2:
        raise ValueError(f"embeddings must be [M >= 2, D], got {tuple(embeddings.shape)}")
    m = embeddings.shape[0]
    if tuple(weights.shape) != (m, m):
        raise ValueError(f"weights must be [{m}, {m}], got {tuple(weights.shape)}")
    z = F.normalize(embeddings.float(), dim=1)
    logits = (z @ z.T) / temperature
    eye = torch.eye(m, dtype=torch.bool, device=z.device)
    w = weights.to(device=z.device, dtype=torch.float32).masked_fill(eye, 0.0)
    log_denominator = torch.logsumexp(logits.masked_fill(eye, float("-inf")), dim=1)
    total = w.sum(dim=1)
    used = total > 0
    positive = (w * logits).sum(dim=1) / total.clamp_min(torch.finfo(torch.float32).tiny)
    per_anchor = torch.where(used, log_denominator - positive, torch.zeros_like(total))
    count = used.sum()
    return SupConResult(loss=per_anchor.sum() / count.clamp_min(1), anchors_used=count, anchors_excluded=m - count)


def normalized_hsic(style: torch.Tensor, tag_probs: torch.Tensor) -> torch.Tensor:
    """Gaussian-kernel CKA; the tag side and median bandwidths carry no gradient."""
    if style.ndim != 2 or tag_probs.ndim != 2 or style.shape[0] != tag_probs.shape[0]:
        raise ValueError("HSIC inputs must be [batch, features] with the same batch size")
    if style.shape[0] < 4:
        raise ValueError("HSIC needs at least four images")

    def centered_kernel(x: torch.Tensor) -> torch.Tensor:
        distances = torch.pdist(x, p=2).square()
        positive = distances.detach()[distances.detach() > 0]
        bandwidth = positive.median() if positive.numel() else x.new_tensor(1.0)
        squared = torch.cdist(x, x).square()
        kernel = torch.exp(-squared / bandwidth.clamp_min(1e-12))
        return kernel - kernel.mean(dim=0, keepdim=True) - kernel.mean(dim=1, keepdim=True) + kernel.mean()

    k = centered_kernel(style.float())
    l = centered_kernel(tag_probs.detach().float())
    denominator = (k.square().sum() * l.square().sum()).sqrt()
    return torch.where(denominator > 1e-12, (k * l).sum() / denominator.clamp_min(1e-12),
                       k.sum() * 0.0)
