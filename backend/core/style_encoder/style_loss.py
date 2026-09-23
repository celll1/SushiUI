"""Style encoder losses (design §8)."""

from __future__ import annotations

from typing import Sequence

import torch


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
