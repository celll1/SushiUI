"""Optional local-view SigLIP2 distillation used only as a diagnostic ablation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DistillationOutput:
    loss: torch.Tensor
    student_anchors: torch.Tensor
    valid_anchors: torch.Tensor


class LocalSigLIP2DistillationHead(nn.Module):
    """Align a stride-16-or-finer student map to a local teacher's 16x16 grid."""

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        *,
        anchor_grid_size: int = 16,
    ) -> None:
        super().__init__()
        if student_dim <= 0 or teacher_dim <= 0 or anchor_grid_size <= 0:
            raise ValueError("feature dimensions and anchor_grid_size must be positive")
        self.anchor_grid_size = anchor_grid_size
        self.projection = nn.Linear(student_dim, teacher_dim)

    def forward(
        self,
        student_map: torch.Tensor,
        student_coordinates: torch.Tensor,
        view_geometry: torch.Tensor,
        teacher_tokens: torch.Tensor,
        teacher_mask: Optional[torch.Tensor] = None,
    ) -> DistillationOutput:
        if student_map.ndim != 4:
            raise ValueError("student_map must have shape [batch, channels, height, width]")
        batch, _, height, width = student_map.shape
        grid = self.anchor_grid_size
        if height < grid or width < grid:
            raise ValueError(
                f"student map must be at least {grid}x{grid}; got {height}x{width}"
            )
        if student_coordinates.shape != (batch, height, width, 2):
            raise ValueError("student_coordinates must align with student_map")
        if view_geometry.ndim != 2 or view_geometry.shape[0] != batch or view_geometry.shape[1] < 4:
            raise ValueError("view_geometry must have shape [batch, at least 4]")
        if teacher_tokens.shape[:2] != (batch, grid * grid):
            raise ValueError(f"teacher_tokens must have shape [batch, {grid * grid}, dim]")

        pooled, populated = pool_to_source_anchors(
            student_map,
            student_coordinates,
            view_geometry,
            grid_size=grid,
        )
        projected = self.projection(pooled)
        valid = populated
        if teacher_mask is not None:
            if teacher_mask.shape != (batch, grid * grid):
                raise ValueError(f"teacher_mask must have shape [batch, {grid * grid}]")
            valid = valid & teacher_mask.to(device=valid.device, dtype=torch.bool)
        if valid.any():
            cosine = 1.0 - F.cosine_similarity(
                projected,
                teacher_tokens.detach().to(device=projected.device, dtype=projected.dtype),
                dim=-1,
            )
            loss = cosine[valid].mean()
        else:
            loss = projected.sum() * 0.0
        return DistillationOutput(loss, projected, valid)


def pool_to_source_anchors(
    student_map: torch.Tensor,
    student_coordinates: torch.Tensor,
    view_geometry: torch.Tensor,
    *,
    grid_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average student cells into teacher cells using normalized source coordinates."""
    batch, channels, height, width = student_map.shape
    if student_coordinates.shape != (batch, height, width, 2):
        raise ValueError("student_coordinates must align with student_map")
    if view_geometry.shape[0] != batch or view_geometry.shape[1] < 4:
        raise ValueError("view_geometry must provide center and field size")
    if (view_geometry[:, 2:4] <= 0).any():
        raise ValueError("view field dimensions must be positive")

    left_top = view_geometry[:, None, None, 0:2] - view_geometry[:, None, None, 2:4] / 2.0
    normalized = (student_coordinates - left_top) / view_geometry[:, None, None, 2:4]
    cells = torch.floor(normalized * grid_size).to(torch.long).clamp(0, grid_size - 1)
    indices = cells[..., 1] * grid_size + cells[..., 0]
    flat_features = student_map.flatten(2).transpose(1, 2)
    flat_indices = indices.flatten(1)
    anchors = student_map.new_zeros((batch, grid_size * grid_size, channels))
    counts = student_map.new_zeros((batch, grid_size * grid_size, 1))
    anchors.scatter_add_(1, flat_indices.unsqueeze(-1).expand(-1, -1, channels), flat_features)
    counts.scatter_add_(
        1,
        flat_indices.unsqueeze(-1),
        torch.ones((*flat_indices.shape, 1), device=student_map.device, dtype=student_map.dtype),
    )
    populated = counts.squeeze(-1) > 0
    anchors = anchors / counts.clamp_min(1.0)
    return anchors, populated
