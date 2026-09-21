"""Complete-coverage target partitioning for Qwen-Image 2.1 training."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
from core.training.partition import (
    DiTPartitionAdapter,
    PartitionBox,
    PartitionPlan,
    PartitionRegion,
    PartitionTopology,
    build_fixed_partition_plan as _build_fixed_partition_plan,
    flatten_region,
)


def build_fixed_partition_plan(*args, **kwargs) -> PartitionPlan:
    """Qwen compatibility wrapper applying its target-slot multiple."""
    return _build_fixed_partition_plan(*args, input_token_multiple=4, **kwargs)


def full_canvas_position_ids(
    full_height: int,
    full_width: int,
    box: PartitionBox,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return row-major `(height, width)` RoPE indices from the full canvas."""
    full_y = torch.arange(
        -(full_height - full_height // 2), full_height // 2, device=device, dtype=torch.long
    )
    full_x = torch.arange(
        -(full_width - full_width // 2), full_width // 2, device=device, dtype=torch.long
    )
    yy = full_y[box.top : box.bottom, None].expand(box.height, box.width)
    xx = full_x[box.left : box.right][None, :].expand(box.height, box.width)
    return torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)


class QwenImage21PartitionAdapter(DiTPartitionAdapter):
    topology = PartitionTopology(
        prefix_attends_target=False,
        global_position_ids=True,
        input_token_multiple=4,
    )

    def position_ids(
        self, full_height: int, full_width: int, box: PartitionBox, *, device=None
    ) -> torch.Tensor:
        return full_canvas_position_ids(
            full_height, full_width, box, device=device
        )


class QwenPartitionGlobalAdapter(nn.Module):
    """Small trainable full-canvas communication path for partitioned training."""

    ADAPTER_ALGORITHM = "qwen_partition_global"
    WEIGHT_DECOMPOSE = False

    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int,
        *,
        rank: int = 64,
        summary_tokens: int = 16,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if rank <= 0 or summary_tokens <= 0:
            raise ValueError("Qwen partition global rank and summary tokens must be positive")
        self.rank = int(rank)
        self.summary_tokens = int(summary_tokens)
        input_dim = int(latent_channels) + 2
        self.key = nn.Linear(input_dim, self.rank, bias=False, dtype=dtype)
        self.value = nn.Linear(input_dim, self.rank, bias=False, dtype=dtype)
        self.query = nn.Linear(input_dim, self.rank, bias=False, dtype=dtype)
        self.summary_queries = nn.Parameter(
            torch.randn(self.summary_tokens, self.rank, dtype=dtype) / self.rank**0.5
        )
        self.out = nn.Linear(self.rank, int(hidden_dim), bias=False, dtype=dtype)
        nn.init.zeros_(self.out.weight)

    @staticmethod
    def _features(grid: torch.Tensor, box: PartitionBox | None = None) -> torch.Tensor:
        batch, height, width, _ = grid.shape
        y = torch.linspace(-1, 1, height, device=grid.device, dtype=grid.dtype)
        x = torch.linspace(-1, 1, width, device=grid.device, dtype=grid.dtype)
        yy = y[:, None].expand(height, width)
        xx = x[None, :].expand(height, width)
        coords = torch.stack((yy, xx), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        features = torch.cat((grid, coords), dim=-1)
        if box is not None:
            features = features[:, box.top : box.bottom, box.left : box.right]
        return features.reshape(batch, -1, features.shape[-1])

    def forward(self, full_grid: torch.Tensor, box: PartitionBox) -> torch.Tensor:
        full = self._features(full_grid)
        local = self._features(full_grid, box)
        if not torch.is_autocast_enabled(full.device.type):
            parameter_dtype = self.key.weight.dtype
            full = full.to(parameter_dtype)
            local = local.to(parameter_dtype)
        keys = self.key(full)
        values = self.value(full)
        scale = self.rank**-0.5
        summary_weights = torch.softmax(
            torch.einsum("sr,bnr->bsn", self.summary_queries, keys) * scale,
            dim=-1,
            dtype=torch.float32,
        ).to(values.dtype)
        summaries = torch.einsum("bsn,bnr->bsr", summary_weights, values)
        queries = self.query(local)
        local_weights = torch.softmax(
            torch.einsum("bnr,bsr->bns", queries, summaries) * scale,
            dim=-1,
            dtype=torch.float32,
        ).to(summaries.dtype)
        context = torch.einsum("bns,bsr->bnr", local_weights, summaries)
        return self.out(context)

    def branch_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "key.weight": self.key.weight,
            "value.weight": self.value.weight,
            "query.weight": self.query.weight,
            "summary_queries": self.summary_queries,
            "out.weight": self.out.weight,
        }

    def trainable_parameters(self):
        return iter(self.parameters())

    def export_tensors(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.branch_tensors().items()}

    def load_tensors(self, tensors: Mapping[str, torch.Tensor]) -> None:
        for name, parameter in self.branch_tensors().items():
            value = tensors.get(name)
            if value is not None:
                parameter.data.copy_(value)

    def spec_constants(self) -> tuple[str, ...]:
        return ()
