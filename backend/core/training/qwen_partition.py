"""Complete-coverage target partitioning for Qwen-Image 2.1 training."""

from __future__ import annotations

from core.models.qwen_image_21.partition_global import QwenPartitionGlobalAdapter

import torch
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
