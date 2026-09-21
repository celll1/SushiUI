"""Architecture contract for complete-coverage DiT partition execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .core import PartitionBox, PartitionPlan, build_fixed_partition_plan


@dataclass(frozen=True)
class PartitionTopology:
    prefix_attends_target: bool
    global_position_ids: bool
    input_token_multiple: int


class DiTPartitionAdapter(ABC):
    """Only architectures with an audited attention/position contract implement this."""

    topology: PartitionTopology

    def build_fixed_plan(self, height: int, width: int, **kwargs) -> PartitionPlan:
        return build_fixed_partition_plan(
            height,
            width,
            input_token_multiple=self.topology.input_token_multiple,
            **kwargs,
        )

    @abstractmethod
    def position_ids(
        self, full_height: int, full_width: int, box: PartitionBox, *, device=None
    ) -> torch.Tensor:
        raise NotImplementedError
