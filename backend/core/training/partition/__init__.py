"""Architecture-gated complete-coverage DiT partition primitives."""

from .core import (
    PartitionBox,
    PartitionPlan,
    PartitionRegion,
    build_fixed_partition_plan,
    flatten_region,
    validate_partition_regions,
)
from .adapter import DiTPartitionAdapter, PartitionTopology

__all__ = [
    "PartitionBox",
    "PartitionPlan",
    "PartitionRegion",
    "build_fixed_partition_plan",
    "flatten_region",
    "validate_partition_regions",
    "DiTPartitionAdapter",
    "PartitionTopology",
]
