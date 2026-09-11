from __future__ import annotations

from typing import Iterable, List

import torch


def snapshot_schedule_scalars(values: Iterable) -> List[float]:
    """Copy a schedule to Python scalars with one device transfer."""
    if isinstance(values, torch.Tensor):
        return [float(value) for value in values.detach().cpu().tolist()]
    return [
        float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
        for value in values
    ]
