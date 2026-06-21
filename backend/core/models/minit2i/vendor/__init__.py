"""Vendored MiniT2I model code (MIT). Independent of the upstream package."""

from .mmjit import MMJiT, MMJiTConfig, DiffusionModel
from .transformer import MiniT2IMMJiTModel, MiniT2IFlowMatchScheduler
from .single_file import (
    KNOWN_VARIANTS,
    detect_variant_from_state_dict,
    load_single_file,
    save_single_file,
)

__all__ = [
    "MMJiT",
    "MMJiTConfig",
    "DiffusionModel",
    "MiniT2IMMJiTModel",
    "MiniT2IFlowMatchScheduler",
    "KNOWN_VARIANTS",
    "detect_variant_from_state_dict",
    "load_single_file",
    "save_single_file",
]
