"""
Model-specific adapters for LoRA, Full Parameter, and ControlNet training.

This module provides a clean separation between core training logic and
model-specific implementations (SD1.5, SDXL, Z-Image, FLUX.2).
"""

from .base_adapter import BaseLoRAAdapter, BaseFullParameterAdapter
from .sd15_adapter import SD15LoRAAdapter, SD15FullParameterAdapter
from .sdxl_adapter import SDXLLoRAAdapter, SDXLFullParameterAdapter
from .zimage_adapter import ZImageLoRAAdapter, ZImageFullParameterAdapter
# DEUS support removed - architecture no longer maintained
# from .deus_adapter import DEUSLoRAAdapter, DEUSFullParameterAdapter
from .flux2_adapter import FLUX2LoRAAdapter, FLUX2FullParameterAdapter
from .anima_adapter import AnimaLoRAAdapter, AnimaFullParameterAdapter
from .lens_adapter import LensLoRAAdapter, LensFullParameterAdapter
from .base_controlnet_adapter import BaseControlNetAdapter
from .controlnet_sd15_adapter import ControlNetSD15Adapter
from .controlnet_sdxl_adapter import ControlNetSDXLAdapter

__all__ = [
    "BaseLoRAAdapter",
    "BaseFullParameterAdapter",
    "SD15LoRAAdapter",
    "SD15FullParameterAdapter",
    "SDXLLoRAAdapter",
    "SDXLFullParameterAdapter",
    "ZImageLoRAAdapter",
    "ZImageFullParameterAdapter",
    # DEUS support removed
    # "DEUSLoRAAdapter",
    # "DEUSFullParameterAdapter",
    "FLUX2LoRAAdapter",
    "FLUX2FullParameterAdapter",
    "AnimaLoRAAdapter",
    "AnimaFullParameterAdapter",
    "LensLoRAAdapter",
    "LensFullParameterAdapter",
    "BaseControlNetAdapter",
    "ControlNetSD15Adapter",
    "ControlNetSDXLAdapter",
]
