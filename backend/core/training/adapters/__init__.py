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
from .ideogram4_adapter import Ideogram4LoRAAdapter, Ideogram4FullParameterAdapter
from .minit2i_adapter import MiniT2ILoRAAdapter, MiniT2IFullParameterAdapter
from .krea2_adapter import Krea2LoRAAdapter, Krea2FullParameterAdapter
from .ltx2_adapter import Ltx2LoRAAdapter, Ltx2FullParameterAdapter
# MiniMax-H3 exports a LoRA adapter ONLY: full fine-tuning is refused for this
# architecture (design section 7) and the absence of a FullParameterAdapter class
# here is one of the three layers of that refusal.
from .minimax_h3_adapter import MiniMaxH3LoRAAdapter
from .acestep_adapter import AceStepLoRAAdapter, AceStepFullParameterAdapter
from .sensenova_adapter import SenseNovaLoRAAdapter, SenseNovaFullParameterAdapter
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
    "Ideogram4LoRAAdapter",
    "Ideogram4FullParameterAdapter",
    "Krea2LoRAAdapter",
    "Krea2FullParameterAdapter",
    "Ltx2LoRAAdapter",
    "Ltx2FullParameterAdapter",
    "MiniMaxH3LoRAAdapter",
    "AceStepLoRAAdapter",
    "AceStepFullParameterAdapter",
    "SenseNovaLoRAAdapter",
    "SenseNovaFullParameterAdapter",
    "BaseControlNetAdapter",
    "ControlNetSD15Adapter",
    "ControlNetSDXLAdapter",
]
