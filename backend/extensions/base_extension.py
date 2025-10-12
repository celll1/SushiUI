from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image
import torch

class BaseExtension(ABC):
    """Base class for all extensions (Hires Fix, ControlNet, Tiled VAE, etc.)"""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def get_ui_params(self) -> Dict[str, Any]:
        """Return UI parameter definitions for this extension"""
        pass

    @abstractmethod
    def process_before_generation(
        self,
        pipeline: Any,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process parameters before generation starts"""
        pass

    @abstractmethod
    def process_after_generation(
        self,
        image: Image.Image,
        params: Dict[str, Any]
    ) -> Image.Image:
        """Process image after generation"""
        pass

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate extension parameters"""
        return True
