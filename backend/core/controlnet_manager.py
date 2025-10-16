"""
ControlNet Manager for loading and applying ControlNet models.

Supports:
- Standard ControlNet models
- ControlNet-LLLite models
- Layer-wise weight control
- Step range control
"""

import os
import torch
from typing import Optional, Dict, List, Any
from pathlib import Path
from diffusers import ControlNetModel
from PIL import Image
import numpy as np
from config.settings import settings


class ControlNetConfig:
    """Configuration for a single ControlNet application"""
    def __init__(
        self,
        model_path: str,
        image: Image.Image,
        strength: float = 1.0,
        start_step: float = 0.0,
        end_step: float = 1.0,
        layer_weights: Optional[Dict[str, float]] = None,
        prompt: Optional[str] = None,
        is_lllite: bool = False,
    ):
        self.model_path = model_path
        self.image = image
        self.strength = strength
        self.start_step = start_step
        self.end_step = end_step
        self.layer_weights = layer_weights or {"down": 1.0, "mid": 1.0, "up": 1.0}
        self.prompt = prompt  # Optional separate prompt for this ControlNet
        self.is_lllite = is_lllite


class ControlNetManager:
    """Manages ControlNet models and their application"""

    def __init__(self, controlnet_dir: Optional[str] = None):
        if controlnet_dir is None:
            controlnet_dir = settings.controlnet_dir
        self.controlnet_dir = Path(controlnet_dir)
        print(f"[ControlNetManager] ControlNet directory: {self.controlnet_dir}")

        self.loaded_controlnets: Dict[str, ControlNetModel] = {}
        self.loaded_lllites: Dict[str, Any] = {}

    def get_available_controlnets(self) -> List[str]:
        """Get list of available ControlNet models"""
        controlnets = []

        if not self.controlnet_dir.exists():
            return controlnets

        # Look for .safetensors and .pth files
        for file in self.controlnet_dir.glob("**/*"):
            if file.suffix in [".safetensors", ".pth", ".pt", ".bin"]:
                relative_path = file.relative_to(self.controlnet_dir)
                controlnets.append(str(relative_path))

        return sorted(controlnets)

    def load_controlnet(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        is_lllite: bool = False
    ) -> Optional[ControlNetModel]:
        """Load a ControlNet model"""

        # Check if already loaded
        if model_path in self.loaded_controlnets:
            return self.loaded_controlnets[model_path]

        full_path = self.controlnet_dir / model_path

        if not full_path.exists():
            print(f"ControlNet model not found: {full_path}")
            return None

        try:
            if is_lllite:
                # Load ControlNet-LLLite
                # TODO: Implement LLLite loading
                print(f"Loading ControlNet-LLLite from {full_path}")
                # For now, return None - will implement LLLite support separately
                return None
            else:
                # Load standard ControlNet
                print(f"Loading ControlNet from {full_path}")

                # Try to load as diffusers ControlNet
                if full_path.is_dir():
                    controlnet = ControlNetModel.from_pretrained(
                        str(full_path),
                        torch_dtype=dtype
                    )
                else:
                    # Load from single file
                    controlnet = ControlNetModel.from_single_file(
                        str(full_path),
                        torch_dtype=dtype
                    )

                controlnet = controlnet.to(device)
                self.loaded_controlnets[model_path] = controlnet

                print(f"ControlNet loaded successfully: {model_path}")
                return controlnet

        except Exception as e:
            print(f"Failed to load ControlNet {model_path}: {e}")
            return None

    def unload_controlnet(self, model_path: str):
        """Unload a ControlNet model to free memory"""
        if model_path in self.loaded_controlnets:
            del self.loaded_controlnets[model_path]
            torch.cuda.empty_cache()
            print(f"ControlNet unloaded: {model_path}")

    def unload_all(self):
        """Unload all ControlNet models"""
        self.loaded_controlnets.clear()
        self.loaded_lllites.clear()
        torch.cuda.empty_cache()
        print("All ControlNets unloaded")

    def prepare_controlnet_image(
        self,
        image: Image.Image,
        width: int,
        height: int
    ) -> torch.Tensor:
        """Prepare ControlNet conditioning image"""
        # Resize image to target dimensions
        image = image.convert("RGB")
        image = image.resize((width, height), Image.LANCZOS)

        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

        return image_tensor

    def apply_layer_weights(
        self,
        controlnet: ControlNetModel,
        layer_weights: Dict[str, float]
    ):
        """Apply layer-wise weights to ControlNet"""
        # Scale ControlNet outputs by layer weights
        # This is applied during the controlnet forward pass
        # Store weights for use during generation
        if hasattr(controlnet, 'layer_weights'):
            controlnet.layer_weights = layer_weights
        else:
            # Add as attribute if not exists
            controlnet.layer_weights = layer_weights


# Global ControlNet manager instance
controlnet_manager = ControlNetManager()
