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
        start_step: int = 0,
        end_step: int = 1000,
        layer_weights: Optional[Dict[str, float]] = None,
        prompt: Optional[str] = None,
        is_lllite: bool = False,
    ):
        self.model_path = model_path
        self.image = image
        self.strength = strength
        self.start_step = start_step  # 0-1000 step range
        self.end_step = end_step      # 0-1000 step range
        self.layer_weights = layer_weights  # Per-layer weights like {"IN00": 1.0, ..., "MID": 1.0}
        self.prompt = prompt  # Optional separate prompt for this ControlNet
        self.is_lllite = is_lllite


class ControlNetManager:
    """Manages ControlNet models and their application"""

    def __init__(self, controlnet_dir: Optional[str] = None):
        if controlnet_dir is None:
            controlnet_dir = settings.controlnet_dir
        self.controlnet_dir = Path(controlnet_dir)
        self.additional_dirs: List[Path] = []  # User-configured additional directories
        print(f"[ControlNetManager] ControlNet directory: {self.controlnet_dir}")

        self.loaded_controlnets: Dict[str, ControlNetModel] = {}
        self.loaded_lllites: Dict[str, Any] = {}

    def set_additional_dirs(self, dirs: List[str]):
        """Set additional directories to scan for ControlNets"""
        self.additional_dirs = [Path(d) for d in dirs if d.strip()]
        print(f"[ControlNetManager] Additional directories set: {self.additional_dirs}")

    def _resolve_controlnet_path(self, controlnet_path: str) -> Optional[Path]:
        """Resolve controlnet path, checking default and additional directories"""
        # Try default directory first
        full_path = self.controlnet_dir / controlnet_path
        if full_path.exists():
            return full_path

        # Try additional directories
        for additional_dir in self.additional_dirs:
            full_path = additional_dir / controlnet_path
            if full_path.exists():
                return full_path

        return None

    def get_available_controlnets(self) -> List[str]:
        """Get list of available ControlNet models from default and additional directories"""
        controlnets = []

        # Combine default directory with additional directories
        all_dirs = [self.controlnet_dir] + self.additional_dirs

        for controlnet_dir in all_dirs:
            if not controlnet_dir.exists():
                print(f"[ControlNetManager] Skipping non-existent directory: {controlnet_dir}")
                continue

            print(f"[ControlNetManager] Scanning directory: {controlnet_dir}")
            # Look for .safetensors and .pth files
            for file in controlnet_dir.glob("**/*"):
                if file.suffix in [".safetensors", ".pth", ".pt", ".bin"]:
                    relative_path = file.relative_to(controlnet_dir)
                    controlnets.append(str(relative_path))

        print(f"[ControlNetManager] Total ControlNet models found: {len(controlnets)}")
        return sorted(list(set(controlnets)))  # Remove duplicates

    def is_lllite_model(self, controlnet_path: str) -> bool:
        """Check if a ControlNet model is LLLite format

        LLLite models have characteristic keys like:
        - lllite_unet_input_blocks_X_1_transformer_blocks_0_attn1_to_q.lora_down.weight
        - lllite_unet_middle_block_1_transformer_blocks_0_attn1_to_q.lora_down.weight

        Standard ControlNet models have keys like:
        - input_blocks.X.1.proj_in.weight
        - middle_block.1.proj_in.weight
        """
        full_path = self._resolve_controlnet_path(controlnet_path)

        if full_path is None:
            return False

        try:
            # Load state dict keys
            if full_path.suffix == '.safetensors':
                from safetensors import safe_open
                with safe_open(str(full_path), framework="pt", device="cpu") as f:
                    keys = list(f.keys())
            else:
                import torch
                state_dict = torch.load(str(full_path), map_location="cpu")
                keys = list(state_dict.keys())

            # Check for LLLite-specific keys
            lllite_indicators = [
                'lllite_unet',
                'lllite_mid',
                'lora_down.weight',
                'lora_up.weight'
            ]

            # If any key contains LLLite indicators, it's likely LLLite
            for key in keys[:50]:  # Check first 50 keys
                key_lower = key.lower()
                if any(indicator in key_lower for indicator in lllite_indicators):
                    print(f"[ControlNetManager] Detected LLLite model: {controlnet_path}")
                    print(f"[ControlNetManager] Sample key: {key}")
                    return True

            # Standard ControlNet has different structure
            print(f"[ControlNetManager] Detected standard ControlNet model: {controlnet_path}")
            return False

        except Exception as e:
            print(f"[ControlNetManager] Error detecting model type: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_controlnet_layers(self, controlnet_path: str) -> List[str]:
        """Get layer structure for a ControlNet model

        ControlNet uses Zero Convolution and has:
        - down_blocks (IN00-IN11): Input blocks
        - mid_block (MID): Middle block
        - NO up_blocks: ControlNet doesn't have output blocks (zero convolution ends at mid)

        Returns list like: ['IN00', 'IN01', ..., 'IN11', 'MID']
        """
        full_path = self._resolve_controlnet_path(controlnet_path)

        if full_path is None:
            print(f"[ControlNetManager] ControlNet not found: {controlnet_path}")
            return []

        try:
            # Standard ControlNet has 12 input blocks + 1 middle block
            # IN00-IN11 correspond to down_blocks.0-11
            # MID corresponds to mid_block
            layers = []

            # Add input blocks (IN00-IN11)
            for i in range(12):
                layers.append(f"IN{i:02d}")

            # Add middle block
            layers.append("MID")

            print(f"[ControlNetManager] ControlNet layers for {controlnet_path}: {layers}")
            return layers

        except Exception as e:
            print(f"[ControlNetManager] Error getting ControlNet layers: {e}")
            import traceback
            traceback.print_exc()
            return []

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

        full_path = self._resolve_controlnet_path(model_path)

        if full_path is None:
            print(f"[ControlNetManager] ControlNet model not found: {model_path}")
            return None

        try:
            if is_lllite:
                # Load ControlNet-LLLite
                print(f"Loading ControlNet-LLLite from {full_path}")
                lllite_model = self._load_lllite_model(full_path, device, dtype)
                if lllite_model is not None:
                    self.loaded_lllites[model_path] = lllite_model
                    print(f"ControlNet-LLLite loaded successfully: {model_path}")
                return lllite_model
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

    def _load_lllite_model(self, model_path: Path, device: str, dtype: torch.dtype):
        """Load ControlNet-LLLite model

        ControlNet-LLLite is a lightweight adaptation method that modifies
        intermediate features without a separate ControlNet model.
        """
        try:
            from safetensors.torch import load_file

            # Load state dict
            if model_path.suffix == '.safetensors':
                state_dict = load_file(str(model_path), device=device)
            else:
                state_dict = torch.load(str(model_path), map_location=device)

            # LLLite models are applied differently - they modify U-Net intermediate layers
            # Store the state dict for later application to the U-Net
            lllite_model = {
                'state_dict': state_dict,
                'dtype': dtype,
                'device': device,
                'model_path': str(model_path)
            }

            return lllite_model

        except Exception as e:
            print(f"Failed to load LLLite model: {e}")
            import traceback
            traceback.print_exc()
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

    def apply_lllite_to_unet(self, unet, lllite_model: dict, control_image: torch.Tensor):
        """Apply LLLite ControlNet to U-Net

        LLLite works by injecting LoRA-like modules into U-Net attention layers.
        The control image is processed and the LLLite weights modulate the attention.

        Args:
            unet: The U-Net model to modify
            lllite_model: Dict containing LLLite state_dict and metadata
            control_image: Preprocessed control image tensor
        """
        print(f"[ControlNetManager] Applying LLLite to U-Net")

        # Store control image in the lllite model for later use
        lllite_model['control_image'] = control_image

        # Get layer weights if available
        layer_weights = lllite_model.get('_layer_weights', None)

        # Apply LLLite weights to U-Net
        # LLLite modifies attention layers, we need to patch the forward hooks
        self._patch_unet_with_lllite(unet, lllite_model, layer_weights)

        print(f"[ControlNetManager] LLLite applied to U-Net")

    def _patch_unet_with_lllite(self, unet, lllite_model: dict, layer_weights=None):
        """Patch U-Net forward hooks to apply LLLite conditioning

        LLLite uses LoRA-like modules to condition the U-Net.
        Each attention layer gets modified based on the control image.
        """
        state_dict = lllite_model['state_dict']
        control_image = lllite_model.get('control_image')

        # Store LLLite info on U-Net for access during forward pass
        if not hasattr(unet, '_lllite_models'):
            unet._lllite_models = []

        unet._lllite_models.append({
            'state_dict': state_dict,
            'control_image': control_image,
            'layer_weights': layer_weights
        })

        print(f"[ControlNetManager] LLLite conditioning registered on U-Net")

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
        controlnet,
        layer_weights: Dict[str, float]
    ):
        """Apply layer-wise weights to ControlNet

        ControlNet architecture:
        - down_blocks (IN00-IN11): 12 input blocks
        - mid_block (MID): 1 middle block

        Layer weights map like:
        - IN00-IN11 -> down_block_res_samples[0-11]
        - MID -> mid_block_res_sample
        """
        if not layer_weights:
            return

        print(f"[ControlNetManager] Applying layer weights to ControlNet")

        # Convert layer names (IN00, IN01, ..., MID) to indices
        # Store as a list for easier indexing during forward pass
        down_weights = []
        mid_weight = 1.0

        for i in range(12):
            layer_name = f"IN{i:02d}"
            weight = layer_weights.get(layer_name, 1.0)
            down_weights.append(weight)

        mid_weight = layer_weights.get("MID", 1.0)

        # Store weights in a format compatible with ControlNet output
        # ControlNet returns (down_block_res_samples, mid_block_res_sample)
        layer_weight_data = {
            'down': down_weights,  # List of 12 weights
            'mid': mid_weight      # Single weight
        }

        print(f"[ControlNetManager] Layer weights applied: down={down_weights}, mid={mid_weight}")

        # Handle LLLite models (stored as dict) vs standard ControlNet models
        if isinstance(controlnet, dict):
            # LLLite model - store weights in the dict
            controlnet['_layer_weights'] = layer_weight_data
        else:
            # Standard ControlNet model - store as attribute
            controlnet._layer_weights = layer_weight_data
            # Monkey-patch the forward method to apply weights
            self._patch_controlnet_forward(controlnet)

    def _patch_controlnet_forward(self, controlnet: ControlNetModel):
        """Patch ControlNet forward method to apply layer weights"""

        # Check if already patched
        if hasattr(controlnet, '_original_forward'):
            return

        # Save original forward method
        controlnet._original_forward = controlnet.forward

        def weighted_forward(*args, **kwargs):
            # Call original forward
            output = controlnet._original_forward(*args, **kwargs)

            # Apply layer weights if set
            if hasattr(controlnet, '_layer_weights'):
                weights = controlnet._layer_weights

                # Output is either:
                # - tuple: (down_block_res_samples, mid_block_res_sample)
                # - ControlNetOutput object with .down_block_res_samples and .mid_block_res_sample

                if isinstance(output, tuple):
                    down_samples, mid_sample = output
                else:
                    # ControlNetOutput object
                    down_samples = output.down_block_res_samples
                    mid_sample = output.mid_block_res_sample

                # Apply down block weights
                weighted_down_samples = []
                for i, sample in enumerate(down_samples):
                    if i < len(weights['down']):
                        weight = weights['down'][i]
                        weighted_down_samples.append(sample * weight)
                    else:
                        weighted_down_samples.append(sample)

                # Apply mid block weight
                weighted_mid_sample = mid_sample * weights['mid']

                # Return in same format as input
                if isinstance(output, tuple):
                    return (weighted_down_samples, weighted_mid_sample)
                else:
                    # Reconstruct ControlNetOutput
                    output.down_block_res_samples = weighted_down_samples
                    output.mid_block_res_sample = weighted_mid_sample
                    return output

            return output

        # Replace forward method
        controlnet.forward = weighted_forward

    def apply_lllite_to_unet(self, unet, lllite_model_dict: Dict, strength: float = 1.0):
        """Apply ControlNet-LLLite to U-Net

        LLLite works by adding learned modifications to intermediate layer outputs.
        Based on kohya-ss implementation.
        """
        try:
            state_dict = lllite_model_dict['state_dict']
            device = lllite_model_dict['device']
            dtype = lllite_model_dict['dtype']

            print(f"[ControlNetManager] Applying LLLite to U-Net with strength {strength}")

            # LLLite modules are applied to specific U-Net layers
            # The state dict contains weights for modules like:
            # - input_blocks.X.1.transformer_blocks.0.attn1.to_q.lora_down.weight
            # - middle_block.1.transformer_blocks.0.attn1.to_q.lora_up.weight
            # etc.

            # For now, store the LLLite state for custom forward hooks
            # Full implementation would require hooking into U-Net forward pass
            if not hasattr(unet, '_lllite_modules'):
                unet._lllite_modules = []

            unet._lllite_modules.append({
                'state_dict': state_dict,
                'strength': strength,
                'device': device,
                'dtype': dtype
            })

            print(f"[ControlNetManager] LLLite module registered (strength={strength})")
            print(f"[ControlNetManager] Note: Full LLLite support requires custom U-Net hooks")

            return True

        except Exception as e:
            print(f"[ControlNetManager] Failed to apply LLLite: {e}")
            import traceback
            traceback.print_exc()
            return False


# Global ControlNet manager instance
controlnet_manager = ControlNetManager()
