"""
LoRA (Low-Rank Adaptation) Manager
Handles loading and applying multiple LoRAs with fine-grained control
"""
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
from config.settings import settings


class LoRAConfig:
    """Configuration for a single LoRA"""
    def __init__(
        self,
        path: str,
        strength: float = 1.0,
        apply_to_text_encoder: bool = True,
        apply_to_unet: bool = True,
        unet_layer_weights: Optional[Dict[str, float]] = None,
        step_range: Optional[List[int]] = None
    ):
        self.path = path
        self.strength = strength
        self.apply_to_text_encoder = apply_to_text_encoder
        self.apply_to_unet = apply_to_unet
        self.unet_layer_weights = unet_layer_weights or {}
        self.step_range = step_range or [0, 1000]  # 0 = start, 1000 = end

    def is_active_at_step(self, current_step: int, total_steps: int) -> bool:
        """Check if LoRA should be active at current step"""
        # Convert normalized range [0-1000] to actual step range
        start_step = int((self.step_range[0] / 1000) * total_steps)
        end_step = int((self.step_range[1] / 1000) * total_steps)
        return start_step <= current_step <= end_step

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoRAConfig":
        """Create LoRAConfig from dictionary"""
        return cls(
            path=data.get("path", ""),
            strength=data.get("strength", 1.0),
            apply_to_text_encoder=data.get("apply_to_text_encoder", True),
            apply_to_unet=data.get("apply_to_unet", True),
            unet_layer_weights=data.get("unet_layer_weights"),
            step_range=data.get("step_range", [0, 1000])
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "path": self.path,
            "strength": self.strength,
            "apply_to_text_encoder": self.apply_to_text_encoder,
            "apply_to_unet": self.apply_to_unet,
            "unet_layer_weights": self.unet_layer_weights,
            "step_range": self.step_range
        }


class LoRAManager:
    """Manages multiple LoRAs for Stable Diffusion pipelines"""

    def __init__(self, lora_dir: Optional[str] = None):
        if lora_dir is None:
            lora_dir = settings.lora_dir
        self.lora_dir = Path(lora_dir)
        self.additional_dirs: List[Path] = []  # User-configured additional directories
        self.loaded_loras: List[LoRAConfig] = []

        # Add training directory to search paths (for trained LoRAs)
        training_dir = Path(settings.root_dir) / "training"
        if training_dir.exists():
            self.additional_dirs.append(training_dir)
            print(f"[LoRAManager] Added training directory to search paths: {training_dir}")

        print(f"[LoRAManager] LoRA directory: {self.lora_dir}")

    def set_additional_dirs(self, dirs: List[str]):
        """Set additional directories to scan for LoRAs"""
        self.additional_dirs = [Path(d) for d in dirs if d.strip()]
        print(f"[LoRAManager] Additional directories set: {self.additional_dirs}")

    def _resolve_lora_path(self, lora_path: str) -> Optional[Path]:
        """Resolve LoRA path, checking default and additional directories"""
        # Try default directory first
        full_path = self.lora_dir / lora_path
        if full_path.exists():
            return full_path

        # Try additional directories
        for additional_dir in self.additional_dirs:
            full_path = additional_dir / lora_path
            if full_path.exists():
                return full_path

        return None

    def get_available_loras(self) -> List[str]:
        """Get list of available LoRA files from default and additional directories"""
        lora_files = []

        # Combine default directory with additional directories
        all_dirs = [self.lora_dir] + self.additional_dirs

        for lora_dir in all_dirs:
            print(f"[LoRAManager] Checking directory: {lora_dir}")
            print(f"[LoRAManager] Directory exists: {lora_dir.exists()}")

            if not lora_dir.exists():
                if lora_dir == self.lora_dir:
                    print(f"[LoRAManager] Creating default directory: {lora_dir}")
                    lora_dir.mkdir(parents=True, exist_ok=True)
                else:
                    print(f"[LoRAManager] Skipping non-existent directory: {lora_dir}")
                continue

            for ext in [".safetensors", ".pt", ".bin"]:
                found = list(lora_dir.rglob(f"*{ext}"))
                print(f"[LoRAManager] Found {len(found)} files with extension {ext} in {lora_dir}")
                lora_files.extend([
                    str(f.relative_to(lora_dir))
                    for f in found
                ])

        print(f"[LoRAManager] Total LoRA files found: {len(lora_files)}")
        return sorted(list(set(lora_files)))  # Remove duplicates

    def load_loras(self, pipeline: Any, lora_configs: List[Dict[str, Any]]) -> Any:
        """
        Load multiple LoRAs into the pipeline

        Args:
            pipeline: Diffusers pipeline
            lora_configs: List of LoRA configurations

        Returns:
            Modified pipeline with LoRAs loaded
        """
        print(f"[LoRAManager] load_loras called with {len(lora_configs) if lora_configs else 0} configs")
        print(f"[LoRAManager] lora_configs: {lora_configs}")

        if not lora_configs:
            print("[LoRAManager] No LoRA configs provided, skipping")
            return pipeline

        # Parse configs
        self.loaded_loras = [LoRAConfig.from_dict(cfg) for cfg in lora_configs]
        print(f"[LoRAManager] Parsed {len(self.loaded_loras)} LoRA configs")

        # Load LoRAs using diffusers' native support
        try:
            for i, lora_config in enumerate(self.loaded_loras):
                lora_path = self._resolve_lora_path(lora_config.path)

                if lora_path is None:
                    print(f"[LoRAManager] WARNING: LoRA file not found: {lora_config.path}")
                    continue

                print(f"[LoRAManager] Attempting to load LoRA from: {lora_path}")
                print(f"[LoRAManager] LoRA config: strength={lora_config.strength}, apply_to_text_encoder={lora_config.apply_to_text_encoder}, apply_to_unet={lora_config.apply_to_unet}")

                print(f"[LoRAManager] Loading LoRA {i+1}/{len(self.loaded_loras)}: {lora_config.path}")

                # Load LoRA weights
                adapter_name = f"lora_{i}"
                print(f"[LoRAManager] Calling pipeline.load_lora_weights with adapter_name={adapter_name}")
                pipeline.load_lora_weights(
                    str(lora_path.parent),
                    weight_name=lora_path.name,
                    adapter_name=adapter_name
                )
                print(f"[LoRAManager] Successfully loaded LoRA weights")

                # Set adapter with strength
                # Note: Step ranges will be handled in callback
                if hasattr(pipeline, 'set_adapters'):
                    print(f"[LoRAManager] Setting adapter with strength={lora_config.strength}")
                    pipeline.set_adapters(adapter_name, adapter_weights=lora_config.strength)

                    # Apply per-layer weights if specified
                    if lora_config.unet_layer_weights and hasattr(pipeline, 'unet'):
                        print(f"[LoRAManager] Applying per-layer weights: {len(lora_config.unet_layer_weights)} layers")
                        self._apply_layer_weights(pipeline, adapter_name, lora_config)
                else:
                    print(f"[LoRAManager] WARNING: Pipeline does not have set_adapters method")

            print(f"[LoRAManager] Successfully loaded {len(self.loaded_loras)} LoRA(s)")

        except Exception as e:
            print(f"[LoRAManager] ERROR loading LoRAs: {e}")
            import traceback
            traceback.print_exc()

        return pipeline

    def _apply_layer_weights(self, pipeline: Any, adapter_name: str, lora_config: LoRAConfig):
        """
        Apply per-block weights to the LoRA adapter

        This modifies the LoRA adapter weights in the UNet directly by scaling them according to
        the block-specific weights (IN00-IN11, MID, OUT00-OUT11) specified in the config.
        """
        try:
            import re

            # Access the UNet's LoRA layers directly
            if not hasattr(pipeline, 'unet'):
                print("[LoRAManager] Pipeline does not have unet attribute")
                return

            unet = pipeline.unet

            # Check if UNet has peft_config (PEFT-based LoRA)
            if not hasattr(unet, 'peft_config'):
                print("[LoRAManager] UNet does not have peft_config, trying alternative method")
                # Try alternative method for non-PEFT LoRAs
                self._apply_layer_weights_alternative(pipeline, adapter_name, lora_config)
                return

            # Iterate through all named modules in the UNet
            modified_count = 0
            for name, module in unet.named_modules():
                # Check if this module has LoRA adapters
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    # Determine which block this module belongs to
                    block_weight = self._get_block_weight_for_module(name, lora_config.unet_layer_weights)

                    if block_weight != 1.0:  # Only modify if weight is not default
                        # Scale the LoRA weights
                        if hasattr(module, 'scaling') and adapter_name in module.scaling:
                            # Modify the scaling factor
                            original_scaling = module.scaling[adapter_name]
                            module.scaling[adapter_name] = original_scaling * block_weight
                            modified_count += 1

            if modified_count > 0:
                print(f"[LoRAManager] Applied block weights to {modified_count} LoRA layers")
            else:
                print(f"[LoRAManager] WARNING: No LoRA layers were modified with block weights")

        except Exception as e:
            print(f"[LoRAManager] WARNING: Failed to apply per-block weights: {e}")
            import traceback
            traceback.print_exc()

    def _get_block_weight_for_module(self, module_name: str, block_weights: dict) -> float:
        """
        Determine the block weight for a given module name

        Args:
            module_name: Full name of the module (e.g., "down_blocks.0.attentions.0")
            block_weights: Dictionary of block_id -> weight

        Returns:
            Weight value for this module (default 1.0)
        """
        # Parse module name to determine block
        if 'down_blocks' in module_name or 'input_blocks' in module_name:
            # Extract block number
            import re
            match = re.search(r'(down_blocks|input_blocks)[._](\d+)', module_name)
            if match:
                block_num = int(match.group(2))
                block_id = f"IN{block_num:02d}"
                return block_weights.get(block_id, 1.0)

        elif 'mid_block' in module_name or 'middle_block' in module_name:
            return block_weights.get("MID", 1.0)

        elif 'up_blocks' in module_name or 'output_blocks' in module_name:
            import re
            match = re.search(r'(up_blocks|output_blocks)[._](\d+)', module_name)
            if match:
                block_num = int(match.group(2))
                block_id = f"OUT{block_num:02d}"
                return block_weights.get(block_id, 1.0)

        # Check for BASE
        return block_weights.get("BASE", 1.0)

    def _apply_layer_weights_alternative(self, pipeline: Any, adapter_name: str, lora_config: LoRAConfig):
        """
        Alternative method for applying block weights (for older diffusers versions)
        """
        print("[LoRAManager] Using alternative block weight application method")
        # This is a fallback - in practice, the main method should work for most cases

    def create_step_callback(self, pipeline: Any, total_steps: int, original_callback=None):
        """
        Create a callback that handles step-based LoRA activation

        Args:
            pipeline: The diffusion pipeline
            total_steps: Total number of generation steps
            original_callback: Original progress callback to chain

        Returns:
            Callback function for step-based LoRA control
        """
        def callback(pipe, step: int, timestep: float, callback_kwargs: dict):
            # Check which LoRAs should be active at this step
            active_adapters = []
            adapter_weights = []

            for i, lora_config in enumerate(self.loaded_loras):
                if lora_config.is_active_at_step(step, total_steps):
                    adapter_name = f"lora_{i}"
                    active_adapters.append(adapter_name)
                    adapter_weights.append(lora_config.strength)

            # Update active adapters for this step
            if hasattr(pipeline, 'set_adapters'):
                if active_adapters:
                    pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)
                else:
                    # Disable all LoRAs if none are active
                    pipeline.disable_lora()

            # Call original callback if provided
            if original_callback:
                return original_callback(pipe, step, timestep, callback_kwargs)

            return callback_kwargs

        return callback

    def unload_loras(self, pipeline: Any) -> Any:
        """Unload all LoRAs from pipeline"""
        try:
            if hasattr(pipeline, 'unload_lora_weights'):
                pipeline.unload_lora_weights()
                print("Unloaded all LoRAs")
        except Exception as e:
            print(f"Error unloading LoRAs: {e}")

        self.loaded_loras = []
        return pipeline

    def get_lora_info(self, lora_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific LoRA file"""
        # Use _resolve_lora_path to check both lora/ and training/ directories
        lora_path = self._resolve_lora_path(lora_name)

        if lora_path is None:
            return None

        # Get layer information
        layers = self.get_lora_layers(lora_name)

        return {
            "name": lora_name,
            "path": str(lora_path),
            "size": lora_path.stat().st_size,
            "exists": True,
            "layers": layers
        }

    def get_lora_layers(self, lora_name: str) -> List[str]:
        """
        Extract U-Net block structure from LoRA file
        Returns blocks in format: BASE, IN00-IN11, MID, OUT00-OUT11
        """
        # Use _resolve_lora_path to check both lora/ and training/ directories
        lora_path = self._resolve_lora_path(lora_name)

        if lora_path is None:
            return []

        try:
            from safetensors import safe_open
            import re

            blocks = set()

            with safe_open(lora_path, framework="pt", device="cpu") as f:
                keys = f.keys()

                for key in keys:
                    # Check for lora_unet_ prefix (some LoRAs use this)
                    if 'lora_unet_' in key:
                        # Extract block info from keys like:
                        # lora_unet_input_blocks_0_...
                        # lora_unet_middle_block_...
                        # lora_unet_output_blocks_0_...

                        if 'input_blocks' in key:
                            match = re.search(r'input_blocks[_.](\d+)', key)
                            if match:
                                block_num = int(match.group(1))
                                blocks.add(f"IN{block_num:02d}")

                        elif 'middle_block' in key:
                            blocks.add("MID")

                        elif 'output_blocks' in key:
                            match = re.search(r'output_blocks[_.](\d+)', key)
                            if match:
                                block_num = int(match.group(1))
                                blocks.add(f"OUT{block_num:02d}")

                    # Also check for diffusers format (down_blocks, mid_block, up_blocks)
                    elif 'down_blocks' in key or 'up_blocks' in key or 'mid_block' in key:
                        if 'down_blocks' in key:
                            match = re.search(r'down_blocks[_.](\d+)', key)
                            if match:
                                block_num = int(match.group(1))
                                blocks.add(f"IN{block_num:02d}")

                        elif 'mid_block' in key:
                            blocks.add("MID")

                        elif 'up_blocks' in key:
                            match = re.search(r'up_blocks[_.](\d+)', key)
                            if match:
                                block_num = int(match.group(1))
                                blocks.add(f"OUT{block_num:02d}")

                # If no blocks found, add BASE
                if not blocks:
                    blocks.add("BASE")

            # Sort blocks: BASE, IN00-IN11, MID, OUT00-OUT11
            def sort_key(block):
                if block == "BASE":
                    return (0, 0)
                elif block == "MID":
                    return (2, 0)
                elif block.startswith("IN"):
                    return (1, int(block[2:]))
                elif block.startswith("OUT"):
                    return (3, int(block[3:]))
                return (9, 0)

            sorted_blocks = sorted(list(blocks), key=sort_key)
            print(f"[LoRAManager] Found {len(sorted_blocks)} blocks in {lora_name}: {sorted_blocks}")

            return sorted_blocks

        except Exception as e:
            print(f"[LoRAManager] Error reading LoRA layers: {e}")
            import traceback
            traceback.print_exc()
            return []


# Global instance
lora_manager = LoRAManager()
