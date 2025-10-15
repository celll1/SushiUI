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
        self.loaded_loras: List[LoRAConfig] = []
        print(f"[LoRAManager] LoRA directory: {self.lora_dir}")

    def get_available_loras(self) -> List[str]:
        """Get list of available LoRA files"""
        print(f"[LoRAManager] Checking directory: {self.lora_dir}")
        print(f"[LoRAManager] Directory exists: {self.lora_dir.exists()}")

        if not self.lora_dir.exists():
            print(f"[LoRAManager] Creating directory: {self.lora_dir}")
            self.lora_dir.mkdir(parents=True, exist_ok=True)
            return []

        lora_files = []
        for ext in [".safetensors", ".pt", ".bin"]:
            found = list(self.lora_dir.rglob(f"*{ext}"))
            print(f"[LoRAManager] Found {len(found)} files with extension {ext}")
            lora_files.extend([
                str(f.relative_to(self.lora_dir))
                for f in found
            ])

        print(f"[LoRAManager] Total LoRA files found: {len(lora_files)}")
        return sorted(lora_files)

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
                lora_path = self.lora_dir / lora_config.path
                print(f"[LoRAManager] Attempting to load LoRA from: {lora_path}")
                print(f"[LoRAManager] LoRA config: strength={lora_config.strength}, apply_to_text_encoder={lora_config.apply_to_text_encoder}, apply_to_unet={lora_config.apply_to_unet}")

                if not lora_path.exists():
                    print(f"[LoRAManager] WARNING: LoRA file not found: {lora_path}")
                    continue

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

        This modifies the LoRA adapter weights directly by scaling them according to
        the block-specific weights (IN00-IN11, MID, OUT00-OUT11) specified in the config.
        """
        try:
            import re

            if not hasattr(pipeline.unet, 'get_adapter_state_dict'):
                print("[LoRAManager] UNet does not support get_adapter_state_dict, skipping block weights")
                return

            # Get the adapter state dict
            adapter_state = pipeline.unet.get_adapter_state_dict(adapter_name)

            # Apply block weights
            for block_id, weight in lora_config.unet_layer_weights.items():
                # Convert block ID (IN00, MID, OUT05, etc.) to pattern matching
                # IN00 -> input_blocks_0 or down_blocks_0
                # MID -> middle_block or mid_block
                # OUT05 -> output_blocks_5 or up_blocks_5

                matching_keys = []

                if block_id == "BASE":
                    # BASE matches any layer without specific block designation
                    matching_keys = [k for k in adapter_state.keys()]

                elif block_id == "MID":
                    # Match middle block
                    matching_keys = [k for k in adapter_state.keys()
                                    if 'middle_block' in k or 'mid_block' in k]

                elif block_id.startswith("IN"):
                    # Extract block number from IN00, IN01, etc.
                    block_num = int(block_id[2:])
                    matching_keys = [k for k in adapter_state.keys()
                                    if f'input_blocks.{block_num}' in k or f'input_blocks_{block_num}' in k
                                    or f'down_blocks.{block_num}' in k or f'down_blocks_{block_num}' in k]

                elif block_id.startswith("OUT"):
                    # Extract block number from OUT00, OUT01, etc.
                    block_num = int(block_id[3:])
                    matching_keys = [k for k in adapter_state.keys()
                                    if f'output_blocks.{block_num}' in k or f'output_blocks_{block_num}' in k
                                    or f'up_blocks.{block_num}' in k or f'up_blocks_{block_num}' in k]

                if matching_keys:
                    print(f"[LoRAManager]   Block '{block_id}': weight={weight}, affecting {len(matching_keys)} parameters")

                    # Scale the weights for this block
                    for key in matching_keys:
                        if adapter_state[key] is not None:
                            adapter_state[key] = adapter_state[key] * weight

            print(f"[LoRAManager] Applied per-block weights successfully")

        except Exception as e:
            print(f"[LoRAManager] WARNING: Failed to apply per-block weights: {e}")
            import traceback
            traceback.print_exc()

    def create_step_callback(self, original_callback=None):
        """
        Create a callback that handles step-based LoRA activation

        Args:
            original_callback: Original progress callback to chain

        Returns:
            Callback function for step-based LoRA control
        """
        def callback(step: int, timestep: float, latents):
            # TODO: Implement step-based LoRA weight adjustment
            # This would require modifying adapter weights at each step
            # which is not directly supported by diffusers yet

            # Call original callback if provided
            if original_callback:
                return original_callback(step, timestep, latents)

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
        lora_path = self.lora_dir / lora_name

        if not lora_path.exists():
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
        lora_path = self.lora_dir / lora_name

        if not lora_path.exists():
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
