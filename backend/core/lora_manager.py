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
        self.unet_layer_weights = unet_layer_weights or {"down": 1.0, "mid": 1.0, "up": 1.0}
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
        if not lora_configs:
            return pipeline

        # Parse configs
        self.loaded_loras = [LoRAConfig.from_dict(cfg) for cfg in lora_configs]

        # Load LoRAs using diffusers' native support
        try:
            for i, lora_config in enumerate(self.loaded_loras):
                lora_path = self.lora_dir / lora_config.path

                if not lora_path.exists():
                    print(f"Warning: LoRA file not found: {lora_path}")
                    continue

                print(f"Loading LoRA {i+1}/{len(self.loaded_loras)}: {lora_config.path}")

                # Load LoRA weights
                adapter_name = f"lora_{i}"
                pipeline.load_lora_weights(
                    str(lora_path.parent),
                    weight_name=lora_path.name,
                    adapter_name=adapter_name
                )

                # Set adapter with strength
                # Note: Layer-specific weights and step ranges will be handled in callback
                if hasattr(pipeline, 'set_adapters'):
                    pipeline.set_adapters(adapter_name, adapter_weights=lora_config.strength)

            print(f"Successfully loaded {len(self.loaded_loras)} LoRA(s)")

        except Exception as e:
            print(f"Error loading LoRAs: {e}")
            import traceback
            traceback.print_exc()

        return pipeline

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

        return {
            "name": lora_name,
            "path": str(lora_path),
            "size": lora_path.stat().st_size,
            "exists": True
        }


# Global instance
lora_manager = LoRAManager()
