"""
LoRA (Low-Rank Adaptation) Trainer for Stable Diffusion models.

This is a modular implementation using model-specific adapters:
- SD15LoRAAdapter: SD1.5 models
- SDXLLoRAAdapter: SDXL models
- ZImageLoRAAdapter: Z-Image models
- DEUSLoRAAdapter: DEUS models

Key improvements:
- Model-specific logic separated into adapters
- Supports SD1.5, SDXL, Z-Image, and DEUS
- Clean separation of concerns
- Easy to extend with new model types

References:
- sd-scripts (Apache-2 license) by kohya-ss
- ai-toolkit (MIT license) by ostris
- musubi-tuner (Apache-2 license) by kohya-ss (Z-Image support)

Author: Claude (2026-01-04)
Last Updated: Claude (2026-01-08) - Added DEUS support
"""

from pathlib import Path
from typing import Dict, List
import torch.nn as nn

from .base_trainer import BaseTrainer
from .adapters import (
    SD15LoRAAdapter,
    SDXLLoRAAdapter,
    ZImageLoRAAdapter,
    DEUSLoRAAdapter,
)


class LoRATrainer(BaseTrainer):
    """
    LoRA Trainer for SD/SDXL/Z-Image models.

    Uses model-specific adapters for LoRA injection, parameter collection,
    and checkpoint saving.
    """

    def __init__(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dtype: str = 'fp32',
        train_unet: bool = True,
        train_text_encoder: bool = False,
        **kwargs
    ):
        """
        Initialize LoRA Trainer.

        Args:
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha (scaling factor = alpha / rank)
            lora_dtype: Data type for LoRA weights ('fp32', 'fp16', 'bf16')
            train_unet: Whether to train U-Net/Transformer
            train_text_encoder: Whether to train Text Encoder(s)
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # LoRA-specific settings (set before super().__init__)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / lora_rank
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder

        # LoRA modules storage
        self.lora_layers: Dict[str, nn.Module] = {}

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Convert lora_dtype string to torch.dtype (after super().__init__ to have access to get_torch_dtype)
        from .base_trainer import get_torch_dtype
        self.lora_dtype = get_torch_dtype(lora_dtype)

        # Override log prefix
        self.log_prefix = "[LoRA Trainer]"

        # Create model-specific adapter
        self._create_adapter()

        # Apply LoRA using adapter
        self._apply_lora()

        print(f"{self.log_prefix} Initialized (rank={self.lora_rank}, alpha={self.lora_alpha})")
        print(f"{self.log_prefix} Training U-Net: {self.train_unet}, Text Encoder: {self.train_text_encoder}")

    def _create_adapter(self):
        """Create model-specific LoRA adapter based on detected model type."""
        if self.is_zimage:
            self.adapter = ZImageLoRAAdapter(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
            print(f"{self.log_prefix} Using ZImageLoRAAdapter")
        elif self.is_deus:
            self.adapter = DEUSLoRAAdapter(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
            print(f"{self.log_prefix} Using DEUSLoRAAdapter")
        elif self.is_sdxl:
            self.adapter = SDXLLoRAAdapter(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
            print(f"{self.log_prefix} Using SDXLLoRAAdapter")
        else:
            self.adapter = SD15LoRAAdapter(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
            print(f"{self.log_prefix} Using SD15LoRAAdapter")

    def _apply_lora(self):
        """Apply LoRA to U-Net/Transformer and Text Encoders using adapter."""
        print(f"{self.log_prefix} Applying LoRA layers...")

        # Apply LoRA to U-Net/Transformer
        if self.train_unet:
            unet_count = self.adapter.apply_lora_to_unet(self.lora_layers)
            print(f"{self.log_prefix} Injected {unet_count} LoRA layers into U-Net/Transformer")
        else:
            print(f"{self.log_prefix} U-Net/Transformer LoRA skipped (train_unet=False)")

        # Apply LoRA to Text Encoder(s)
        if self.train_text_encoder:
            te_count = self.adapter.apply_lora_to_text_encoders(self.lora_layers)
            print(f"{self.log_prefix} Injected {te_count} LoRA layers into Text Encoder(s)")
        else:
            print(f"{self.log_prefix} Text Encoder LoRA skipped (train_text_encoder=False)")

        print(f"{self.log_prefix} Total LoRA layers: {len(self.lora_layers)}")

    def setup_trainable_parameters(self) -> List[Dict]:
        """
        Collect trainable parameters with per-component learning rates.

        Uses adapter to handle model-specific parameter grouping.

        Returns:
            List of parameter groups for optimizer
        """
        return self.adapter.setup_trainable_parameters(self.lora_layers)

    def save_checkpoint(self, step: int, epoch: int):
        """
        Save LoRA checkpoint.

        Uses adapter to handle model-specific checkpoint format.

        Args:
            step: Current training step
            epoch: Current training epoch
        """
        checkpoint_path = self.output_dir / f"{self.run_name}_step_{step:06d}.safetensors"
        self.adapter.save_checkpoint(self.lora_layers, step, epoch, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load LoRA checkpoint for resuming training.

        Args:
            checkpoint_path: Path to LoRA checkpoint (.safetensors)

        Returns:
            Step number from checkpoint
        """
        from safetensors import safe_open
        from safetensors.torch import load_file
        import torch

        print(f"{self.log_prefix} Loading LoRA checkpoint: {checkpoint_path}")

        # Extract metadata using safe_open
        step = 0
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if metadata and 'step' in metadata:
                step = int(metadata['step'])

        # Load checkpoint weights
        checkpoint = load_file(checkpoint_path)

        # Load LoRA weights into existing layers
        for lora_name, lora_layer in self.lora_layers.items():
            # Load lora_down weight
            down_key = f"{lora_name}.lora_down.weight"
            if down_key in checkpoint:
                lora_layer.lora_down.weight.data.copy_(checkpoint[down_key])

            # Load lora_up weight
            up_key = f"{lora_name}.lora_up.weight"
            if up_key in checkpoint:
                lora_layer.lora_up.weight.data.copy_(checkpoint[up_key])

        print(f"{self.log_prefix} Loaded LoRA checkpoint from step {step}")
        return step
