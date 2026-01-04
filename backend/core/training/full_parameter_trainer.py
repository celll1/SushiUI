"""
Full Parameter Trainer for Stable Diffusion models.

This is a modular implementation using model-specific adapters:
- SD15FullParameterAdapter: SD1.5 models
- SDXLFullParameterAdapter: SDXL models
- ZImageFullParameterAdapter: Z-Image models

Key improvements:
- Model-specific logic separated into adapters
- Supports SD1.5, SDXL, and Z-Image
- Clean separation of concerns
- Easy to extend with new model types

References:
- sd-scripts (Apache-2 license) by kohya-ss
- ai-toolkit (MIT license) by ostris
- musubi-tuner (Apache-2 license) by kohya-ss (Z-Image support)

Author: Claude (2026-01-04)
"""

from pathlib import Path
from typing import Dict, List

from .base_trainer import BaseTrainer
from .adapters import (
    SD15FullParameterAdapter,
    SDXLFullParameterAdapter,
    ZImageFullParameterAdapter,
)


class FullParameterTrainer(BaseTrainer):
    """
    Full Parameter Trainer for SD/SDXL/Z-Image models.

    Uses model-specific adapters for parameter preparation, collection,
    and checkpoint saving.
    """

    def __init__(
        self,
        train_unet: bool = True,
        train_text_encoder: bool = False,
        **kwargs
    ):
        """
        Initialize Full Parameter Trainer.

        Args:
            train_unet: Whether to train U-Net/Transformer
            train_text_encoder: Whether to train Text Encoder(s)
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # Full fine-tune settings (set before super().__init__)
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Override log prefix
        self.log_prefix = "[Full Parameter Trainer]"

        # Create model-specific adapter
        self._create_adapter()

        # Prepare models for training using adapter
        self._prepare_models()

        print(f"{self.log_prefix} Initialized")
        print(f"{self.log_prefix} Training U-Net: {self.train_unet}, Text Encoder: {self.train_text_encoder}")

    def _create_adapter(self):
        """Create model-specific Full Parameter adapter based on detected model type."""
        if self.is_zimage:
            self.adapter = ZImageFullParameterAdapter(self)
            print(f"{self.log_prefix} Using ZImageFullParameterAdapter")
        elif self.is_sdxl:
            self.adapter = SDXLFullParameterAdapter(self)
            print(f"{self.log_prefix} Using SDXLFullParameterAdapter")
        else:
            self.adapter = SD15FullParameterAdapter(self)
            print(f"{self.log_prefix} Using SD15FullParameterAdapter")

    def _prepare_models(self):
        """Prepare models for full parameter training using adapter."""
        self.adapter.prepare_models_for_training()

    def setup_trainable_parameters(self) -> List[Dict]:
        """
        Collect trainable parameters with per-component learning rates.

        Uses adapter to handle model-specific parameter grouping.

        Returns:
            List of parameter groups for optimizer
        """
        return self.adapter.setup_trainable_parameters()

    def save_checkpoint(self, step: int, epoch: int):
        """
        Save full parameter checkpoint.

        Uses adapter to handle model-specific checkpoint format.

        Args:
            step: Current training step
            epoch: Current training epoch
        """
        checkpoint_path = self.output_dir / f"{self.run_name}_step_{step:06d}"
        self.adapter.save_checkpoint(step, epoch, checkpoint_path)
