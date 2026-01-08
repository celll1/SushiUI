"""
Full Parameter Trainer for Stable Diffusion models.

This is a modular implementation using model-specific adapters:
- SD15FullParameterAdapter: SD1.5 models
- SDXLFullParameterAdapter: SDXL models
- ZImageFullParameterAdapter: Z-Image models
- DEUSFullParameterAdapter: DEUS models

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

from .base_trainer import BaseTrainer
from .adapters import (
    SD15FullParameterAdapter,
    SDXLFullParameterAdapter,
    ZImageFullParameterAdapter,
    DEUSFullParameterAdapter,
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
        train_image_encoder: bool = False,  # DEUS Image Encoder (future T2I support)
        **kwargs
    ):
        """
        Initialize Full Parameter Trainer.

        Args:
            train_unet: Whether to train U-Net/Transformer
            train_text_encoder: Whether to train Text Encoder(s)
            train_image_encoder: Whether to train Image Encoder (DEUS only, future T2I)
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # Full fine-tune settings (set before super().__init__)
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder
        self.train_image_encoder = train_image_encoder

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Override log prefix
        self.log_prefix = "[Full Parameter Trainer]"

        # Create model-specific adapter
        self._create_adapter()

        # Prepare models for training using adapter
        self._prepare_models()

        print(f"{self.log_prefix} Initialized")
        print(f"{self.log_prefix} Training U-Net: {self.train_unet}, Text Encoder: {self.train_text_encoder}, Image Encoder: {self.train_image_encoder}")

    def _create_adapter(self):
        """Create model-specific Full Parameter adapter based on detected model type."""
        if self.is_zimage:
            self.adapter = ZImageFullParameterAdapter(self)
            print(f"{self.log_prefix} Using ZImageFullParameterAdapter")
        elif self.is_deus:
            self.adapter = DEUSFullParameterAdapter(self)
            print(f"{self.log_prefix} Using DEUSFullParameterAdapter")
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

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load full parameter checkpoint for resuming training.

        Args:
            checkpoint_path: Path to checkpoint directory (diffusers format)

        Returns:
            Step number from checkpoint
        """
        import json

        print(f"{self.log_prefix} Loading checkpoint: {checkpoint_path}")

        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        # Load metadata to get step number
        metadata_path = checkpoint_dir / "metadata.json"
        step = 0
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                step = metadata.get('step', 0)

        # Load U-Net
        if self.train_unet:
            unet_path = checkpoint_dir / "unet"
            if unet_path.exists() and self.unet is not None:
                from diffusers import UNet2DConditionModel
                loaded_unet = UNet2DConditionModel.from_pretrained(unet_path)
                self.unet.load_state_dict(loaded_unet.state_dict())
                print(f"{self.log_prefix} Loaded U-Net from {unet_path}")

        # Load Text Encoders
        if self.train_text_encoder:
            te1_path = checkpoint_dir / "text_encoder"
            if te1_path.exists() and self.text_encoder is not None:
                from transformers import CLIPTextModel
                loaded_te1 = CLIPTextModel.from_pretrained(te1_path)
                self.text_encoder.load_state_dict(loaded_te1.state_dict())
                print(f"{self.log_prefix} Loaded Text Encoder 1 from {te1_path}")

            if self.is_sdxl:
                te2_path = checkpoint_dir / "text_encoder_2"
                if te2_path.exists() and self.text_encoder_2 is not None:
                    from transformers import CLIPTextModelWithProjection
                    loaded_te2 = CLIPTextModelWithProjection.from_pretrained(te2_path)
                    self.text_encoder_2.load_state_dict(loaded_te2.state_dict())
                    print(f"{self.log_prefix} Loaded Text Encoder 2 from {te2_path}")

        print(f"{self.log_prefix} Loaded checkpoint from step {step}")
        return step
