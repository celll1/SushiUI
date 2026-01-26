"""
ControlNet Trainer for Stable Diffusion models.

This is a modular implementation using model-specific adapters:
- ControlNetSD15Adapter: SD1.5 Standard ControlNet / LLLite
- ControlNetSDXLAdapter: SDXL Standard ControlNet / LLLite [Phase 3]

Implements the third training mode alongside LoRA and Full Parameter.
ControlNet training freezes UNet/VAE/TE entirely and only trains
the ControlNet module.

References:
- diffusers ControlNetModel (Apache-2 license)
- sd-scripts (Apache-2 license) by kohya-ss (LLLite implementation)

Author: Claude (2026-01-26)
"""

from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from .adapters import ControlNetSD15Adapter, ControlNetSDXLAdapter


class ControlNetTrainer(BaseTrainer):
    """
    ControlNet Trainer for SD1.5/SDXL models.

    Uses model-specific adapters for ControlNet creation, training,
    and checkpoint management.

    Supports:
    - Standard ControlNet (diffusers ControlNetModel)
    - ControlNet-LLLite (kohya-ss sd-scripts compatible) [Phase 2]
    """

    def __init__(
        self,
        controlnet_type: str = "standard",
        controlnet_pretrained_path: Optional[str] = None,
        init_from_unet: bool = True,
        # LLLite parameters (Phase 2)
        lllite_conditioning_channels: int = 32,
        lllite_rank: int = 64,
        # Condition generation (Phase 4)
        condition_preprocessors: Optional[List[str]] = None,
        condition_cache_mode: str = "on_the_fly",
        **kwargs
    ):
        """
        Initialize ControlNet Trainer.

        Args:
            controlnet_type: "standard" (diffusers ControlNetModel) or "lllite" (sd-scripts compatible)
            controlnet_pretrained_path: Path to existing ControlNet checkpoint for resume
            init_from_unet: Initialize ControlNet weights from base UNet (standard only)
            lllite_conditioning_channels: Number of conditioning channels for LLLite (Phase 2)
            lllite_rank: Rank for LLLite linear layers (Phase 2)
            condition_preprocessors: List of controlnet-aux preprocessor types (Phase 4)
            condition_cache_mode: "pre_generate" or "on_the_fly" (Phase 4)
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # ControlNet-specific settings (set before super().__init__)
        self.controlnet_type = controlnet_type
        self.controlnet_pretrained_path = controlnet_pretrained_path
        self.init_from_unet = init_from_unet
        self.lllite_conditioning_channels = lllite_conditioning_channels
        self.lllite_rank = lllite_rank
        self.condition_preprocessors = condition_preprocessors
        self.condition_cache_mode = condition_cache_mode

        # ControlNet module storage (set by _create_controlnet)
        self.controlnet: Optional[nn.Module] = None

        # ControlNet training does NOT train UNet/TE
        self.train_unet = False
        self.train_text_encoder = False
        self.train_image_encoder = False

        # Flag to signal base_trainer to load condition images
        self.use_condition_images = True

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Override log prefix
        self.log_prefix = "[ControlNet Trainer]"

        # Validate model type (only SD1.5/SDXL supported)
        if self.is_zimage or self.is_flux2:
            model_type = "Z-Image" if self.is_zimage else "FLUX.2"
            raise ValueError(
                f"ControlNet training is only supported for SD1.5 and SDXL models. "
                f"Detected model type: {model_type}"
            )
        if self.is_deus:
            raise ValueError(
                f"ControlNet training is only supported for SD1.5 and SDXL models. "
                f"Detected model type: DEUS"
            )

        # Freeze all base model components
        self._freeze_base_models()

        # Create model-specific adapter
        self._create_adapter()

        # Create ControlNet using adapter
        self._create_controlnet()

        print(f"{self.log_prefix} Initialized")
        print(f"{self.log_prefix} ControlNet type: {self.controlnet_type}")
        print(f"{self.log_prefix} Model type: {'SDXL' if self.is_sdxl else 'SD1.5'}")

    def _freeze_base_models(self):
        """Freeze all base model components (UNet, VAE, TE)."""
        print(f"{self.log_prefix} Freezing all base model components...")

        if self.unet is not None:
            self.unet.requires_grad_(False)
            self.unet.eval()
            print(f"  UNet: frozen")

        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae.eval()
            print(f"  VAE: frozen")

        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
            print(f"  Text Encoder 1: frozen")

        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)
            self.text_encoder_2.eval()
            print(f"  Text Encoder 2: frozen")

    def _create_adapter(self):
        """Create model-specific ControlNet adapter based on detected model type."""
        if self.is_sdxl:
            self.adapter = ControlNetSDXLAdapter(self, self.controlnet_type)
            print(f"{self.log_prefix} Using ControlNetSDXLAdapter ({self.controlnet_type})")
        else:
            self.adapter = ControlNetSD15Adapter(self, self.controlnet_type)
            print(f"{self.log_prefix} Using ControlNetSD15Adapter ({self.controlnet_type})")

    def _create_controlnet(self):
        """Create ControlNet model using adapter."""
        print(f"{self.log_prefix} Creating ControlNet...")

        self.controlnet = self.adapter.create_controlnet(
            init_from_unet=self.init_from_unet,
            pretrained_path=self.controlnet_pretrained_path,
        )

        # Enable gradient checkpointing for ControlNet
        if hasattr(self.controlnet, 'enable_gradient_checkpointing'):
            self.controlnet.enable_gradient_checkpointing()
            print(f"{self.log_prefix} Gradient checkpointing enabled for ControlNet")

        print(f"{self.log_prefix} ControlNet created successfully")

    def setup_trainable_parameters(self) -> List[Dict]:
        """
        Collect trainable parameters from ControlNet.

        Returns:
            List of parameter groups for optimizer
        """
        return self.adapter.setup_trainable_parameters(self.controlnet)

    def save_checkpoint(self, step: int, epoch: int):
        """
        Save ControlNet checkpoint.

        Standard: saves as diffusers-compatible directory
        LLLite: saves as sd-scripts compatible .safetensors [Phase 2]

        Args:
            step: Current training step
            epoch: Current training epoch
        """
        if self.controlnet_type == "standard":
            # Directory format: {run_name}_controlnet_step_001000/
            checkpoint_path = self.output_dir / f"{self.run_name}_controlnet_step_{step:06d}"
        else:
            # LLLite: single file format
            checkpoint_path = self.output_dir / f"{self.run_name}_lllite_step_{step:06d}.safetensors"

        self.adapter.save_checkpoint(self.controlnet, step, epoch, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load ControlNet checkpoint for resume training.

        Args:
            checkpoint_path: Path to checkpoint directory or file

        Returns:
            Step number from checkpoint
        """
        print(f"{self.log_prefix} Loading checkpoint: {checkpoint_path}")
        step = self.adapter.load_checkpoint(self.controlnet, checkpoint_path)
        print(f"{self.log_prefix} Loaded checkpoint from step {step}")
        return step
