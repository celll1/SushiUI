"""
Base adapter class for ControlNet training.

Supports two ControlNet types:
- Standard ControlNet (diffusers ControlNetModel)
- ControlNet-LLLite (kohya-ss sd-scripts compatible)

Author: Claude (2026-01-26)
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn


class BaseControlNetAdapter(ABC):
    """
    Abstract base class for model-specific ControlNet training adapters.

    Each model architecture (SD1.5, SDXL) implements this interface
    to provide model-specific ControlNet creation, parameter collection,
    and checkpoint saving logic.
    """

    def __init__(self, trainer, controlnet_type: str = "standard"):
        """
        Initialize adapter.

        Args:
            trainer: Parent ControlNetTrainer instance
            controlnet_type: "standard" (diffusers ControlNetModel) or "lllite" (sd-scripts compatible)
        """
        self.trainer = trainer
        self.controlnet_type = controlnet_type

    @abstractmethod
    def create_controlnet(
        self,
        init_from_unet: bool = True,
        pretrained_path: Optional[str] = None,
    ) -> nn.Module:
        """
        Create or load ControlNet model.

        Standard mode:
        - init_from_unet=True, pretrained_path=None: Initialize ControlNet from base UNet weights
        - init_from_unet=False, pretrained_path=path: Load from existing ControlNet checkpoint
        - init_from_unet=False, pretrained_path=None: Initialize ControlNet with random weights

        LLLite mode:
        - pretrained_path=None: Create new LLLite modules (zero-initialized up projection)
        - pretrained_path=path: Load existing LLLite checkpoint

        Args:
            init_from_unet: Initialize ControlNet weights from base UNet (standard only)
            pretrained_path: Path to existing ControlNet checkpoint to resume from

        Returns:
            Trainable ControlNet module (ControlNetModel or LLLiteModule)
        """
        pass

    @abstractmethod
    def setup_trainable_parameters(self, controlnet: nn.Module) -> List[Dict[str, Any]]:
        """
        Collect trainable parameters from ControlNet for optimizer.

        Args:
            controlnet: ControlNet module (ControlNetModel or LLLiteModule)

        Returns:
            List of parameter groups for optimizer
            Format: [{"params": [...], "lr": float}, ...]
        """
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        controlnet: nn.Module,
        step: int,
        epoch: int,
        output_path: Path,
    ):
        """
        Save ControlNet checkpoint.

        Standard: saves as diffusers-compatible directory
        LLLite: saves as sd-scripts compatible .safetensors

        Args:
            controlnet: ControlNet module
            step: Current training step
            epoch: Current training epoch
            output_path: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, controlnet: nn.Module, checkpoint_path: str) -> int:
        """
        Load ControlNet checkpoint for resume training.

        Args:
            controlnet: ControlNet module to load weights into
            checkpoint_path: Path to checkpoint

        Returns:
            Training step number extracted from checkpoint filename (0 if not determinable)
        """
        pass

    @abstractmethod
    def controlnet_forward(
        self,
        controlnet: nn.Module,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        condition_images: torch.Tensor,
        added_cond_kwargs: Optional[Dict] = None,
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor]]:
        """
        ControlNet forward pass.

        Standard: Returns (down_block_residuals, mid_block_residual)
        LLLite: Applies patches to UNet internally, returns None

        Args:
            controlnet: ControlNet module
            noisy_latents: Noisy latent tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            text_embeddings: Text embedding tensor [B, seq_len, dim]
            condition_images: Condition image tensor [B, 3, H, W] in [0, 1] range
            added_cond_kwargs: Additional conditioning (SDXL: pooled_embeddings + time_ids)

        Returns:
            Standard: (down_block_res_samples, mid_block_res_sample)
            LLLite: None (patches applied directly to UNet)
        """
        pass

    def _extract_step_from_path(self, path: Path) -> int:
        """Extract training step number from checkpoint path name."""
        name = path.stem if path.is_file() else path.name

        # Pattern: step_NNNN or step-NNNN
        match = re.search(r'step[_-](\d+)', name, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Pattern: sNNNN
        match = re.search(r's(\d+)', name, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return 0
