"""
SD1.5 ControlNet training adapter.

Supports:
- Standard ControlNet (diffusers ControlNetModel)
- ControlNet-LLLite (kohya-ss sd-scripts compatible)

Author: Claude (2026-01-26)
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from diffusers import ControlNetModel
from safetensors.torch import save_file as safetensors_save_file, load_file as safetensors_load_file

from .base_controlnet_adapter import BaseControlNetAdapter
from ..lllite_module import LLLiteModule


class ControlNetSD15Adapter(BaseControlNetAdapter):
    """
    ControlNet training adapter for SD1.5.

    Standard mode:
    - Creates ControlNetModel from UNet weights or loads from checkpoint
    - Forward: controlnet(latents, t, embeds, cond) -> (down_res, mid_res) -> UNet with residuals
    - Save: diffusers-compatible directory (config.json + safetensors)
    """

    def create_controlnet(
        self,
        init_from_unet: bool = True,
        pretrained_path: Optional[str] = None,
    ) -> nn.Module:
        """
        Create or load SD1.5 ControlNet.

        Args:
            init_from_unet: Initialize from base UNet weights (Standard only)
            pretrained_path: Path to existing checkpoint to resume from

        Returns:
            ControlNetModel instance
        """
        if self.controlnet_type == "standard":
            return self._create_standard_controlnet(init_from_unet, pretrained_path)
        elif self.controlnet_type == "lllite":
            return self._create_lllite_controlnet(pretrained_path)
        else:
            raise ValueError(
                f"Unknown ControlNet type '{self.controlnet_type}' for SD1.5. "
                f"Supported types: 'standard', 'lllite'"
            )

    def _create_standard_controlnet(
        self,
        init_from_unet: bool,
        pretrained_path: Optional[str],
    ) -> ControlNetModel:
        """Create Standard ControlNet for SD1.5."""
        unet = self.trainer.unet

        if pretrained_path is not None:
            # Load from existing checkpoint
            pretrained = Path(pretrained_path)
            print(f"[ControlNetSD15] Loading ControlNet from: {pretrained}")

            if pretrained.is_dir():
                controlnet = ControlNetModel.from_pretrained(
                    str(pretrained),
                    torch_dtype=unet.dtype,
                )
            else:
                controlnet = ControlNetModel.from_single_file(
                    str(pretrained),
                    torch_dtype=unet.dtype,
                )
            print(f"[ControlNetSD15] Loaded ControlNet from checkpoint")

        elif init_from_unet:
            # Initialize from UNet weights
            print(f"[ControlNetSD15] Initializing ControlNet from UNet weights")
            controlnet = ControlNetModel.from_unet(
                unet,
                load_weights_from_unet=True,
                conditioning_channels=3,
            )
            print(f"[ControlNetSD15] ControlNet initialized from UNet")

        else:
            # Initialize with random weights (from UNet architecture but no weight copy)
            print(f"[ControlNetSD15] Initializing ControlNet with random weights (UNet architecture)")
            controlnet = ControlNetModel.from_unet(
                unet,
                load_weights_from_unet=False,
                conditioning_channels=3,
            )
            print(f"[ControlNetSD15] ControlNet initialized with random weights")

        # Move to same device/dtype as UNet
        controlnet = controlnet.to(device=unet.device, dtype=unet.dtype)

        # Set to training mode
        controlnet.train()
        controlnet.requires_grad_(True)

        # Log parameter count
        total_params = sum(p.numel() for p in controlnet.parameters())
        trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
        print(f"[ControlNetSD15] ControlNet parameters: {total_params:,} total, {trainable_params:,} trainable")

        return controlnet

    def _create_lllite_controlnet(
        self,
        pretrained_path: Optional[str],
    ) -> LLLiteModule:
        """Create LLLite ControlNet for SD1.5."""
        unet = self.trainer.unet
        conditioning_channels = self.trainer.lllite_conditioning_channels
        rank = self.trainer.lllite_rank

        if pretrained_path is not None:
            # Load from existing kohya-ss compatible checkpoint
            pretrained = Path(pretrained_path)
            print(f"[ControlNetSD15] Loading LLLite from: {pretrained}")

            state_dict = safetensors_load_file(str(pretrained))
            lllite = LLLiteModule.from_kohya_state_dict(state_dict, unet)
            print(f"[ControlNetSD15] Loaded LLLite from checkpoint")
        else:
            # Create new LLLite modules from UNet structure
            print(f"[ControlNetSD15] Creating LLLite modules (cond_ch={conditioning_channels}, rank={rank})")
            lllite = LLLiteModule.from_unet(
                unet,
                conditioning_channels=conditioning_channels,
                rank=rank,
                is_sdxl=False,
            )
            print(f"[ControlNetSD15] LLLite modules created")

        # Move to same device/dtype as UNet
        lllite = lllite.to(device=unet.device, dtype=unet.dtype)

        # Set to training mode
        lllite.train()
        lllite.requires_grad_(True)

        return lllite

    def setup_trainable_parameters(self, controlnet: nn.Module) -> List[Dict[str, Any]]:
        """
        Collect all ControlNet parameters for optimizer.

        Args:
            controlnet: ControlNetModel or LLLiteModule instance

        Returns:
            List with single parameter group containing all ControlNet parameters
        """
        if self.controlnet_type == "standard":
            return self._setup_standard_parameters(controlnet)
        elif self.controlnet_type == "lllite":
            return self._setup_lllite_parameters(controlnet)
        else:
            raise ValueError(f"Unknown ControlNet type '{self.controlnet_type}'")

    def _setup_standard_parameters(self, controlnet: ControlNetModel) -> List[Dict[str, Any]]:
        """Setup trainable parameters for Standard ControlNet."""
        params = [p for p in controlnet.parameters() if p.requires_grad]

        if not params:
            raise ValueError("[ControlNetSD15] No trainable parameters found in ControlNet")

        # Single parameter group with UNet learning rate
        # (ControlNet mirrors UNet architecture, so UNet LR is appropriate)
        param_groups = [
            {"params": params, "lr": self.trainer.unet_lr}
        ]

        print(f"[ControlNetSD15] Trainable parameters: {sum(p.numel() for p in params):,}")
        return param_groups

    def _setup_lllite_parameters(self, controlnet: LLLiteModule) -> List[Dict[str, Any]]:
        """Setup trainable parameters for LLLite ControlNet."""
        params = [p for p in controlnet.parameters() if p.requires_grad]

        if not params:
            raise ValueError("[ControlNetSD15] No trainable parameters found in LLLite module")

        param_groups = [
            {"params": params, "lr": self.trainer.unet_lr}
        ]

        print(f"[ControlNetSD15] LLLite trainable parameters: {sum(p.numel() for p in params):,}")
        return param_groups

    def save_checkpoint(
        self,
        controlnet: nn.Module,
        step: int,
        epoch: int,
        output_path: Path,
    ):
        """
        Save ControlNet checkpoint.

        Standard: saves as diffusers-compatible directory (config.json + safetensors)
        LLLite: saves as kohya-ss sd-scripts compatible .safetensors

        Args:
            controlnet: ControlNetModel or LLLiteModule instance
            step: Current training step
            epoch: Current training epoch
            output_path: Directory path (standard) or file path (lllite)
        """
        if self.controlnet_type == "standard":
            self._save_standard_checkpoint(controlnet, step, epoch, output_path)
        elif self.controlnet_type == "lllite":
            self._save_lllite_checkpoint(controlnet, step, epoch, output_path)
        else:
            raise ValueError(f"Unknown ControlNet type '{self.controlnet_type}'")

    def _save_standard_checkpoint(
        self,
        controlnet: ControlNetModel,
        step: int,
        epoch: int,
        output_path: Path,
    ):
        """Save Standard ControlNet checkpoint."""
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        # Save using diffusers save_pretrained (creates config.json + safetensors)
        controlnet.save_pretrained(
            str(output_path),
            safe_serialization=True,
        )

        print(f"[ControlNetSD15] Saved Standard ControlNet checkpoint: {output_path}")
        print(f"  Step: {step}, Epoch: {epoch}")

    def _save_lllite_checkpoint(
        self,
        controlnet: LLLiteModule,
        step: int,
        epoch: int,
        output_path: Path,
    ):
        """Save LLLite ControlNet checkpoint in kohya-ss compatible format."""
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export as kohya-ss compatible state dict
        state_dict = controlnet.to_kohya_state_dict()

        # Save as safetensors
        safetensors_save_file(state_dict, str(output_path))

        print(f"[ControlNetSD15] Saved LLLite checkpoint: {output_path}")
        print(f"  Step: {step}, Epoch: {epoch}, Keys: {len(state_dict)}")

    def load_checkpoint(self, controlnet: nn.Module, checkpoint_path: str) -> int:
        """
        Load ControlNet checkpoint for resume training.

        Args:
            controlnet: ControlNet module
            checkpoint_path: Path to checkpoint directory or file

        Returns:
            Training step number extracted from directory/filename (0 if not determinable)
        """
        if self.controlnet_type == "standard":
            return self._load_standard_checkpoint(controlnet, checkpoint_path)
        elif self.controlnet_type == "lllite":
            return self._load_lllite_checkpoint(controlnet, checkpoint_path)
        else:
            raise ValueError(f"Unknown ControlNet type '{self.controlnet_type}'")

    def _load_standard_checkpoint(self, controlnet: nn.Module, checkpoint_path: str) -> int:
        """Load Standard ControlNet checkpoint."""
        path = Path(checkpoint_path)

        # Load weights into existing model
        if path.is_dir():
            loaded = ControlNetModel.from_pretrained(
                str(path),
                torch_dtype=controlnet.dtype,
            )
        else:
            loaded = ControlNetModel.from_single_file(
                str(path),
                torch_dtype=controlnet.dtype,
            )

        # Copy weights
        controlnet.load_state_dict(loaded.state_dict())
        del loaded

        # Extract step from directory/filename
        step = self._extract_step_from_path(path)
        print(f"[ControlNetSD15] Loaded checkpoint from {path} (step={step})")

        return step

    def _load_lllite_checkpoint(self, controlnet: LLLiteModule, checkpoint_path: str) -> int:
        """Load LLLite ControlNet checkpoint."""
        path = Path(checkpoint_path)

        # Load state dict from safetensors
        state_dict = safetensors_load_file(str(path))

        # Re-create LLLite from loaded state dict and copy weights
        loaded = LLLiteModule.from_kohya_state_dict(state_dict, self.trainer.unet)

        # Copy weights into existing module
        controlnet.load_state_dict(loaded.state_dict())
        del loaded

        step = self._extract_step_from_path(path)
        print(f"[ControlNetSD15] Loaded LLLite checkpoint from {path} (step={step})")

        return step

    def _extract_step_from_path(self, path: Path) -> int:
        """Extract training step number from checkpoint path name."""
        # Try patterns like: *_step_001000, *_step001000, *_s001000
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
        Standard ControlNet forward pass for SD1.5.

        Args:
            controlnet: ControlNetModel instance
            noisy_latents: Noisy latent tensor [B, 4, H, W]
            timesteps: Timestep tensor [B]
            text_embeddings: Text embedding tensor [B, seq_len, 768]
            condition_images: Condition image tensor [B, 3, H, W] in [0, 1] range
            added_cond_kwargs: Not used for SD1.5 (SDXL only)

        Returns:
            Tuple of (down_block_res_samples, mid_block_res_sample)
            - down_block_res_samples: List of tensors for each down block
            - mid_block_res_sample: Single tensor for mid block
        """
        if self.controlnet_type == "standard":
            return self._standard_forward(
                controlnet, noisy_latents, timesteps,
                text_embeddings, condition_images
            )
        elif self.controlnet_type == "lllite":
            # LLLite does not return residuals - it patches UNet internally
            # The caller (train_step_controlnet) should use apply_patches/remove_patches instead
            return None
        else:
            raise ValueError(f"Unknown ControlNet type '{self.controlnet_type}'")

    def _standard_forward(
        self,
        controlnet: ControlNetModel,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        condition_images: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Standard ControlNet forward pass.

        ControlNet takes the same inputs as UNet plus condition images,
        and outputs residuals that are added to UNet's intermediate features.
        """
        output = controlnet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=condition_images,
            return_dict=True,
        )

        return (
            output.down_block_res_samples,
            output.mid_block_res_sample,
        )
