"""
SDXL ControlNet training adapter.

Supports:
- Standard ControlNet (diffusers ControlNetModel)
- ControlNet-LLLite (kohya-ss sd-scripts compatible)

SDXL-specific additions over SD1.5:
- added_cond_kwargs (pooled_embeddings + time_ids) in ControlNet forward
- Different block indexing for LLLite

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


class ControlNetSDXLAdapter(BaseControlNetAdapter):
    """
    ControlNet training adapter for SDXL.

    Standard mode:
    - Creates ControlNetModel from UNet weights or loads from checkpoint
    - Forward includes added_cond_kwargs (pooled_embeddings + time_ids)
    - Save: diffusers-compatible directory (config.json + safetensors)

    LLLite mode:
    - Creates LLLiteModule from UNet attention structure
    - Patches UNet attention layers with conditioning
    - Save: kohya-ss sd-scripts compatible .safetensors
    """

    def create_controlnet(
        self,
        init_from_unet: bool = True,
        pretrained_path: Optional[str] = None,
    ) -> nn.Module:
        """Create or load SDXL ControlNet."""
        if self.controlnet_type == "standard":
            return self._create_standard_controlnet(init_from_unet, pretrained_path)
        elif self.controlnet_type == "lllite":
            return self._create_lllite_controlnet(pretrained_path)
        else:
            raise ValueError(
                f"Unknown ControlNet type '{self.controlnet_type}' for SDXL. "
                f"Supported types: 'standard', 'lllite'"
            )

    def _create_standard_controlnet(
        self,
        init_from_unet: bool,
        pretrained_path: Optional[str],
    ) -> ControlNetModel:
        """Create Standard ControlNet for SDXL."""
        unet = self.trainer.unet

        if pretrained_path is not None:
            pretrained = Path(pretrained_path)
            print(f"[ControlNetSDXL] Loading ControlNet from: {pretrained}")

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
            print(f"[ControlNetSDXL] Loaded ControlNet from checkpoint")

        elif init_from_unet:
            print(f"[ControlNetSDXL] Initializing ControlNet from UNet weights")
            controlnet = ControlNetModel.from_unet(
                unet,
                load_weights_from_unet=True,
                conditioning_channels=3,
            )
            print(f"[ControlNetSDXL] ControlNet initialized from UNet")

        else:
            print(f"[ControlNetSDXL] Initializing ControlNet with random weights (UNet architecture)")
            controlnet = ControlNetModel.from_unet(
                unet,
                load_weights_from_unet=False,
                conditioning_channels=3,
            )
            print(f"[ControlNetSDXL] ControlNet initialized with random weights")

        controlnet = controlnet.to(device=unet.device, dtype=unet.dtype)
        controlnet.train()
        controlnet.requires_grad_(True)

        total_params = sum(p.numel() for p in controlnet.parameters())
        trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
        print(f"[ControlNetSDXL] ControlNet parameters: {total_params:,} total, {trainable_params:,} trainable")

        return controlnet

    def _create_lllite_controlnet(
        self,
        pretrained_path: Optional[str],
    ) -> LLLiteModule:
        """Create LLLite ControlNet for SDXL."""
        unet = self.trainer.unet
        conditioning_channels = self.trainer.lllite_conditioning_channels
        rank = self.trainer.lllite_rank

        if pretrained_path is not None:
            pretrained = Path(pretrained_path)
            print(f"[ControlNetSDXL] Loading LLLite from: {pretrained}")

            state_dict = safetensors_load_file(str(pretrained))
            lllite = LLLiteModule.from_kohya_state_dict(state_dict, unet)
            print(f"[ControlNetSDXL] Loaded LLLite from checkpoint")
        else:
            print(f"[ControlNetSDXL] Creating LLLite modules (cond_ch={conditioning_channels}, rank={rank})")
            lllite = LLLiteModule.from_unet(
                unet,
                conditioning_channels=conditioning_channels,
                rank=rank,
                is_sdxl=True,
            )
            print(f"[ControlNetSDXL] LLLite modules created")

        lllite = lllite.to(device=unet.device, dtype=unet.dtype)
        lllite.train()
        lllite.requires_grad_(True)

        return lllite

    def setup_trainable_parameters(self, controlnet: nn.Module) -> List[Dict[str, Any]]:
        """Collect all ControlNet parameters for optimizer."""
        params = [p for p in controlnet.parameters() if p.requires_grad]

        if not params:
            raise ValueError("[ControlNetSDXL] No trainable parameters found")

        param_groups = [
            {"params": params, "lr": self.trainer.unet_lr}
        ]

        param_type = "Standard ControlNet" if self.controlnet_type == "standard" else "LLLite"
        print(f"[ControlNetSDXL] {param_type} trainable parameters: {sum(p.numel() for p in params):,}")
        return param_groups

    def save_checkpoint(
        self,
        controlnet: nn.Module,
        step: int,
        epoch: int,
        output_path: Path,
    ):
        """Save ControlNet checkpoint."""
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
        output_path.mkdir(parents=True, exist_ok=True)
        controlnet.save_pretrained(
            str(output_path),
            safe_serialization=True,
        )
        print(f"[ControlNetSDXL] Saved Standard ControlNet checkpoint: {output_path}")
        print(f"  Step: {step}, Epoch: {epoch}")

    def _save_lllite_checkpoint(
        self,
        controlnet: LLLiteModule,
        step: int,
        epoch: int,
        output_path: Path,
    ):
        """Save LLLite ControlNet checkpoint in kohya-ss compatible format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = controlnet.to_kohya_state_dict()
        safetensors_save_file(state_dict, str(output_path))
        print(f"[ControlNetSDXL] Saved LLLite checkpoint: {output_path}")
        print(f"  Step: {step}, Epoch: {epoch}, Keys: {len(state_dict)}")

    def load_checkpoint(self, controlnet: nn.Module, checkpoint_path: str) -> int:
        """Load ControlNet checkpoint for resume training."""
        if self.controlnet_type == "standard":
            return self._load_standard_checkpoint(controlnet, checkpoint_path)
        elif self.controlnet_type == "lllite":
            return self._load_lllite_checkpoint(controlnet, checkpoint_path)
        else:
            raise ValueError(f"Unknown ControlNet type '{self.controlnet_type}'")

    def _load_standard_checkpoint(self, controlnet: nn.Module, checkpoint_path: str) -> int:
        """Load Standard ControlNet checkpoint."""
        path = Path(checkpoint_path)

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

        controlnet.load_state_dict(loaded.state_dict())
        del loaded

        step = self._extract_step_from_path(path)
        print(f"[ControlNetSDXL] Loaded checkpoint from {path} (step={step})")
        return step

    def _load_lllite_checkpoint(self, controlnet: LLLiteModule, checkpoint_path: str) -> int:
        """Load LLLite ControlNet checkpoint."""
        path = Path(checkpoint_path)
        state_dict = safetensors_load_file(str(path))

        loaded = LLLiteModule.from_kohya_state_dict(state_dict, self.trainer.unet)
        controlnet.load_state_dict(loaded.state_dict())
        del loaded

        step = self._extract_step_from_path(path)
        print(f"[ControlNetSDXL] Loaded LLLite checkpoint from {path} (step={step})")
        return step

    def _extract_step_from_path(self, path: Path) -> int:
        """Extract training step number from checkpoint path name."""
        name = path.stem if path.is_file() else path.name

        match = re.search(r'step[_-](\d+)', name, re.IGNORECASE)
        if match:
            return int(match.group(1))

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
        ControlNet forward pass for SDXL.

        Args:
            controlnet: ControlNetModel or LLLiteModule instance
            noisy_latents: Noisy latent tensor [B, 4, H, W]
            timesteps: Timestep tensor [B]
            text_embeddings: Text embedding tensor [B, seq_len, 2048]
            condition_images: Condition image tensor [B, 3, H, W] in [0, 1] range
            added_cond_kwargs: SDXL conditioning (pooled_embeddings + time_ids)

        Returns:
            Standard: (down_block_res_samples, mid_block_res_sample)
            LLLite: None (patches applied via apply_patches)
        """
        if self.controlnet_type == "standard":
            return self._standard_forward(
                controlnet, noisy_latents, timesteps,
                text_embeddings, condition_images, added_cond_kwargs
            )
        elif self.controlnet_type == "lllite":
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
        added_cond_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Standard ControlNet forward pass for SDXL.

        SDXL requires added_cond_kwargs containing:
        - text_embeds: pooled text embeddings [B, 1280]
        - time_ids: micro-conditioning [B, 6]
        """
        forward_kwargs = {
            "sample": noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": text_embeddings,
            "controlnet_cond": condition_images,
            "return_dict": True,
        }

        if added_cond_kwargs is not None:
            forward_kwargs["added_cond_kwargs"] = added_cond_kwargs

        output = controlnet(**forward_kwargs)

        return (
            output.down_block_res_samples,
            output.mid_block_res_sample,
        )
