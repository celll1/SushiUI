"""
FLUX VAE Wrapper

FLUX VAE uses 16-channel latents (vs SDXL's 4-channel).
This provides better detail preservation and higher quality.

VAE Architecture:
- Encoder: RGB (3ch) → Latent (16ch)
- Decoder: Latent (16ch) → RGB (3ch)
- Latent resolution: H/8 x W/8 (same as SDXL)
- Latent channels: 16 (4x more than SDXL)
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from typing import Optional


class FluxVAEWrapper(nn.Module):
    """
    FLUX VAE wrapper for 16-channel latent encoding/decoding.

    Key differences from SDXL VAE:
    - Latent channels: 16 (vs SDXL's 4)
    - Better detail preservation
    - Higher quality reconstruction
    """

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-dev",
        subfolder: str = "vae",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        load_from_checkpoint: bool = False
    ):
        super().__init__()

        self.model_name = model_name
        self.dtype = dtype
        self.device_name = device

        if load_from_checkpoint:
            # Create empty VAE structure (weights will be loaded via load_state_dict)
            print(f"[FLUX VAE] Creating VAE structure (loading from checkpoint)...")

            # Load config only (no weights)
            from diffusers.models import AutoencoderKL
            from transformers import PretrainedConfig
            import json
            import tempfile
            import os

            # FLUX VAE config (hardcoded, as we know the structure)
            # This avoids downloading config from HF
            config_dict = {
                "_class_name": "AutoencoderKL",
                "_diffusers_version": "0.21.0",
                "act_fn": "silu",
                "block_out_channels": [128, 256, 512, 512],
                "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
                "in_channels": 3,
                "latent_channels": 16,
                "layers_per_block": 2,
                "norm_num_groups": 32,
                "out_channels": 3,
                "sample_size": 256,
                "scaling_factor": 0.3611,
                "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"]
            }

            # Create VAE from config
            from diffusers import ConfigMixin
            self.vae = AutoencoderKL(**config_dict)
            self.vae = self.vae.to(dtype).to(device)

            # Get config
            self.latent_channels = self.vae.config.latent_channels  # 16
            self.scaling_factor = self.vae.config.scaling_factor  # ~0.3611

            print(f"[FLUX VAE] VAE structure created (weights pending):")
            print(f"  Latent channels: {self.latent_channels}")
            print(f"  Scaling factor: {self.scaling_factor:.4f}")
            print(f"  Block out channels: {self.vae.config.block_out_channels}")
        else:
            # Load from HuggingFace (with pretrained weights)
            print(f"[FLUX VAE] Loading VAE from {model_name}/{subfolder}...")

            # Load FLUX VAE
            self.vae = AutoencoderKL.from_pretrained(
                model_name,
                subfolder=subfolder,
                torch_dtype=dtype
            )

            # Move to device
            self.vae = self.vae.to(device)

            # Get config
            self.latent_channels = self.vae.config.latent_channels  # 16
            self.scaling_factor = self.vae.config.scaling_factor  # ~0.3611

            print(f"[FLUX VAE] VAE loaded:")
            print(f"  Latent channels: {self.latent_channels}")
            print(f"  Scaling factor: {self.scaling_factor:.4f}")
            print(f"  Block out channels: {self.vae.config.block_out_channels}")

    def encode(
        self,
        images: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Encode images to latents.

        Args:
            images: Input images [batch_size, 3, height, width] in range [-1, 1]
            return_dict: Return dict with 'latent_dist' (for sampling)

        Returns:
            Latents [batch_size, 16, height//8, width//8]
        """
        # Encode to latent distribution
        latent_dist = self.vae.encode(images).latent_dist

        if return_dict:
            return {"latent_dist": latent_dist}

        # Sample from distribution and scale
        latents = latent_dist.sample() * self.scaling_factor

        return latents

    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            latents: Latents [batch_size, 16, height//8, width//8]
            return_dict: Return dict with 'sample'

        Returns:
            Decoded images [batch_size, 3, height, width] in range [-1, 1]
        """
        # Unscale latents
        latents = latents / self.scaling_factor

        # Decode
        decoded = self.vae.decode(latents, return_dict=return_dict)

        if return_dict:
            return decoded

        return decoded.sample

    def forward(
        self,
        sample: torch.Tensor,
        return_dict: bool = False
    ):
        """
        Forward pass (encode -> decode).

        Args:
            sample: Input images [batch_size, 3, height, width]
            return_dict: Return dict

        Returns:
            Reconstruction
        """
        return self.vae(sample, return_dict=return_dict)

    def to(self, device):
        """Move VAE to device."""
        self.vae = self.vae.to(device)
        self.device_name = device
        return self

    def eval(self):
        """Set VAE to eval mode."""
        self.vae.eval()
        return self

    def train(self, mode: bool = True):
        """Set VAE to train mode."""
        self.vae.train(mode)
        return self


def get_flux_vae(
    model_name: str = "black-forest-labs/FLUX.1-dev",
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> FluxVAEWrapper:
    """
    Helper function to load FLUX VAE.

    Args:
        model_name: Model name or path
        dtype: Data type
        device: Device to load on

    Returns:
        FluxVAEWrapper instance
    """
    return FluxVAEWrapper(
        model_name=model_name,
        dtype=dtype,
        device=device
    )
