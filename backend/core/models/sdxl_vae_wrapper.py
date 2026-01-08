"""
SDXL VAE Wrapper

SDXL VAE uses 4-channel latents (standard for SDXL).
This wrapper uses the FP16-fixed version from madebyollin/sdxl-vae-fp16-fix
to avoid NaN issues when running in float16 precision.

VAE Architecture:
- Encoder: RGB (3ch) → Latent (4ch)
- Decoder: Latent (4ch) → RGB (3ch)
- Latent resolution: H/8 x W/8
- Latent channels: 4
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from typing import Optional


class SDXLVAEWrapper(nn.Module):
    """
    SDXL VAE wrapper for 4-channel latent encoding/decoding.
    
    Uses madebyollin/sdxl-vae-fp16-fix which is modified to run in fp16
    precision without generating NaNs.
    
    Key differences from FLUX VAE:
    - Latent channels: 4 (vs FLUX's 16)
    - Scaling factor: 0.13025 (vs FLUX's 0.3611)
    - FP16-safe implementation
    """

    def __init__(
        self,
        model_name: str = "madebyollin/sdxl-vae-fp16-fix",
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
            print(f"[SDXL VAE] Creating VAE structure (loading from checkpoint)...")

            # SDXL VAE config (hardcoded, as we know the structure)
            # This avoids downloading config from HF
            config_dict = {
                "_class_name": "AutoencoderKL",
                "_diffusers_version": "0.21.0",
                "act_fn": "silu",
                "block_out_channels": [128, 256, 512, 512],
                "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
                "in_channels": 3,
                "latent_channels": 4,  # SDXL uses 4 channels
                "layers_per_block": 2,
                "norm_num_groups": 32,
                "out_channels": 3,
                "sample_size": 512,
                "scaling_factor": 0.13025,  # SDXL scaling factor
                "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"]
            }

            # Create VAE from config
            self.vae = AutoencoderKL(**config_dict)
            self.vae = self.vae.to(dtype).to(device)

            # Get config
            self.latent_channels = self.vae.config.latent_channels  # 4
            self.scaling_factor = self.vae.config.scaling_factor  # ~0.13025

            print(f"[SDXL VAE] VAE structure created (weights pending):")
            print(f"  Latent channels: {self.latent_channels}")
            print(f"  Scaling factor: {self.scaling_factor:.4f}")
            print(f"  Block out channels: {self.vae.config.block_out_channels}")
        else:
            # Load from HuggingFace (with pretrained weights)
            print(f"[SDXL VAE] Loading VAE from {model_name}...")

            # Load SDXL VAE (FP16-fixed version)
            self.vae = AutoencoderKL.from_pretrained(
                model_name,
                torch_dtype=dtype
            )

            # Move to device
            self.vae = self.vae.to(device)

            # Get config
            self.latent_channels = self.vae.config.latent_channels  # 4
            self.scaling_factor = self.vae.config.scaling_factor  # ~0.13025

            print(f"[SDXL VAE] VAE loaded:")
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
            Latents [batch_size, 4, height//8, width//8]
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
            latents: Latents [batch_size, 4, height//8, width//8]
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

        # Handle different return types from vae.decode()
        # When return_dict=False, vae.decode() may return:
        # - A tuple: (sample,) -> use decoded[0]
        # - An object with .sample attribute -> use decoded.sample
        # - A tensor directly -> use decoded
        if isinstance(decoded, tuple):
            return decoded[0]
        elif hasattr(decoded, 'sample'):
            return decoded.sample
        else:
            # Direct tensor return
            return decoded

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


def get_sdxl_vae(
    model_name: str = "madebyollin/sdxl-vae-fp16-fix",
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> SDXLVAEWrapper:
    """
    Helper function to load SDXL VAE.

    Args:
        model_name: Model name or path
        dtype: Data type
        device: Device to load on

    Returns:
        SDXLVAEWrapper instance
    """
    return SDXLVAEWrapper(
        model_name=model_name,
        dtype=dtype,
        device=device
    )
