"""
Create Unified Checkpoint for Original Architecture

Creates a single safetensors file containing:
- SigLIP-2 text encoder (from HuggingFace)
- SigLIP-2 image encoder (from HuggingFace)
- Original U-Net (randomly initialized)
- FLUX VAE (from HuggingFace)

Similar to SDXL format - all components in one file.
"""

import sys
import torch
from pathlib import Path

# Add backend to path
sys.path.insert(0, 'backend')

from core.models.siglip2_wrapper import SigLIP2TextEncoder, SigLIP2ImageEncoder
from core.models.unet_original import OriginalUNet, UNetConfig, count_parameters
from core.models.flux_vae_wrapper import FluxVAEWrapper
from core.models.checkpoint_utils import save_unified_checkpoint


def create_unified_checkpoint(
    variant: str = "medium",
    output_path: str = "models/original_model_medium.safetensors",
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    include_text_encoder: bool = True,
    include_image_encoder: bool = True,
    include_vae: bool = True
):
    """
    Create unified checkpoint with all components.

    Args:
        variant: U-Net variant ("small", "medium", "large")
        output_path: Path to save checkpoint
        dtype: Data type
        device: Device to use for loading pretrained models
        include_text_encoder: Include SigLIP-2 text encoder
        include_image_encoder: Include SigLIP-2 image encoder
        include_vae: Include FLUX VAE
    """
    print("=" * 80)
    print(f"Creating Unified Checkpoint ({variant})")
    print("=" * 80)
    print()

    components = {}

    # Create U-Net (randomly initialized)
    print(f"[1/4] Creating U-Net ({variant} variant)...")
    config = UNetConfig.from_variant(variant)
    unet = OriginalUNet(config)
    unet = unet.to(dtype)

    num_params = count_parameters(unet)
    print(f"  U-Net parameters: {num_params / 1e9:.2f}B")
    print()

    # Load SigLIP-2 text encoder
    text_encoder = None
    if include_text_encoder:
        print(f"[2/4] Loading SigLIP-2 text encoder...")
        try:
            text_encoder = SigLIP2TextEncoder(dtype=dtype, device=device)
            print(f"  Text encoder loaded")
        except Exception as e:
            print(f"  WARNING: Failed to load text encoder: {e}")
            print(f"  Skipping text encoder...")
            text_encoder = None
        print()
    else:
        print(f"[2/4] Skipping text encoder (include_text_encoder=False)")
        print()

    # Load SigLIP-2 image encoder
    image_encoder = None
    if include_image_encoder:
        print(f"[3/4] Loading SigLIP-2 image encoder...")
        try:
            image_encoder = SigLIP2ImageEncoder(dtype=dtype, device=device)
            print(f"  Image encoder loaded")
        except Exception as e:
            print(f"  WARNING: Failed to load image encoder: {e}")
            print(f"  Skipping image encoder...")
            image_encoder = None
        print()
    else:
        print(f"[3/4] Skipping image encoder (include_image_encoder=False)")
        print()

    # Load FLUX VAE
    vae = None
    if include_vae:
        print(f"[4/4] Loading FLUX VAE...")
        try:
            vae = FluxVAEWrapper(dtype=dtype, device=device)
            print(f"  VAE loaded (latent channels: {vae.latent_channels})")
        except Exception as e:
            print(f"  WARNING: Failed to load VAE: {e}")
            print(f"  Skipping VAE...")
            vae = None
        print()
    else:
        print(f"[4/4] Skipping VAE (include_vae=False)")
        print()

    # Save unified checkpoint
    print("=" * 80)
    print("Saving Unified Checkpoint")
    print("=" * 80)
    print()

    metadata = {
        "created_by": "SushiUI Original Architecture",
        "variant": variant,
        "unet_parameters": str(num_params),
        "description": "Unified checkpoint for Original architecture (SigLIP-2 + U-Net + FLUX VAE)",
    }

    save_unified_checkpoint(
        unet=unet,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        vae=vae,
        output_path=output_path,
        metadata=metadata
    )

    print()
    print("=" * 80)
    print("Checkpoint created successfully!")
    print("=" * 80)
    print()
    print(f"File: {output_path}")
    print(f"Variant: {variant}")
    print(f"U-Net parameters: {num_params / 1e9:.2f}B")
    print()
    print("Components included:")
    print(f"  - U-Net: ✓ (randomly initialized)")
    print(f"  - Text Encoder: {'✓' if text_encoder is not None else '✗'}")
    print(f"  - Image Encoder: {'✓' if image_encoder is not None else '✗'}")
    print(f"  - VAE: {'✓' if vae is not None else '✗'}")
    print()
    print("NOTE: U-Net is randomly initialized and requires training before use.")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create unified checkpoint")
    parser.add_argument(
        "--variant",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="U-Net variant (default: medium)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: models/original_model_{variant}.safetensors)"
    )
    parser.add_argument(
        "--no-text-encoder",
        action="store_true",
        help="Exclude text encoder (U-Net + VAE only)"
    )
    parser.add_argument(
        "--no-image-encoder",
        action="store_true",
        help="Exclude image encoder"
    )
    parser.add_argument(
        "--no-vae",
        action="store_true",
        help="Exclude VAE (U-Net only)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for loading models (default: cuda)"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_path = f"models/original_model_{args.variant}.safetensors"
    else:
        output_path = args.output

    # Create checkpoint
    create_unified_checkpoint(
        variant=args.variant,
        output_path=output_path,
        dtype=torch.float16,
        device=args.device,
        include_text_encoder=not args.no_text_encoder,
        include_image_encoder=not args.no_image_encoder,
        include_vae=not args.no_vae
    )
