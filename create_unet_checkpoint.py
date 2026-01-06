"""
Create initialized U-Net checkpoint for Original Architecture

Creates a safetensors file containing only the U-Net weights (randomly initialized).
This can be loaded as a base model for training.
"""

import sys
import torch
from safetensors.torch import save_file
from pathlib import Path

# Add backend to path
sys.path.insert(0, 'backend')

from core.models.unet_original import OriginalUNet, UNetConfig, count_parameters


def create_unet_checkpoint(
    variant: str = "medium",
    output_path: str = "models/original_unet_medium.safetensors",
    dtype: torch.dtype = torch.float16
):
    """
    Create U-Net checkpoint with initialized weights.

    Args:
        variant: "small", "medium", or "large"
        output_path: Path to save checkpoint
        dtype: Data type for weights
    """
    print("=" * 80)
    print(f"Creating Original U-Net Checkpoint ({variant})")
    print("=" * 80)
    print()

    # Create U-Net
    print(f"Creating U-Net ({variant} variant)...")
    config = UNetConfig.from_variant(variant)
    unet = OriginalUNet(config)

    # Count parameters
    num_params = count_parameters(unet)
    print(f"\nU-Net parameters: {num_params / 1e9:.2f}B")
    print()

    # Convert to target dtype
    print(f"Converting to {dtype}...")
    unet = unet.to(dtype)

    # Get state dict
    state_dict = unet.state_dict()

    # Add metadata
    metadata = {
        "model_type": "original_unet",
        "variant": variant,
        "num_parameters": str(num_params),
        "architecture": "Original U-Net with RoPE and sparse skip connections",
        "latent_channels": str(config.in_channels),
        "context_dim": str(config.context_dim),
        "model_channels": str(config.model_channels),
        "channel_mult": str(config.channel_mult),
        "skip_connection_interval": str(config.skip_connection_interval),
        "num_attention_heads": str(config.num_attention_heads),
        "transformer_depth": str(config.transformer_depth),
        "description": "Randomly initialized U-Net for Original architecture. Requires training before use.",
        "encoder": "SigLIP-2 (google/siglip2-so400m-patch16-naflex)",
        "vae": "FLUX VAE (16-channel latents)",
    }

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    print(f"Saving checkpoint to: {output_path}")
    save_file(state_dict, str(output_path), metadata=metadata)

    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Checkpoint saved: {file_size_mb:.1f} MB")
    print()

    # Print summary
    print("=" * 80)
    print("Checkpoint created successfully!")
    print("=" * 80)
    print()
    print(f"File: {output_path}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Variant: {variant}")
    print(f"Parameters: {num_params / 1e9:.2f}B")
    print()
    print("NOTE: This U-Net is randomly initialized and requires training before use.")
    print("It will not produce meaningful images in its current state.")
    print()

    # Print config for reference
    print("Configuration:")
    print(f"  Model channels: {config.model_channels}")
    print(f"  Channel mult: {config.channel_mult}")
    print(f"  Skip connection interval: {config.skip_connection_interval}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Transformer depth: {config.transformer_depth}")
    print(f"  Latent channels: {config.in_channels} -> {config.out_channels}")
    print(f"  Context dim: {config.context_dim} (SigLIP-2)")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create U-Net checkpoint")
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
        help="Output path (default: models/original_unet_{variant}.safetensors)"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_path = f"models/original_unet_{args.variant}.safetensors"
    else:
        output_path = args.output

    # Create checkpoint
    create_unet_checkpoint(
        variant=args.variant,
        output_path=output_path,
        dtype=torch.float16
    )
