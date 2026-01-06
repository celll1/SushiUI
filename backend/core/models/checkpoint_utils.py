"""
Checkpoint utilities for DEUS Architecture

Handles saving and loading unified checkpoints containing:
- SigLIP-2 text/image encoders
- DEUS U-Net (Dual-Embeddings U-Net Structure)
- FLUX VAE

Format similar to SDXL safetensors (all components in one file).
"""

import torch
from safetensors.torch import save_file, load_file
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .siglip2_wrapper import SigLIP2TextEncoder, SigLIP2ImageEncoder
from .unet_deus import DeusUNet, UNetConfig
from .flux_vae_wrapper import FluxVAEWrapper


def save_unified_checkpoint(
    unet: DeusUNet,
    text_encoder: Optional[SigLIP2TextEncoder] = None,
    image_encoder: Optional[SigLIP2ImageEncoder] = None,
    vae: Optional[FluxVAEWrapper] = None,
    output_path: str = "models/deus_model.safetensors",
    metadata: Optional[Dict[str, str]] = None
):
    """
    Save unified checkpoint containing all components.

    Args:
        unet: DEUS U-Net model
        text_encoder: SigLIP-2 text encoder (if None, not saved)
        image_encoder: SigLIP-2 image encoder (if None, not saved)
        vae: FLUX VAE (if None, not saved)
        output_path: Path to save checkpoint
        metadata: Additional metadata
    """
    print(f"[Checkpoint] Saving unified checkpoint to: {output_path}")

    # Collect state dicts with prefixes (SDXL-style)
    unified_state_dict = {}

    # U-Net (required)
    print(f"[Checkpoint] Adding U-Net weights...")
    unet_state = unet.state_dict()
    for key, value in unet_state.items():
        unified_state_dict[f"model.diffusion_model.{key}"] = value

    # Text Encoder (optional)
    if text_encoder is not None:
        print(f"[Checkpoint] Adding text encoder weights...")
        text_state = text_encoder.text_model.state_dict()
        for key, value in text_state.items():
            unified_state_dict[f"conditioner.embedders.0.transformer.{key}"] = value

    # Image Encoder (optional)
    if image_encoder is not None:
        print(f"[Checkpoint] Adding image encoder weights...")
        image_state = image_encoder.vision_model.state_dict()
        for key, value in image_state.items():
            unified_state_dict[f"conditioner.embedders.1.model.{key}"] = value

    # VAE (optional)
    if vae is not None:
        print(f"[Checkpoint] Adding VAE weights...")
        vae_state = vae.vae.state_dict()
        for key, value in vae_state.items():
            unified_state_dict[f"first_stage_model.{key}"] = value

    # Prepare metadata
    checkpoint_metadata = {
        "model_type": "deus",
        "architecture": "DEUS Architecture (Dual-Embeddings U-Net Structure: SigLIP-2 + U-Net + FLUX VAE)",
        "unet_variant": unet.config.variant,
        "latent_channels": str(unet.config.latent_channels),
        "context_dim": str(unet.config.context_dim),
        "model_channels": str(unet.config.model_channels),
        "channel_mult": str(unet.config.channel_mult),
        "skip_connection_interval": str(unet.config.skip_connection_interval),
        "has_text_encoder": str(text_encoder is not None),
        "has_image_encoder": str(image_encoder is not None),
        "has_vae": str(vae is not None),
    }

    if metadata is not None:
        checkpoint_metadata.update(metadata)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    print(f"[Checkpoint] Saving {len(unified_state_dict)} tensors...")
    save_file(unified_state_dict, str(output_path), metadata=checkpoint_metadata)

    # Report file size
    file_size_gb = output_path.stat().st_size / (1024 ** 3)
    print(f"[Checkpoint] Saved: {file_size_gb:.2f} GB")
    print(f"[Checkpoint] Components included:")
    print(f"  - U-Net: ✓")
    print(f"  - Text Encoder: {'✓' if text_encoder is not None else '✗'}")
    print(f"  - Image Encoder: {'✓' if image_encoder is not None else '✗'}")
    print(f"  - VAE: {'✓' if vae is not None else '✗'}")


def load_unified_checkpoint(
    checkpoint_path: str,
    unet_variant: str = "medium",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    load_text_encoder: bool = True,
    load_image_encoder: bool = True,
    load_vae: bool = True
) -> Dict[str, Any]:
    """
    Load unified checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        unet_variant: U-Net variant (used for creating model structure)
        device: Device to load on
        dtype: Data type
        load_text_encoder: Load text encoder from checkpoint
        load_image_encoder: Load image encoder from checkpoint
        load_vae: Load VAE from checkpoint

    Returns:
        Dict with keys: 'unet', 'text_encoder', 'image_encoder', 'vae', 'metadata'
    """
    print(f"[Checkpoint] Loading unified checkpoint from: {checkpoint_path}")

    # Load checkpoint
    state_dict = load_file(checkpoint_path, device=str(device))

    # Load metadata
    from safetensors import safe_open
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    print(f"[Checkpoint] Loaded {len(state_dict)} tensors")
    print(f"[Checkpoint] Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # Separate state dicts by prefix
    unet_state = {}
    text_encoder_state = {}
    image_encoder_state = {}
    vae_state = {}

    for key, value in state_dict.items():
        if key.startswith("model.diffusion_model."):
            # U-Net
            new_key = key.replace("model.diffusion_model.", "")
            unet_state[new_key] = value
        elif key.startswith("conditioner.embedders.0.transformer."):
            # Text Encoder
            new_key = key.replace("conditioner.embedders.0.transformer.", "")
            text_encoder_state[new_key] = value
        elif key.startswith("conditioner.embedders.1.model."):
            # Image Encoder
            new_key = key.replace("conditioner.embedders.1.model.", "")
            image_encoder_state[new_key] = value
        elif key.startswith("first_stage_model."):
            # VAE
            new_key = key.replace("first_stage_model.", "")
            vae_state[new_key] = value

    print(f"[Checkpoint] Component tensors:")
    print(f"  - U-Net: {len(unet_state)} tensors")
    print(f"  - Text Encoder: {len(text_encoder_state)} tensors")
    print(f"  - Image Encoder: {len(image_encoder_state)} tensors")
    print(f"  - VAE: {len(vae_state)} tensors")

    # Create U-Net and load weights
    print(f"[Checkpoint] Creating U-Net ({unet_variant})...")
    config = UNetConfig.from_variant(unet_variant)
    unet = DeusUNet(config)
    unet = unet.to(dtype).to(device)

    if len(unet_state) > 0:
        print(f"[Checkpoint] Loading U-Net weights...")
        unet.load_state_dict(unet_state)
    else:
        print(f"[Checkpoint] WARNING: No U-Net weights found in checkpoint!")

    # Create text encoder and load weights
    text_encoder = None
    if load_text_encoder and len(text_encoder_state) > 0:
        print(f"[Checkpoint] Creating text encoder...")
        text_encoder = SigLIP2TextEncoder(dtype=dtype, device=device)
        print(f"[Checkpoint] Loading text encoder weights...")
        text_encoder.text_model.load_state_dict(text_encoder_state)
    elif load_text_encoder:
        print(f"[Checkpoint] WARNING: No text encoder weights found in checkpoint!")

    # Create image encoder and load weights
    image_encoder = None
    if load_image_encoder and len(image_encoder_state) > 0:
        print(f"[Checkpoint] Creating image encoder...")
        image_encoder = SigLIP2ImageEncoder(dtype=dtype, device=device)
        print(f"[Checkpoint] Loading image encoder weights...")
        image_encoder.vision_model.load_state_dict(image_encoder_state)
    elif load_image_encoder:
        print(f"[Checkpoint] WARNING: No image encoder weights found in checkpoint!")

    # Create VAE and load weights
    vae = None
    if load_vae and len(vae_state) > 0:
        print(f"[Checkpoint] Creating VAE...")
        vae = FluxVAEWrapper(dtype=dtype, device=device)
        print(f"[Checkpoint] Loading VAE weights...")
        vae.vae.load_state_dict(vae_state)
    elif load_vae:
        print(f"[Checkpoint] WARNING: No VAE weights found in checkpoint!")

    print(f"[Checkpoint] Checkpoint loaded successfully!")

    return {
        "unet": unet,
        "text_encoder": text_encoder,
        "image_encoder": image_encoder,
        "vae": vae,
        "metadata": metadata
    }


def detect_checkpoint_components(checkpoint_path: str) -> Dict[str, bool]:
    """
    Detect which components are present in a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dict with keys: 'has_unet', 'has_text_encoder', 'has_image_encoder', 'has_vae'
    """
    from safetensors import safe_open

    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    has_unet = any(k.startswith("model.diffusion_model.") for k in keys)
    has_text_encoder = any(k.startswith("conditioner.embedders.0.transformer.") for k in keys)
    has_image_encoder = any(k.startswith("conditioner.embedders.1.model.") for k in keys)
    has_vae = any(k.startswith("first_stage_model.") for k in keys)

    return {
        "has_unet": has_unet,
        "has_text_encoder": has_text_encoder,
        "has_image_encoder": has_image_encoder,
        "has_vae": has_vae
    }
