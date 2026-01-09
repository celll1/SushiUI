"""
Checkpoint utilities for DEUS Architecture

Handles saving and loading unified checkpoints containing:
- SigLIP-2 text/image encoders
- DEUS U-Net (Dual-Embeddings U-Net Structure)
- SDXL VAE

Format similar to SDXL safetensors (all components in one file).
"""

import torch
from safetensors.torch import save_file, load_file
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time

from .siglip2_wrapper import SigLIP2TextEncoder, SigLIP2ImageEncoder
from .unet_deus import DeusUNet, UNetConfig
from .unet_deus_v2 import DeusUNet as DeusUNetV2, UNetConfig as UNetConfigV2
from .sdxl_vae_wrapper import SDXLVAEWrapper


def save_unified_checkpoint(
    unet: DeusUNet,
    text_encoder: Optional[SigLIP2TextEncoder] = None,
    image_encoder: Optional[SigLIP2ImageEncoder] = None,
    vae: Optional[SDXLVAEWrapper] = None,
    output_path: str = "models/deus_model.safetensors",
    metadata: Optional[Dict[str, str]] = None
):
    """
    Save unified checkpoint containing all components.

    Args:
        unet: DEUS U-Net model
        text_encoder: SigLIP-2 text encoder (if None, not saved)
        image_encoder: SigLIP-2 image encoder (if None, not saved)
        vae: SDXL VAE (if None, not saved)
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
        "architecture": "DEUS Architecture (Dual-Embeddings U-Net Structure: SigLIP-2 + U-Net + SDXL VAE)",
        "unet_variant": unet.config.variant,
        "latent_channels": str(unet.config.latent_channels),
        "context_dim": str(unet.config.context_dim),
        "model_channels": str(unet.config.model_channels),
        "channel_mult": str(unet.config.channel_mult),
        "attention_head_dim": str(unet.config.attention_head_dim),
        "num_attention_heads": str(unet.config.num_attention_heads),
        "transformer_layers_per_block": str(unet.config.transformer_layers_per_block),
        "transformer_layers_per_mid_block": str(unet.config.transformer_layers_per_mid_block),
        "skip_connection_interval": str(unet.config.skip_connection_interval),
        "model_channels": str(unet.config.model_channels),
        "channel_mult": str(unet.config.channel_mult),
        "skip_connection_interval": str(unet.config.skip_connection_interval),
        "attention_head_dim": str(unet.config.attention_head_dim),
        "num_attention_heads": str(unet.config.num_attention_heads),
        "transformer_layers_per_block": str(unet.config.transformer_layers_per_block),
        "transformer_layers_per_mid_block": str(unet.config.transformer_layers_per_mid_block),
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
    load_vae: bool = True,
    load_unet: bool = True
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
        load_unet: Load U-Net weights from checkpoint (if False, U-Net will be randomly initialized)

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
        # DEUS v2 format (diffusers-style)
        if key.startswith("unet."):
            # U-Net
            new_key = key.replace("unet.", "")
            unet_state[new_key] = value
        elif key.startswith("text_encoder."):
            # Text Encoder
            new_key = key.replace("text_encoder.", "")
            text_encoder_state[new_key] = value
        elif key.startswith("vae."):
            # VAE
            new_key = key.replace("vae.", "")
            vae_state[new_key] = value
        # DEUS v1 format (ComfyUI-style, legacy)
        elif key.startswith("model.diffusion_model."):
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

    # Detect DEUS version from metadata
    model_type = metadata.get("model_type", "deus")
    is_deus_v2 = model_type == "deus_v2"

    # Create U-Net and load weights
    if is_deus_v2:
        print(f"[Checkpoint] Creating DEUS v2 U-Net ({unet_variant})...")
        config = UNetConfigV2.from_variant(unet_variant)
        unet = DeusUNetV2(config)
    else:
        print(f"[Checkpoint] Creating DEUS v1 U-Net ({unet_variant})...")
        config = UNetConfig.from_variant(unet_variant)
        unet = DeusUNet(config)
    unet = unet.to(dtype).to(device)

    if load_unet and len(unet_state) > 0:
        print(f"[Checkpoint] Loading U-Net weights...")
        try:
            # Check for shape mismatches before loading
            model_state = unet.state_dict()
            shape_mismatches = []
            missing_keys = []
            unexpected_keys = []
            
            for key in unet_state.keys():
                if key not in model_state:
                    unexpected_keys.append(key)
                elif unet_state[key].shape != model_state[key].shape:
                    shape_mismatches.append({
                        'key': key,
                        'checkpoint_shape': list(unet_state[key].shape),
                        'model_shape': list(model_state[key].shape)
                    })
            
            for key in model_state.keys():
                if key not in unet_state:
                    missing_keys.append(key)
            
            if shape_mismatches:
                print(f"[Checkpoint] ERROR: Found {len(shape_mismatches)} shape mismatches:")
                for mismatch in shape_mismatches[:20]:  # Show first 20
                    print(f"  {mismatch['key']}: checkpoint={mismatch['checkpoint_shape']}, model={mismatch['model_shape']}")
                if len(shape_mismatches) > 20:
                    print(f"  ... and {len(shape_mismatches) - 20} more")
                print(f"[Checkpoint] Cannot load U-Net weights due to shape mismatches!")
                print(f"[Checkpoint] U-Net will remain randomly initialized")
                raise RuntimeError(f"Shape mismatches detected: {len(shape_mismatches)} keys")
            
            if unexpected_keys:
                print(f"[Checkpoint] WARNING: Found {len(unexpected_keys)} unexpected keys in checkpoint (will be ignored)")
                if len(unexpected_keys) <= 10:
                    for key in unexpected_keys:
                        print(f"  {key}")
                else:
                    for key in unexpected_keys[:10]:
                        print(f"  {key}")
                    print(f"  ... and {len(unexpected_keys) - 10} more")
            
            if missing_keys:
                print(f"[Checkpoint] WARNING: Found {len(missing_keys)} missing keys (will remain randomly initialized)")
                if len(missing_keys) <= 10:
                    for key in missing_keys:
                        print(f"  {key}")
                else:
                    for key in missing_keys[:10]:
                        print(f"  {key}")
                    print(f"  ... and {len(missing_keys) - 10} more")
            
            # Load with strict=False to allow missing keys, but shape mismatches will raise error
            unet.load_state_dict(unet_state, strict=False)
            print(f"[Checkpoint] U-Net weights loaded successfully!")
        except RuntimeError as e:
            print(f"[Checkpoint] ERROR: Failed to load U-Net weights: {e}")
            print(f"[Checkpoint] U-Net will remain randomly initialized")
            import traceback
            traceback.print_exc()
    elif load_unet and len(unet_state) == 0:
        print(f"[Checkpoint] WARNING: No U-Net weights found in checkpoint!")
    elif not load_unet:
        print(f"[Checkpoint] Skipping U-Net weight loading (load_unet=False)")
        print(f"[Checkpoint] U-Net will remain randomly initialized")

    # Load shared config once (text encoder and image encoder use the same model)
    shared_config = None
    if (load_text_encoder and len(text_encoder_state) > 0) or (load_image_encoder and len(image_encoder_state) > 0):
        from transformers import AutoConfig
        model_name = "google/siglip2-so400m-patch16-naflex"
        print(f"[Checkpoint] Loading shared config (for text/image encoders)...")
        start_time = time.time()
        shared_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        config_time = time.time() - start_time
        print(f"[Checkpoint] Shared config loaded in {config_time:.2f}s")

    # Get max_position_embeddings from metadata (if available)
    max_position_embeddings = None
    if 'max_position_embeddings' in metadata:
        max_position_embeddings = int(metadata['max_position_embeddings'])
        print(f"[Checkpoint] max_position_embeddings from metadata: {max_position_embeddings}")

    # Create text encoder and load weights
    text_encoder = None
    if load_text_encoder and len(text_encoder_state) > 0:
        print(f"[Checkpoint] Creating text encoder (from checkpoint)...")
        start_time = time.time()
        text_encoder = SigLIP2TextEncoder(
            dtype=dtype,
            device=device,
            load_from_checkpoint=True,
            shared_config=shared_config,
            max_position_embeddings=max_position_embeddings
        )
        encoder_create_time = time.time() - start_time
        print(f"[Checkpoint] Text encoder structure created in {encoder_create_time:.2f}s")
        
        print(f"[Checkpoint] Loading text encoder weights...")
        start_time = time.time()

        # Optimized: Load weights directly (faster than load_state_dict for large models)
        # Convert weights to dtype and device before loading
        with torch.no_grad():
            for name, param in text_encoder.text_model.named_parameters():
                if name in text_encoder_state:
                    param.data = text_encoder_state[name].to(dtype=dtype, device=device)
                else:
                    print(f"[Checkpoint] WARNING: Missing weight for {name}")

            # Also handle buffers (e.g., LayerNorm running_mean/running_var)
            for name, buffer in text_encoder.text_model.named_buffers():
                if name in text_encoder_state:
                    buffer.data = text_encoder_state[name].to(dtype=dtype, device=device)

        weights_load_time = time.time() - start_time
        print(f"[Checkpoint] Text encoder weights loaded from checkpoint in {weights_load_time:.2f}s!")

        # Move model to device (the structure itself, not just weights)
        text_encoder.text_model = text_encoder.text_model.to(device=device, dtype=dtype)

        # Update text_encoder.config to match text_model.config
        text_encoder.config = text_encoder.text_model.config

        print(f"[Checkpoint] Text encoder ready on {device}")
        print(f"[Checkpoint] Text encoder config.max_position_embeddings: {text_encoder.config.max_position_embeddings}")
    elif load_text_encoder:
        print(f"[Checkpoint] WARNING: No text encoder weights found in checkpoint!")

    # Create image encoder and load weights
    image_encoder = None
    if load_image_encoder and len(image_encoder_state) > 0:
        print(f"[Checkpoint] Creating image encoder (from checkpoint)...")
        start_time = time.time()
        image_encoder = SigLIP2ImageEncoder(
            dtype=dtype,
            device=device,
            load_from_checkpoint=True,
            shared_config=shared_config,
            max_position_embeddings=max_position_embeddings
        )
        encoder_create_time = time.time() - start_time
        print(f"[Checkpoint] Image encoder structure created in {encoder_create_time:.2f}s")
        
        print(f"[Checkpoint] Loading image encoder weights...")
        start_time = time.time()
        
        # Optimized: Load weights directly (faster than load_state_dict for large models)
        # Convert weights to dtype and device before loading
        with torch.no_grad():
            for name, param in image_encoder.vision_model.named_parameters():
                if name in image_encoder_state:
                    param.data = image_encoder_state[name].to(dtype=dtype, device=device)
                else:
                    print(f"[Checkpoint] WARNING: Missing weight for {name}")
            
            # Also handle buffers (e.g., LayerNorm running_mean/running_var)
            for name, buffer in image_encoder.vision_model.named_buffers():
                if name in image_encoder_state:
                    buffer.data = image_encoder_state[name].to(dtype=dtype, device=device)
        
        weights_load_time = time.time() - start_time
        print(f"[Checkpoint] Image encoder weights loaded from checkpoint in {weights_load_time:.2f}s!")

        # Move model to device (the structure itself, not just weights)
        image_encoder.vision_model = image_encoder.vision_model.to(device=device, dtype=dtype)

        print(f"[Checkpoint] Image encoder ready on {device}")
    elif load_image_encoder:
        print(f"[Checkpoint] WARNING: No image encoder weights found in checkpoint!")

    # Create VAE and load weights
    vae = None
    if load_vae and len(vae_state) > 0:
        print(f"[Checkpoint] Creating VAE (from checkpoint)...")
        vae = SDXLVAEWrapper(dtype=dtype, device=device, load_from_checkpoint=True)
        print(f"[Checkpoint] Loading VAE weights...")
        # Use strict=False to allow missing keys (quant_conv, post_quant_conv are optional)
        missing_keys, unexpected_keys = vae.vae.load_state_dict(vae_state, strict=False)
        if missing_keys:
            print(f"[Checkpoint] VAE missing keys (optional): {missing_keys}")
        if unexpected_keys:
            print(f"[Checkpoint] VAE unexpected keys: {unexpected_keys}")
        print(f"[Checkpoint] VAE weights loaded from checkpoint!")
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
