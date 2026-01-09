"""
Create DEUS v2 initial checkpoint.

Components:
- SigLIP-2 text encoder (google/siglip2-so400m-patch16-naflex, pre-trained)
- SDXL VAE (madebyollin/sdxl-vae-fp16-fix, pre-trained)
- DEUS v2 U-Net (randomly initialized)

Output: D:/celll1/webui_cl/models/deus_model_medium_v2.safetensors
"""

import sys
sys.path.insert(0, 'backend')

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer
from diffusers import AutoencoderKL
from core.models.unet_deus_v2 import DeusUNet, UNetConfig

print("=" * 80)
print("DEUS v2 Initial Model Creation")
print("=" * 80)
print()

# Device
device = "cpu"  # Load to CPU first, will be moved to GPU during inference
dtype = torch.float16

# 1. Load SigLIP-2 (text encoder)
print("1. Loading SigLIP-2 text encoder...")
siglip_model_name = "google/siglip2-so400m-patch16-naflex"
text_encoder = AutoModel.from_pretrained(
    siglip_model_name,
    trust_remote_code=True,
    torch_dtype=dtype
).text_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(siglip_model_name, trust_remote_code=True)

print(f"   Loaded: {siglip_model_name}")
print(f"   Hidden size: {text_encoder.config.hidden_size}")
print(f"   Num layers: {text_encoder.config.num_hidden_layers}")
print()

# 2. Load SDXL VAE
print("2. Loading SDXL VAE...")
vae_model_name = "madebyollin/sdxl-vae-fp16-fix"
vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=dtype).to(device)

print(f"   Loaded: {vae_model_name}")
print(f"   Latent channels: {vae.config.latent_channels}")
print(f"   Scaling factor: {vae.config.scaling_factor}")
print()

# 3. Initialize DEUS v2 U-Net
print("3. Initializing DEUS v2 U-Net (medium variant)...")
unet_config = UNetConfig.from_variant("medium")
unet = DeusUNet(unet_config).to(device, dtype=dtype)

# Count parameters
unet_params = sum(p.numel() for p in unet.parameters())
print(f"   U-Net parameters: {unet_params / 1e9:.3f}B")
print()

# 4. Collect state dicts
print("4. Collecting state dicts...")
state_dict = {}

# Text encoder
for key, value in text_encoder.state_dict().items():
    state_dict[f"text_encoder.{key}"] = value.to(dtype)

# VAE
for key, value in vae.state_dict().items():
    state_dict[f"vae.{key}"] = value.to(dtype)

# U-Net
for key, value in unet.state_dict().items():
    state_dict[f"unet.{key}"] = value.to(dtype)

print(f"   Total tensors: {len(state_dict)}")
print()

# 5. Add metadata
metadata = {
    "model_type": "deus_v2",
    "architecture": "sdxl_compatible",
    "text_encoder": siglip_model_name,
    "vae": vae_model_name,
    "unet_variant": "medium",
    "unet_params": f"{unet_params / 1e9:.3f}B",
    "block_out_channels": str(unet_config.block_out_channels),
    "skip_connection_blocks": str(unet_config.skip_connection_blocks),
    "skip_connections_per_up_block": str(unet_config.skip_connections_per_up_block),
    "attention_head_dim": str(unet_config.attention_head_dim),
    "num_attention_heads": str(unet_config.num_attention_heads),
    "transformer_layers_per_block": str(unet_config.transformer_layers_per_block),
    "context_dim": str(unet_config.context_dim),
    "max_position_embeddings": "77",  # SigLIP-2 default
    "dtype": "float16",
    "created_by": "SushiUI DEUS v2 Initial Model Creator",
}

# 6. Save checkpoint
output_path = "D:/celll1/webui_cl/models/deus_model_medium_v2.safetensors"
print(f"5. Saving checkpoint to {output_path}...")

save_file(state_dict, output_path, metadata=metadata)

# Calculate file size
import os
file_size_gb = os.path.getsize(output_path) / (1024**3)
print(f"   File size: {file_size_gb:.2f} GB")
print()

print("=" * 80)
print("DEUS v2 Initial Model Created Successfully!")
print("=" * 80)
print()
print(f"Output: {output_path}")
print()
print("Components:")
print(f"  - Text Encoder: SigLIP-2 ({text_encoder.config.hidden_size}D, pre-trained)")
print(f"  - VAE: SDXL VAE (madebyollin, pre-trained)")
print(f"  - U-Net: DEUS v2 Medium ({unet_params / 1e9:.3f}B params, randomly initialized)")
print()
print("Next steps:")
print("  1. Train the U-Net on your dataset")
print("  2. Use the trained model for inference")
print()
