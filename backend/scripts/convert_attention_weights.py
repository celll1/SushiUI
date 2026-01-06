"""
Convert DEUS checkpoint from nn.MultiheadAttention to Flash Attention format.

Old format (nn.MultiheadAttention):
  - self_attn.in_proj_weight  [3*C, C]  (Q,K,V fused)
  - self_attn.in_proj_bias    [3*C]
  - self_attn.out_proj.weight [C, C]
  - self_attn.out_proj.bias   [C]
  - cross_attn.q_proj_weight  [C, C]
  - cross_attn.k_proj_weight  [C, context_dim]
  - cross_attn.v_proj_weight  [C, context_dim]
  - cross_attn.in_proj_bias   [3*C]  (only for K,V; Q has separate weight)
  - cross_attn.out_proj.weight [C, C]
  - cross_attn.out_proj.bias   [C]

New format (Flash Attention):
  - to_q.weight         [C, C]
  - to_k.weight         [C, C]
  - to_v.weight         [C, C]
  - to_out.weight       [C, C]
  - to_out.bias         [C]
  - to_q_cross.weight   [C, C]
  - to_k_cross.weight   [C, context_dim]
  - to_v_cross.weight   [C, context_dim]
  - to_out_cross.weight [C, C]
  - to_out_cross.bias   [C]
"""

import torch
from safetensors import safe_open
from safetensors.torch import save_file
import argparse
from pathlib import Path


def convert_self_attention_weights(prefix: str, state_dict: dict, channels: int):
    """Convert self-attention weights from MultiheadAttention to manual Q/K/V."""

    # Old keys
    in_proj_weight_key = f"{prefix}.self_attn.in_proj_weight"
    in_proj_bias_key = f"{prefix}.self_attn.in_proj_bias"
    out_proj_weight_key = f"{prefix}.self_attn.out_proj.weight"
    out_proj_bias_key = f"{prefix}.self_attn.out_proj.bias"

    # Get old weights
    in_proj_weight = state_dict[in_proj_weight_key]  # [3*C, C]
    in_proj_bias = state_dict[in_proj_bias_key]      # [3*C]
    out_proj_weight = state_dict[out_proj_weight_key]  # [C, C]
    out_proj_bias = state_dict[out_proj_bias_key]      # [C]

    # Split in_proj into Q, K, V
    q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)  # Each: [C, C]
    q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)          # Each: [C]

    # New keys (no bias for Q/K/V in Flash Attention version)
    new_weights = {
        f"{prefix}.to_q.weight": q_weight,
        f"{prefix}.to_k.weight": k_weight,
        f"{prefix}.to_v.weight": v_weight,
        f"{prefix}.to_out.weight": out_proj_weight,
        f"{prefix}.to_out.bias": out_proj_bias,
    }

    # Remove old keys
    del state_dict[in_proj_weight_key]
    del state_dict[in_proj_bias_key]
    del state_dict[out_proj_weight_key]
    del state_dict[out_proj_bias_key]

    return new_weights


def convert_cross_attention_weights(prefix: str, state_dict: dict, channels: int, context_dim: int):
    """Convert cross-attention weights from MultiheadAttention to manual Q/K/V."""

    # Old keys
    q_proj_weight_key = f"{prefix}.cross_attn.q_proj_weight"
    k_proj_weight_key = f"{prefix}.cross_attn.k_proj_weight"
    v_proj_weight_key = f"{prefix}.cross_attn.v_proj_weight"
    in_proj_bias_key = f"{prefix}.cross_attn.in_proj_bias"
    out_proj_weight_key = f"{prefix}.cross_attn.out_proj.weight"
    out_proj_bias_key = f"{prefix}.cross_attn.out_proj.bias"

    # Get old weights
    q_weight = state_dict[q_proj_weight_key]  # [C, C]
    k_weight = state_dict[k_proj_weight_key]  # [C, context_dim]
    v_weight = state_dict[v_proj_weight_key]  # [C, context_dim]
    in_proj_bias = state_dict[in_proj_bias_key]  # [3*C] (but only K,V portions used)
    out_proj_weight = state_dict[out_proj_weight_key]  # [C, C]
    out_proj_bias = state_dict[out_proj_bias_key]      # [C]

    # New keys (no bias for Q/K/V in Flash Attention version)
    new_weights = {
        f"{prefix}.to_q_cross.weight": q_weight,
        f"{prefix}.to_k_cross.weight": k_weight,
        f"{prefix}.to_v_cross.weight": v_weight,
        f"{prefix}.to_out_cross.weight": out_proj_weight,
        f"{prefix}.to_out_cross.bias": out_proj_bias,
    }

    # Remove old keys
    del state_dict[q_proj_weight_key]
    del state_dict[k_proj_weight_key]
    del state_dict[v_proj_weight_key]
    del state_dict[in_proj_bias_key]
    del state_dict[out_proj_weight_key]
    del state_dict[out_proj_bias_key]

    return new_weights


def convert_checkpoint(input_path: str, output_path: str):
    """Convert entire checkpoint from old to new format."""

    print(f"Loading checkpoint from: {input_path}")

    # Load checkpoint
    state_dict = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    print(f"Loaded {len(state_dict)} tensors")

    # Get model config from metadata
    channels = int(metadata.get("model_channels", 384))
    context_dim = int(metadata.get("context_dim", 1152))

    print(f"Model config: channels={channels}, context_dim={context_dim}")

    # Convert all attention layers
    attention_prefixes = []

    # Collect all U-Net attention layer prefixes (skip text/image encoders)
    for key in list(state_dict.keys()):
        if ".self_attn." in key or ".cross_attn." in key:
            # Skip non-U-Net layers (conditioner, text_encoder, image_encoder, vae)
            if any(skip in key for skip in ["conditioner", "text_encoder", "first_stage_model"]):
                continue

            # Extract prefix (e.g., "down_blocks.0.attentions.0")
            parts = key.split(".")
            if "self_attn" in parts:
                idx = parts.index("self_attn")
            else:
                idx = parts.index("cross_attn")
            prefix = ".".join(parts[:idx])

            if prefix not in attention_prefixes:
                attention_prefixes.append(prefix)

    print(f"Found {len(attention_prefixes)} attention layers to convert")

    # Convert each attention layer
    new_weights_all = {}
    for prefix in attention_prefixes:
        print(f"Converting: {prefix}")

        # Convert self-attention
        self_attn_weights = convert_self_attention_weights(prefix, state_dict, channels)
        new_weights_all.update(self_attn_weights)

        # Convert cross-attention
        cross_attn_weights = convert_cross_attention_weights(prefix, state_dict, channels, context_dim)
        new_weights_all.update(cross_attn_weights)

    # Add new weights to state_dict
    state_dict.update(new_weights_all)

    print(f"Converted checkpoint has {len(state_dict)} tensors")

    # Update metadata
    metadata["attention_format"] = "flash_attention"
    metadata["converted_from"] = "nn.MultiheadAttention"

    # Save converted checkpoint
    print(f"Saving to: {output_path}")
    save_file(state_dict, output_path, metadata=metadata)
    print("Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DEUS checkpoint to Flash Attention format")
    parser.add_argument("input", type=str, help="Input checkpoint path")
    parser.add_argument("output", type=str, help="Output checkpoint path")

    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)
