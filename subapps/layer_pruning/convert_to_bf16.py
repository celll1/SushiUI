#!/usr/bin/env python3
"""
Convert safetensors model to bfloat16 precision.

Usage:
    python convert_to_bf16.py --input model.safetensors --output model_bf16.safetensors
"""

import argparse
import torch
from safetensors.torch import load_file, save_file


def convert_to_bf16(input_path: str, output_path: str):
    """
    Convert all tensors in a safetensors file to bfloat16.

    Args:
        input_path: Path to input safetensors file
        output_path: Path to output safetensors file (bf16)
    """
    print(f"[Convert] Loading model from: {input_path}")
    state_dict = load_file(input_path)

    print(f"[Convert] Converting {len(state_dict)} tensors to bfloat16...")
    bf16_state_dict = {}
    total_size_before = 0
    total_size_after = 0

    for key, tensor in state_dict.items():
        total_size_before += tensor.numel() * tensor.element_size()

        # Convert to bf16
        bf16_tensor = tensor.to(dtype=torch.bfloat16)
        bf16_state_dict[key] = bf16_tensor

        total_size_after += bf16_tensor.numel() * bf16_tensor.element_size()

    print(f"[Convert] Size before: {total_size_before / (1024**3):.2f} GB")
    print(f"[Convert] Size after:  {total_size_after / (1024**3):.2f} GB")
    print(f"[Convert] Reduction:   {(1 - total_size_after/total_size_before)*100:.1f}%")

    print(f"[Convert] Saving to: {output_path}")
    save_file(bf16_state_dict, output_path)

    print("[Convert] Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert safetensors model to bfloat16")
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file")
    parser.add_argument("--output", type=str, required=True, help="Output safetensors file (bf16)")

    args = parser.parse_args()

    convert_to_bf16(args.input, args.output)


if __name__ == "__main__":
    main()
