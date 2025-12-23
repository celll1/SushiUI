"""
Evaluate Pruned Model

Compare the 1-step inference loss between original and pruned models.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from prune_layers import LayerPruner


def main():
    parser = argparse.ArgumentParser(description="Evaluate pruned model vs original")
    parser.add_argument("--original-model", type=str, required=True, help="Path to original model (safetensors)")
    parser.add_argument("--pruned-model", type=str, required=True, help="Path to pruned model (safetensors)")
    parser.add_argument("--samples", type=str, required=True, help="Path to samples JSON")
    parser.add_argument("--device", type=str, default="cuda", help="Device for computation")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Data type")

    args = parser.parse_args()

    print("=" * 60)
    print("Evaluate Pruned Model")
    print("=" * 60)
    print(f"Original model: {args.original_model}")
    print(f"Pruned model: {args.pruned_model}")
    print(f"Samples: {args.samples}")
    print("=" * 60)

    # Check paths
    if not Path(args.original_model).exists():
        print(f"ERROR: Original model not found: {args.original_model}")
        sys.exit(1)

    if not Path(args.pruned_model).exists():
        print(f"ERROR: Pruned model not found: {args.pruned_model}")
        sys.exit(1)

    if not Path(args.samples).exists():
        print(f"ERROR: Samples file not found: {args.samples}")
        sys.exit(1)

    # Load samples
    with open(args.samples, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"[Eval] Loaded {len(samples)} samples")

    # Setup device and dtype
    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Evaluate original model
    print("\n[Eval] Evaluating original model...")
    original_pruner = LayerPruner(
        model_path=args.original_model,
        samples=samples,
        device=device,
        dtype=dtype
    )
    original_loss = original_pruner.evaluate_model(original_pruner.original_transformer)
    original_layers = len(original_pruner.original_transformer.layers)

    print(f"[Eval] Original model: {original_layers} layers, loss={original_loss:.6f}")

    # Evaluate pruned model
    print("\n[Eval] Evaluating pruned model...")
    pruned_pruner = LayerPruner(
        model_path=args.pruned_model,
        samples=samples,
        device=device,
        dtype=dtype
    )
    pruned_loss = pruned_pruner.evaluate_model(pruned_pruner.original_transformer)
    pruned_layers = len(pruned_pruner.original_transformer.layers)

    print(f"[Eval] Pruned model: {pruned_layers} layers, loss={pruned_loss:.6f}")

    # Calculate delta
    delta_loss = pruned_loss - original_loss
    delta_pct = (delta_loss / original_loss) * 100 if original_loss > 0 else 0

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Original: {original_layers} layers, loss={original_loss:.6f}")
    print(f"Pruned:   {pruned_layers} layers, loss={pruned_loss:.6f}")
    print(f"Delta:    {delta_loss:.6f} ({delta_pct:+.2f}%)")
    print(f"Layer reduction: {original_layers} -> {pruned_layers} ({pruned_layers/original_layers*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
