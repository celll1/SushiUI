"""
Fix DEUS Model FeedForward Layer Dimensions

Issue: TransformerBlock FeedForward layer was initialized with wrong dimensions.
- Old (incorrect): nn.Linear(dim * 2, dim)
- New (correct): nn.Linear(dim * 4, dim)

This script converts existing checkpoint weights to match the correct structure.
"""

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
import sys

def fix_feedforward_weights(checkpoint_path: str, output_path: str):
    """
    Fix FeedForward layer dimensions in DEUS checkpoint.

    TransformerBlock FeedForward structure:
    - GEGLU: dim → dim*4*2 (Linear), then chunk(2) → dim*4 output
    - Linear: dim*4 → dim (this was incorrectly dim*2 → dim)

    The fix: Pad the weight matrix from [dim, dim*2] to [dim, dim*4]
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    state_dict = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    print(f"Loaded {len(state_dict)} tensors")

    # Find all FeedForward second linear layers (ff.2.weight and ff.2.bias)
    ff_keys = [k for k in state_dict.keys() if k.endswith('.ff.2.weight')]

    print(f"\nFound {len(ff_keys)} FeedForward layers to fix:")

    fixed_count = 0
    for ff_weight_key in ff_keys:
        ff_bias_key = ff_weight_key.replace('.weight', '.bias')

        # Get current weight and bias
        weight = state_dict[ff_weight_key]  # [dim, dim*2]
        bias = state_dict[ff_bias_key]      # [dim]

        dim_out, dim_in = weight.shape
        expected_dim_in = dim_out * 4  # Should be dim*4, but currently dim*2

        print(f"  {ff_weight_key}")
        print(f"    Current: [{dim_out}, {dim_in}]")
        print(f"    Expected: [{dim_out}, {expected_dim_in}]")

        if dim_in == expected_dim_in:
            print(f"    [OK] Already correct, skipping")
            continue
        elif dim_in * 2 == expected_dim_in:
            print(f"    [FIX] Padding with duplicated weights")

            # Pad weight: [dim, dim*2] → [dim, dim*4]
            # Strategy: Duplicate the weight matrix (better than zeros for initialization)
            # This allows the model to still function, though it needs retraining
            weight_padded = torch.cat([weight, weight], dim=1)  # [dim, dim*4]

            # Update state dict
            state_dict[ff_weight_key] = weight_padded

            fixed_count += 1
            print(f"    [DONE] Fixed: [{dim_out}, {dim_in}] -> [{weight_padded.shape[0]}, {weight_padded.shape[1]}]")
        else:
            print(f"    [SKIP] Unexpected dimension ratio")

    print(f"\nFixed {fixed_count} FeedForward layers")

    # Save fixed checkpoint
    print(f"\nSaving fixed checkpoint: {output_path}")
    save_file(state_dict, output_path, metadata=metadata)

    # Report file sizes
    input_size = Path(checkpoint_path).stat().st_size / (1024 ** 3)
    output_size = Path(output_path).stat().st_size / (1024 ** 3)

    print(f"\nDone!")
    print(f"  Input:  {input_size:.2f} GB")
    print(f"  Output: {output_size:.2f} GB")
    print(f"  Fixed:  {fixed_count} layers")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_deus_feedforward.py <checkpoint_path> [output_path]")
        print("\nExample:")
        print('  python fix_deus_feedforward.py "D:\\celll1\\webui_cl\\models\\deus_model_medium.safetensors"')
        print('  python fix_deus_feedforward.py input.safetensors output.safetensors')
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Default: Add "_fixed" suffix
        p = Path(checkpoint_path)
        output_path = str(p.parent / f"{p.stem}_fixed{p.suffix}")

    fix_feedforward_weights(checkpoint_path, output_path)
