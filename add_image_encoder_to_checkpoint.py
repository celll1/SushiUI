"""
Add Image Encoder from base model to existing training checkpoint.
"""
import sys
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch

def add_image_encoder_to_checkpoint(checkpoint_path: str, base_model_path: str):
    """
    Add Image Encoder from base model to training checkpoint.

    Args:
        checkpoint_path: Path to training checkpoint (to be modified)
        base_model_path: Path to base model with Image Encoder
    """
    print(f"Loading training checkpoint: {checkpoint_path}")
    checkpoint = load_file(checkpoint_path)
    checkpoint_metadata = {}

    # Load with metadata
    from safetensors import safe_open
    with safe_open(checkpoint_path, framework='pt', device='cpu') as f:
        if f.metadata():
            checkpoint_metadata = dict(f.metadata())

    print(f"  Total keys: {len(checkpoint)}")
    print(f"  Metadata: {checkpoint_metadata}")

    # Check if image encoder already exists
    image_encoder_keys = [k for k in checkpoint.keys() if k.startswith('image_encoder.')]
    if image_encoder_keys:
        print(f"  WARNING: Image Encoder already exists ({len(image_encoder_keys)} keys)")
        user_input = input("Continue and replace? (y/n): ")
        if user_input.lower() != 'y':
            print("Aborted.")
            return

        # Remove existing image encoder keys
        for key in image_encoder_keys:
            del checkpoint[key]
        print(f"  Removed {len(image_encoder_keys)} existing Image Encoder keys")

    print(f"\nLoading base model: {base_model_path}")
    base_model = load_file(base_model_path)

    # Extract Image Encoder keys from base model
    base_image_encoder_keys = [k for k in base_model.keys() if k.startswith('image_encoder.')]
    print(f"  Found {len(base_image_encoder_keys)} Image Encoder keys in base model")

    if not base_image_encoder_keys:
        print("  ERROR: No Image Encoder found in base model!")
        return

    # Get dtype from checkpoint (text_encoder weights)
    checkpoint_dtype = None
    text_encoder_keys = [k for k in checkpoint.keys() if k.startswith('text_encoder.')]
    if text_encoder_keys:
        checkpoint_dtype = checkpoint[text_encoder_keys[0]].dtype
        print(f"  Checkpoint dtype: {checkpoint_dtype}")
    else:
        # Fallback: check unet
        unet_keys = [k for k in checkpoint.keys() if k.startswith('unet.')]
        if unet_keys:
            checkpoint_dtype = checkpoint[unet_keys[0]].dtype
            print(f"  Checkpoint dtype (from unet): {checkpoint_dtype}")

    # Get dtype from base model image encoder
    base_dtype = base_model[base_image_encoder_keys[0]].dtype
    print(f"  Base model Image Encoder dtype: {base_dtype}")

    # Copy Image Encoder with dtype conversion if needed
    added_keys = 0
    for key in base_image_encoder_keys:
        tensor = base_model[key]

        # Convert dtype if needed
        if checkpoint_dtype and tensor.dtype != checkpoint_dtype:
            tensor = tensor.to(dtype=checkpoint_dtype)

        checkpoint[key] = tensor
        added_keys += 1

    print(f"  Added {added_keys} Image Encoder keys (dtype: {checkpoint_dtype or base_dtype})")

    # Update metadata
    checkpoint_metadata['has_image_encoder'] = 'True'
    if 'train_image_encoder' not in checkpoint_metadata:
        checkpoint_metadata['train_image_encoder'] = 'False'

    print(f"\nUpdated metadata: {checkpoint_metadata}")

    # Save modified checkpoint
    output_path = checkpoint_path
    print(f"\nSaving modified checkpoint: {output_path}")
    save_file(checkpoint, output_path, metadata=checkpoint_metadata)

    print(f"✓ Successfully added Image Encoder to checkpoint")
    print(f"  Total keys now: {len(checkpoint)}")
    print(f"  Image Encoder keys: {len([k for k in checkpoint.keys() if k.startswith('image_encoder.')])}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_image_encoder_to_checkpoint.py <checkpoint_path> <base_model_path>")
        print()
        print("Example:")
        print('  python add_image_encoder_to_checkpoint.py \\')
        print('    "M:\\sushiUI\\training\\20260110_155445_21f4a8fd\\20260110_155445_21f4a8fd_step_010000.safetensors" \\')
        print('    "D:\\celll1\\webui_cl\\models\\deus_model_medium_v2.safetensors"')
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    base_model_path = sys.argv[2]

    # Verify files exist
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not Path(base_model_path).exists():
        print(f"ERROR: Base model not found: {base_model_path}")
        sys.exit(1)

    add_image_encoder_to_checkpoint(checkpoint_path, base_model_path)
