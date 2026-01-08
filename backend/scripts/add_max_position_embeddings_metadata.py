"""
Add max_position_embeddings metadata to DEUS checkpoint

This allows the model to be loaded with configurable max sequence length
instead of hardcoding values in the code.
"""

import sys
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path


def add_metadata(checkpoint_path: str, output_path: str, max_pos_embeddings: int = 512):
    """
    Add max_position_embeddings to checkpoint metadata.

    Args:
        checkpoint_path: Input checkpoint path
        output_path: Output checkpoint path
        max_pos_embeddings: Maximum position embeddings (default: 512)
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    state_dict = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    print(f"Loaded {len(state_dict)} tensors")
    print(f"Current metadata: {metadata}")

    # Check position embedding tensor shapes
    text_pos_emb_key = "conditioner.embedders.0.transformer.embeddings.position_embedding.weight"
    image_pos_emb_key = "conditioner.embedders.1.model.embeddings.position_embedding.weight"

    if text_pos_emb_key in state_dict:
        text_pos_shape = state_dict[text_pos_emb_key].shape
        print(f"\nText encoder position embedding: {text_pos_shape}")
        print(f"  Current max_position_embeddings: {text_pos_shape[0]}")

    if image_pos_emb_key in state_dict:
        image_pos_shape = state_dict[image_pos_emb_key].shape
        print(f"Image encoder position embedding: {image_pos_shape}")
        print(f"  Current max_position_embeddings: {image_pos_shape[0]}")

    # Add metadata
    metadata = metadata.copy() if metadata else {}
    metadata["max_position_embeddings"] = str(max_pos_embeddings)
    metadata["architecture"] = "deus"
    metadata["model_type"] = "diffusion_transformer"
    metadata["text_encoder_type"] = "siglip2"
    metadata["image_encoder_type"] = "siglip2"
    metadata["vae_type"] = "sdxl"
    metadata["variant"] = "medium"

    print(f"\nAdding metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # Save checkpoint
    print(f"\nSaving checkpoint: {output_path}")
    save_file(state_dict, output_path, metadata=metadata)

    # Report file sizes
    input_size = Path(checkpoint_path).stat().st_size / (1024 ** 3)
    output_size = Path(output_path).stat().st_size / (1024 ** 3)

    print(f"\nDone!")
    print(f"  Input:  {input_size:.2f} GB")
    print(f"  Output: {output_size:.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_max_position_embeddings_metadata.py <checkpoint_path> [max_pos_embeddings] [output_path]")
        print("\nExample:")
        print('  python add_max_position_embeddings_metadata.py "D:\\celll1\\webui_cl\\models\\deus_model_medium.safetensors" 512')
        print('  python add_max_position_embeddings_metadata.py input.safetensors 1024 output.safetensors')
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    if len(sys.argv) >= 3:
        max_pos_embeddings = int(sys.argv[2])
    else:
        max_pos_embeddings = 512  # Default: 512 tokens

    if len(sys.argv) >= 4:
        output_path = sys.argv[3]
    else:
        # Default: Create temporary file, then rename
        p = Path(checkpoint_path)
        output_path = str(p.parent / f"{p.stem}_with_metadata{p.suffix}")

    add_metadata(checkpoint_path, output_path, max_pos_embeddings)

    # If output is different from input and user didn't specify output, replace original
    if len(sys.argv) < 4 and output_path != checkpoint_path:
        print(f"\nReplacing original file...")
        import os
        backup_path = str(Path(checkpoint_path).parent / f"{Path(checkpoint_path).stem}_no_metadata{Path(checkpoint_path).suffix}")
        os.rename(checkpoint_path, backup_path)
        os.rename(output_path, checkpoint_path)
        print(f"  Original backed up to: {backup_path}")
        print(f"  New file is now: {checkpoint_path}")
