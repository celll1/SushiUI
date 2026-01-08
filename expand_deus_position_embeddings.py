"""
Expand position_embeddings in DEUS checkpoint from 64 to 512 tokens.

This script:
1. Loads the DEUS checkpoint
2. Expands text encoder position_embedding from 64 to 512 using linear interpolation
3. Updates metadata
4. Saves the modified checkpoint
"""

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
import sys
import os

def expand_position_embeddings(checkpoint_path: str, output_path: str, target_size: int = 512):
    """
    Expand position_embeddings in DEUS checkpoint.

    Args:
        checkpoint_path: Input checkpoint path
        output_path: Output checkpoint path
        target_size: Target position embedding size (default: 512)
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    state_dict = load_file(checkpoint_path)

    # Load metadata
    from safetensors import safe_open
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    print(f"Original metadata: {metadata}")
    print(f"Total keys: {len(state_dict)}")

    # Find text encoder position_embedding
    pos_emb_key = "conditioner.embedders.0.transformer.embeddings.position_embedding.weight"

    if pos_emb_key not in state_dict:
        print(f"ERROR: Position embedding key not found: {pos_emb_key}")
        print("Available keys (first 20):")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            print(f"  {i+1}. {key}")
        return False

    # Get current position embedding
    old_weight = state_dict[pos_emb_key]
    current_size = old_weight.shape[0]
    hidden_size = old_weight.shape[1]

    print(f"\nCurrent position_embedding:")
    print(f"  Shape: {old_weight.shape}")
    print(f"  Current size: {current_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Target size: {target_size}")

    if current_size >= target_size:
        print(f"\nPosition embedding already has {current_size} positions (>= target {target_size})")
        print("No expansion needed.")
        return True

    # Expand using linear interpolation
    print(f"\nExpanding position_embedding: {current_size} -> {target_size}")

    # Reshape for interpolation: [current_size, hidden_size] -> [1, hidden_size, current_size]
    old_weight_transposed = old_weight.t().unsqueeze(0)  # [1, hidden_size, current_size]

    # Interpolate to target size
    new_weight_transposed = F.interpolate(
        old_weight_transposed,
        size=target_size,
        mode='linear',
        align_corners=True
    )  # [1, hidden_size, target_size]

    # Reshape back: [1, hidden_size, target_size] -> [target_size, hidden_size]
    new_weight = new_weight_transposed.squeeze(0).t()

    # Make contiguous (required for safetensors)
    new_weight = new_weight.contiguous()

    print(f"New position_embedding shape: {new_weight.shape}")
    print(f"Is contiguous: {new_weight.is_contiguous()}")

    # Update state_dict
    state_dict[pos_emb_key] = new_weight

    # Update metadata
    new_metadata = metadata.copy()
    new_metadata["max_position_embeddings"] = str(target_size)
    new_metadata["position_embedding_expanded"] = "true"
    new_metadata["original_position_embedding_size"] = str(current_size)

    print(f"\nUpdated metadata:")
    for key, value in new_metadata.items():
        if key in metadata and metadata[key] != value:
            print(f"  {key}: {metadata[key]} -> {value}")
        elif key not in metadata:
            print(f"  {key}: (new) {value}")

    # Save modified checkpoint
    print(f"\nSaving modified checkpoint to: {output_path}")
    save_file(state_dict, output_path, metadata=new_metadata)

    print("\n[OK] Checkpoint saved successfully!")

    # Verify saved checkpoint
    print("\nVerifying saved checkpoint...")
    with safe_open(output_path, framework="pt", device="cpu") as f:
        saved_metadata = f.metadata() or {}
        saved_pos_emb = f.get_tensor(pos_emb_key)

    print(f"  Saved position_embedding shape: {saved_pos_emb.shape}")
    print(f"  Saved metadata max_position_embeddings: {saved_metadata.get('max_position_embeddings', 'N/A')}")

    if saved_pos_emb.shape[0] == target_size:
        print("\n[OK] Verification passed!")
        return True
    else:
        print(f"\n[ERROR] Verification failed: Expected {target_size}, got {saved_pos_emb.shape[0]}")
        return False

if __name__ == "__main__":
    # Default paths
    checkpoint_path = r"D:\celll1\webui_cl\models\deus_model_medium.safetensors"

    # Create backup path and output path
    backup_path = checkpoint_path.replace(".safetensors", "_backup_original.safetensors")
    output_path = checkpoint_path.replace(".safetensors", "_expanded.safetensors")  # Save to new file

    # Check if backup already exists
    if os.path.exists(backup_path):
        print(f"Backup already exists: {backup_path}")
        print("Using existing backup.")
    else:
        # Create backup
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(checkpoint_path, backup_path)
        print("[OK] Backup created successfully!")

    print(f"\nOriginal checkpoint: {checkpoint_path}")
    print(f"Backup checkpoint: {backup_path}")
    print(f"Output checkpoint: {output_path}")
    print("\n" + "="*80)

    # Expand position embeddings
    success = expand_position_embeddings(checkpoint_path, output_path, target_size=512)

    if success:
        print("\n" + "="*80)
        print("[OK] Position embedding expansion completed successfully!")
        print(f"[OK] Original checkpoint backed up to: {backup_path}")
        print(f"[OK] Modified checkpoint saved to: {output_path}")
    else:
        print("\n" + "="*80)
        print("[ERROR] Position embedding expansion failed!")
        sys.exit(1)
