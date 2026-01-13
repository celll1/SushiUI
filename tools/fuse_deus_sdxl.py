"""
DEUS-SDXL U-Net Weight Fusion Tool

Fuses SDXL U-Net weights into DEUS model v2 to potentially accelerate fine-tuning.

Architecture Analysis:
- DEUS and SDXL share identical channel progression: 320 -> 640 -> 1280
- Compatible: conv_in, conv_out, time_embed, self-attention, feedforward, resnet, normalization
- Incompatible: cross-attention to_k/to_v (DEUS: 1152 context dim, SDXL: 2048 context dim)

Fusion Strategy (Option A - Maximum Transfer):
- Transfer 91.6% of weights (all compatible layers)
- Keep DEUS original cross-attention to_k/to_v layers (8.4%)
"""

import argparse
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def build_layer_mapping(deus_keys: list, sdxl_keys: list) -> dict:
    """
    Build mapping from DEUS U-Net keys to SDXL U-Net keys.

    DEUS format: unet.down_blocks.X.attentions.Y.transformer_blocks.Z...
    SDXL format: model.diffusion_model.input_blocks.N.M.transformer_blocks.Z...

    Returns:
        dict: {deus_key: sdxl_key} for compatible layers
    """
    mapping = {}

    # DEUS to SDXL block index mapping (approximate)
    # DEUS down_blocks: 0, 1, 2 -> SDXL input_blocks: 1-3, 4-6, 7-9
    # DEUS mid_block -> SDXL middle_block
    # DEUS up_blocks: 0, 1, 2 -> SDXL output_blocks: 0-2, 3-5, 6-8

    # For now, we'll use shape-based matching which is more robust
    return mapping


def is_cross_attention_kv(key: str) -> bool:
    """Check if key is a cross-attention to_k or to_v layer (incompatible)."""
    return 'attn2' in key and ('to_k' in key or 'to_v' in key)


def categorize_layer(key: str) -> str:
    """Categorize a layer by its type."""
    if 'time_embed' in key:
        return 'time_embed'
    elif 'conv_in' in key or 'input_blocks.0.0' in key:
        return 'conv_in'
    elif 'conv_out' in key or 'out.' in key:
        return 'conv_out'
    elif 'attn1' in key:
        return 'self_attention'
    elif 'attn2' in key:
        if 'to_k' in key or 'to_v' in key:
            return 'cross_attention_kv'  # Incompatible
        else:
            return 'cross_attention_qo'  # Compatible (to_q, to_out)
    elif 'ff.net' in key or 'ff.' in key:
        return 'feedforward'
    elif 'norm' in key:
        return 'normalization'
    elif 'proj_in' in key or 'proj_out' in key:
        return 'projection'
    elif 'resnet' in key or 'in_layers' in key or 'out_layers' in key:
        return 'resnet'
    elif 'downsample' in key or 'upsample' in key:
        return 'sampling'
    else:
        return 'other'


def find_best_sdxl_match(deus_key: str, deus_shape: tuple, sdxl_by_shape: dict,
                          sdxl_used: set, deus_category: str) -> str:
    """
    Find the best matching SDXL key for a DEUS key.

    Matching criteria:
    1. Same shape
    2. Same layer category
    3. Not already used
    """
    if deus_shape not in sdxl_by_shape:
        return None

    candidates = sdxl_by_shape[deus_shape]

    for sdxl_key in candidates:
        if sdxl_key in sdxl_used:
            continue

        sdxl_category = categorize_layer(sdxl_key)

        # Match categories
        if deus_category == sdxl_category:
            return sdxl_key

        # Allow some flexibility for similar categories
        if deus_category == 'cross_attention_qo' and sdxl_category == 'cross_attention_qo':
            return sdxl_key

    return None


def fuse_cross_attention_kv(deus_tensor: torch.Tensor, sdxl_tensor: torch.Tensor,
                             method: str = "truncate", transfer_ratio: float = 1.0) -> torch.Tensor:
    """
    Fuse cross-attention to_k/to_v weights with different context dimensions.

    DEUS: [out_dim, 1152] (context_dim=1152, SigLIP2)
    SDXL: [out_dim, 2048] (context_dim=2048, CLIP+OpenCLIP)

    Methods:
    - truncate: Use first 1152 dims of SDXL (simple but loses information)
    - pca: Project SDXL 2048 -> 1152 using SVD (preserves variance)
    - average_chunks: Average SDXL dims in chunks to get 1152 (smooths features)
    - weighted_sample: Sample SDXL dims with importance weighting

    Args:
        deus_tensor: DEUS weight tensor [out_dim, 1152]
        sdxl_tensor: SDXL weight tensor [out_dim, 2048]
        method: Fusion method
        transfer_ratio: Blend ratio (0=DEUS, 1=SDXL)

    Returns:
        Fused tensor [out_dim, 1152]
    """
    out_dim, deus_context = deus_tensor.shape
    _, sdxl_context = sdxl_tensor.shape

    if method == "truncate":
        # Simple truncation: use first 1152 dimensions
        # SDXL's CLIP text encoder uses first ~768 dims, rest is OpenCLIP
        # This preserves the CLIP portion which is more semantically aligned
        fused = sdxl_tensor[:, :deus_context].clone()

    elif method == "pca":
        # SVD-based projection: project 2048 -> 1152 preserving max variance
        # We need to create a linear projection from SDXL's 2048-dim space to DEUS's 1152-dim space
        #
        # Approach: Learn a projection matrix that maps 2048 -> 1152
        # using SVD on SDXL weights to find the most important directions
        #
        # sdxl_tensor: [out_dim, 2048]
        # We want: fused [out_dim, 1152]

        sdxl_float = sdxl_tensor.float()

        # SVD on sdxl_tensor: [out_dim, 2048] = U @ S @ Vh
        # U: [out_dim, k], S: [k], Vh: [k, 2048] where k = min(out_dim, 2048)
        U, S, Vh = torch.linalg.svd(sdxl_float, full_matrices=False)

        # Vh contains the principal directions in 2048-dim space
        # Take first 1152 principal directions: Vh[:1152, :] is [1152, 2048]
        # Project: sdxl @ Vh[:1152, :].T gives [out_dim, 1152]

        # But if out_dim < 1152, we only have out_dim principal directions
        # In that case, we need to pad or use interpolation
        k = min(S.shape[0], deus_context)  # Available principal components

        if k >= deus_context:
            # Have enough components: use first deus_context
            projection = Vh[:deus_context, :].T  # [2048, 1152]
            fused = (sdxl_float @ projection).to(deus_tensor.dtype)
        else:
            # Not enough components (out_dim < 1152): use interpolation fallback
            # First project to k dimensions, then interpolate to 1152
            projection = Vh[:k, :].T  # [2048, k]
            projected = sdxl_float @ projection  # [out_dim, k]

            # Interpolate from k to 1152
            fused = torch.nn.functional.interpolate(
                projected.unsqueeze(0),  # [1, out_dim, k]
                size=deus_context,
                mode='linear',
                align_corners=True
            ).squeeze(0).to(deus_tensor.dtype)  # [out_dim, 1152]

    elif method == "average_chunks":
        # Average adjacent dimensions to reduce 2048 -> 1152
        # 2048 / 1152 ≈ 1.778, so we need adaptive chunking
        # Use interpolation-like approach
        sdxl_float = sdxl_tensor.float()

        # Reshape and interpolate
        # [out, 2048] -> [out, 1152] using linear interpolation
        fused = torch.nn.functional.interpolate(
            sdxl_float.unsqueeze(0),  # [1, out, 2048]
            size=deus_context,
            mode='linear',
            align_corners=True
        ).squeeze(0).to(deus_tensor.dtype)  # [out, 1152]

    elif method == "weighted_sample":
        # Sample dimensions with L2-norm based importance weighting
        # Dimensions with higher norm are more important
        sdxl_float = sdxl_tensor.float()

        # Compute importance per dimension (L2 norm across output dim)
        importance = sdxl_float.pow(2).sum(dim=0).sqrt()  # [2048]

        # Select top-1152 most important dimensions
        _, top_indices = importance.topk(deus_context)
        top_indices = top_indices.sort().values  # Sort for consistency

        fused = sdxl_float[:, top_indices].to(deus_tensor.dtype)

    else:
        raise ValueError(f"Unknown cross-attention fusion method: {method}")

    # Blend with original DEUS weights
    if transfer_ratio < 1.0:
        fused = (1.0 - transfer_ratio) * deus_tensor + transfer_ratio * fused

    return fused


def fuse_weights(deus_state: dict, sdxl_state: dict,
                 transfer_ratio: float = 1.0,
                 cross_attn_method: str = "none",
                 cross_attn_ratio: float = 1.0,
                 verbose: bool = True) -> tuple:
    """
    Fuse SDXL U-Net weights into DEUS model.

    Args:
        deus_state: DEUS model state dict
        sdxl_state: SDXL model state dict
        transfer_ratio: Ratio of SDXL weights to use (0.0 = DEUS only, 1.0 = full SDXL)
        cross_attn_method: Method for cross-attention fusion ("none", "truncate", "pca", "average_chunks", "weighted_sample")
        cross_attn_ratio: Ratio for cross-attention fusion (separate from transfer_ratio)
        verbose: Print detailed progress

    Returns:
        tuple: (fused_state_dict, stats_dict)
    """
    # Extract U-Net keys
    deus_unet = {k: v for k, v in deus_state.items() if k.startswith('unet.')}
    sdxl_unet = {k: v for k, v in sdxl_state.items() if k.startswith('model.diffusion_model.')}

    print(f"DEUS U-Net layers: {len(deus_unet)}")
    print(f"SDXL U-Net layers: {len(sdxl_unet)}")

    # Build SDXL shape -> keys mapping
    sdxl_by_shape = defaultdict(list)
    for k, v in sdxl_unet.items():
        sdxl_by_shape[tuple(v.shape)].append(k)

    # Track statistics
    stats = {
        'transferred': 0,
        'kept_deus': 0,
        'incompatible_shape': 0,
        'incompatible_cross_attn': 0,
        'cross_attn_fused': 0,
        'cross_attn_method': cross_attn_method,
        'by_category': defaultdict(lambda: {'transferred': 0, 'kept': 0, 'fused': 0})
    }

    # Build SDXL cross-attention mapping for dimension-mismatched fusion
    # DEUS [out, 1152] <-> SDXL [out, 2048]
    sdxl_cross_attn_by_out_dim = defaultdict(list)
    for k, v in sdxl_unet.items():
        if 'attn2' in k and ('to_k.weight' in k or 'to_v.weight' in k):
            out_dim = v.shape[0]
            sdxl_cross_attn_by_out_dim[out_dim].append((k, v))

    # Create fused state dict (start with full DEUS model)
    fused_state = deus_state.copy()
    sdxl_used = set()

    # Process each DEUS U-Net layer
    for deus_key, deus_tensor in tqdm(deus_unet.items(), desc="Fusing weights"):
        deus_shape = tuple(deus_tensor.shape)
        category = categorize_layer(deus_key)

        # Handle cross-attention to_k/to_v layers (dimension mismatch)
        if is_cross_attention_kv(deus_key):
            if cross_attn_method != "none":
                # Try to fuse with dimension conversion
                out_dim = deus_tensor.shape[0]
                layer_type = 'to_k' if 'to_k' in deus_key else 'to_v'

                # Find matching SDXL layer by output dimension and layer type
                sdxl_match = None
                for sdxl_key, sdxl_tensor in sdxl_cross_attn_by_out_dim.get(out_dim, []):
                    if layer_type in sdxl_key and sdxl_key not in sdxl_used:
                        sdxl_match = (sdxl_key, sdxl_tensor)
                        break

                if sdxl_match:
                    sdxl_key, sdxl_tensor = sdxl_match
                    sdxl_used.add(sdxl_key)

                    # Fuse with dimension conversion
                    fused_tensor = fuse_cross_attention_kv(
                        deus_tensor, sdxl_tensor,
                        method=cross_attn_method,
                        transfer_ratio=cross_attn_ratio
                    )
                    fused_state[deus_key] = fused_tensor

                    stats['cross_attn_fused'] += 1
                    stats['by_category'][category]['fused'] += 1
                    continue

            # No fusion - keep DEUS original
            stats['incompatible_cross_attn'] += 1
            stats['kept_deus'] += 1
            stats['by_category'][category]['kept'] += 1
            continue

        # Find matching SDXL layer
        sdxl_key = find_best_sdxl_match(deus_key, deus_shape, sdxl_by_shape, sdxl_used, category)

        if sdxl_key is None:
            # No match found - keep DEUS original
            stats['incompatible_shape'] += 1
            stats['kept_deus'] += 1
            stats['by_category'][category]['kept'] += 1
            continue

        # Transfer weights (with optional blending)
        sdxl_tensor = sdxl_unet[sdxl_key]
        sdxl_used.add(sdxl_key)

        if transfer_ratio >= 1.0:
            # Full transfer
            fused_state[deus_key] = sdxl_tensor.clone()
        elif transfer_ratio <= 0.0:
            # Keep DEUS
            pass
        else:
            # Blend: (1-ratio) * DEUS + ratio * SDXL
            fused_state[deus_key] = (1.0 - transfer_ratio) * deus_tensor + transfer_ratio * sdxl_tensor

        stats['transferred'] += 1
        stats['by_category'][category]['transferred'] += 1

    return fused_state, stats


def print_stats(stats: dict):
    """Print fusion statistics."""
    print("\n" + "=" * 60)
    print("FUSION STATISTICS")
    print("=" * 60)

    total = stats['transferred'] + stats['kept_deus'] + stats['cross_attn_fused']
    print(f"\nTotal U-Net layers: {total}")
    print(f"Transferred from SDXL (same shape): {stats['transferred']} ({stats['transferred']/total*100:.1f}%)")
    print(f"Cross-attn fused (dim conversion): {stats['cross_attn_fused']} ({stats['cross_attn_fused']/total*100:.1f}%)")
    print(f"Kept from DEUS: {stats['kept_deus']} ({stats['kept_deus']/total*100:.1f}%)")
    print(f"  - Incompatible cross-attention: {stats['incompatible_cross_attn']}")
    print(f"  - Incompatible shape: {stats['incompatible_shape']}")

    if stats['cross_attn_method'] != "none":
        print(f"\nCross-attention fusion method: {stats['cross_attn_method']}")

    print("\nBy category:")
    for category, cat_stats in sorted(stats['by_category'].items()):
        cat_total = cat_stats['transferred'] + cat_stats['kept'] + cat_stats.get('fused', 0)
        if cat_total > 0:
            parts = []
            if cat_stats['transferred'] > 0:
                parts.append(f"{cat_stats['transferred']} transferred")
            if cat_stats.get('fused', 0) > 0:
                parts.append(f"{cat_stats['fused']} fused")
            if cat_stats['kept'] > 0:
                parts.append(f"{cat_stats['kept']} kept")
            print(f"  {category}: {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description="Fuse SDXL U-Net weights into DEUS model")
    parser.add_argument(
        "--deus-model",
        type=str,
        default="D:/celll1/webui_cl/models/deus_model_medium_v2.safetensors",
        help="Path to DEUS model",
    )
    parser.add_argument(
        "--sdxl-model",
        type=str,
        default="M:/model/sdxl/Illustrious-XL-v2.0.safetensors",
        help="Path to SDXL model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="D:/celll1/webui_cl/models/deus_model_medium_v2_sdxl_fused.safetensors",
        help="Output path for fused model",
    )
    parser.add_argument(
        "--transfer-ratio",
        type=float,
        default=1.0,
        help="Ratio of SDXL weights (0.0=DEUS only, 1.0=full SDXL transfer)",
    )
    parser.add_argument(
        "--unet-only",
        action="store_true",
        help="Only save U-Net weights (smaller file)",
    )
    parser.add_argument(
        "--cross-attn-method",
        type=str,
        default="none",
        choices=["none", "truncate", "pca", "average_chunks", "weighted_sample"],
        help="Method for cross-attention fusion (none=skip, truncate=use first 1152 dims, pca=SVD projection, average_chunks=interpolation, weighted_sample=importance sampling)",
    )
    parser.add_argument(
        "--cross-attn-ratio",
        type=float,
        default=1.0,
        help="Ratio for cross-attention fusion (separate from transfer-ratio)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DEUS-SDXL U-Net Weight Fusion")
    print("=" * 60)
    print(f"DEUS model: {args.deus_model}")
    print(f"SDXL model: {args.sdxl_model}")
    print(f"Output: {args.output}")
    print(f"Transfer ratio: {args.transfer_ratio}")
    print(f"Cross-attn method: {args.cross_attn_method}")
    if args.cross_attn_method != "none":
        print(f"Cross-attn ratio: {args.cross_attn_ratio}")

    # Load models
    print("\n[1/3] Loading models...")
    print("  Loading DEUS model...")
    deus_state = load_file(args.deus_model)
    print(f"  DEUS keys: {len(deus_state)}")

    # Load DEUS metadata (important for model detection)
    deus_metadata = None
    with safe_open(args.deus_model, framework='pt') as f:
        deus_metadata = f.metadata()
    if deus_metadata:
        print(f"  DEUS metadata: {deus_metadata.get('model_type', 'unknown')}")

    print("  Loading SDXL model...")
    sdxl_state = load_file(args.sdxl_model)
    print(f"  SDXL keys: {len(sdxl_state)}")

    # Fuse weights
    print("\n[2/3] Fusing weights...")
    fused_state, stats = fuse_weights(
        deus_state, sdxl_state,
        transfer_ratio=args.transfer_ratio,
        cross_attn_method=args.cross_attn_method,
        cross_attn_ratio=args.cross_attn_ratio,
        verbose=True
    )

    print_stats(stats)

    # Save fused model
    print("\n[3/3] Saving fused model...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare metadata (preserve DEUS metadata + add fusion info)
    output_metadata = {}
    if deus_metadata:
        output_metadata = dict(deus_metadata)
    output_metadata['sdxl_fused'] = 'true'
    output_metadata['sdxl_source'] = Path(args.sdxl_model).name
    output_metadata['transfer_ratio'] = str(args.transfer_ratio)
    output_metadata['transferred_layers'] = str(stats['transferred'])
    output_metadata['kept_deus_layers'] = str(stats['kept_deus'])
    output_metadata['cross_attn_fused'] = str(stats['cross_attn_fused'])
    output_metadata['cross_attn_method'] = args.cross_attn_method

    if args.unet_only:
        # Save only U-Net weights
        unet_state = {k: v for k, v in fused_state.items() if k.startswith('unet.')}
        save_file(unet_state, str(output_path), metadata=output_metadata)
        print(f"  Saved U-Net only: {len(unet_state)} keys")
    else:
        # Save full model with metadata
        save_file(fused_state, str(output_path), metadata=output_metadata)
        print(f"  Saved full model: {len(fused_state)} keys")
    print(f"  Metadata preserved: model_type={output_metadata.get('model_type', 'unknown')}")

    print(f"\nOutput saved to: {output_path}")
    print("\n" + "=" * 60)
    print("Fusion complete!")
    print("=" * 60)
    print("\nNotes:")
    if args.cross_attn_method == "none":
        print("- Cross-attention to_k/to_v layers kept from DEUS (incompatible context dim)")
    else:
        print(f"- Cross-attention to_k/to_v fused using '{args.cross_attn_method}' method")
        print("  WARNING: This is experimental - dimension mismatch may cause quality issues")
    print("- This fused model may need fine-tuning to reconcile SDXL features with DEUS text encoder")
    print("- Self-attention and feedforward layers should benefit most from SDXL knowledge")


if __name__ == "__main__":
    main()
