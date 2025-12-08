"""
Transformer Registry for Block Offloading

Auto-detect transformer architecture and create appropriate block offloader.
"""

import torch
import torch.nn as nn
from typing import Optional

from .block_offloading import TransformerBlockOffloader


def detect_transformer_architecture(transformer: nn.Module) -> str:
    """
    Detect transformer architecture by inspecting module structure

    Args:
        transformer: Transformer model

    Returns:
        Architecture name: "zimage", "flux", "sd3", "unknown"
    """
    # Z-Image: has layers attribute with ZImageTransformerBlock
    if hasattr(transformer, 'layers'):
        first_layer = transformer.layers[0] if len(transformer.layers) > 0 else None
        if first_layer is not None:
            layer_class_name = first_layer.__class__.__name__
            if "ZImage" in layer_class_name:
                return "zimage"
            elif "Flux" in layer_class_name:
                return "flux"
            elif "SD3" in layer_class_name:
                return "sd3"

    return "unknown"


def create_block_offloader_for_model(
    transformer: nn.Module,
    blocks_to_swap: int,
    device: torch.device,
    target_dtype: Optional[torch.dtype] = None,
    use_pinned_memory: bool = False
) -> TransformerBlockOffloader:
    """
    Create block offloader for transformer model (auto-detect architecture)

    Args:
        transformer: Transformer model
        blocks_to_swap: Number of blocks to swap
        device: Target device
        target_dtype: Target dtype for computation
        use_pinned_memory: Use pinned memory for faster transfer

    Returns:
        TransformerBlockOffloader instance
    """
    architecture = detect_transformer_architecture(transformer)
    print(f"[TransformerRegistry] Detected architecture: {architecture}")

    # Get blocks
    if hasattr(transformer, 'layers'):
        blocks = transformer.layers
    else:
        raise ValueError(f"Transformer does not have 'layers' attribute")

    # Default dtype
    if target_dtype is None:
        first_param = next(transformer.parameters())
        target_dtype = first_param.dtype
        print(f"[TransformerRegistry] Auto-detected dtype: {target_dtype}")

    # Create offloader
    offloader = TransformerBlockOffloader(
        blocks=blocks,
        blocks_to_swap=blocks_to_swap,
        device=device,
        target_dtype=target_dtype,
        use_pinned_memory=use_pinned_memory,
        transformer=transformer
    )

    return offloader
