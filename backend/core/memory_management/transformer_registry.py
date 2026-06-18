"""
Transformer Registry for Block Offloading

Auto-detect transformer architecture and create appropriate block offloader.
"""

import torch
import torch.nn as nn
from typing import Optional, Union

from .block_offloading import TransformerBlockOffloader


def detect_transformer_architecture(transformer: nn.Module) -> str:
    """
    Detect transformer architecture by inspecting module structure

    Args:
        transformer: Transformer model

    Returns:
        Architecture name: "zimage", "flux2", "flux", "sd3", "unknown"
    """
    # FLUX.2: has both transformer_blocks and single_transformer_blocks (diffusers style)
    if hasattr(transformer, 'transformer_blocks') and hasattr(transformer, 'single_transformer_blocks'):
        # Check first block class name
        if len(transformer.transformer_blocks) > 0:
            block_class = transformer.transformer_blocks[0].__class__.__name__
            if "Flux" in block_class:
                return "flux2"

    # Z-Image: has layers attribute with ZImageTransformerBlock
    if hasattr(transformer, 'layers'):
        first_layer = transformer.layers[0] if len(transformer.layers) > 0 else None
        if first_layer is not None:
            layer_class_name = first_layer.__class__.__name__
            if "ZImage" in layer_class_name:
                return "zimage"
            elif "Ideogram4" in layer_class_name:
                return "ideogram4"
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
    use_pinned_memory: bool = False,
    supports_backward: bool = False
) -> Union[TransformerBlockOffloader, "FluxBlockOffloader"]:
    """
    Create block offloader for transformer model (auto-detect architecture)

    Args:
        transformer: Transformer model
        blocks_to_swap: Number of blocks to swap
        device: Target device
        target_dtype: Target dtype for computation
        use_pinned_memory: Use pinned memory for faster transfer
        supports_backward: Enable backward pass support (for training)

    Returns:
        TransformerBlockOffloader or FluxBlockOffloader instance
    """
    architecture = detect_transformer_architecture(transformer)
    print(f"[TransformerRegistry] Detected architecture: {architecture}")

    # FLUX.2: Use specialized FluxBlockOffloader
    if architecture == "flux2":
        from .flux_block_offloading import create_flux_block_offloader
        return create_flux_block_offloader(
            transformer=transformer,
            blocks_to_swap=blocks_to_swap,
            device=device,
            target_dtype=target_dtype,
            use_pinned_memory=use_pinned_memory,
            supports_backward=supports_backward
        )

    # Z-Image and other single-list architectures
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
        transformer=transformer,
        supports_backward=supports_backward
    )

    return offloader
