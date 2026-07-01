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
    supports_backward: bool = False,
    h2d_only: bool = False,
    ring_size: int = 2,
    block_list: Optional[nn.ModuleList] = None,
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
        block_list: Explicit swappable block ModuleList. Use for architectures whose
            heavy block list is not named ``layers`` (e.g. Anima ``blocks``, Lens
            ``transformer_blocks``, MiniT2I ``double_blocks``). When given, it is used
            directly for the single-list offloader (and for the clamp), bypassing the
            ``layers`` auto-detection. Ignored for the FLUX.2 dual-list path.

    Returns:
        TransformerBlockOffloader or FluxBlockOffloader instance
    """
    architecture = detect_transformer_architecture(transformer)
    print(f"[TransformerRegistry] Detected architecture: {architecture}")

    # Clamp blocks_to_swap to a valid range [0, num_blocks - 1] so an over-large value
    # (e.g. from a stale config) cannot index past the block list. At least one block must
    # stay resident, so the maximum is num_blocks - 1. Covers both the FLUX.2 dual/single
    # unified sequence and single-list (Z-Image etc.) architectures.
    if block_list is not None:
        num_blocks = len(block_list)
    elif hasattr(transformer, 'transformer_blocks') and hasattr(transformer, 'single_transformer_blocks'):
        num_blocks = len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks)
    elif hasattr(transformer, 'layers'):
        num_blocks = len(transformer.layers)
    else:
        num_blocks = None
    if num_blocks is not None:
        clamped = max(0, min(int(blocks_to_swap), num_blocks - 1))
        if clamped != blocks_to_swap:
            print(f"[TransformerRegistry] blocks_to_swap={blocks_to_swap} out of range; "
                  f"clamped to {clamped} (num_blocks={num_blocks})")
        blocks_to_swap = clamped

    # torchao / tensor-subclass weights (e.g. AffineQuantizedTensor from uint quantization)
    # are not handled by the Linear weight-data swap path: block swap streams plain Linear
    # weights only, so subclass weights are either left GPU-resident (no VRAM saving) or the
    # pinned-buffer copy is unreliable. Warn clearly rather than silently under-offloading.
    def _has_subclass_linear_weight(mod):
        for m in mod.modules():
            if m.__class__.__name__.endswith("Linear") and getattr(m, "weight", None) is not None:
                if type(m.weight.data) is not torch.Tensor:
                    return True
        return False
    if _has_subclass_linear_weight(transformer):
        print("[TransformerRegistry] WARNING: transformer has tensor-subclass (e.g. torchao "
              "uint-quantized) Linear weights. Block swap streams plain Linear weights only "
              "and will NOT offload these (reduced VRAM saving; transfers may be unreliable). "
              "Use FP8 (fp8_e4m3fn / fp8_e5m2) instead of uint/torchao quantization with block swap.")

    # FLUX.2: Use specialized FluxBlockOffloader
    if architecture == "flux2":
        from .flux_block_offloading import create_flux_block_offloader
        return create_flux_block_offloader(
            transformer=transformer,
            blocks_to_swap=blocks_to_swap,
            device=device,
            target_dtype=target_dtype,
            use_pinned_memory=use_pinned_memory,
            supports_backward=supports_backward,
            h2d_only=h2d_only,
            ring_size=ring_size,
        )

    # Z-Image and other single-list architectures. An explicit block_list wins (for models
    # whose heavy block list is not named 'layers'); otherwise fall back to 'layers'.
    if block_list is not None:
        blocks = block_list
    elif hasattr(transformer, 'layers'):
        blocks = transformer.layers
    else:
        raise ValueError(f"Transformer does not have 'layers' attribute and no block_list was provided")

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
        supports_backward=supports_backward,
        h2d_only=h2d_only,
        ring_size=ring_size,
    )

    return offloader
