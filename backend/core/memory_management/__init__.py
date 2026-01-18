"""Memory Management for Transformer Models"""

from .block_offloading import TransformerBlockOffloader
from .flux_block_offloading import FluxBlockOffloader, create_flux_block_offloader
from .transformer_registry import create_block_offloader_for_model
from .ring_buffer_allocator import RingBufferAllocator, TensorAllocator, DynamicActivationAllocator
from .tensor_utils import (
    extract_tensors,
    replace_tensor_data,
    move_tensors_to_device,
    async_copy_to_device,
    cuda_stream_context,
)
from .layer_offload_strategy import LayerOffloadStrategy
from .layer_offload_conductor import LayerOffloadConductor
from .fused_block_swap import FusedBlockSwapTrainer

__all__ = [
    # Block offloading (existing, production-ready)
    "TransformerBlockOffloader",
    "create_block_offloader_for_model",
    # FLUX.2 block offloading
    "FluxBlockOffloader",
    "create_flux_block_offloader",
    # Ring buffer allocation (new, for advanced training)
    "RingBufferAllocator",
    "TensorAllocator",
    "DynamicActivationAllocator",
    # Tensor utilities
    "extract_tensors",
    "replace_tensor_data",
    "move_tensors_to_device",
    "async_copy_to_device",
    "cuda_stream_context",
    # Layer offload (alternative implementation)
    "LayerOffloadStrategy",
    "LayerOffloadConductor",
    # Fused block swap (complete VRAM-efficient training)
    "FusedBlockSwapTrainer",
]
