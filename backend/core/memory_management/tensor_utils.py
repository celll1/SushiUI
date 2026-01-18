"""
Tensor Utility Functions for Memory Management

Provides utilities for extracting, replacing, and moving tensors with custom allocators.
"""

import torch
from typing import Any, Optional, Callable, List
from contextlib import contextmanager


def extract_tensors(obj: Any, tensors: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
    """
    Extract all tensors from nested data structure.

    Args:
        obj: Object to extract tensors from (dict, list, tuple, tensor, etc.)
        tensors: Accumulator list (for recursion)

    Returns:
        List of all tensors found
    """
    if tensors is None:
        tensors = []

    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            extract_tensors(value, tensors)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            extract_tensors(item, tensors)

    return tensors


def replace_tensor_data(target: torch.Tensor, source: torch.Tensor):
    """
    Replace tensor data in-place without changing tensor object identity.

    This is critical for maintaining gradient tracking when offloading.

    Args:
        target: Tensor whose data will be replaced
        source: Tensor to copy data from
    """
    if target.shape != source.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, source {source.shape}")

    if target.dtype != source.dtype:
        raise ValueError(f"Dtype mismatch: target {target.dtype}, source {source.dtype}")

    # Replace data in-place
    target.data = source.data


def move_tensors_to_device(
    obj: Any,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    allocator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    non_blocking: bool = False
) -> Any:
    """
    Move all tensors in nested structure to device.

    Args:
        obj: Object containing tensors (dict, list, tuple, tensor, etc.)
        device: Target device
        dtype: Target dtype (None to keep original)
        allocator: Optional custom allocator function
        non_blocking: Use non-blocking transfer (requires pinned memory)

    Returns:
        Object with tensors moved to device
    """
    if isinstance(obj, torch.Tensor):
        if allocator:
            # Allocate with custom allocator
            new_tensor = allocator(obj)
            # Copy data
            new_tensor.copy_(obj, non_blocking=non_blocking)
            return new_tensor
        else:
            # Standard PyTorch move
            if dtype is not None:
                return obj.to(device=device, dtype=dtype, non_blocking=non_blocking)
            else:
                return obj.to(device=device, non_blocking=non_blocking)

    elif isinstance(obj, dict):
        return {
            key: move_tensors_to_device(value, device, dtype, allocator, non_blocking)
            for key, value in obj.items()
        }

    elif isinstance(obj, list):
        return [
            move_tensors_to_device(item, device, dtype, allocator, non_blocking)
            for item in obj
        ]

    elif isinstance(obj, tuple):
        return tuple(
            move_tensors_to_device(item, device, dtype, allocator, non_blocking)
            for item in obj
        )

    else:
        # Non-tensor object (int, str, None, etc.)
        return obj


def is_same_device(tensor: torch.Tensor, device: torch.device) -> bool:
    """
    Check if tensor is on specified device.

    Args:
        tensor: Tensor to check
        device: Device to compare

    Returns:
        True if tensor is on device
    """
    return tensor.device == device


def get_tensor_memory_size(tensor: torch.Tensor) -> int:
    """
    Get memory size of tensor in bytes.

    Args:
        tensor: Tensor to measure

    Returns:
        Memory size in bytes
    """
    return tensor.numel() * tensor.element_size()


@contextmanager
def cuda_stream_context(stream: Optional[torch.cuda.Stream] = None):
    """
    Context manager for CUDA stream.

    Usage:
        with cuda_stream_context(my_stream):
            # Operations run on my_stream
            tensor.copy_(other, non_blocking=True)

    Args:
        stream: CUDA stream (None for default stream)
    """
    if stream is None:
        yield
    else:
        with torch.cuda.stream(stream):
            yield


def create_pinned_copy(tensor: torch.Tensor) -> torch.Tensor:
    """
    Create pinned CPU copy of tensor for faster GPU transfer.

    Args:
        tensor: Source tensor (any device)

    Returns:
        Pinned CPU tensor
    """
    cpu_tensor = tensor.to(device='cpu', non_blocking=False)

    if not cpu_tensor.is_pinned():
        cpu_tensor = cpu_tensor.pin_memory()

    return cpu_tensor


def async_copy_to_device(
    source: torch.Tensor,
    target: torch.Tensor,
    stream: Optional[torch.cuda.Stream] = None
) -> torch.cuda.Event:
    """
    Async copy tensor data from source to target.

    Args:
        source: Source tensor (must be pinned if CPU → GPU)
        target: Target tensor (must be pre-allocated)
        stream: CUDA stream for async copy

    Returns:
        CUDA event that will be signaled when copy completes
    """
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: source {source.shape}, target {target.shape}")

    if source.dtype != target.dtype:
        raise ValueError(f"Dtype mismatch: source {source.dtype}, target {target.dtype}")

    # Create event
    event = torch.cuda.Event()

    # Perform async copy
    with cuda_stream_context(stream):
        target.copy_(source, non_blocking=True)
        event.record(stream)

    return event


def sync_stream(stream: Optional[torch.cuda.Stream] = None):
    """
    Synchronize CUDA stream.

    Args:
        stream: CUDA stream (None for default stream)
    """
    if stream is None:
        torch.cuda.synchronize()
    else:
        stream.synchronize()


def wait_for_event(event: torch.cuda.Event, stream: Optional[torch.cuda.Stream] = None):
    """
    Make stream wait for event.

    Args:
        event: CUDA event to wait for
        stream: CUDA stream that will wait (None for default stream)
    """
    if stream is None:
        event.synchronize()
    else:
        stream.wait_event(event)
