"""
Ring Buffer Memory Allocator for VRAM-Efficient Training

Implements custom memory allocation to avoid PyTorch caching allocator fragmentation.
Uses ring buffer strategy for layer parameters and dynamic allocation for activations.
"""

import torch
import math
from typing import Optional, Callable


def align_bytes(size: int, alignment: int = 16) -> int:
    """Align size to boundary."""
    return (size + alignment - 1) // alignment * alignment


class RingBufferAllocator:
    """
    Ring buffer allocator for layer parameters.

    Allocates large byte buffers and provides views to avoid fragmentation.
    Supports bidirectional allocation (forward: left-to-right, backward: right-to-left).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.pinned = (device.type == "cpu")

        self.buffers: list[Optional[torch.Tensor]] = []
        self.buffer_size = 0
        self.start_offset = 0
        self.end_offset = 0

        self.layer_allocators: list[Optional['TensorAllocator']] = []

    def initialize(self, layers: list[torch.nn.Module], target_bytes: int):
        """Initialize buffer cache."""
        if self.buffers:
            return

        # Calculate layer sizes
        max_param_bytes = 0
        for layer in layers:
            for param in layer.parameters(recurse=False):
                param_bytes = param.numel() * param.element_size()
                max_param_bytes = max(max_param_bytes, param_bytes)

        # Determine buffer count (balance overhead vs flexibility)
        num_buffers = min(
            max(1, int(target_bytes * 0.1) // max(max_param_bytes, 1)),
            max(1, target_bytes // max(max_param_bytes * 2, 1)),
            10
        )

        self.buffer_size = (target_bytes // num_buffers) + max_param_bytes + 4096
        self.buffers = [None] * num_buffers
        self.layer_allocators = [None] * len(layers)
        self.start_offset = 0
        self.end_offset = 0

    def allocate_buffer(self, idx: int):
        """Lazy buffer allocation."""
        if self.buffers[idx] is None:
            buf = torch.zeros((self.buffer_size,), dtype=torch.int8, device=self.device)
            if self.pinned:
                buf = buf.pin_memory()
            self.buffers[idx] = buf

    def get_layer_allocator(self, layer_idx: int, forward: bool) -> 'TensorAllocator':
        """Get allocator for layer."""
        alloc = TensorAllocator(self, layer_idx, forward)
        self.layer_allocators[layer_idx] = alloc
        return alloc

    def free_layer(self, layer_idx: int, forward: bool):
        """Free layer allocation."""
        if self.layer_allocators[layer_idx]:
            self.layer_allocators[layer_idx].free(forward)
            self.layer_allocators[layer_idx] = None

    def cleanup(self):
        """Cleanup all buffers."""
        if self.pinned:
            for buf in self.buffers:
                if buf is not None and buf.is_pinned():
                    buf.data = buf.clone().data
        self.buffers = [None] * len(self.buffers)
        self.layer_allocators = [None] * len(self.layer_allocators)


class TensorAllocator:
    """Allocator for tensors within a layer."""

    def __init__(self, parent: RingBufferAllocator, layer_idx: int, forward: bool):
        self.parent = parent
        self.layer_idx = layer_idx
        self.forward = forward

        if forward:
            self.start = parent.end_offset
            self.end = parent.end_offset
        else:
            self.start = parent.start_offset
            self.end = parent.start_offset

    def allocate(self, template: torch.Tensor) -> torch.Tensor:
        """Allocate tensor matching template."""
        num_bytes = template.numel() * template.element_size()
        buf_size = self.parent.buffer_size
        total_size = buf_size * len(self.parent.buffers)

        if self.forward:
            buf_idx = self.end // buf_size
            offset = align_bytes(self.end % buf_size)

            if offset + num_bytes > buf_size:
                buf_idx += 1
                offset = 0

            if buf_idx * buf_size + offset + num_bytes > total_size:
                buf_idx = 0
                offset = 0

            self.end = buf_idx * buf_size + offset
            self.parent.allocate_buffer(buf_idx)
            allocated = self.parent.buffers[buf_idx][offset:offset + num_bytes]
            self.end += num_bytes
            self.parent.end_offset = self.end
        else:
            buf_idx = self.start // buf_size
            offset = self.start % buf_size

            if offset - num_bytes < 0:
                buf_idx -= 1
                offset = buf_size

            if buf_idx < 0:
                buf_idx = len(self.parent.buffers) - 1
                offset = buf_size

            new_offset = (offset - num_bytes) // 16 * 16
            self.parent.allocate_buffer(buf_idx)
            allocated = self.parent.buffers[buf_idx][new_offset:new_offset + num_bytes]
            self.start = buf_idx * buf_size + new_offset
            self.parent.start_offset = self.start

        return allocated.view(dtype=template.dtype).view(size=template.shape)

    def free(self, forward: bool):
        """Free allocation."""
        if forward:
            self.parent.start_offset = self.end
        else:
            self.parent.end_offset = self.start


class DynamicActivationAllocator:
    """Dynamic allocator for activations."""

    def __init__(self, device: torch.device):
        self.device = device
        self.pinned = (device.type == "cpu")
        self.buffers: list[torch.Tensor] = []
        self.current_idx = 0
        self.current_offset = 0
        self.allocated = 0
        self.peak = 0

    def reserve(self, tensors: list[torch.Tensor]):
        """Reserve space for tensors."""
        total = sum(t.numel() * t.element_size() + 16 for t in tensors)
        if total == 0:
            return

        if not self.buffers:
            total = max(total, self.peak)

        found = False
        while self.current_idx < len(self.buffers):
            remaining = self.buffers[self.current_idx].shape[0] - self.current_offset
            if remaining >= total:
                found = True
                break
            self.current_idx += 1
            self.current_offset = 0

        if not found:
            buf = torch.zeros((total,), dtype=torch.int8, device=self.device)
            if self.pinned:
                buf = buf.pin_memory()
            self.buffers.append(buf)
            self.allocated += total

        self.peak = max(self.peak, self.allocated)

    def allocate(self, template: torch.Tensor) -> torch.Tensor:
        """Allocate tensor matching template."""
        num_bytes = template.numel() * template.element_size()
        buf = self.buffers[self.current_idx]
        allocated = buf[self.current_offset:self.current_offset + num_bytes]
        self.current_offset += align_bytes(num_bytes)
        return allocated.view(dtype=template.dtype).view(size=template.shape)

    def reset(self):
        """Reset for next iteration."""
        if len(self.buffers) > 1:
            if self.pinned:
                for buf in self.buffers:
                    if buf.is_pinned():
                        buf.data = buf.clone().data
            self.buffers.clear()
            total = self.allocated + 4096
            buf = torch.zeros((total,), dtype=torch.int8, device=self.device)
            if self.pinned:
                buf = buf.pin_memory()
            self.buffers = [buf]

        self.current_idx = 0
        self.current_offset = 0
        self.allocated = sum(b.shape[0] for b in self.buffers)

    def cleanup(self):
        """Cleanup all buffers."""
        if self.pinned:
            for buf in self.buffers:
                if buf.is_pinned():
                    buf.data = buf.clone().data
        self.buffers.clear()
