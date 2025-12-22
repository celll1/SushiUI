"""
Layer Offload Conductor

Orchestrates layer loading/offloading with async transfers and custom allocators.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import copy

from .ring_buffer_allocator import RingBufferAllocator, DynamicActivationAllocator
from .layer_offload_strategy import LayerOffloadStrategy
from .tensor_utils import (
    extract_tensors,
    replace_tensor_data,
    create_pinned_copy,
    async_copy_to_device,
    wait_for_event,
    get_tensor_memory_size
)


class LayerOffloadConductor:
    """
    Orchestrates layer offloading for VRAM-efficient training.

    Features:
    - Async CPU ↔ GPU transfer with dedicated streams
    - Custom memory allocators to avoid fragmentation
    - Activation offloading for gradient checkpointing
    - Hook-based integration with transformer layers
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = True,
        cpu_buffer_size_mb: int = 8192,
        activation_buffer_size_mb: int = 4096,
        enable_prefetch: bool = True,
        enable_activation_offload: bool = True
    ):
        """
        Initialize conductor.

        Args:
            layers: Transformer layers (ModuleList)
            blocks_to_swap: Number of layers to swap to CPU
            device: GPU device
            use_pinned_memory: Use pinned CPU memory for faster transfer
            cpu_buffer_size_mb: CPU buffer size in MB (for layer params)
            activation_buffer_size_mb: CPU buffer size in MB (for activations)
            enable_prefetch: Enable prefetching next layer
            enable_activation_offload: Enable activation offloading
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.use_pinned_memory = use_pinned_memory
        self.enable_prefetch = enable_prefetch
        self.enable_activation_offload = enable_activation_offload

        # Strategy
        self.strategy = LayerOffloadStrategy(
            num_layers=self.num_layers,
            blocks_to_swap=blocks_to_swap,
            device=device
        )

        # Memory allocators
        cpu_device = torch.device('cpu')

        self.param_allocator = RingBufferAllocator(device=cpu_device)
        self.param_allocator.initialize(
            layers=list(layers),
            target_bytes=cpu_buffer_size_mb * 1024 * 1024
        )

        if enable_activation_offload:
            self.activation_allocator = DynamicActivationAllocator(device=cpu_device)
        else:
            self.activation_allocator = None

        # CUDA streams for async transfer
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.compute_stream = torch.cuda.current_stream(device=device)

        # State tracking
        self.layer_states: Dict[int, str] = {}  # layer_idx -> 'cpu' | 'gpu' | 'loading' | 'offloading'
        self.layer_gpu_copies: Dict[int, Dict[str, torch.Tensor]] = {}  # layer_idx -> {param_name: tensor}
        self.layer_cpu_copies: Dict[int, Dict[str, torch.Tensor]] = {}  # layer_idx -> {param_name: tensor}
        self.pending_events: Dict[int, torch.cuda.Event] = {}  # layer_idx -> event
        self.saved_activations: Dict[int, Any] = {}  # layer_idx -> activations

        # Hook handles
        self.hook_handles: List[Any] = []

        # Initialize layers
        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize layer states and create CPU copies."""
        print("[LayerOffloadConductor] Initializing layers...")

        for layer_idx, layer in enumerate(self.layers):
            if self.strategy.is_resident(layer_idx):
                # Resident layers stay on GPU
                layer.to(self.device)
                self.layer_states[layer_idx] = 'gpu'
                print(f"  Layer {layer_idx}: Resident (GPU)")

            else:
                # Offloadable layers: create CPU copy and move to CPU
                print(f"  Layer {layer_idx}: Offloadable (CPU)")

                # Create CPU copy with custom allocator
                cpu_params = {}
                allocator = self.param_allocator.get_layer_allocator(layer_idx, forward=True)

                for name, param in layer.named_parameters():
                    # Allocate CPU buffer
                    cpu_param = allocator.allocate(param)

                    # Copy data to CPU
                    if self.use_pinned_memory:
                        pinned = create_pinned_copy(param)
                        cpu_param.copy_(pinned, non_blocking=False)
                    else:
                        cpu_param.copy_(param.cpu(), non_blocking=False)

                    cpu_params[name] = cpu_param

                self.layer_cpu_copies[layer_idx] = cpu_params

                # Move layer to CPU (PyTorch default allocator)
                layer.to('cpu')

                # CRITICAL: Replace layer parameters with ring buffer allocations
                # This ensures layer uses custom allocator memory, not PyTorch default
                for name, param in layer.named_parameters():
                    if name in cpu_params:
                        param.data = cpu_params[name]

                self.layer_states[layer_idx] = 'cpu'

        self.strategy.print_strategy()
        print(f"[LayerOffloadConductor] Initialization complete")

    def load_layer_to_gpu(self, layer_idx: int, async_transfer: bool = True):
        """
        Load layer from CPU to GPU.

        Args:
            layer_idx: Layer index
            async_transfer: Use async transfer
        """
        if self.layer_states[layer_idx] == 'gpu':
            return  # Already on GPU

        if self.layer_states[layer_idx] == 'loading':
            # Wait for pending load
            if layer_idx in self.pending_events:
                wait_for_event(self.pending_events[layer_idx], self.compute_stream)
                del self.pending_events[layer_idx]
            self.layer_states[layer_idx] = 'gpu'
            return

        # Mark as loading
        self.layer_states[layer_idx] = 'loading'

        layer = self.layers[layer_idx]
        cpu_params = self.layer_cpu_copies[layer_idx]

        # Move layer to GPU
        layer.to(self.device)

        # Copy parameters from CPU buffer to GPU
        if async_transfer:
            with torch.cuda.stream(self.transfer_stream):
                for name, param in layer.named_parameters():
                    cpu_param = cpu_params[name]
                    param.data.copy_(cpu_param, non_blocking=True)

                # Record event
                event = torch.cuda.Event()
                event.record(self.transfer_stream)
                self.pending_events[layer_idx] = event
        else:
            for name, param in layer.named_parameters():
                cpu_param = cpu_params[name]
                param.data.copy_(cpu_param, non_blocking=False)

            self.layer_states[layer_idx] = 'gpu'

    def offload_layer_to_cpu(self, layer_idx: int, async_transfer: bool = True):
        """
        Offload layer from GPU to CPU.

        Args:
            layer_idx: Layer index
            async_transfer: Use async transfer
        """
        if self.layer_states[layer_idx] == 'cpu':
            return  # Already on CPU

        if self.layer_states[layer_idx] == 'offloading':
            # Wait for pending offload
            if layer_idx in self.pending_events:
                wait_for_event(self.pending_events[layer_idx], self.compute_stream)
                del self.pending_events[layer_idx]
            self.layer_states[layer_idx] = 'cpu'
            return

        # Mark as offloading
        self.layer_states[layer_idx] = 'offloading'

        layer = self.layers[layer_idx]
        cpu_params = self.layer_cpu_copies[layer_idx]

        # Copy parameters from GPU to CPU buffer
        if async_transfer:
            with torch.cuda.stream(self.transfer_stream):
                for name, param in layer.named_parameters():
                    cpu_param = cpu_params[name]
                    cpu_param.copy_(param.data, non_blocking=True)

                # Record event
                event = torch.cuda.Event()
                event.record(self.transfer_stream)
                self.pending_events[layer_idx] = event
        else:
            for name, param in layer.named_parameters():
                cpu_param = cpu_params[name]
                cpu_param.copy_(param.data, non_blocking=False)

        # Move layer to CPU after transfer
        if async_transfer:
            # Wait for transfer to complete before moving
            wait_for_event(self.pending_events[layer_idx], self.transfer_stream)
            del self.pending_events[layer_idx]

        layer.to('cpu')
        self.layer_states[layer_idx] = 'cpu'

    def sync_layer(self, layer_idx: int):
        """
        Wait for layer transfer to complete.

        Args:
            layer_idx: Layer index
        """
        if layer_idx in self.pending_events:
            wait_for_event(self.pending_events[layer_idx], self.compute_stream)
            del self.pending_events[layer_idx]

            # Update state
            if self.layer_states[layer_idx] == 'loading':
                self.layer_states[layer_idx] = 'gpu'
            elif self.layer_states[layer_idx] == 'offloading':
                self.layer_states[layer_idx] = 'cpu'

    def forward_layer(self, layer_idx: int, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Execute forward pass for layer with offloading.

        Args:
            layer_idx: Layer index
            hidden_states: Input tensor
            *args, **kwargs: Additional layer arguments

        Returns:
            Output tensor
        """
        # Load layer to GPU
        self.load_layer_to_gpu(layer_idx, async_transfer=True)

        # Wait for load to complete
        self.sync_layer(layer_idx)

        # Prefetch next layer
        if self.enable_prefetch:
            next_layer_idx = self.strategy.should_prefetch(layer_idx, 'forward')
            if next_layer_idx is not None:
                self.load_layer_to_gpu(next_layer_idx, async_transfer=True)

        # Execute layer
        layer = self.layers[layer_idx]
        output = layer(hidden_states, *args, **kwargs)

        # Save activations for backward (if enabled)
        if self.enable_activation_offload and self.activation_allocator is not None:
            self.saved_activations[layer_idx] = self._offload_activation(hidden_states)

        # Offload layer after forward (if not needed for backward immediately)
        if self.strategy.is_offloadable(layer_idx):
            # Don't offload immediately - wait for backward
            pass

        return output

    def _offload_activation(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Offload activation to CPU.

        Args:
            activation: Activation tensor

        Returns:
            CPU copy
        """
        if self.use_pinned_memory:
            cpu_activation = create_pinned_copy(activation)
        else:
            cpu_activation = activation.cpu()

        return cpu_activation

    def _restore_activation(self, cpu_activation: torch.Tensor) -> torch.Tensor:
        """
        Restore activation from CPU.

        Args:
            cpu_activation: CPU activation

        Returns:
            GPU tensor
        """
        return cpu_activation.to(self.device, non_blocking=True)

    def register_hooks(self):
        """
        Register forward/backward hooks for automatic offloading.

        This enables integration with PyTorch autograd.
        """
        print("[LayerOffloadConductor] Registering hooks...")

        for layer_idx, layer in enumerate(self.layers):
            if self.strategy.is_offloadable(layer_idx):
                # Forward pre-hook: Load layer
                def forward_pre_hook(module, inputs, idx=layer_idx):
                    self.load_layer_to_gpu(idx, async_transfer=True)
                    self.sync_layer(idx)

                handle = layer.register_forward_pre_hook(forward_pre_hook)
                self.hook_handles.append(handle)

                # Backward hook: Offload after gradient calculation
                def backward_hook(module, grad_input, grad_output, idx=layer_idx):
                    # Offload layer after backward
                    self.offload_layer_to_cpu(idx, async_transfer=True)

                handle = layer.register_full_backward_hook(backward_hook)
                self.hook_handles.append(handle)

        print(f"[LayerOffloadConductor] Registered {len(self.hook_handles)} hooks")

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hook_handles:
            handle.remove()

        self.hook_handles = []
        print("[LayerOffloadConductor] Removed hooks")

    def cleanup(self):
        """Cleanup allocators and restore layers to GPU."""
        print("[LayerOffloadConductor] Cleaning up...")

        # Remove hooks
        self.remove_hooks()

        # Move all layers to GPU
        for layer_idx, layer in enumerate(self.layers):
            if self.layer_states[layer_idx] != 'gpu':
                self.load_layer_to_gpu(layer_idx, async_transfer=False)

        # Cleanup allocators
        self.param_allocator.cleanup()

        if self.activation_allocator is not None:
            self.activation_allocator.cleanup()

        print("[LayerOffloadConductor] Cleanup complete")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory stats
        """
        stats = {
            "num_layers": self.num_layers,
            "blocks_to_swap": self.blocks_to_swap,
            "layer_states": dict(self.layer_states),
        }

        # Count layers by state
        state_counts = {'cpu': 0, 'gpu': 0, 'loading': 0, 'offloading': 0}
        for state in self.layer_states.values():
            state_counts[state] += 1

        stats["state_counts"] = state_counts

        # GPU memory
        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved(self.device) / (1024 ** 2)

        return stats

    def print_memory_stats(self):
        """Print memory usage statistics."""
        stats = self.get_memory_stats()

        print("=" * 60)
        print("[LayerOffloadConductor] Memory Statistics")
        print("=" * 60)
        print(f"  Layer States: CPU={stats['state_counts']['cpu']}, "
              f"GPU={stats['state_counts']['gpu']}, "
              f"Loading={stats['state_counts']['loading']}, "
              f"Offloading={stats['state_counts']['offloading']}")

        if "gpu_allocated_mb" in stats:
            print(f"  GPU Allocated: {stats['gpu_allocated_mb']:.2f} MB")
            print(f"  GPU Reserved:  {stats['gpu_reserved_mb']:.2f} MB")

        print("=" * 60)
