"""
FLUX.2 Transformer Block Offloading for Low VRAM Environments

FLUX.2 has a unique dual-list architecture:
- transformer_blocks (dual stream): Joint attention on image + text
- single_transformer_blocks (single stream): Self-attention on concatenated sequence

This module extends the base TransformerBlockOffloader to handle FLUX.2's
architecture with proper block swapping across both block lists.

Block counts:
- Klein 4B: 5 dual + 20 single = 25 total blocks
- Klein 9B: 8 dual + 24 single = 32 total blocks
- Full: 19 dual + 38 single = 57 total blocks (FLUX.1 style)
"""

import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

from .block_offloading import weighs_to_device, _synchronize_device


class FluxBlockOffloader:
    """
    Block offloader for FLUX.2 Transformer (dual block list architecture)

    Strategy:
    - Treat transformer_blocks and single_transformer_blocks as a unified block sequence
    - First N blocks stay on GPU (partial offload)
    - Remaining blocks swap between CPU and GPU during forward pass

    Unified index mapping:
    - Index 0 to (num_dual - 1): transformer_blocks
    - Index num_dual to (num_dual + num_single - 1): single_transformer_blocks
    """

    def __init__(
        self,
        transformer_blocks: nn.ModuleList,
        single_transformer_blocks: nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device,
        target_dtype: torch.dtype = torch.bfloat16,
        use_pinned_memory: bool = False,
        transformer: Optional[nn.Module] = None,
        supports_backward: bool = False
    ):
        """
        Initialize FLUX.2 Block Offloader

        Args:
            transformer_blocks: Dual stream blocks (FluxTransformerBlock)
            single_transformer_blocks: Single stream blocks (FluxSingleTransformerBlock)
            blocks_to_swap: Number of blocks to keep on CPU
            device: Target device (cuda:0)
            target_dtype: Target dtype for computation
            use_pinned_memory: Use pinned memory for faster transfer
            transformer: Parent transformer (for auxiliary modules)
            supports_backward: Enable backward pass support (for training)
        """
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.num_dual_blocks = len(transformer_blocks)
        self.num_single_blocks = len(single_transformer_blocks)
        self.num_blocks = self.num_dual_blocks + self.num_single_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.target_dtype = target_dtype
        self.use_pinned_memory = use_pinned_memory
        self.transformer = transformer
        self.supports_backward = supports_backward
        self.forward_only = not supports_backward

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        # Staging buffers for weight swapping
        self.staging_buffer_a = None
        self.staging_buffer_b = None
        self.pinned_buffer = None

        # Backward hook handles (for training)
        self.backward_hook_handles = []

        mode_str = "training (backward enabled)" if supports_backward else "inference (forward-only)"
        print(f"[FluxBlockOffloader] Initialized: {self.num_blocks} total blocks "
              f"({self.num_dual_blocks} dual + {self.num_single_blocks} single), "
              f"{self.blocks_to_swap} to swap ({mode_str})")
        print(f"[FluxBlockOffloader] Device: {self.device}, dtype: {self.target_dtype}, "
              f"pinned_memory: {self.use_pinned_memory}")

    def _get_block(self, unified_idx: int) -> nn.Module:
        """Get block by unified index"""
        if unified_idx < self.num_dual_blocks:
            return self.transformer_blocks[unified_idx]
        else:
            return self.single_transformer_blocks[unified_idx - self.num_dual_blocks]

    def _set_block(self, unified_idx: int, block: nn.Module):
        """Set block by unified index"""
        if unified_idx < self.num_dual_blocks:
            self.transformer_blocks[unified_idx] = block
        else:
            self.single_transformer_blocks[unified_idx - self.num_dual_blocks] = block

    def prepare_block_devices_before_forward(self):
        """
        Prepare block device placement before forward pass

        - First (num_blocks - blocks_to_swap) blocks: full model on GPU
        - Last blocks_to_swap blocks: weights on CPU, buffers on GPU
        """
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        print(f"[FluxBlockOffloader] Preparing block devices...")

        num_blocks_on_gpu = self.num_blocks - self.blocks_to_swap

        # Move first N blocks to GPU (full)
        print(f"[FluxBlockOffloader] Moving first {num_blocks_on_gpu} blocks to GPU (full)...")
        for i in range(num_blocks_on_gpu):
            block = self._get_block(i)
            block = block.to(self.device)
            weighs_to_device(block, self.device)
            self._set_block(i, block)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            print(f"[FluxBlockOffloader] GPU synchronization complete")

        # Move last M blocks: buffers to GPU, weights to CPU
        print(f"[FluxBlockOffloader] Moving last {self.blocks_to_swap} blocks: buffers to GPU, weights to CPU...")
        cpu_device = torch.device("cpu")
        for i in range(num_blocks_on_gpu, self.num_blocks):
            block = self._get_block(i)
            # First move entire block to GPU (ensures buffers are on GPU)
            block = block.to(self.device)
            # Then move weights back to CPU
            weighs_to_device(block, cpu_device)
            self._set_block(i, block)

        _synchronize_device(self.device)

        # Move auxiliary modules to GPU
        self._move_auxiliary_modules_to_gpu()

        print(f"[FluxBlockOffloader] Block device preparation complete")

        # Log device status
        self.log_device_status("Ready for forward pass")

    def _move_auxiliary_modules_to_gpu(self):
        """
        Move FLUX.2 auxiliary modules to GPU

        FLUX.2 has these auxiliary modules:
        - pos_embed (FluxPosEmbed)
        - time_text_embed (CombinedTimestepTextProjEmbeddings)
        - context_embedder (Linear)
        - x_embedder (Linear)
        - norm_out (AdaLayerNormContinuous)
        - proj_out (Linear)
        """
        if self.transformer is None:
            return

        print(f"[FluxBlockOffloader] Moving auxiliary modules to GPU...")

        auxiliary_module_names = [
            "pos_embed",
            "time_text_embed",
            "context_embedder",
            "x_embedder",
            "norm_out",
            "proj_out",
        ]

        parent = self.transformer
        for module_name in auxiliary_module_names:
            if hasattr(parent, module_name):
                module = getattr(parent, module_name)
                if module is not None and isinstance(module, nn.Module):
                    module._apply(lambda t: t.to(self.device) if isinstance(t, torch.Tensor) else t)
                    print(f"[FluxBlockOffloader]   - Moved {module_name} to {self.device}")

        # Move transformer-level buffers/parameters
        for name, param in parent.named_parameters(recurse=False):
            if param.device != self.device:
                param.data = param.data.to(self.device)
                print(f"[FluxBlockOffloader]   - Moved parameter {name} to {self.device}")

        for name, buffer in parent.named_buffers(recurse=False):
            if buffer.device != self.device:
                buffer.data = buffer.data.to(self.device)
                print(f"[FluxBlockOffloader]   - Moved buffer {name} to {self.device}")

        print(f"[FluxBlockOffloader] Auxiliary modules moved to GPU")

    def wait_for_block(self, unified_idx: int):
        """
        Wait for block transfer to complete

        Args:
            unified_idx: Unified block index (0 to num_blocks-1)
        """
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        num_blocks_on_gpu = self.num_blocks - self.blocks_to_swap

        # First N blocks stay on GPU permanently, no wait needed
        if unified_idx < num_blocks_on_gpu:
            return

        # If block has a pending transfer, wait for it
        if unified_idx in self.futures:
            future = self.futures.pop(unified_idx)
            _, bidx_to_cuda, sync_event = future.result()

            assert unified_idx == bidx_to_cuda, f"Block index mismatch: {unified_idx} != {bidx_to_cuda}"

            if self.cuda_available and sync_event is not None:
                torch.cuda.current_stream().wait_event(sync_event)
        else:
            # No pending transfer - check if block weights are on CPU
            block = self._get_block(unified_idx)
            first_param = next(block.parameters(), None)
            if first_param is not None and first_param.device.type == "cpu":
                print(f"[FluxBlockOffloader DEBUG] Block {unified_idx} weights on CPU, moving to GPU synchronously...")
                weighs_to_device(block, self.device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                print(f"[FluxBlockOffloader DEBUG] Block {unified_idx} weights moved to GPU")

    def submit_move_blocks_forward(self, unified_idx: int):
        """
        Submit block swap for forward pass

        Args:
            unified_idx: Current unified block index (just executed)
        """
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        num_blocks_on_gpu = self.num_blocks - self.blocks_to_swap

        if not self.forward_only:
            # Backward-enabled mode: only swap first blocks_to_swap blocks
            if unified_idx >= self.blocks_to_swap:
                return
            block_idx_to_cpu = unified_idx
            block_idx_to_gpu = unified_idx + 1
        else:
            # Forward-only mode: rotate among swappable blocks
            if unified_idx < num_blocks_on_gpu:
                return

            block_idx_to_cpu = unified_idx
            next_block = unified_idx + 1
            if next_block >= self.num_blocks:
                next_block = num_blocks_on_gpu
            block_idx_to_gpu = next_block

        self._submit_block_swap(block_idx_to_cpu, block_idx_to_gpu)

    def _submit_block_swap(self, block_idx_to_cpu: int, block_idx_to_gpu: int):
        """
        Submit asynchronous block swap

        Args:
            block_idx_to_cpu: Unified index of block to move to CPU
            block_idx_to_gpu: Unified index of block to move to GPU
        """
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_gpu, block_to_gpu):
            dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(dev)

            sync_event = self.swap_weight_devices(block_to_cpu, block_to_gpu)
            return bidx_to_cpu, bidx_to_gpu, sync_event

        block_to_cpu = self._get_block(block_idx_to_cpu)
        block_to_gpu = self._get_block(block_idx_to_gpu)

        self.futures[block_idx_to_gpu] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_gpu, block_to_gpu
        )

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        """
        Swap weights between two blocks

        Note: FLUX.2 has FluxTransformerBlock (dual) and FluxSingleTransformerBlock (single)
        which have different structures. We only swap weights from Linear modules.
        """
        weight_swap_jobs = []

        # Find Linear modules to swap
        modules_to_cpu = {k: v for k, v in block_to_cpu.named_modules()}
        for module_to_cuda_name, module_to_cuda in block_to_cuda.named_modules():
            if (
                hasattr(module_to_cuda, "weight")
                and module_to_cuda.weight is not None
                and module_to_cuda.__class__.__name__.endswith("Linear")
            ):
                module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
                if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                    weight_swap_jobs.append(
                        (module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data)
                    )
                else:
                    if module_to_cuda.weight.data.device.type != self.device.type:
                        module_to_cuda.weight.data = module_to_cuda.weight.data.to(self.device)

        # Synchronize before swap
        torch.cuda.current_stream().synchronize()

        if not self.use_pinned_memory:
            # Strategy 1: Use staging buffers (less pinned memory)
            stream = self.stream
            with torch.cuda.stream(stream):
                if self.staging_buffer_a is None:
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=self.device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=self.device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]

                event_b = None
                for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                    self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
                ):
                    # CUDA to staging buffer A
                    event_a = torch.cuda.Event()
                    sbuf_a.copy_(cuda_data_view.data, non_blocking=True)
                    event_a.record(stream)

                    # Wait for staging buffer B
                    if event_b is not None:
                        event_b.synchronize()

                    # CPU to staging buffer B
                    sbuf_b.copy_(module_to_cuda.weight.data)

                    # Wait for staging buffer A
                    event_a.synchronize()

                    # Staging buffer B to CUDA
                    event_b = torch.cuda.Event()
                    cuda_data_view.copy_(sbuf_b, non_blocking=True)
                    event_b.record(stream)

                    # Staging buffer A to CPU
                    cpu_data_view.copy_(sbuf_a)

            # Update references
            for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = cpu_data_view

            sync_event = event_b

        else:
            # Strategy 2: Use full pinned memory (faster but more memory)
            if self.pinned_buffer is None:
                with torch.cuda.stream(self.stream):
                    self.pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=self.device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                self.stream.synchronize()
            released_pinned_buffer = []

            events = [torch.cuda.Event() for _ in weight_swap_jobs]

            # Copy weights to CPU
            for event, module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                events, self.pinned_buffer, weight_swap_jobs
            ):
                with torch.cuda.stream(self.stream):
                    module_pin_buf.copy_(cuda_data_view, non_blocking=True)
                    event.record(self.stream)

            # CPU to CUDA
            for event, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(events, weight_swap_jobs):
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(event)
                    cuda_data_view.copy_(cpu_data_view, non_blocking=True)

            # Update references
            for module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.pinned_buffer, weight_swap_jobs
            ):
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = module_pin_buf
                released_pinned_buffer.append(cpu_data_view)

            # Reuse released pinned buffers
            if not released_pinned_buffer[0].is_pinned():
                with torch.cuda.stream(self.stream):
                    released_pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=self.device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
            self.pinned_buffer = released_pinned_buffer

            sync_event = self.stream.record_event()

        return sync_event

    def log_device_status(self, status_message: str = "Device Status"):
        """Log current device status of blocks"""
        print(f"============================================================")
        print(f"[FluxBlockOffloader] {status_message}")
        print(f"============================================================")

        num_blocks_on_gpu = self.num_blocks - self.blocks_to_swap

        # Log first dual block (GPU)
        if self.num_dual_blocks > 0 and num_blocks_on_gpu > 0:
            block = self.transformer_blocks[0]
            params = list(block.parameters())
            if params:
                first_param_device = params[0].device
                print(f"  Dual Block 0 (GPU): device={first_param_device}")

        # Log first single block (if any on GPU)
        if self.num_single_blocks > 0 and num_blocks_on_gpu > self.num_dual_blocks:
            block = self.single_transformer_blocks[0]
            params = list(block.parameters())
            if params:
                first_param_device = params[0].device
                print(f"  Single Block 0 (GPU): device={first_param_device}")

        # Log first CPU block
        if self.blocks_to_swap > 0:
            cpu_block_idx = num_blocks_on_gpu
            block = self._get_block(cpu_block_idx)
            params = list(block.parameters())
            if params:
                first_param_device = params[0].device
                block_type = "Dual" if cpu_block_idx < self.num_dual_blocks else "Single"
                local_idx = cpu_block_idx if cpu_block_idx < self.num_dual_blocks else cpu_block_idx - self.num_dual_blocks
                print(f"  {block_type} Block {local_idx} (CPU weights): device={first_param_device}")

        # Log VRAM usage
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print(f"  VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        print(f"============================================================")

    def register_backward_hooks(self):
        """
        Register backward hooks for training-time block swapping
        """
        if not self.supports_backward:
            print(f"[FluxBlockOffloader] Backward hooks not registered (forward-only mode)")
            return

        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        print(f"[FluxBlockOffloader] Registering backward hooks for {self.num_blocks} blocks...")

        hooks_registered = 0
        for i in range(self.num_blocks):
            hook = self._create_backward_hook(i)
            if hook is not None:
                block = self._get_block(i)
                handle = block.register_full_backward_hook(hook)
                self.backward_hook_handles.append(handle)
                hooks_registered += 1

        print(f"[FluxBlockOffloader] Registered {hooks_registered} backward hooks")

    def _create_backward_hook(self, block_index: int):
        """
        Create backward hook for specific block
        """
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_gpu = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if swapping:
                self._submit_block_swap(block_idx_to_cpu, block_idx_to_gpu)
            if waiting:
                self.wait_for_block(block_idx_to_wait)
            return None

        return backward_hook

    def remove_backward_hooks(self):
        """Remove all registered backward hooks"""
        if not self.backward_hook_handles:
            return

        for handle in self.backward_hook_handles:
            handle.remove()

        num_removed = len(self.backward_hook_handles)
        self.backward_hook_handles = []
        print(f"[FluxBlockOffloader] Removed {num_removed} backward hooks")

    def cleanup(self):
        """Cleanup offloader resources"""
        print(f"[FluxBlockOffloader] Cleaning up...")

        self.remove_backward_hooks()
        self.thread_pool.shutdown(wait=True)

        self.staging_buffer_a = None
        self.staging_buffer_b = None
        self.pinned_buffer = None
        self.futures.clear()

        print(f"[FluxBlockOffloader] Cleanup complete")

    def clear_activations(self):
        """
        Clear saved activations after backward pass (for training).

        Called after each backward pass to free activation memory and
        prevent VRAM leaks. For FLUX.2, this clears any internal
        state used for gradient computation.
        """
        # Clear pending futures
        self.futures.clear()

        # Synchronize CUDA stream to ensure all operations complete
        if self.cuda_available and self.stream is not None:
            self.stream.synchronize()

        # Force garbage collection
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def create_flux_block_offloader(
    transformer: nn.Module,
    blocks_to_swap: int,
    device: torch.device,
    target_dtype: Optional[torch.dtype] = None,
    use_pinned_memory: bool = False,
    supports_backward: bool = False
) -> FluxBlockOffloader:
    """
    Create block offloader for FLUX.2 transformer

    Args:
        transformer: FLUX.2 transformer model (FluxTransformer2DModel)
        blocks_to_swap: Number of blocks to swap
        device: Target device
        target_dtype: Target dtype for computation
        use_pinned_memory: Use pinned memory for faster transfer
        supports_backward: Enable backward pass support (for training)

    Returns:
        FluxBlockOffloader instance
    """
    # Validate FLUX.2 architecture
    if not hasattr(transformer, 'transformer_blocks'):
        raise ValueError("Transformer does not have 'transformer_blocks' attribute")
    if not hasattr(transformer, 'single_transformer_blocks'):
        raise ValueError("Transformer does not have 'single_transformer_blocks' attribute")

    # Default dtype
    if target_dtype is None:
        first_param = next(transformer.parameters())
        target_dtype = first_param.dtype
        print(f"[FluxBlockOffloader] Auto-detected dtype: {target_dtype}")

    print(f"[FluxBlockOffloader] Creating offloader for FLUX.2 transformer")
    print(f"  - transformer_blocks: {len(transformer.transformer_blocks)}")
    print(f"  - single_transformer_blocks: {len(transformer.single_transformer_blocks)}")

    offloader = FluxBlockOffloader(
        transformer_blocks=transformer.transformer_blocks,
        single_transformer_blocks=transformer.single_transformer_blocks,
        blocks_to_swap=blocks_to_swap,
        device=device,
        target_dtype=target_dtype,
        use_pinned_memory=use_pinned_memory,
        transformer=transformer,
        supports_backward=supports_backward
    )

    return offloader
