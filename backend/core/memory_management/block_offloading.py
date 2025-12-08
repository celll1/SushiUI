"""
Transformer Block Offloading for Low VRAM Environments

Based on musubi-tuner's approach:
- Weight-only offloading (Linear/Conv weights on CPU, buffers on GPU)
- Forward-only strategy (keeps first N blocks on GPU permanently)
- Async weight swapping with staging buffers
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import torch
import torch.nn as nn


def _synchronize_device(device: torch.device):
    """Synchronize device operations"""
    if device.type == "cuda":
        torch.cuda.synchronize()


def weighs_to_device(layer: nn.Module, device: torch.device):
    """Move Linear layer weights to device (non-blocking)"""
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            if module.__class__.__name__.endswith("Linear"):
                module.weight.data = module.weight.data.to(device, non_blocking=device.type != "cpu")


class TransformerBlockOffloader:
    """
    Block offloader for Transformer models (forward-only inference)

    Strategy:
    - Keep first N blocks on GPU (full model)
    - Keep last M blocks on CPU (weights only, buffers on GPU)
    - During forward pass, swap blocks asynchronously
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device,
        target_dtype: torch.dtype = torch.bfloat16,
        use_pinned_memory: bool = False,
        transformer: Optional[nn.Module] = None
    ):
        """
        Initialize Block Offloader

        Args:
            blocks: Transformer blocks (nn.ModuleList)
            blocks_to_swap: Number of blocks to keep on CPU
            device: Target device (cuda:0)
            target_dtype: Target dtype for computation
            use_pinned_memory: Use pinned memory for faster transfer
            transformer: Parent transformer (for auxiliary modules)
        """
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.target_dtype = target_dtype
        self.use_pinned_memory = use_pinned_memory
        self.transformer = transformer

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        # Staging buffers for weight swapping
        self.staging_buffer_a = None
        self.staging_buffer_b = None
        self.pinned_buffer = None

        print(f"[BlockOffloader] Initialized: {self.num_blocks} total blocks, {self.blocks_to_swap} to swap")
        print(f"[BlockOffloader] Device: {self.device}, dtype: {self.target_dtype}, pinned_memory: {self.use_pinned_memory}")

    def prepare_block_devices_before_forward(self):
        """
        Prepare block device placement before forward pass

        - First (num_blocks - blocks_to_swap) blocks: full model on GPU
        - Last blocks_to_swap blocks: weights on CPU, buffers on GPU
        """
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        print(f"[BlockOffloader] Preparing block devices...")

        num_blocks_on_gpu = self.num_blocks - self.blocks_to_swap

        # Move first N blocks to GPU (full)
        print(f"[BlockOffloader] Moving first {num_blocks_on_gpu} blocks to GPU (full)...")
        for i in range(num_blocks_on_gpu):
            self.blocks[i] = self.blocks[i].to(self.device)
            weighs_to_device(self.blocks[i], self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            print(f"[BlockOffloader] GPU synchronization complete")

        # Move last M blocks: buffers to GPU, weights to CPU
        print(f"[BlockOffloader] Moving last {self.blocks_to_swap} blocks: buffers to GPU, weights to CPU...")
        cpu_device = torch.device("cpu")
        for i in range(num_blocks_on_gpu, self.num_blocks):
            # First move entire block to GPU (ensures buffers are on GPU)
            self.blocks[i] = self.blocks[i].to(self.device)
            # Then move weights back to CPU
            weighs_to_device(self.blocks[i], cpu_device)

        _synchronize_device(self.device)

        # Move auxiliary modules to GPU
        self._move_auxiliary_modules_to_gpu()

        print(f"[BlockOffloader] Block device preparation complete")

        # Log device status
        self.log_device_status("Ready for forward pass")

    def _move_auxiliary_modules_to_gpu(self):
        """
        Move Z-Image auxiliary modules to GPU

        Z-Image has these auxiliary modules outside self.layers:
        - t_embedder (TimestepEmbedder)
        - cap_embedder (nn.Sequential)
        - all_x_embedder (nn.ModuleDict of patch embedders)
        - all_final_layer (nn.ModuleDict of final layers)
        - noise_refiner (nn.ModuleList)
        - context_refiner (nn.ModuleList)
        """
        if self.transformer is None:
            return

        print(f"[BlockOffloader] Moving auxiliary modules to GPU...")

        auxiliary_module_names = [
            "all_x_embedder",
            "all_final_layer",
            "noise_refiner",
            "context_refiner",
            "t_embedder",
            "cap_embedder",
        ]

        parent = self.transformer
        for module_name in auxiliary_module_names:
            if hasattr(parent, module_name):
                module = getattr(parent, module_name)
                if module is not None and isinstance(module, nn.Module):
                    module._apply(lambda t: t.to(self.device) if isinstance(t, torch.Tensor) else t)
                    print(f"[BlockOffloader]   - Moved {module_name} to {self.device}")

        # Move transformer-level buffers/parameters (x_pad_token, etc.)
        for name, param in parent.named_parameters(recurse=False):
            if param.device != self.device:
                param.data = param.data.to(self.device)
                print(f"[BlockOffloader]   - Moved parameter {name} to {self.device}")

        for name, buffer in parent.named_buffers(recurse=False):
            if buffer.device != self.device:
                buffer.data = buffer.data.to(self.device)
                print(f"[BlockOffloader]   - Moved buffer {name} to {self.device}")

        print(f"[BlockOffloader] Auxiliary modules moved to GPU")

    def wait_for_block(self, block_idx: int):
        """
        Wait for block transfer to complete
        If block is on CPU and not being transferred, move it to GPU synchronously

        Args:
            block_idx: Block index to wait for
        """
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        num_blocks_on_gpu = self.num_blocks - self.blocks_to_swap

        # First N blocks stay on GPU permanently, no wait needed
        if block_idx < num_blocks_on_gpu:
            return

        # If block has a pending transfer, wait for it
        if block_idx in self.futures:
            future = self.futures.pop(block_idx)
            _, bidx_to_cuda, sync_event = future.result()

            assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

            if self.cuda_available and sync_event is not None:
                torch.cuda.current_stream().wait_event(sync_event)
        else:
            # No pending transfer - check if block weights are on CPU and move them synchronously
            block = self.blocks[block_idx]
            first_param = next(block.parameters(), None)
            if first_param is not None and first_param.device.type == "cpu":
                # Block weights are on CPU - move to GPU synchronously
                weighs_to_device(block, self.device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

    def submit_move_blocks_forward(self, block_idx: int):
        """
        Submit block swap for forward-only offloading

        Strategy:
        - First N blocks stay on GPU permanently
        - Last M blocks rotate among swappable slots

        Args:
            block_idx: Current block index (just executed)
        """
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        num_blocks_on_gpu = self.num_blocks - self.blocks_to_swap

        # First N blocks stay on GPU permanently, no swap needed
        if block_idx < num_blocks_on_gpu:
            return

        # For blocks >= num_blocks_on_gpu, rotate among the swappable blocks
        block_idx_to_cpu = block_idx
        next_block = block_idx + 1
        if next_block >= self.num_blocks:
            next_block = num_blocks_on_gpu

        block_idx_to_gpu = next_block
        self._submit_block_swap(block_idx_to_cpu, block_idx_to_gpu)

    def _submit_block_swap(self, block_idx_to_cpu: int, block_idx_to_gpu: int):
        """
        Submit asynchronous block swap

        Args:
            block_idx_to_cpu: Block to move to CPU
            block_idx_to_gpu: Block to move to GPU
        """
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_gpu, block_to_gpu):
            dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(dev)

            sync_event = self.swap_weight_devices(block_to_cpu, block_to_gpu)
            return bidx_to_cpu, bidx_to_gpu, sync_event

        block_to_cpu = self.blocks[block_idx_to_cpu]
        block_to_gpu = self.blocks[block_idx_to_gpu]

        self.futures[block_idx_to_gpu] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_gpu, block_to_gpu
        )

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        """
        Swap weights between two blocks

        Args:
            block_to_cpu: Block whose weights will be moved to CPU
            block_to_cuda: Block whose weights will be moved to GPU

        Returns:
            sync_event: CUDA event for synchronization
        """
        assert block_to_cpu.__class__ == block_to_cuda.__class__

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
        print(f"[BlockOffloader] {status_message}")
        print(f"============================================================")

        num_blocks_on_gpu = self.num_blocks - self.blocks_to_swap

        # Log first GPU block
        if num_blocks_on_gpu > 0:
            block = self.blocks[0]
            params = list(block.parameters())
            if params:
                first_param_device = params[0].device
                print(f"  Block 0 (GPU): device={first_param_device}")

        # Log first CPU block
        if self.blocks_to_swap > 0:
            block = self.blocks[num_blocks_on_gpu]
            params = list(block.parameters())
            if params:
                first_param_device = params[0].device
                print(f"  Block {num_blocks_on_gpu} (CPU weights): device={first_param_device}")

        # Log VRAM usage
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print(f"  VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        print(f"============================================================")
