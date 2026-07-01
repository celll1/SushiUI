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
        transformer: Optional[nn.Module] = None,
        supports_backward: bool = False,
        h2d_only: bool = False,
        ring_size: int = 2,
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
            supports_backward: Enable backward pass support (for training)
            h2d_only: H2D-only block swap (inference / frozen weights). Keeps a permanent
                pinned CPU master per swappable block and only ever copies host->device into
                a fixed ring of GPU buffers, eliminating the redundant device->host eviction
                of read-only weights (~halves PCIe traffic). Forward-only; ignored when
                supports_backward is True.
            ring_size: Number of GPU weight-buffer slots in the H2D-only ring (>=1). 1 keeps
                the minimum VRAM (fully serial loads); 2 (default) double-buffers so the next
                block's H2D overlaps the current block's compute.
        """
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.target_dtype = target_dtype
        self.use_pinned_memory = use_pinned_memory
        self.transformer = transformer
        self.supports_backward = supports_backward
        self.forward_only = not supports_backward

        # H2D-only mode is forward-only (read-only weights). Fall back to the normal swap
        # path for training (backward) until backward-direction H2D-only is implemented.
        self.h2d_only = bool(h2d_only) and self.forward_only
        self.ring_size = max(1, int(ring_size))
        if h2d_only and not self.forward_only:
            print("[BlockOffloader] h2d_only requested but backward is enabled; "
                  "falling back to normal block swap (H2D-only is inference-only for now).")

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        # Staging buffers for weight swapping
        self.staging_buffer_a = None
        self.staging_buffer_b = None
        self.pinned_buffer = None

        # H2D-only state (built in prepare when h2d_only is active)
        self.h2d_masters = None       # block_idx -> list[(module, pinned_cpu_master)]
        self.h2d_ring = None          # slot -> list[gpu_buffer] (one per Linear)
        self.h2d_slot_futures = None  # slot -> pending load future (or None)
        self.h2d_loaded_block = None  # slot -> block_idx currently (being) loaded (or None)
        self.h2d_swappable = None     # list of swappable block indices
        self.h2d_num_on_gpu = None

        # Backward hook handles (for training)
        self.backward_hook_handles = []

        mode_str = "training (backward enabled)" if supports_backward else "inference (forward-only)"
        h2d_str = f", H2D-only ring_size={self.ring_size}" if self.h2d_only else ""
        print(f"[BlockOffloader] Initialized: {self.num_blocks} total blocks, {self.blocks_to_swap} to swap ({mode_str}){h2d_str}")
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

        # Build H2D-only state (permanent pinned masters + GPU ring) from the CPU weights.
        if self.h2d_only:
            self._h2d_setup()

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

        if self.h2d_only:
            self._h2d_wait(block_idx)
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
            # No pending transfer - check if block weights are on CPU and move them synchronously.
            # Detect via a representative Linear weight (which is what weighs_to_device moves)
            # rather than the first parameter: for weight-only-FP8 models the Linear weight is a
            # buffer and the only parameters are GPU-resident norms, so a param-device check would
            # never trip. Falls back to the first parameter for plain models with no Linear weight.
            block = self.blocks[block_idx]
            weight_device = None
            for module in block.modules():
                if (
                    module.__class__.__name__.endswith("Linear")
                    and getattr(module, "weight", None) is not None
                ):
                    weight_device = module.weight.data.device
                    break
            if weight_device is None:
                first_param = next(block.parameters(), None)
                weight_device = first_param.device if first_param is not None else self.device
            if weight_device.type == "cpu":
                # Block weights are on CPU - move to GPU synchronously
                print(f"[BlockOffloader DEBUG] Block {block_idx} weights on CPU, moving to GPU synchronously...")
                weighs_to_device(block, self.device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                print(f"[BlockOffloader DEBUG] Block {block_idx} weights moved to GPU")

    def submit_move_blocks_forward(self, block_idx: int):
        """
        Submit block swap for forward pass

        Strategy (forward-only mode):
        - First N blocks stay on GPU permanently
        - Last M blocks rotate among swappable slots

        Strategy (backward-enabled mode):
        - Only swap first blocks_to_swap blocks
        - Remaining blocks stay on GPU for backward pass

        Args:
            block_idx: Current block index (just executed)
        """
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.h2d_only:
            self._h2d_submit(block_idx)
            return

        num_blocks_on_gpu = self.num_blocks - self.blocks_to_swap

        if not self.forward_only:
            # Backward-enabled mode: only swap first blocks_to_swap blocks
            if block_idx >= self.blocks_to_swap:
                return
            block_idx_to_cpu = block_idx
            block_idx_to_gpu = block_idx + 1
        else:
            # Forward-only mode: rotate among swappable blocks
            if block_idx < num_blocks_on_gpu:
                return

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

    # ------------------------------------------------------------------
    # H2D-only block swap (inference / read-only weights)
    #
    # Standard block swap copies the just-used block's weights back to CPU (D2H) and the
    # next block's weights to GPU (H2D). During inference the transformer weights never
    # change, so the D2H eviction of read-only weights is redundant PCIe traffic. H2D-only
    # keeps a permanent pinned CPU master per swappable block (never written) and streams
    # only host->device into a fixed ring of GPU buffers, halving PCIe bytes and removing
    # the per-tensor D2H sync. A ring of ring_size>=2 lets the next block's H2D overlap the
    # current block's compute.
    # ------------------------------------------------------------------
    @staticmethod
    def _linear_weight_modules(block: nn.Module):
        """List (module, weight_name-agnostic) Linear modules that carry a weight tensor,
        in a deterministic order shared across identically-structured blocks."""
        out = []
        for _name, m in block.named_modules():
            if m.__class__.__name__.endswith("Linear") and getattr(m, "weight", None) is not None:
                out.append(m)
        return out

    def _h2d_setup(self):
        """Build permanent pinned flat CPU masters and the GPU ring. Called from prepare
        after the swappable blocks' weights are on CPU.

        Coalescing (Tier 2C): each swappable block's Linear weights are concatenated into a
        single flat pinned CPU tensor, and each ring slot is a single flat GPU tensor, so a
        block's whole weight set moves in ONE host->device copy instead of one per Linear.
        Each Linear's weight.data becomes a (reshaped) view into the flat buffer."""
        self.h2d_num_on_gpu = self.num_blocks - self.blocks_to_swap
        self.h2d_swappable = list(range(self.h2d_num_on_gpu, self.num_blocks))
        num_swappable = len(self.h2d_swappable)
        if num_swappable == 0:
            self.h2d_only = False
            return
        self.ring_size = max(1, min(self.ring_size, num_swappable))

        # Coalescing into one flat buffer requires a single dtype across all swappable Linear
        # weights. Fall back to standard block swap if mixed (keeps correctness).
        dtypes = set()
        for bidx in self.h2d_swappable:
            for m in self._linear_weight_modules(self.blocks[bidx]):
                dtypes.add(m.weight.data.dtype)
        if len(dtypes) != 1:
            print(f"[BlockOffloader] H2D-only disabled: mixed Linear weight dtypes {dtypes}; "
                  f"using standard block swap.")
            self.h2d_only = False
            return
        flat_dtype = dtypes.pop()

        # Permanent pinned flat CPU master per swappable block. Never overwritten (read-only).
        # h2d_masters[bidx] = (flat_cpu, [(module, offset, numel, shape), ...])
        self.h2d_masters = {}
        for bidx in self.h2d_swappable:
            mods = self._linear_weight_modules(self.blocks[bidx])
            total = sum(m.weight.data.numel() for m in mods)
            flat_cpu = torch.empty(total, dtype=flat_dtype, device="cpu")
            if self.cuda_available:
                flat_cpu = flat_cpu.pin_memory(device=self.device)
            layout = []
            off = 0
            for m in mods:
                w = m.weight.data
                n = w.numel()
                shape = tuple(w.shape)
                flat_cpu[off:off + n].copy_(w.reshape(-1))
                m.weight.data = flat_cpu[off:off + n].view(shape)  # master view (CPU)
                layout.append((m, off, n, shape))
                off += n
            self.h2d_masters[bidx] = (flat_cpu, layout)

        # GPU ring: ring_size flat buffers (all swappable blocks share size/structure).
        flat_numel = self.h2d_masters[self.h2d_swappable[0]][0].numel()
        for bidx in self.h2d_swappable:
            assert self.h2d_masters[bidx][0].numel() == flat_numel, (
                "H2D-only requires identically-structured swappable blocks")
        self.h2d_ring = [
            torch.empty(flat_numel, dtype=flat_dtype, device=self.device)
            for _ in range(self.ring_size)
        ]
        self.h2d_slot_futures = [None] * self.ring_size
        self.h2d_loaded_block = [None] * self.ring_size

        # Prime the first ring_size swappable blocks into slots 0..ring_size-1.
        for j in range(self.ring_size):
            self._h2d_submit_load(self.h2d_swappable[j], j)

        print(f"[BlockOffloader] H2D-only ready: {num_swappable} swappable blocks, "
              f"ring_size={self.ring_size}, coalesced flat pinned CPU masters (no D2H eviction)")

    def _h2d_submit_load(self, block_idx: int, slot: int):
        """Submit an async single-copy H2D load of block_idx's flat master into ring[slot]."""
        flat_cpu = self.h2d_masters[block_idx][0]
        flat_gpu = self.h2d_ring[slot]
        self.h2d_loaded_block[slot] = block_idx
        if not self.cuda_available:
            flat_gpu.copy_(flat_cpu)
            self.h2d_slot_futures[slot] = None
            return
        # Order the H2D after the compute that last used this slot (the block vacating it),
        # captured as an event on the compute stream at submit time.
        compute_done = torch.cuda.current_stream().record_event()

        def load():
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(compute_done)
                flat_gpu.copy_(flat_cpu, non_blocking=True)
                ev = self.stream.record_event()
            return block_idx, slot, ev

        self.h2d_slot_futures[slot] = self.thread_pool.submit(load)

    def _h2d_point_weights(self, block_idx: int, flat_buf):
        """Point each Linear's weight.data at its slice/view of the given flat buffer."""
        for (m, off, n, shape) in self.h2d_masters[block_idx][1]:
            m.weight.data = flat_buf[off:off + n].view(shape)

    def _h2d_wait(self, block_idx: int):
        """Ensure block_idx's weights are resident in its ring slot, then point weight.data
        at views into the slot's flat GPU buffer. Self-heals at step boundaries."""
        if block_idx < self.h2d_num_on_gpu:
            return
        slot = (block_idx - self.h2d_num_on_gpu) % self.ring_size
        fut = self.h2d_slot_futures[slot]
        if fut is not None and self.h2d_loaded_block[slot] == block_idx:
            bidx, s, ev = fut.result()
            self.h2d_slot_futures[slot] = None
            assert bidx == block_idx and s == slot, f"H2D slot mismatch: {bidx}/{s} != {block_idx}/{slot}"
            if self.cuda_available and ev is not None:
                torch.cuda.current_stream().wait_event(ev)
        elif self.h2d_loaded_block[slot] != block_idx:
            # Slot does not hold this block (e.g. first blocks of a new denoise step) -> load
            # synchronously.
            if fut is not None:
                fut.result()
                self.h2d_slot_futures[slot] = None
            self.h2d_ring[slot].copy_(self.h2d_masters[block_idx][0])
            if self.cuda_available:
                torch.cuda.synchronize()
            self.h2d_loaded_block[slot] = block_idx
        self._h2d_point_weights(block_idx, self.h2d_ring[slot])

    def _h2d_submit(self, block_idx: int):
        """After block_idx ran: repoint it to its CPU master (no copy) and prefetch the
        block ring_size ahead into the freed slot."""
        if block_idx < self.h2d_num_on_gpu:
            return
        i = block_idx - self.h2d_num_on_gpu
        slot = i % self.ring_size
        # Repoint the just-run block back to its permanent flat CPU master (no D2H).
        self._h2d_point_weights(block_idx, self.h2d_masters[block_idx][0])
        # Prefetch ring_size blocks ahead into this freed slot (no wrap across the step end;
        # the first blocks of the next step self-heal in _h2d_wait).
        next_i = i + self.ring_size
        if next_i < len(self.h2d_swappable):
            self._h2d_submit_load(self.h2d_swappable[next_i], slot)
        else:
            self.h2d_slot_futures[slot] = None
            self.h2d_loaded_block[slot] = None

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

        # Order the swap AFTER the compute that just used these weights, but do it on the
        # transfer stream via a CUDA event instead of draining the whole compute stream on
        # the host. record_event() on the compute stream captures all work enqueued so far
        # (the block that just executed, enqueued before this swap was submitted); the
        # transfer stream then waits for that event before it evicts (D2H) / overwrites
        # (H2D) the GPU weight buffers. This removes a full current_stream().synchronize()
        # that was paid on every one of ~20 swaps per denoise step (draining unrelated
        # compute + blocking the host thread) and replaces it with a GPU-side dependency
        # that preserves the exact same ordering guarantee.
        compute_done = torch.cuda.current_stream().record_event()
        self.stream.wait_event(compute_done)

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

    def register_backward_hooks(self):
        """
        Register backward hooks for training-time block swapping

        This method registers hooks that swap blocks during backward pass,
        moving blocks from GPU to CPU in reverse order to free VRAM.
        """
        if not self.supports_backward:
            print(f"[BlockOffloader] Backward hooks not registered (forward-only mode)")
            return

        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        print(f"[BlockOffloader] Registering backward hooks for {self.num_blocks} blocks...")

        hooks_registered = 0
        for i in range(self.num_blocks):
            hook = self._create_backward_hook(i)
            if hook is not None:
                handle = self.blocks[i].register_full_backward_hook(hook)
                self.backward_hook_handles.append(handle)
                hooks_registered += 1

        print(f"[BlockOffloader] Registered {hooks_registered} backward hooks")

    def _create_backward_hook(self, block_index: int):
        """
        Create backward hook for specific block

        Args:
            block_index: Block index to create hook for

        Returns:
            Hook function or None if hook not needed for this block
        """
        # Calculate which blocks need hooks
        # Backward propagates from last block to first block
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # Calculate indices for swapping
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_gpu = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            """Backward hook: swap blocks as gradients propagate"""
            if swapping:
                self._submit_block_swap(block_idx_to_cpu, block_idx_to_gpu)
            if waiting:
                self.wait_for_block(block_idx_to_wait)
            return None

        return backward_hook

    def remove_backward_hooks(self):
        """
        Remove all registered backward hooks

        Call this method when switching from training to inference mode.
        """
        if not self.backward_hook_handles:
            return

        for handle in self.backward_hook_handles:
            handle.remove()

        num_removed = len(self.backward_hook_handles)
        self.backward_hook_handles = []
        print(f"[BlockOffloader] Removed {num_removed} backward hooks")

    def cleanup(self):
        """
        Cleanup offloader resources

        - Remove backward hooks
        - Shutdown thread pool
        - Clear staging buffers
        """
        print(f"[BlockOffloader] Cleaning up...")

        # Remove hooks
        self.remove_backward_hooks()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        # Clear staging buffers
        self.staging_buffer_a = None
        self.staging_buffer_b = None
        self.pinned_buffer = None

        # Clear futures
        self.futures.clear()

        print(f"[BlockOffloader] Cleanup complete")
