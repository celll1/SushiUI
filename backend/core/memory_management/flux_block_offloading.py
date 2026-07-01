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
        supports_backward: bool = False,
        h2d_only: bool = False,
        ring_size: int = 2,
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
            h2d_only: H2D-only block swap (inference / read-only weights). Permanent pinned
                CPU masters + fixed GPU ring, no device->host eviction. Forward-only.
            ring_size: Number of GPU weight-buffer slots in the H2D-only ring (>=1).
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

        # H2D-only mode is forward-only (read-only weights).
        self.h2d_only = bool(h2d_only) and self.forward_only
        self.ring_size = max(1, int(ring_size))
        if h2d_only and not self.forward_only:
            print("[FluxBlockOffloader] h2d_only requested but backward is enabled; "
                  "falling back to normal block swap (H2D-only is inference-only for now).")
        # H2D-only state (built in prepare)
        self.h2d_masters = None       # unified_idx -> (flat_cpu, [(module, offset, numel, shape)])
        self.h2d_ring = None          # slot -> flat GPU buffer (max block size)
        self.h2d_slot_futures = None
        self.h2d_loaded_block = None
        self.h2d_swappable = None
        self.h2d_num_on_gpu = None

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        # Staging buffers for weight swapping (separate for dual and single blocks)
        # FLUX.2 has different block structures, so we need separate buffers
        self.staging_buffer_dual_a = None
        self.staging_buffer_dual_b = None
        self.staging_buffer_single_a = None
        self.staging_buffer_single_b = None
        self.pinned_buffer_dual = None
        self.pinned_buffer_single = None

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

        # Build H2D-only state (permanent pinned flat masters + GPU ring) from CPU weights.
        if self.h2d_only:
            self._h2d_setup()

        # Move auxiliary modules to GPU
        self._move_auxiliary_modules_to_gpu()

        print(f"[FluxBlockOffloader] Block device preparation complete")

        # Log device status
        self.log_device_status("Ready for forward pass")

    def _move_auxiliary_modules_to_gpu(self):
        """
        Move FLUX.2 auxiliary modules to GPU

        FLUX.2 (Klein) has these auxiliary modules:
        - pos_embed (FluxPosEmbed)
        - time_guidance_embed (Flux2CombinedTimestepGuidanceEmbedding) - FLUX.2 specific
        - time_text_embed (CombinedTimestepTextProjEmbeddings) - FLUX.1 compatibility
        - double_stream_modulation_img (Flux2ModulationOut) - FLUX.2 specific
        - double_stream_modulation_txt (Flux2ModulationOut) - FLUX.2 specific
        - single_stream_modulation (Flux2ModulationOut) - FLUX.2 specific
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
            "time_guidance_embed",  # FLUX.2 uses this instead of time_text_embed
            "time_text_embed",  # FLUX.1 compatibility
            "double_stream_modulation_img",  # FLUX.2 specific
            "double_stream_modulation_txt",  # FLUX.2 specific
            "single_stream_modulation",  # FLUX.2 specific
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

        if self.h2d_only:
            self._h2d_wait(unified_idx)
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

        if self.h2d_only:
            self._h2d_submit(unified_idx)
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

    # ------------------------------------------------------------------
    # H2D-only block swap (inference / read-only weights) — FLUX.2 variant.
    # Handles the dual+single structure: swappable blocks may differ in size, so the GPU
    # ring buffers are sized to the largest swappable block and each block copies only its
    # own bytes (flat_gpu[:n]). Views index within [0, block_total).
    # ------------------------------------------------------------------
    @staticmethod
    def _h2d_linear_modules(block):
        out = []
        for _n, m in block.named_modules():
            if m.__class__.__name__.endswith("Linear") and getattr(m, "weight", None) is not None:
                out.append(m)
        return out

    def _h2d_setup(self):
        self.h2d_num_on_gpu = self.num_blocks - self.blocks_to_swap
        self.h2d_swappable = list(range(self.h2d_num_on_gpu, self.num_blocks))
        num_swappable = len(self.h2d_swappable)
        if num_swappable == 0:
            self.h2d_only = False
            return
        self.ring_size = max(1, min(self.ring_size, num_swappable))

        dtypes = set()
        for uidx in self.h2d_swappable:
            for m in self._h2d_linear_modules(self._get_block(uidx)):
                dtypes.add(m.weight.data.dtype)
        if len(dtypes) != 1:
            print(f"[FluxBlockOffloader] H2D-only disabled: mixed Linear weight dtypes {dtypes}; "
                  f"using standard block swap.")
            self.h2d_only = False
            return
        flat_dtype = dtypes.pop()

        # Permanent pinned flat CPU master per swappable block (dual and single differ in
        # size). h2d_masters[uidx] = (flat_cpu, [(module, offset, numel, shape)]).
        self.h2d_masters = {}
        max_numel = 0
        for uidx in self.h2d_swappable:
            mods = self._h2d_linear_modules(self._get_block(uidx))
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
                m.weight.data = flat_cpu[off:off + n].view(shape)
                layout.append((m, off, n, shape))
                off += n
            self.h2d_masters[uidx] = (flat_cpu, layout)
            max_numel = max(max_numel, total)

        # GPU ring sized to the largest swappable block; each load copies only its own bytes.
        self.h2d_ring = [
            torch.empty(max_numel, dtype=flat_dtype, device=self.device)
            for _ in range(self.ring_size)
        ]
        self.h2d_slot_futures = [None] * self.ring_size
        self.h2d_loaded_block = [None] * self.ring_size
        for j in range(self.ring_size):
            self._h2d_submit_load(self.h2d_swappable[j], j)
        print(f"[FluxBlockOffloader] H2D-only ready: {num_swappable} swappable blocks, "
              f"ring_size={self.ring_size}, coalesced flat pinned CPU masters (no D2H eviction)")

    def _h2d_submit_load(self, unified_idx: int, slot: int):
        flat_cpu = self.h2d_masters[unified_idx][0]
        n = flat_cpu.numel()
        flat_gpu = self.h2d_ring[slot]
        self.h2d_loaded_block[slot] = unified_idx
        if not self.cuda_available:
            flat_gpu[:n].copy_(flat_cpu)
            self.h2d_slot_futures[slot] = None
            return
        compute_done = torch.cuda.current_stream().record_event()

        def load():
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(compute_done)
                flat_gpu[:n].copy_(flat_cpu, non_blocking=True)
                ev = self.stream.record_event()
            return unified_idx, slot, ev

        self.h2d_slot_futures[slot] = self.thread_pool.submit(load)

    def _h2d_point_weights(self, unified_idx: int, flat_buf):
        for (m, off, n, shape) in self.h2d_masters[unified_idx][1]:
            m.weight.data = flat_buf[off:off + n].view(shape)

    def _h2d_wait(self, unified_idx: int):
        if unified_idx < self.h2d_num_on_gpu:
            return
        slot = (unified_idx - self.h2d_num_on_gpu) % self.ring_size
        fut = self.h2d_slot_futures[slot]
        if fut is not None and self.h2d_loaded_block[slot] == unified_idx:
            bidx, s, ev = fut.result()
            self.h2d_slot_futures[slot] = None
            assert bidx == unified_idx and s == slot, f"H2D slot mismatch: {bidx}/{s} != {unified_idx}/{slot}"
            if self.cuda_available and ev is not None:
                torch.cuda.current_stream().wait_event(ev)
        elif self.h2d_loaded_block[slot] != unified_idx:
            if fut is not None:
                fut.result()
                self.h2d_slot_futures[slot] = None
            flat_cpu = self.h2d_masters[unified_idx][0]
            self.h2d_ring[slot][:flat_cpu.numel()].copy_(flat_cpu)
            if self.cuda_available:
                torch.cuda.synchronize()
            self.h2d_loaded_block[slot] = unified_idx
        self._h2d_point_weights(unified_idx, self.h2d_ring[slot])

    def _h2d_submit(self, unified_idx: int):
        if unified_idx < self.h2d_num_on_gpu:
            return
        i = unified_idx - self.h2d_num_on_gpu
        slot = i % self.ring_size
        self._h2d_point_weights(unified_idx, self.h2d_masters[unified_idx][0])
        next_i = i + self.ring_size
        if next_i < len(self.h2d_swappable):
            self._h2d_submit_load(self.h2d_swappable[next_i], slot)
        else:
            self.h2d_slot_futures[slot] = None
            self.h2d_loaded_block[slot] = None

    def _is_dual_block(self, block: nn.Module) -> bool:
        """Check if block is a dual stream block (has different structure than single)"""
        # Dual blocks have attn (FluxAttention) with separate to_q, to_k, to_v
        # Single blocks have attn with combined qkv_proj
        # Use class name check as primary indicator
        class_name = block.__class__.__name__
        return "Single" not in class_name

    def _get_or_create_staging_buffers(self, weight_swap_jobs, is_dual: bool):
        """Get or create staging buffers for the specified block type"""
        if is_dual:
            # Check if dual buffers exist and match
            if self.staging_buffer_dual_a is not None:
                if len(self.staging_buffer_dual_a) == len(weight_swap_jobs):
                    # Check if shapes match
                    shapes_match = all(
                        buf.shape == job[2].shape
                        for buf, job in zip(self.staging_buffer_dual_a, weight_swap_jobs)
                    )
                    if shapes_match:
                        return self.staging_buffer_dual_a, self.staging_buffer_dual_b

            # Create new dual buffers
            self.staging_buffer_dual_a = [
                torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=self.device)
                for _, _, cuda_data_view, _ in weight_swap_jobs
            ]
            self.staging_buffer_dual_b = [
                torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=self.device)
                for _, _, cuda_data_view, _ in weight_swap_jobs
            ]
            return self.staging_buffer_dual_a, self.staging_buffer_dual_b
        else:
            # Check if single buffers exist and match
            if self.staging_buffer_single_a is not None:
                if len(self.staging_buffer_single_a) == len(weight_swap_jobs):
                    # Check if shapes match
                    shapes_match = all(
                        buf.shape == job[2].shape
                        for buf, job in zip(self.staging_buffer_single_a, weight_swap_jobs)
                    )
                    if shapes_match:
                        return self.staging_buffer_single_a, self.staging_buffer_single_b

            # Create new single buffers
            self.staging_buffer_single_a = [
                torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=self.device)
                for _, _, cuda_data_view, _ in weight_swap_jobs
            ]
            self.staging_buffer_single_b = [
                torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=self.device)
                for _, _, cuda_data_view, _ in weight_swap_jobs
            ]
            return self.staging_buffer_single_a, self.staging_buffer_single_b

    def _get_or_create_pinned_buffer(self, weight_swap_jobs, is_dual: bool):
        """Get or create pinned buffer for the specified block type"""
        if is_dual:
            if self.pinned_buffer_dual is not None:
                if len(self.pinned_buffer_dual) == len(weight_swap_jobs):
                    shapes_match = all(
                        buf.shape == job[2].shape
                        for buf, job in zip(self.pinned_buffer_dual, weight_swap_jobs)
                    )
                    if shapes_match:
                        return self.pinned_buffer_dual

            # Create new dual pinned buffer
            self.pinned_buffer_dual = [
                torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=self.device)
                for _, _, cuda_data_view, _ in weight_swap_jobs
            ]
            return self.pinned_buffer_dual
        else:
            if self.pinned_buffer_single is not None:
                if len(self.pinned_buffer_single) == len(weight_swap_jobs):
                    shapes_match = all(
                        buf.shape == job[2].shape
                        for buf, job in zip(self.pinned_buffer_single, weight_swap_jobs)
                    )
                    if shapes_match:
                        return self.pinned_buffer_single

            # Create new single pinned buffer
            self.pinned_buffer_single = [
                torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=self.device)
                for _, _, cuda_data_view, _ in weight_swap_jobs
            ]
            return self.pinned_buffer_single

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        """
        Swap weights between two blocks

        Note: FLUX.2 has FluxTransformerBlock (dual) and FluxSingleTransformerBlock (single)
        which have different structures. We use separate staging buffer pools for each type.
        """
        weight_swap_jobs = []

        # Dual/single boundary crossing: when blocks_to_swap exceeds the number of single
        # blocks, the rotation can pair a dual block (FluxTransformerBlock) with a single
        # block (FluxSingleTransformerBlock). Their Linear layouts/shapes differ, so the
        # name/shape-paired pointer swap below does not apply. Move each block's weights to
        # its target device independently instead of relying on the wait_for_block
        # synchronous fallback (which also leaves the outgoing block GPU-resident). This is
        # correct and fully offloads the outgoing block.
        if block_to_cpu.__class__ != block_to_cuda.__class__:
            if not getattr(self, "_warned_boundary_swap", False):
                print("[FluxBlockOffloader] blocks_to_swap crosses the dual/single block "
                      "boundary; using independent per-block moves there. Set blocks_to_swap "
                      "<= number of single blocks to keep the fast paired swap.")
                self._warned_boundary_swap = True
            compute_done = torch.cuda.current_stream().record_event()
            self.stream.wait_event(compute_done)
            with torch.cuda.stream(self.stream):
                weighs_to_device(block_to_cuda, self.device)
                weighs_to_device(block_to_cpu, torch.device("cpu"))
            return self.stream.record_event()

        # Determine block type for buffer selection
        is_dual = self._is_dual_block(block_to_cuda)

        # Find Linear modules to swap
        modules_to_cpu = {k: v for k, v in block_to_cpu.named_modules()}
        for module_to_cuda_name, module_to_cuda in block_to_cuda.named_modules():
            # Skip non-Linear modules (ModuleList, Sequential, etc.)
            if not module_to_cuda.__class__.__name__.endswith("Linear"):
                continue
            if not hasattr(module_to_cuda, "weight") or module_to_cuda.weight is None:
                continue

            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is None:
                continue
            # Check module_to_cpu also has weight attribute
            if not hasattr(module_to_cpu, "weight") or module_to_cpu.weight is None:
                continue

            if module_to_cpu.weight.shape == module_to_cuda.weight.shape:
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
        # that was paid on every swap per denoise step (draining unrelated compute +
        # blocking the host thread) and replaces it with a GPU-side dependency that
        # preserves the exact same ordering guarantee.
        compute_done = torch.cuda.current_stream().record_event()
        self.stream.wait_event(compute_done)

        if not self.use_pinned_memory:
            # Strategy 1: Use staging buffers (less pinned memory)
            # Get or create cached buffers for this block type
            stream = self.stream
            staging_buffer_a, staging_buffer_b = self._get_or_create_staging_buffers(weight_swap_jobs, is_dual)

            with torch.cuda.stream(stream):
                event_b = None
                for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                    staging_buffer_a, staging_buffer_b, weight_swap_jobs
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
                staging_buffer_a, staging_buffer_b, weight_swap_jobs
            ):
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = cpu_data_view

            sync_event = event_b

        else:
            # Strategy 2: Use full pinned memory (faster but more memory)
            # Get or create cached pinned buffer for this block type
            pinned_buffer = self._get_or_create_pinned_buffer(weight_swap_jobs, is_dual)

            events = [torch.cuda.Event() for _ in weight_swap_jobs]

            # Copy weights to CPU
            for event, module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                events, pinned_buffer, weight_swap_jobs
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
                pinned_buffer, weight_swap_jobs
            ):
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = module_pin_buf

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

        # Clear dual block buffers
        self.staging_buffer_dual_a = None
        self.staging_buffer_dual_b = None
        self.pinned_buffer_dual = None

        # Clear single block buffers
        self.staging_buffer_single_a = None
        self.staging_buffer_single_b = None
        self.pinned_buffer_single = None

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
    supports_backward: bool = False,
    h2d_only: bool = False,
    ring_size: int = 2,
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
        supports_backward=supports_backward,
        h2d_only=h2d_only,
        ring_size=ring_size,
    )

    return offloader
