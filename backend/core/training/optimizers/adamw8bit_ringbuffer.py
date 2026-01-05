"""
AdamW 8-bit Optimizer with Ring Buffer Support

Based on bitsandbytes 8-bit optimizer (MIT License)
https://github.com/TimDettmers/bitsandbytes

Modified for SushiUI Ring Buffer integration:
- Optimizer states (exp_avg, exp_avg_sq) allocated on CPU via Ring Buffer
- Automatic GPU transfer during backward pass
- VRAM savings: ~75% for optimizer states (largest VRAM consumer)

Implementation:
- Uses bitsandbytes quantization algorithm (dynamic map, CUB reduce, bias correction)
- Supports FP32, FP16, BF16 parameters
- Per-parameter fused updates via register_post_accumulate_grad_hook
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Callable
import os

# Import CUDA extension (lazy loading to avoid compilation on import)
from .adamw8bit_cuda import get_extension

# Import quantization map generator
from .quantization_map import create_quantization_map


def quantize_blockwise_inplace(tensor: torch.Tensor, blocksize: int = 256):
    """
    Quantize a tensor to UINT8 using blockwise quantization (for z initialization).

    Args:
        tensor: Input tensor (FP16/FP32) on GPU
        blocksize: Block size for quantization (default: 256)

    Returns:
        quantized: UINT8 tensor (same shape as input)
        absmax: FP32 absmax values per block [num_blocks]
    """
    n = tensor.numel()
    num_blocks = (n + blocksize - 1) // blocksize

    # Allocate output
    device = tensor.device
    quantized = torch.zeros(n, dtype=torch.uint8, device=device)
    absmax = torch.zeros(num_blocks, dtype=torch.float32, device=device)

    # Flatten tensor for blockwise processing
    flat = tensor.flatten()

    # Process each block
    for i in range(num_blocks):
        start = i * blocksize
        end = min(start + blocksize, n)
        block = flat[start:end]

        # Compute absmax for this block
        block_absmax = block.abs().max()
        absmax[i] = block_absmax

        # Quantize: map [-absmax, absmax] -> [0, 255]
        if block_absmax > 0:
            # Normalize to [-1, 1], then map to [0, 255]
            normalized = block / block_absmax  # [-1, 1]
            quantized_block = ((normalized + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            quantized[start:end] = quantized_block
        else:
            # Zero block
            quantized[start:end] = 127  # Middle of [0, 255]

    return quantized, absmax


class AdamW8bit_RingBuffer(Optimizer):
    """
    AdamW optimizer with 8-bit blockwise quantization and Ring Buffer support.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for momentum and variance (default: (0.9, 0.999))
        eps: Epsilon for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        use_8bit: Enable 8-bit quantization (default: True)
        cautious: Enable cautious masking (default: False)
        get_state_buffer: Callable to allocate Ring Buffer state (from RingBufferAllocator)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        use_8bit: bool = True,
        cautious: bool = False,
        schedule_free: bool = False,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        use_radam: bool = False,
        stochastic_rounding: bool = False,
        get_state_buffer: Optional[Callable] = None,
    ):
        # Lazy load CUDA extension (compile on first optimizer creation)
        try:
            self.ext = get_extension()
        except Exception as e:
            raise RuntimeError(
                f"[AdamW8bit_RingBuffer] CUDA extension compilation failed: {e}\n"
                "Please ensure CUDA toolkit and ninja are installed."
            )

        # Schedule-Free: cautious is incompatible (no exp_avg momentum to mask)
        if schedule_free and cautious:
            print("[AdamW8bit_RingBuffer] WARNING: cautious is disabled when schedule_free=True")
            cautious = False

        # RAdam: warmup is incompatible (automatic adaptive LR)
        if use_radam and warmup_steps > 0:
            print("[AdamW8bit_RingBuffer] WARNING: warmup_steps is ignored when use_radam=True (RAdam uses automatic adaptive LR)")
            warmup_steps = 0

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_8bit=use_8bit,
            cautious=cautious,
            schedule_free=schedule_free,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            use_radam=use_radam,
            stochastic_rounding=stochastic_rounding,
        )
        super().__init__(params, defaults)

        self.get_state_buffer = get_state_buffer
        self.step_count = 0
        self.cautious = cautious
        self.schedule_free = schedule_free
        self.use_radam = use_radam
        self.stochastic_rounding = stochastic_rounding

        # Schedule-Free specific state
        if schedule_free:
            self.k = 0  # Step counter
            self.weight_sum = 0.0  # FP32 accumulator for weighted average
            self.lr_max = 0.0  # Maximum learning rate seen
            self.train_mode = False  # Training mode flag

        # Keys that must preserve dtype (UINT8 states, FP32 absmax)
        # Based on bitsandbytes.optim.optimizer.Optimizer8bit.non_castable_tensor_keys
        self.non_castable_tensor_keys = {
            "exp_avg",      # state1 (UINT8 or FP32)
            "exp_avg_sq",   # state2 (UINT8 or FP32)
            "absmax1",      # FP32 absmax tracking for exp_avg
            "absmax2",      # FP32 absmax tracking for exp_avg_sq
            "z",            # Schedule-Free: z sequence (UINT8 or FP32)
            "absmax_z",     # FP32 absmax tracking for z
        }

        # Create quantization maps (once, shared across all parameters)
        if use_8bit:
            self._init_quantization_maps()

    def _init_quantization_maps(self):
        """Initialize quantization maps on device."""
        # Create dynamic quantization maps
        qmap_signed = create_quantization_map(signed=True)       # For exp_avg
        qmap_unsigned = create_quantization_map(signed=False)    # For exp_avg_sq

        # Initialize on device (copies to constant memory)
        self.ext.init_quantization_maps(qmap_signed, qmap_unsigned)

        print("[AdamW8bit_RingBuffer] Quantization maps initialized on device")

    @staticmethod
    def _copy_stochastic_bf16(target: torch.Tensor, source: torch.Tensor):
        """
        Stochastic rounding from FP32 to BF16.

        Based on: https://github.com/pytorch/pytorch/issues/120376

        BF16 has 7-bit mantissa (vs FP32's 23-bit), so rounding error is significant
        for small updates. Stochastic rounding removes bias in repeated small updates
        by randomly rounding up or down based on the fractional part.

        Args:
            target: Target tensor in BF16 (modified in-place)
            source: Source tensor in FP32
        """
        assert target.dtype == torch.bfloat16, f"Target must be BF16, got {target.dtype}"
        assert source.dtype == torch.float32, f"Source must be FP32, got {source.dtype}"

        # Create random 16-bit integer [0, 65536)
        # This will be added to the lower 16 bits of FP32 mantissa
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # Add random to lower 16 bits of mantissa (probabilistic rounding)
        # View as int32 to manipulate bits directly
        result.add_(source.view(dtype=torch.int32))

        # Mask off lower 16 bits (keep upper 16 bits = BF16 format)
        result.bitwise_and_(-65536)  # 0xFFFF0000 as signed int32

        # Copy upper 16 bits to target (BF16)
        # This effectively does: target = round(source) with stochastic rounding
        target.copy_(result.view(dtype=torch.float32))

    def _init_param_state(self, p: nn.Parameter):
        """Initialize optimizer state for a parameter."""
        state = self.state[p]
        group = None
        for g in self.param_groups:
            # Use id() comparison to avoid tensor shape mismatch errors
            if any(id(param) == id(p) for param in g['params']):
                group = g
                break

        if group is None:
            raise RuntimeError(f"Parameter {p.shape} not found in param_groups")

        use_8bit = group['use_8bit']
        schedule_free = group.get('schedule_free', False)

        if use_8bit:
            # ============================================================
            # 8-bit Quantized States (Ring Buffer Allocation)
            # ============================================================

            blocksize = 256  # Must match QUANTIZATION_BLOCKSIZE in CUDA kernel
            n = p.numel()
            num_blocks = (n + blocksize - 1) // blocksize

            # Allocate quantized states
            if self.get_state_buffer is not None:
                # Ring Buffer enabled: CPU allocation (for Block Swap integration)
                if schedule_free:
                    # Schedule-Free: only exp_avg_sq and z (no exp_avg)
                    state['exp_avg_sq'] = self.get_state_buffer(p, dtype=torch.uint8)

                    # Initialize z by quantizing p, then transfer to CPU
                    device = p.device if p.device.type == 'cuda' else torch.device('cuda:0')
                    z_quantized, absmax_z_init = quantize_blockwise_inplace(p.detach().clone().to(device), blocksize)

                    # Allocate CPU buffer and copy quantized z
                    state['z'] = self.get_state_buffer(p, dtype=torch.uint8)
                    state['z'].copy_(z_quantized.cpu())
                    state['_absmax_z_init'] = absmax_z_init  # Temporary storage (on GPU)

                    # Use pinned memory for faster CPU-GPU transfer
                    if hasattr(state['exp_avg_sq'], 'pin_memory'):
                        state['exp_avg_sq'] = state['exp_avg_sq'].pin_memory()
                        state['z'] = state['z'].pin_memory()
                else:
                    # Standard AdamW: exp_avg and exp_avg_sq
                    state['exp_avg'] = self.get_state_buffer(p, dtype=torch.uint8)
                    state['exp_avg_sq'] = self.get_state_buffer(p, dtype=torch.uint8)

                    # Use pinned memory for faster CPU-GPU transfer
                    if hasattr(state['exp_avg'], 'pin_memory'):
                        state['exp_avg'] = state['exp_avg'].pin_memory()
                        state['exp_avg_sq'] = state['exp_avg_sq'].pin_memory()
            else:
                # Ring Buffer disabled: GPU allocation (bitsandbytes-compatible)
                # Avoids CPU-GPU transfer overhead (~256ms/step for 350M params)
                device = p.device if p.device.type == 'cuda' else torch.device('cuda:0')

                if schedule_free:
                    # Schedule-Free: only exp_avg_sq and z (no exp_avg)
                    state['exp_avg_sq'] = torch.zeros(n, dtype=torch.uint8, device=device)

                    # Initialize z by quantizing p (z starts as a quantized copy of p)
                    z_quantized, absmax_z_init = quantize_blockwise_inplace(p.detach().clone().to(device), blocksize)
                    state['z'] = z_quantized
                    # absmax_z will be allocated later, initialized with absmax_z_init
                    state['_absmax_z_init'] = absmax_z_init  # Temporary storage
                else:
                    # Standard AdamW: exp_avg and exp_avg_sq
                    state['exp_avg'] = torch.zeros(n, dtype=torch.uint8, device=device)
                    state['exp_avg_sq'] = torch.zeros(n, dtype=torch.uint8, device=device)

            # Absmax metadata (small, ALWAYS keep on GPU even if param moves to CPU)
            # Must be on CUDA for CUDA kernel execution
            device = p.device if p.device.type == 'cuda' else torch.device('cuda:0')

            if schedule_free:
                # Schedule-Free: absmax for exp_avg_sq and z
                state['absmax2'] = torch.zeros(num_blocks, dtype=torch.float32, device=device)

                # Initialize absmax_z from quantized p (if available)
                if '_absmax_z_init' in state:
                    state['absmax_z'] = state['_absmax_z_init'].to(device)
                    del state['_absmax_z_init']  # Clean up temporary storage
                else:
                    state['absmax_z'] = torch.zeros(num_blocks, dtype=torch.float32, device=device)
            else:
                # Standard AdamW: absmax for exp_avg and exp_avg_sq
                state['absmax1'] = torch.zeros(num_blocks, dtype=torch.float32, device=device)
                state['absmax2'] = torch.zeros(num_blocks, dtype=torch.float32, device=device)

            state['is_8bit'] = True

            # Memory reporting (disabled to reduce log verbosity)
            # state_mem_mb = (n * 2) / (1024 ** 2)  # 2 bytes per element (UINT8 x2)
            # absmax_mem_mb = (num_blocks * 2 * 4) / (1024 ** 2)  # FP32 x2
            # device_str = "CPU (Ring Buffer)" if self.get_state_buffer else "CPU"
            # print(f"[AdamW8bit_RingBuffer] Allocated 8-bit states for {p.shape} "
            #       f"({state_mem_mb:.2f} MB on {device_str}, {absmax_mem_mb:.2f} MB absmax on GPU)")

        else:
            # ============================================================
            # FP32 States (Standard AdamW)
            # ============================================================

            if schedule_free:
                # Schedule-Free: only exp_avg_sq and z (no exp_avg)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['z'] = torch.clone(p, memory_format=torch.preserve_format)  # Initialize z to p
            else:
                # Standard AdamW: exp_avg and exp_avg_sq
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            state['is_8bit'] = False

            state_mem_mb = (p.numel() * 2 * p.element_size()) / (1024 ** 2)
            print(f"[AdamW8bit_RingBuffer] Allocated FP32 states for {p.shape} "
                  f"({state_mem_mb:.2f} MB on GPU)")

    def load_state_dict(self, state_dict):
        """
        Load optimizer state while preserving UINT8 dtypes.

        Based on bitsandbytes.optim.optimizer.Optimizer8bit.load_state_dict()
        Standard PyTorch load_state_dict() converts UINT8 states to parameter dtype,
        which breaks 8-bit quantization. This override preserves UINT8 dtypes.
        """
        from copy import deepcopy
        from itertools import chain
        from collections import abc as container_abcs, defaultdict

        # Deepcopy to match PyTorch standard behavior
        state_dict = deepcopy(state_dict)

        # Validate state_dict structure
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")

        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group that doesn't match the size of optimizer's group",
            )

        # Create ID mapping (old param ID -> current param object)
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        }

        def cast(param, value):
            """
            Cast value to match parameter, but preserve UINT8 dtype for optimizer states.
            Based on bitsandbytes cast() function (Line 191-211).
            """
            if isinstance(value, torch.Tensor):
                # CRITICAL: Preserve UINT8 dtype (8-bit quantized states)
                # Only convert floating-point types, never UINT8
                if param.is_floating_point() and value.dtype != torch.uint8:
                    value = value.to(param.dtype)
                # Move to parameter's device
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                # For dict values (optimizer state), protect non_castable_tensor_keys
                for k, v in value.items():
                    if k in self.non_castable_tensor_keys:
                        # Only move device, preserve dtype (UINT8 for exp_avg/exp_avg_sq)
                        if isinstance(v, torch.Tensor):
                            # absmax1/absmax2 must ALWAYS be on GPU (required by CUDA kernel)
                            if k in ('absmax1', 'absmax2'):
                                target_device = param.device if param.device.type == 'cuda' else torch.device('cuda:0')
                                value[k] = v.to(target_device)
                            else:
                                # exp_avg/exp_avg_sq can be on CPU (Ring Buffer)
                                value[k] = v.to(param.device)
                    else:
                        # Other keys: standard cast
                        value[k] = cast(param, v)
                return value
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors appropriately)
        state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups
        def update_group(group, new_group):
            new_group["params"] = group["params"]
            # Add missing keys from current defaults (for backward compatibility)
            for key in group.keys():
                if key not in new_group:
                    new_group[key] = group[key]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        # ============================================================
        # Schedule-Free: Update global state
        # ============================================================
        if self.schedule_free:
            # Ensure optimizer is in train mode
            if not self.train_mode:
                raise RuntimeError(
                    "Optimizer must be in train mode when step() is called. "
                    "Call optimizer.train() before training loop."
                )

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            use_8bit = group['use_8bit']
            schedule_free = group.get('schedule_free', False)

            # ============================================================
            # Schedule-Free: Compute learning rate schedule and weight
            # ============================================================
            if schedule_free:
                use_radam = group.get('use_radam', False)
                r = group['r']
                weight_lr_power = group['weight_lr_power']
                k = self.k

                if use_radam:
                    # ============================================================
                    # RAdam Schedule-Free: Adaptive LR via Rectified Adam
                    # ============================================================
                    # Reference: https://arxiv.org/abs/1908.03265 (RAdam paper)
                    # Based on: schedulefree/radam_schedulefree.py (Facebook Research)

                    import math

                    step = k + 1  # Use k+1 for all calculations

                    # Bias correction for second moment
                    beta2_t = beta2 ** step
                    bias_correction2 = 1 - beta2_t

                    # SMA (Simple Moving Average) length calculation
                    rho_inf = 2 / (1 - beta2) - 1  # Maximum SMA length
                    rho_t = rho_inf - 2 * step * beta2_t / bias_correction2  # Current SMA length

                    # Rectification term (adaptive LR adjustment)
                    if rho_t > 4.0:
                        # Adam mode: Use rectified adaptive LR
                        rect = math.sqrt(
                            (rho_t - 4) * (rho_t - 2) * rho_inf /
                            ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                        )
                    else:
                        # Early training phase: No parameter update (momentum only)
                        # This stabilizes training by ensuring smooth warmup
                        rect = 0.0

                    scheduled_lr = lr * rect

                    # Update lr_max (for weight calculation)
                    self.lr_max = max(scheduled_lr, self.lr_max)

                    # Bias correction for Schedule-Free
                    bias_correction2_sf = bias_correction2
                else:
                    # ============================================================
                    # AdamW Schedule-Free: Linear warmup
                    # ============================================================
                    warmup_steps = group['warmup_steps']

                    # Linear warmup (use k+1 because k increments at end of step)
                    if k < warmup_steps:
                        sched = (k + 1) / warmup_steps
                    else:
                        sched = 1.0

                    scheduled_lr = lr * sched

                    # Update lr_max
                    self.lr_max = max(scheduled_lr, self.lr_max)

                    # Bias correction (use k+1, not step_count)
                    bias_correction2_sf = 1 - beta2 ** (k + 1)

                # Compute weight for averaging (common for both AdamW and RAdam)
                weight = ((k + 1) ** r) * (self.lr_max ** weight_lr_power)
                self.weight_sum += weight

                # Averaging coefficient
                try:
                    ckp1 = weight / self.weight_sum
                except ZeroDivisionError:
                    ckp1 = 0.0
            else:
                scheduled_lr = lr
                ckp1 = 0.0
                bias_correction2_sf = None

            for p in group['params']:
                if p.grad is None:
                    continue

                # Skip parameters on CPU (offloaded by Block Swap)
                # Optimizer updates will be applied when layer returns to GPU
                if not p.is_cuda:
                    continue

                # Initialize state on first use
                if len(self.state[p]) == 0:
                    self._init_param_state(p)

                state = self.state[p]
                grad = p.grad

                # Gradient norm scaling (for gradient clipping, if applied)
                gnorm_scale = 1.0

                # ============================================================
                # Stochastic Rounding for BF16 Parameters
                # ============================================================
                # If stochastic_rounding is enabled and param is BF16:
                # 1. Create FP32 buffer
                # 2. CUDA kernel updates FP32 buffer
                # 3. Python applies stochastic rounding: FP32 → BF16
                use_stochastic_rounding = (
                    group['stochastic_rounding'] and
                    p.dtype == torch.bfloat16 and
                    use_8bit  # Only for 8-bit quantized updates
                )

                # Create FP32 buffer if needed
                if use_stochastic_rounding:
                    if 'p_fp32' not in state:
                        # Allocate FP32 buffer (same shape as param)
                        state['p_fp32'] = p.detach().clone().to(dtype=torch.float32)
                    p_fp32 = state['p_fp32']
                    p_for_kernel = p_fp32  # CUDA kernel updates FP32 buffer
                else:
                    p_for_kernel = p  # CUDA kernel updates param directly

                if use_8bit:
                    # ============================================================
                    # 8-bit Quantized Update (CUDA Kernel)
                    # ============================================================

                    if schedule_free:
                        # ============================================================
                        # Schedule-Free: Update exp_avg_sq and z, then update y (p)
                        # ============================================================

                        # Ring Buffer optimization: Ensure states are on GPU
                        z_gpu = state['z']
                        exp_avg_sq_gpu = state['exp_avg_sq']

                        if not state['z'].is_cuda:
                            # Async transfer for Ring Buffer states (pinned memory)
                            z_gpu = state['z'].cuda(non_blocking=True)
                            exp_avg_sq_gpu = state['exp_avg_sq'].cuda(non_blocking=True)

                        # Call Schedule-Free CUDA kernel
                        self.ext.adamw_8bit_schedulefree_update(
                            p_for_kernel,           # param (y, GPU) - FP32 buffer if stochastic_rounding
                            grad,                   # grad (GPU)
                            z_gpu,                  # z (UINT8, GPU/async transferred)
                            exp_avg_sq_gpu,         # exp_avg_sq (UINT8, GPU/async transferred)
                            state['absmax_z'],      # absmax_z (FP32, GPU)
                            state['absmax2'],       # absmax2 (FP32, GPU)
                            beta1,
                            beta2,
                            eps,
                            scheduled_lr,
                            weight_decay,
                            ckp1,
                            gnorm_scale,
                            bias_correction2_sf
                        )

                        # Ring Buffer: Copy updated states back to CPU
                        if not state['z'].is_cuda:
                            # Async copy back (non_blocking requires pinned memory)
                            state['z'].copy_(z_gpu, non_blocking=True)
                            state['exp_avg_sq'].copy_(exp_avg_sq_gpu, non_blocking=True)

                        # Stochastic rounding: FP32 buffer → BF16 param
                        if use_stochastic_rounding:
                            self._copy_stochastic_bf16(p, p_fp32)

                    else:
                        # ============================================================
                        # Standard AdamW 8-bit Update
                        # ============================================================

                        # Ring Buffer optimization: Ensure states are on GPU
                        # If states are on CPU (Ring Buffer), move to GPU with non_blocking=True
                        exp_avg_gpu = state['exp_avg']
                        exp_avg_sq_gpu = state['exp_avg_sq']

                        if not state['exp_avg'].is_cuda:
                            # Async transfer for Ring Buffer states (pinned memory)
                            exp_avg_gpu = state['exp_avg'].cuda(non_blocking=True)
                            exp_avg_sq_gpu = state['exp_avg_sq'].cuda(non_blocking=True)

                        self.ext.adamw_8bit_update(
                            p_for_kernel,           # param (GPU) - FP32 buffer if stochastic_rounding
                            grad,                   # grad (GPU)
                            exp_avg_gpu,            # state1 (GPU, async transferred if needed)
                            exp_avg_sq_gpu,         # state2 (GPU, async transferred if needed)
                            state['absmax1'],       # absmax1 (GPU)
                            state['absmax2'],       # absmax2 (GPU)
                            beta1,
                            beta2,
                            eps,
                            scheduled_lr,           # Use scheduled_lr instead of lr
                            weight_decay,
                            gnorm_scale,
                            self.step_count,
                            self.cautious           # Cautious masking
                        )

                        # Ring Buffer: Copy updated states back to CPU
                        if not state['exp_avg'].is_cuda:
                            # Async copy back (non_blocking requires pinned memory)
                            state['exp_avg'].copy_(exp_avg_gpu, non_blocking=True)
                            state['exp_avg_sq'].copy_(exp_avg_sq_gpu, non_blocking=True)

                        # Stochastic rounding: FP32 buffer → BF16 param
                        if use_stochastic_rounding:
                            self._copy_stochastic_bf16(p, p_fp32)

                else:
                    # ============================================================
                    # FP32 Update
                    # ============================================================
                    # Note: Even though states remain FP32, parameter updates
                    # still need stochastic rounding when writing to BF16 params

                    if schedule_free:
                        # ============================================================
                        # Schedule-Free FP32 Update
                        # ============================================================

                        # Use p_for_kernel (FP32 buffer) if stochastic rounding enabled
                        if use_stochastic_rounding:
                            y = p_fp32  # Update FP32 buffer
                        else:
                            y = p  # Update param directly

                        z = state['z']
                        exp_avg_sq = state['exp_avg_sq']

                        # Update exp_avg_sq (second moment)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                        # Bias correction for second moment (use Schedule-Free k+1)
                        denom = (exp_avg_sq / bias_correction2_sf).sqrt_().add_(eps)

                        # Normalize gradient (reuse grad buffer for memory efficiency)
                        grad_normalized = grad.div_(denom)

                        # Weight decay at y (decoupled weight decay)
                        if weight_decay > 0:
                            grad_normalized.add_(y, alpha=weight_decay)

                        # Update y (training parameters)
                        # y = (1 - ckp1) * y + ckp1 * z + lr * (beta1 * (1 - ckp1) - 1) * grad_normalized
                        y.lerp_(end=z, weight=ckp1)
                        y.add_(grad_normalized, alpha=scheduled_lr * (beta1 * (1 - ckp1) - 1))

                        # Update z (main sequence)
                        z.sub_(grad_normalized, alpha=scheduled_lr)

                        # Stochastic rounding: FP32 buffer → BF16 param
                        if use_stochastic_rounding:
                            self._copy_stochastic_bf16(p, p_fp32)

                    else:
                        # ============================================================
                        # Standard AdamW FP32 Update
                        # ============================================================

                        exp_avg = state['exp_avg']
                        exp_avg_sq = state['exp_avg_sq']

                        # Update momentum
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                        # Bias correction
                        bias_correction1 = 1 - beta1 ** self.step_count
                        bias_correction2 = 1 - beta2 ** self.step_count

                        corrected_exp_avg = exp_avg / bias_correction1
                        corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                        # AdamW update
                        denom = corrected_exp_avg_sq.sqrt().add_(eps)
                        step_size = scheduled_lr / bias_correction1

                        # Use p_for_kernel (FP32 buffer) if stochastic rounding enabled
                        if use_stochastic_rounding:
                            p_update = p_fp32  # Update FP32 buffer
                        else:
                            p_update = p  # Update param directly

                        # Decoupled weight decay
                        if weight_decay > 0:
                            p_update.mul_(1 - scheduled_lr * weight_decay)

                        # Apply update
                        p_update.addcdiv_(corrected_exp_avg, denom, value=-step_size)

                        # Stochastic rounding: FP32 buffer → BF16 param
                        if use_stochastic_rounding:
                            self._copy_stochastic_bf16(p, p_fp32)

        # Schedule-Free: Increment k after all parameter updates
        if self.schedule_free:
            self.k += 1

        return loss

    @torch.no_grad()
    def train(self):
        """
        Set optimizer to train mode (Schedule-Free).
        Sets parameters to y (training sequence): p = (1 - beta1) * z + beta1 * y
        """
        if not self.schedule_free:
            return

        for group in self.param_groups:
            beta1 = group['betas'][0]

            for p in group['params']:
                state = self.state.get(p)
                if state is None or 'z' not in state:
                    continue

                # Set p to y: p.lerp_(end=z, weight=1-beta1)
                # This is equivalent to: p = beta1 * p + (1 - beta1) * z
                p.lerp_(end=state['z'], weight=1 - beta1)

        self.train_mode = True

    @torch.no_grad()
    def eval(self):
        """
        Set optimizer to eval mode (Schedule-Free).
        Sets parameters to x (evaluation sequence): p = (1 - 1/beta1) * z + (1/beta1) * y
        """
        if not self.schedule_free:
            return

        for group in self.param_groups:
            beta1 = group['betas'][0]

            for p in group['params']:
                state = self.state.get(p)
                if state is None or 'z' not in state:
                    continue

                # Set p to x: p.lerp_(end=z, weight=1-1/beta1)
                # This is equivalent to: p = (1/beta1) * p + (1 - 1/beta1) * z
                p.lerp_(end=state['z'], weight=1 - 1 / beta1)

        self.train_mode = False


def patch_adamw8bit_ringbuffer(model: nn.Module, optimizer: AdamW8bit_RingBuffer):
    """
    Patch model to use per-parameter fused updates via post_accumulate_grad_hook.

    This allows optimizer updates to happen immediately after each parameter's
    gradient is computed, enabling pipelined execution and reduced peak VRAM.

    Args:
        model: Model to patch
        optimizer: AdamW8bit_RingBuffer optimizer instance
    """

    def create_update_hook(p: nn.Parameter):
        """Create a hook that updates this parameter immediately after grad accumulation."""

        def hook(param: nn.Parameter):
            # Skip parameters on CPU (offloaded by Block Swap)
            # Update will be applied when layer returns to GPU
            if not param.is_cuda:
                return

            # Skip if no gradient
            if param.grad is None:
                return

            # Find parameter's group
            group = None
            for g in optimizer.param_groups:
                # Use id() comparison to avoid tensor shape mismatch errors
                if any(id(p) == id(param) for p in g['params']):
                    group = g
                    break

            if group is None:
                return

            # Initialize state if needed
            if len(optimizer.state[param]) == 0:
                optimizer._init_param_state(param)

            state = optimizer.state[param]
            if not state.get('is_8bit', False):
                return  # Skip FP32 params (updated in optimizer.step())

            # Perform 8-bit update
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            gnorm_scale = 1.0

            optimizer.ext.adamw_8bit_update(
                param,
                param.grad,
                state['exp_avg'],
                state['exp_avg_sq'],
                state['absmax1'],
                state['absmax2'],
                beta1, beta2, eps, lr, weight_decay, gnorm_scale,
                optimizer.step_count + 1  # +1 because hook runs before step()
            )

            # Clear gradient (already applied)
            param.grad = None

        return hook

    # Register hooks for all parameters
    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(create_update_hook(p))

    print(f"[AdamW8bit_RingBuffer] Registered post_accumulate_grad hooks for {sum(1 for p in model.parameters() if p.requires_grad)} parameters")
