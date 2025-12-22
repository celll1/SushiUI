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

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_8bit=use_8bit,
        )
        super().__init__(params, defaults)

        self.get_state_buffer = get_state_buffer
        self.step_count = 0

        # Keys that must preserve dtype (UINT8 states, FP32 absmax)
        # Based on bitsandbytes.optim.optimizer.Optimizer8bit.non_castable_tensor_keys
        self.non_castable_tensor_keys = {
            "exp_avg",      # state1 (UINT8 or FP32)
            "exp_avg_sq",   # state2 (UINT8 or FP32)
            "absmax1",      # FP32 absmax tracking for exp_avg
            "absmax2",      # FP32 absmax tracking for exp_avg_sq
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

        if use_8bit:
            # ============================================================
            # 8-bit Quantized States (Ring Buffer Allocation)
            # ============================================================

            blocksize = 256  # Must match QUANTIZATION_BLOCKSIZE in CUDA kernel
            n = p.numel()
            num_blocks = (n + blocksize - 1) // blocksize

            # Allocate quantized states on CPU (via Ring Buffer)
            if self.get_state_buffer is not None:
                state['exp_avg'] = self.get_state_buffer(p, dtype=torch.uint8)
                state['exp_avg_sq'] = self.get_state_buffer(p, dtype=torch.uint8)
            else:
                # Fallback: CPU allocation without Ring Buffer
                state['exp_avg'] = torch.zeros(n, dtype=torch.uint8, device='cpu')
                state['exp_avg_sq'] = torch.zeros(n, dtype=torch.uint8, device='cpu')

            # Absmax metadata (small, ALWAYS keep on GPU even if param moves to CPU)
            # Must be on CUDA for CUDA kernel execution
            device = p.device if p.device.type == 'cuda' else torch.device('cuda:0')
            state['absmax1'] = torch.zeros(num_blocks, dtype=torch.float32, device=device)
            state['absmax2'] = torch.zeros(num_blocks, dtype=torch.float32, device=device)

            state['is_8bit'] = True

            # Memory reporting
            state_mem_mb = (n * 2) / (1024 ** 2)  # 2 bytes per element (UINT8 x2)
            absmax_mem_mb = (num_blocks * 2 * 4) / (1024 ** 2)  # FP32 x2
            device_str = "CPU (Ring Buffer)" if self.get_state_buffer else "CPU"

            print(f"[AdamW8bit_RingBuffer] Allocated 8-bit states for {p.shape} "
                  f"({state_mem_mb:.2f} MB on {device_str}, {absmax_mem_mb:.2f} MB absmax on GPU)")

        else:
            # ============================================================
            # FP32 States (Standard AdamW)
            # ============================================================

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

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            use_8bit = group['use_8bit']

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

                if use_8bit:
                    # ============================================================
                    # 8-bit Quantized Update (CUDA Kernel)
                    # ============================================================

                    self.ext.adamw_8bit_update(
                        p,                      # param (GPU)
                        grad,                   # grad (GPU)
                        state['exp_avg'],       # state1 (CPU or GPU)
                        state['exp_avg_sq'],    # state2 (CPU or GPU)
                        state['absmax1'],       # absmax1 (GPU)
                        state['absmax2'],       # absmax2 (GPU)
                        beta1,
                        beta2,
                        eps,
                        lr,
                        weight_decay,
                        gnorm_scale,
                        self.step_count
                    )

                else:
                    # ============================================================
                    # FP32 Update (Standard PyTorch AdamW)
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
                    step_size = lr / bias_correction1

                    # Decoupled weight decay
                    if weight_decay > 0:
                        p.mul_(1 - lr * weight_decay)

                    # Apply update
                    p.addcdiv_(corrected_exp_avg, denom, value=-step_size)

        return loss


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
