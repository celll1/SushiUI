"""
Lion 8-bit Optimizer with Ring Buffer Support

Based on bitsandbytes 8-bit optimizer (MIT License)
https://github.com/TimDettmers/bitsandbytes

Lion Algorithm (Evolved Sign Momentum):
- https://arxiv.org/abs/2302.06675
- Symbolic Discovery of Optimization Algorithms (Chen et al., 2023)
- Update: sign(β1*m_{t-1} + (1-β1)*g_t) + λ*θ
- Momentum: m_t = β2*m_{t-1} + (1-β2)*g_t

Modified for SushiUI Ring Buffer integration:
- Momentum state (exp_avg) allocated on CPU via Ring Buffer
- Automatic GPU transfer during backward pass
- VRAM savings: ~87.5% for optimizer states (1 state instead of 2)

Implementation:
- Uses bitsandbytes quantization algorithm (dynamic map, CUB reduce)
- Supports FP32, FP16, BF16 parameters
- Per-parameter fused updates via register_post_accumulate_grad_hook
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Callable

# Import CUDA extension (lazy loading)
from .lion8bit_cuda import get_extension

# Import quantization map generator
from .quantization_map import create_quantization_map


class Lion8bit_RingBuffer(Optimizer):
    """
    Lion optimizer with 8-bit blockwise quantization and Ring Buffer support.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-4)
        betas: Coefficients (β1 for interpolation, β2 for momentum EMA) (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient (default: 0.0)
        use_8bit: Enable 8-bit quantization (default: True)
        cautious: Enable cautious masking (default: False)
        get_state_buffer: Callable to allocate Ring Buffer state (from RingBufferAllocator)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_8bit: bool = True,
        cautious: bool = False,
        schedule_free: bool = False,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        use_radam: bool = False,
        get_state_buffer: Optional[Callable] = None,
    ):
        # Lazy load CUDA extension
        try:
            self.ext = get_extension()
        except Exception as e:
            raise RuntimeError(
                f"[Lion8bit_RingBuffer] CUDA extension compilation failed: {e}\n"
                "Please ensure CUDA toolkit and ninja are installed."
            )

        # Schedule-Free warmup validation
        if use_radam and warmup_steps > 0:
            print("[Lion8bit_RingBuffer] WARNING: warmup_steps is ignored when use_radam=True")
            warmup_steps = 0

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            use_8bit=use_8bit,
            cautious=cautious,
            schedule_free=schedule_free,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            use_radam=use_radam,
        )
        super().__init__(params, defaults)

        self.get_state_buffer = get_state_buffer
        self.step_count = 0
        self.cautious = cautious
        self.schedule_free = schedule_free
        self.warmup_steps = warmup_steps
        self.r = r
        self.weight_lr_power = weight_lr_power
        self.use_radam = use_radam

        # Schedule-Free tracking (max scheduled LR for normalization)
        if self.schedule_free:
            self.lr_max = lr

        # Keys that must preserve dtype (UINT8 state, FP32 absmax)
        # Based on bitsandbytes.optim.optimizer.Optimizer8bit.non_castable_tensor_keys
        self.non_castable_tensor_keys = {
            "exp_avg",   # Momentum state (UINT8 or FP32)
            "absmax",    # FP32 absmax tracking for exp_avg
        }

        # Create quantization maps
        if use_8bit:
            self._init_quantization_maps()

    def _init_quantization_maps(self):
        """Initialize quantization maps on device."""
        # Create dynamic quantization map (signed, for momentum)
        qmap_signed = create_quantization_map(signed=True)

        # Initialize on device (copies to constant memory)
        self.ext.init_quantization_maps(qmap_signed)

        print("[Lion8bit_RingBuffer] Quantization maps initialized on device")

    def _init_param_state(self, p: nn.Parameter):
        """Initialize optimizer state for a parameter."""
        state = self.state[p]
        group = None
        for g in self.param_groups:
            # Use id() comparison to avoid tensor shape mismatch
            if any(id(param) == id(p) for param in g['params']):
                group = g
                break

        if group is None:
            raise RuntimeError(f"Parameter {p.shape} not found in param_groups")

        use_8bit = group['use_8bit']
        schedule_free = group.get('schedule_free', False)

        if use_8bit:
            # ============================================================
            # 8-bit Quantized State (Ring Buffer Allocation)
            # ============================================================

            blocksize = 256  # Must match QUANTIZATION_BLOCKSIZE in CUDA kernel
            n = p.numel()
            num_blocks = (n + blocksize - 1) // blocksize

            if schedule_free:
                # ============================================================
                # Schedule-Free: Allocate state_z (momentum, 1 state)
                # ============================================================

                if self.get_state_buffer is not None:
                    # Ring Buffer enabled: CPU allocation
                    state['state_z'] = self.get_state_buffer(p, dtype=torch.uint8)

                    # Use pinned memory for faster CPU-GPU transfer
                    if hasattr(state['state_z'], 'pin_memory'):
                        state['state_z'] = state['state_z'].pin_memory()
                else:
                    # Ring Buffer disabled: GPU allocation
                    device = p.device if p.device.type == 'cuda' else torch.device('cuda:0')
                    state['state_z'] = torch.zeros(n, dtype=torch.uint8, device=device)

                # Absmax for z (ALWAYS on GPU)
                device = p.device if p.device.type == 'cuda' else torch.device('cuda:0')
                state['absmax_z'] = torch.zeros(num_blocks, dtype=torch.float32, device=device)

            else:
                # ============================================================
                # Standard Lion: Allocate exp_avg (momentum)
                # ============================================================

                if self.get_state_buffer is not None:
                    # Ring Buffer enabled: CPU allocation (for Block Swap integration)
                    state['exp_avg'] = self.get_state_buffer(p, dtype=torch.uint8)

                    # Use pinned memory for faster CPU-GPU transfer
                    if hasattr(state['exp_avg'], 'pin_memory'):
                        state['exp_avg'] = state['exp_avg'].pin_memory()
                else:
                    # Ring Buffer disabled: GPU allocation (bitsandbytes-compatible)
                    # Avoids CPU-GPU transfer overhead (~128ms/step for 350M params)
                    device = p.device if p.device.type == 'cuda' else torch.device('cuda:0')
                    state['exp_avg'] = torch.zeros(n, dtype=torch.uint8, device=device)

                # Absmax metadata (small, ALWAYS keep on GPU even if param moves to CPU)
                # Must be on CUDA for CUDA kernel execution
                device = p.device if p.device.type == 'cuda' else torch.device('cuda:0')
                state['absmax'] = torch.zeros(num_blocks, dtype=torch.float32, device=device)

            state['is_8bit'] = True

            # Memory reporting (disabled to reduce log verbosity)
            # state_mem_mb = n / (1024 ** 2)  # 1 byte per element (UINT8)
            # absmax_mem_mb = (num_blocks * 4) / (1024 ** 2)  # FP32
            # device_str = "CPU (Ring Buffer)" if self.get_state_buffer else "CPU"
            # print(f"[Lion8bit_RingBuffer] Allocated 8-bit state for {p.shape} "
            #       f"({state_mem_mb:.2f} MB on {device_str}, {absmax_mem_mb:.2f} MB absmax on GPU)")

        else:
            # ============================================================
            # FP32 State (Standard Lion)
            # ============================================================

            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['is_8bit'] = False

            state_mem_mb = (p.numel() * p.element_size()) / (1024 ** 2)
            print(f"[Lion8bit_RingBuffer] Allocated FP32 state for {p.shape} "
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
            Based on bitsandbytes cast() function.
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
                        # Only move device, preserve dtype (UINT8 for exp_avg)
                        if isinstance(v, torch.Tensor):
                            # absmax must ALWAYS be on GPU (required by CUDA kernel)
                            if k == 'absmax':
                                target_device = param.device if param.device.type == 'cuda' else torch.device('cuda:0')
                                value[k] = v.to(target_device)
                            else:
                                # exp_avg can be on CPU (Ring Buffer)
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
            closure: A closure that reevaluates the model and returns the loss (optional)
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
            schedule_free = group.get('schedule_free', False)

            # ============================================================
            # Schedule-Free: Compute learning rate schedule and ckp1
            # ============================================================
            if schedule_free:
                use_radam = group.get('use_radam', False)
                r = group['r']
                weight_lr_power = group['weight_lr_power']
                k = self.step_count - 1  # k starts from 0

                if use_radam:
                    # ============================================================
                    # RAdam Schedule-Free: Adaptive LR via Rectified Adam
                    # ============================================================
                    import math

                    step = k + 1  # Use k+1 for all calculations

                    # Bias correction for second moment (Lion doesn't have 2nd moment,
                    # but we use beta2 for SMA calculation like AdamW)
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
                        rect = 0.0

                    scheduled_lr = lr * rect

                    # Update lr_max (for weight calculation)
                    self.lr_max = max(scheduled_lr, self.lr_max)
                else:
                    # ============================================================
                    # Lion Schedule-Free: Linear warmup
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

                # Compute weight for averaging (common for both Lion and RAdam)
                if not hasattr(self, 'weight_sum'):
                    self.weight_sum = 0.0

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

            for p in group['params']:
                if p.grad is None:
                    continue

                # Skip parameters on CPU (offloaded by Block Swap)
                if not p.is_cuda:
                    continue

                # Initialize state if needed
                if len(self.state[p]) == 0:
                    self._init_param_state(p)

                state = self.state[p]

                # 8-bit update
                if state.get('is_8bit', False):
                    if schedule_free:
                        # ============================================================
                        # Schedule-Free 8-bit Update (CUDA Kernel)
                        # ============================================================

                        # Ring Buffer optimization: Ensure state is on GPU
                        state_z_gpu = state['state_z']

                        if not state['state_z'].is_cuda:
                            # Async transfer for Ring Buffer state (pinned memory)
                            state_z_gpu = state['state_z'].cuda(non_blocking=True)

                        self.ext.lion_8bit_schedulefree_update(
                            p,
                            p.grad,
                            state_z_gpu,                # z-sequence (GPU, async transferred if needed)
                            state['absmax_z'],
                            beta1, beta2, 0.0,          # eps unused in Lion
                            scheduled_lr,               # Scheduled LR (with RAdam rect if enabled)
                            weight_decay,
                            ckp1,                       # Averaging coefficient
                            1.0,                        # gnorm_scale
                            self.cautious               # Cautious masking
                        )

                        # Ring Buffer: Copy updated state back to CPU
                        if not state['state_z'].is_cuda:
                            # Async copy back (non_blocking requires pinned memory)
                            state['state_z'].copy_(state_z_gpu, non_blocking=True)
                    else:
                        # ============================================================
                        # Standard 8-bit Update (CUDA Kernel)
                        # ============================================================

                        # Ring Buffer optimization: Ensure state is on GPU
                        exp_avg_gpu = state['exp_avg']

                        if not state['exp_avg'].is_cuda:
                            # Async transfer for Ring Buffer state (pinned memory)
                            exp_avg_gpu = state['exp_avg'].cuda(non_blocking=True)

                        self.ext.lion_8bit_update(
                            p,
                            p.grad,
                            exp_avg_gpu,                # state (GPU, async transferred if needed)
                            state['absmax'],
                            beta1, beta2, 0.0,          # eps unused in Lion
                            lr, weight_decay, 1.0,      # gnorm_scale
                            self.step_count,
                            self.cautious               # Cautious masking
                        )

                        # Ring Buffer: Copy updated state back to CPU
                        if not state['exp_avg'].is_cuda:
                            # Async copy back (non_blocking requires pinned memory)
                            state['exp_avg'].copy_(exp_avg_gpu, non_blocking=True)
                else:
                    # FP32 fallback (standard Lion)
                    grad = p.grad.data

                    exp_avg = state['exp_avg']

                    # Interpolate: c_t = β1 * m_{t-1} + (1 - β1) * g_t
                    c_t = beta1 * exp_avg + (1 - beta1) * grad

                    # Update: sign(c_t) + weight_decay * param
                    update = torch.sign(c_t)
                    p.data.mul_(1 - lr * weight_decay).add_(update, alpha=-lr)

                    # Momentum EMA: m_t = β2 * m_{t-1} + (1 - β2) * g_t
                    exp_avg.mul_(beta2).add_(grad, alpha=(1 - beta2))

        return loss


def register_lion8bit_fused_backward(optimizer, model):
    """
    Register post_accumulate_grad hooks for fused backward pass.

    Lion 8-bit optimizer updates are performed immediately after gradient accumulation,
    without waiting for optimizer.step(). This reduces memory fragmentation and
    improves performance with Block Swap.

    Args:
        optimizer: Lion8bit_RingBuffer optimizer instance
        model: PyTorch model
    """
    if not isinstance(optimizer, Lion8bit_RingBuffer):
        raise TypeError("Optimizer must be Lion8bit_RingBuffer")

    def create_update_hook(param: nn.Parameter):
        """Create hook function for a specific parameter."""

        def hook(param: nn.Parameter):
            # Skip parameters on CPU (offloaded by Block Swap)
            if not param.is_cuda:
                return

            # Skip if no gradient
            if param.grad is None:
                return

            # Find parameter's group (use id() comparison)
            group = None
            for g in optimizer.param_groups:
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
                return  # Skip FP32 params

            # Perform 8-bit update
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']

            optimizer.ext.lion_8bit_update(
                param,
                param.grad,
                state['exp_avg'],
                state['absmax'],
                beta1, beta2, 0.0,  # eps unused
                lr, weight_decay, 1.0,  # gnorm_scale
                optimizer.step_count + 1  # +1 because hook runs before step()
            )

            # Clear gradient (already applied)
            param.grad = None

        return hook

    # Register hooks for all parameters
    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(create_update_hook(p))

    print(f"[Lion8bit_RingBuffer] Registered post_accumulate_grad hooks for {sum(1 for p in model.parameters() if p.requires_grad)} parameters")
