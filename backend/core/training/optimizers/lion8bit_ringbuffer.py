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
- Momentum state (exp_avg) CAN be allocated on CPU, with automatic transfer during
  the update -- but ONLY when a ``get_state_buffer`` allocator is passed in.

NOTE: no caller passes one (see AdamW8bit_RingBuffer's docstring for the full
account; the same gap applies here, and dfa7fbbf introduced this class the same
way). ``get_state_buffer`` resolves to None and ``_init_param_state`` takes its
GPU-allocation branch, so what this class delivers by default is a fused 8-bit Lion
with GPU-resident state. The implementation is complete -- the wiring is missing.
See RINGBUFFER_OPTIMIZERS.md and docs/guides/SENSENOVA_TRAINING_DESIGN.md 6.5.

- VRAM savings vs an FP32 two-state optimizer: ~87.5% (1 byte/param instead of 8).
  This one does NOT depend on CPU residency -- it is quantization plus Lion's
  single state, so the default GPU-allocation path gets it. It is arithmetic, not
  a measurement.

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

# Fused-backward hook registration (driven by optimizer.param_groups)
from .fused_backward_registration import register_fused_backward_hooks

# Gradient-norm recording (the hooks clear param.grad before it can be measured)
from .fused_grad_norm import record_fused_grad_norm, record_fused_grad_observation

# Updated-parameter census (G-RB3): which parameters an update actually reached
from .update_census import record_param_update

# Stochastic rounding helpers (shared with AdamW8bit_RingBuffer)
from .stochastic_rounding import (
    Fp32ScratchPool,
    prepare_master_and_grad,
    should_use_stochastic_rounding,
    stochastic_round_,
)


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
        stochastic_rounding: bool = False,
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

        # Schedule-Free is refused here, not just in the trainer: the trainer's
        # factory call is wrapped in `except (ValueError, ImportError)` that falls
        # back to AdamW, so a ValueError raised from inside the constructor would
        # silently substitute a different optimizer. RuntimeError is not caught
        # there, so every caller sees this.
        #
        # The defect is in lion8bit_schedulefree_kernel.cu: Schedule-Free's z is a
        # POSITION sequence, but that kernel uses z for Lion's momentum EMA and
        # then writes x = (1-ckp1)*z + ckp1*y into the parameter. ckp1 is ~1/k, so
        # the parameter becomes the momentum buffer within a few steps -- measured
        # with random gradients, corr(p, z) = 0.994 at step 5, 0.9996 at step 20.
        # A correct implementation needs both a position sequence and a momentum
        # EMA, i.e. a second 8-bit state that _init_param_state does not allocate,
        # so this is a redesign of the state layout and the checkpoint format, not
        # a patch.
        if schedule_free:
            raise RuntimeError(
                "Lion8bit_RingBuffer does not support schedule_free: its Schedule-Free CUDA "
                "kernel writes Lion's momentum EMA into the parameter instead of the "
                "Schedule-Free position sequence, which destroys the weights within a few "
                "steps. Use AdamW8bit_RingBuffer for a Schedule-Free run, or "
                "Lion8bit_RingBuffer with schedule_free=False."
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
            stochastic_rounding=stochastic_rounding,
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
        self.stochastic_rounding = stochastic_rounding

        # FP32 scratch buffers for stochastic rounding (see stochastic_rounding.py).
        # Shared by every parameter, so the cost is one buffer the size of the
        # largest parameter, not an FP32 master copy of the model.
        self._sr_scratch = Fp32ScratchPool()

        # Schedule-Free/RAdam tracking
        if self.schedule_free or self.use_radam:
            self.k = 0  # Step counter
            self.weight_sum = 0.0  # FP32 accumulator for weighted average
            self.lr_max = 0.0  # Maximum learning rate seen
            if self.schedule_free:
                self.train_mode = False  # Training mode flag

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

    def _next_rounding_seed(self) -> int:
        """A fresh seed for the kernel's stochastic quantization of z.

        Drawn from torch's own CPU generator, so ``torch.manual_seed`` still makes
        a run reproducible, and never a GPU sync.
        """
        return int(torch.randint(0, 2 ** 31 - 1, (1,), device='cpu').item())

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
                    if state['state_z'].is_cpu:
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
                    if state['exp_avg'].is_cpu:
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
            # Unquantized State (Standard Lion)
            # ============================================================
            # NOTE: zeros_like takes the PARAMETER's dtype -- for a bf16
            # parameter the momentum state is bf16, not FP32.

            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['is_8bit'] = False

            state_mem_mb = (p.numel() * p.element_size()) / (1024 ** 2)
            print(f"[Lion8bit_RingBuffer] Allocated unquantized {p.dtype} state for {p.shape} "
                  f"({state_mem_mb:.2f} MB on GPU)")

    def _load_state_dict_uint8(self, state_dict):
        """
        Load optimizer state while preserving UINT8 dtypes.

        Based on bitsandbytes.optim.optimizer.Optimizer8bit.load_state_dict()
        Standard PyTorch load_state_dict() converts UINT8 states to parameter dtype,
        which breaks 8-bit quantization. This override preserves UINT8 dtypes.
        """
        from copy import deepcopy
        from itertools import chain
        from collections import abc as container_abcs, defaultdict

        from .host_state_allocator import (
            copy_containers_only,
            is_absmax_key,
            place_loaded_state_tensor,
        )

        host_resident = self.get_state_buffer is not None

        # Deepcopy to match PyTorch standard behavior -- except under host
        # residency, where it would duplicate tens of GiB of pinned-bound state.
        state_dict = copy_containers_only(state_dict) if host_resident else deepcopy(state_dict)

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
                    if k in self.non_castable_tensor_keys or is_absmax_key(k):
                        # Preserve dtype: UINT8 for exp_avg, FP32 for every
                        # absmax* (Schedule-Free's 'absmax_z' is in neither the
                        # key set nor the uint8 arm below, and the kernel takes
                        # it as float32).
                        if isinstance(v, torch.Tensor):
                            value[k] = place_loaded_state_tensor(self, param, k, v)
                    elif host_resident and isinstance(v, torch.Tensor) and v.dtype == torch.uint8:
                        # Schedule-Free 'state_z' is not in non_castable_tensor_keys.
                        value[k] = place_loaded_state_tensor(self, param, k, v)
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

    def state_dict(self):
        """
        Override state_dict to include Schedule-Free/RAdam specific state.

        PyTorch's default Optimizer.state_dict() only saves state and param_groups,
        but Schedule-Free and RAdam need additional counters (k, weight_sum, lr_max).
        """
        state_dict = super().state_dict()

        # Add Schedule-Free/RAdam specific state
        if self.schedule_free or self.use_radam:
            state_dict['k'] = self.k
            state_dict['weight_sum'] = self.weight_sum
            state_dict['lr_max'] = self.lr_max
            if self.schedule_free:
                state_dict['train_mode'] = self.train_mode

        return state_dict

    def load_state_dict(self, state_dict):
        """
        Override load_state_dict to restore Schedule-Free/RAdam specific state.

        This calls our custom UINT8-preserving load_state_dict
        and then restores our additional counters.
        """
        # First, call our custom load_state_dict for UINT8 preservation
        self._load_state_dict_uint8(state_dict)

        # Restore Schedule-Free/RAdam specific state
        if 'k' in state_dict:
            self.k = state_dict['k']
            print(f"[Lion8bit_RingBuffer] Restored step counter k={self.k}")
        if 'weight_sum' in state_dict:
            self.weight_sum = state_dict['weight_sum']
            print(f"[Lion8bit_RingBuffer] Restored weight_sum={self.weight_sum}")
        if 'lr_max' in state_dict:
            self.lr_max = state_dict['lr_max']
            print(f"[Lion8bit_RingBuffer] Restored lr_max={self.lr_max}")
        if 'train_mode' in state_dict:
            self.train_mode = state_dict['train_mode']
            print(f"[Lion8bit_RingBuffer] Restored train_mode={self.train_mode}")

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

                # Was a silent `continue`: step() never revisits what it skips,
                # so a CPU-resident parameter went untrained for the whole run.
                # The 8-bit update is a CUDA kernel and raises; the FP32 path
                # below runs on CPU tensors. Same as AdamW8bit_RingBuffer.step().
                if group['use_8bit'] and not p.is_cuda:
                    raise RuntimeError(
                        f"Lion8bit_RingBuffer.step() reached a parameter "
                        f"{tuple(p.shape)} on {p.device} with 8-bit state. The update "
                        f"is a CUDA kernel and cannot be applied there, and step() "
                        f"does not revisit parameters, so skipping it would leave "
                        f"this parameter untrained for the whole run. Under Block "
                        f"Swap this optimizer updates through its own "
                        f"post-accumulate-grad hooks "
                        f"(register_lion8bit_fused_backward), which run while the "
                        f"parameter is still resident; reaching step() with a CPU "
                        f"parameter means those hooks were not registered. Options: "
                        f"(1) register them "
                        f"(BaseTrainer._setup_fused_backward_pass), (2) keep the "
                        f"parameters resident (blocks_to_swap=0)."
                    )

                # Initialize state if needed
                if len(self.state[p]) == 0:
                    self._init_param_state(p)

                state = self.state[p]

                # ============================================================
                # Stochastic Rounding for BF16 Parameters
                # ============================================================
                # Round-to-nearest into BF16 storage discards every update below
                # half a ULP, deterministically and forever. When enabled, the
                # update is applied to an FP32 image of the parameter which is
                # then written back with stochastic rounding, so sub-ULP updates
                # survive in expectation. The FP32 buffers are scratch shared
                # across parameters -- no per-parameter master weight is kept.
                grad = p.grad
                if should_use_stochastic_rounding(group['stochastic_rounding'], p):
                    # The 8-bit kernels read the gradient as the parameter's own
                    # dtype, so the grad must be lifted to FP32 with the master.
                    p_fp32, grad = prepare_master_and_grad(p, grad, self._sr_scratch)
                    p_for_update = p_fp32
                else:
                    p_fp32 = None
                    p_for_update = p

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

                        stochastic_z = bool(group['stochastic_rounding'])

                        # ``stochastic_z`` is gated on the flag alone, not on the
                        # parameter dtype: z lives in 8-bit codes whatever dtype
                        # the parameter has, and round-to-nearest on those codes
                        # drops every sub-quantum change to the sequence.
                        self.ext.lion_8bit_schedulefree_update(
                            p_for_update,
                            grad,
                            state_z_gpu,             # z-sequence (GPU, async transferred if needed)
                            state['absmax_z'],
                            beta1, beta2, 0.0,          # eps unused in Lion
                            scheduled_lr,               # Scheduled LR (with RAdam rect if enabled)
                            weight_decay,
                            ckp1,                       # Averaging coefficient
                            1.0,                        # gnorm_scale
                            self.cautious,              # Cautious masking
                            stochastic_z,
                            self._next_rounding_seed() if stochastic_z else 0,
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
                            p_for_update,
                            grad,
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
                    exp_avg = state['exp_avg']

                    # Interpolate: c_t = β1 * m_{t-1} + (1 - β1) * g_t
                    c_t = beta1 * exp_avg + (1 - beta1) * grad

                    # Update: sign(c_t) + weight_decay * param
                    update = torch.sign(c_t)
                    p_for_update.mul_(1 - lr * weight_decay).add_(update, alpha=-lr)

                    # Momentum EMA: m_t = β2 * m_{t-1} + (1 - β2) * g_t
                    exp_avg.mul_(beta2).add_(grad, alpha=(1 - beta2))

                # Stochastic rounding: FP32 image -> BF16 param (no-op when off)
                stochastic_round_(p, p_fp32)

                # Every branch above has applied this parameter's update (G-RB3).
                record_param_update(self, p)

        return loss


def register_lion8bit_fused_backward(optimizer, model):
    """
    Register post_accumulate_grad hooks for fused backward pass.

    Lion 8-bit optimizer updates are performed immediately after gradient accumulation,
    without waiting for optimizer.step(). This reduces memory fragmentation and
    improves performance with Block Swap.

    Hooks are registered on every parameter ``optimizer.param_groups`` holds, not
    on the parameters of ``model``: see fused_backward_registration for why the
    optimizer, and not a module walk, is the source of truth.

    Args:
        optimizer: Lion8bit_RingBuffer optimizer instance
        model: Module the hooks are checked against (may be None)
    """
    if not isinstance(optimizer, Lion8bit_RingBuffer):
        raise TypeError("Optimizer must be Lion8bit_RingBuffer")

    # No Schedule-Free gate is needed here (adamw8bit_ringbuffer has one): the
    # constructor refuses schedule_free outright, so a Schedule-Free
    # Lion8bit_RingBuffer cannot be constructed to be passed in.

    def create_update_hook(p: nn.Parameter, group: dict):
        """Create hook function for a specific parameter."""

        # The group is resolved once at registration (it comes from the iteration
        # over param_groups) instead of being searched for on every backward: the
        # previous per-hook scan was O(P) per parameter, i.e. O(P^2) per step.

        def hook(param: nn.Parameter):
            # Was a silent `return` here. Under fused backward nothing applies the
            # skipped update afterwards -- optimizer.step() is never called -- so
            # the parameter would go untrained for the whole run. Block Swap
            # evicts a block only from a later block's full_backward_hook, after
            # that block's own AccumulateGrad leaves have run, so reaching this
            # means the residency ordering broke. See patch_adamw8bit_ringbuffer.
            if not param.is_cuda:
                raise RuntimeError(
                    f"Lion8bit_RingBuffer's fused-backward hook fired for a parameter "
                    f"{tuple(param.shape)} that is on {param.device}. The 8-bit CUDA "
                    f"kernel cannot update it, and under the fused backward pass there is "
                    f"no later optimizer.step() to apply the update instead, so skipping "
                    f"would leave this parameter untrained for the whole run. Block Swap "
                    f"must keep a block resident until its backward (and its parameter "
                    f"hooks) have finished."
                )

            # Skip if no gradient
            if param.grad is None:
                return

            # Before the update: the hook clears param.grad below, and the
            # trainer's grad-norm reporting runs after the whole backward.
            record_fused_grad_norm(optimizer, param)
            record_fused_grad_observation(optimizer, param)

            # Initialize state if needed
            if len(optimizer.state[param]) == 0:
                optimizer._init_param_state(param)

            state = optimizer.state[param]
            if not state.get('is_8bit', False):
                # Registration refuses use_8bit=False groups, so this can only be
                # reached through state loaded from elsewhere. There is no
                # optimizer.step() under fused backward to update it instead.
                raise RuntimeError(
                    f"Lion8bit_RingBuffer's fused-backward hook fired for a parameter "
                    f"{tuple(param.shape)} whose optimizer state is not 8-bit. The hook "
                    f"performs the 8-bit CUDA update only, and under the fused backward "
                    f"pass there is no optimizer.step() to apply an FP32 update instead, "
                    f"so skipping would leave this parameter untrained for the whole run."
                )

            # Perform 8-bit update
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']

            # Stochastic rounding: update an FP32 image of the param, then round
            # back into BF16. Without this the fused-backward path silently
            # ignores the stochastic_rounding setting that optimizer.step() honours.
            grad = param.grad
            if should_use_stochastic_rounding(group['stochastic_rounding'], param):
                p_fp32, grad = prepare_master_and_grad(param, grad, optimizer._sr_scratch)
            else:
                p_fp32 = None

            optimizer.ext.lion_8bit_update(
                p_fp32 if p_fp32 is not None else param,
                grad,
                state['exp_avg'],
                state['absmax'],
                beta1, beta2, 0.0,  # eps unused
                lr, weight_decay, 1.0,  # gnorm_scale
                optimizer.step_count + 1,  # +1 because hook runs before step()
                optimizer.cautious          # cautious masking (matches step())
            )

            # Stochastic rounding: FP32 image -> BF16 param
            stochastic_round_(param, p_fp32)

            # The update reached this parameter (G-RB3). Recorded here rather
            # than at hook entry so an early return above is not counted.
            record_param_update(optimizer, param)

            # Clear gradient (already applied)
            param.grad = None

        return hook

    hooked, frozen = register_fused_backward_hooks(
        optimizer, model, "register_lion8bit_fused_backward", create_update_hook
    )

    print(f"[Lion8bit_RingBuffer] Registered post_accumulate_grad hooks for {hooked} "
          f"parameters (every parameter in optimizer.param_groups)")
    if frozen:
        print(f"[Lion8bit_RingBuffer] {len(frozen)} parameter(s) in param_groups have "
              f"requires_grad=False and get no hook (they receive no gradient), "
              f"e.g. {frozen[:3]}")
