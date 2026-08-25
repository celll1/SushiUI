"""
AdamW 8-bit Optimizer with Ring Buffer Support

Based on bitsandbytes 8-bit optimizer (MIT License)
https://github.com/TimDettmers/bitsandbytes

Modified for SushiUI Ring Buffer integration:
- Optimizer states (exp_avg, exp_avg_sq) CAN be allocated on CPU, with automatic
  transfer during the update -- but ONLY when a ``get_state_buffer`` allocator is
  passed to the constructor.

NOTE: no caller passes one. ``optimizer_factory`` forwards
``kwargs.get("get_state_buffer", None)`` and nothing supplies it (never has, since
190c876e), so ``get_state_buffer`` resolves to None and ``_init_param_state`` takes
its "Ring Buffer disabled: GPU allocation (bitsandbytes-compatible)" branch. What
this class delivers by default is therefore a fused 8-bit AdamW with GPU-resident
state: the 8-bit quantization saving is real, the CPU residency this file is named
for is not wired up. The implementation is complete -- the wiring is what is
missing. See RINGBUFFER_OPTIMIZERS.md and, for the work needed to enable it,
docs/guides/SENSENOVA_TRAINING_DESIGN.md section 6.5.

The "~75% VRAM savings for optimizer states" this docstring used to claim is
arithmetic from RINGBUFFER_OPTIMIZERS.md's hypothetical 350M-parameter table, not a
measurement, and in that table 75% is the 8-bit-vs-FP32 figure (the branch that
actually runs); the CPU-state figure there is 99.6%.

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

# Fused-backward hook registration (driven by optimizer.param_groups)
from .fused_backward_registration import register_fused_backward_hooks

# Gradient-norm recording (the hooks clear param.grad before it can be measured)
from .fused_grad_norm import record_fused_grad_norm

# Updated-parameter census (G-RB3): which parameters an update actually reached
from .update_census import record_param_update

# Stochastic rounding helpers (shared with Lion8bit_RingBuffer)
from .stochastic_rounding import (
    Fp32ScratchPool,
    copy_stochastic_bf16,
    prepare_master_and_grad,
    should_use_stochastic_rounding,
    stochastic_round_,
)


def quantize_blockwise_inplace(tensor: torch.Tensor, blocksize: int = 256):
    """
    Quantize a tensor to UINT8 using blockwise quantization (for z initialization).

    The code assigned to each element is the NEAREST entry of the signed dynamic
    quantization map -- the same map ``dequantize_value()`` in the CUDA kernels
    reads back through (``d_qmap_signed``), and the same nearest-neighbour rule
    its ``quantize_value()`` applies.

    This used to write LINEAR codes (``(x/absmax + 1) * 127.5``) while the kernel
    decoded them through the dynamic map, so the Schedule-Free z sequence started
    the run at a value unrelated to the parameter it is initialised from: measured
    on a real Krea 2 tensor, mean |z - p| was 2.34e-2 against a mean |p| of
    3.32e-2 (70% relative error) instead of the 7.1e-4 (2%) the 8-bit grid can
    actually represent.

    Args:
        tensor: Input tensor (FP16/BF16/FP32), on any device
        blocksize: Block size for quantization (default: 256)

    Returns:
        quantized: UINT8 tensor, flat [numel]
        absmax: FP32 absmax values per block [num_blocks]
    """
    n = tensor.numel()
    num_blocks = (n + blocksize - 1) // blocksize

    # Vectorised over blocks. The previous per-block Python loop issued a handful
    # of CUDA ops per 256 elements -- 885k iterations for Krea 2's largest tensor.
    flat = tensor.detach().reshape(-1).float()
    pad = num_blocks * blocksize - n
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(num_blocks, blocksize)

    qmap = create_quantization_map(signed=True).to(device=flat.device, dtype=torch.float32)

    # Symmetric headroom, matching the kernels: the signed map ends at +1.0 but
    # at -0.992968738, so a block whose extreme is negative cannot be stored at
    # its own absmax and comes back 0.7031% smaller -- which the kernel then
    # adopts as the new absmax, compounding once per step. Scaling by the largest
    # magnitude representable in BOTH directions makes the extreme exact either
    # way. See adamw8bit_schedulefree_kernel.cu's note.
    qmax_symmetric = torch.minimum(-qmap[0], qmap[-1])

    absmax = blocks.abs().amax(dim=1) / qmax_symmetric
    scale = torch.where(absmax > 0, absmax, torch.ones_like(absmax)).unsqueeze(1)
    normalized = (blocks / scale).clamp(-1.0, 1.0).reshape(-1)
    upper = torch.searchsorted(qmap, normalized.contiguous()).clamp(1, qmap.numel() - 1)
    lower = upper - 1
    take_lower = (normalized - qmap[lower]) <= (qmap[upper] - normalized)
    codes = torch.where(take_lower, lower, upper).to(torch.uint8)

    return codes[:n].contiguous(), absmax


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

        # FP32 scratch buffers for stochastic rounding (see stochastic_rounding.py).
        # Shared by every parameter, so the cost is one buffer the size of the
        # largest parameter, not an FP32 master copy of the model.
        self._sr_scratch = Fp32ScratchPool()

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

        Thin wrapper over ``stochastic_rounding.copy_stochastic_bf16`` (kept so
        existing call sites and any external references keep working).

        Args:
            target: Target tensor in BF16 (modified in-place)
            source: Source tensor in FP32
        """
        copy_stochastic_bf16(target, source)

    def _next_rounding_seed(self) -> int:
        """A fresh seed for the kernel's stochastic quantization of z.

        Drawn from torch's own CPU generator, so ``torch.manual_seed`` still
        makes a run reproducible, and never a GPU sync.
        """
        return int(torch.randint(0, 2 ** 31 - 1, (1,), device='cpu').item())

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

                    # Pin only CPU buffers (a get_state_buffer may return a GPU
                    # tensor for resident params -> partial residency; pinning a
                    # CUDA tensor would raise).
                    if state['exp_avg_sq'].is_cpu:
                        state['exp_avg_sq'] = state['exp_avg_sq'].pin_memory()
                    if state['z'].is_cpu:
                        state['z'] = state['z'].pin_memory()
                else:
                    # Standard AdamW: exp_avg and exp_avg_sq
                    state['exp_avg'] = self.get_state_buffer(p, dtype=torch.uint8)
                    state['exp_avg_sq'] = self.get_state_buffer(p, dtype=torch.uint8)

                    # Pin only CPU buffers (a get_state_buffer may return a GPU
                    # tensor for resident params -> partial residency; pinning a
                    # CUDA tensor would raise).
                    if state['exp_avg'].is_cpu:
                        state['exp_avg'] = state['exp_avg'].pin_memory()
                    if state['exp_avg_sq'].is_cpu:
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
            # Unquantized States (Standard AdamW)
            # ============================================================
            # NOTE: these are allocated with zeros_like/clone, so they take the
            # PARAMETER's dtype -- for a bf16 parameter they are bf16, not FP32.
            # The Schedule-Free 'z' sequence is therefore bf16 storage; step()
            # updates it through an FP32 image and writes it back with stochastic
            # rounding when that is enabled, so sub-ULP updates to the sequence
            # survive. exp_avg_sq is still accumulated in the parameter's dtype:
            # it is a second moment whose per-step relative change is (1-beta2),
            # which biases the denominator but not the step's direction.

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
            print(f"[AdamW8bit_RingBuffer] Allocated unquantized {p.dtype} states for {p.shape} "
                  f"({state_mem_mb:.2f} MB on GPU)")

    def _advance_param_step(self, state: dict) -> int:
        """Advance and return this parameter's own step, for bias correction.

        The fused-backward hook fires once per PARAMETER, so a global counter
        incremented there would advance P times per optimizer step. Same idiom as
        ``adamw8bit_fused`` / ``adafactor_fused``, which keep ``state['step']``.

        The fallback is the global counter, not 0, so state restored from a
        checkpoint written before this key existed -- or converted from another
        8-bit implementation, where BaseTrainer carries ``step_count`` -- resumes
        at the right step instead of restarting bias correction.
        """
        step = int(state.get('step', self.step_count)) + 1
        state['step'] = step
        return step

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

    def state_dict(self):
        """
        Override state_dict to include Schedule-Free/RAdam specific state.

        PyTorch's default Optimizer.state_dict() only saves state and param_groups,
        but Schedule-Free and RAdam need additional counters (k, weight_sum, lr_max).
        """
        state_dict = super().state_dict()

        # Adam's bias correction in step() is driven by this counter and nothing
        # else, so leaving it out restarted every ordinary resume at step 1 --
        # 1/(1-beta1) times the intended first update, decaying over ~1/(1-beta2)
        # steps. The per-parameter state['step'] the fused hook keeps is already
        # serialized (it lives in self.state); this is the step() path's half.
        state_dict['step_count'] = self.step_count

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

        # Absent from checkpoints written before step_count was serialized, and
        # from BaseTrainer's prefix-preserving partial load, which rebuilds the
        # dict from state + param_groups only. Both keep the current value, which
        # is what a cross-implementation conversion sets (see _advance_param_step).
        if 'step_count' in state_dict:
            self.step_count = int(state_dict['step_count'])
            print(f"[AdamW8bit_RingBuffer] Restored step_count={self.step_count}")

        # Restore Schedule-Free/RAdam specific state
        if 'k' in state_dict:
            self.k = state_dict['k']
            print(f"[AdamW8bit_RingBuffer] Restored step counter k={self.k}")
        if 'weight_sum' in state_dict:
            self.weight_sum = state_dict['weight_sum']
            print(f"[AdamW8bit_RingBuffer] Restored weight_sum={self.weight_sum}")
        if 'lr_max' in state_dict:
            self.lr_max = state_dict['lr_max']
            print(f"[AdamW8bit_RingBuffer] Restored lr_max={self.lr_max}")
        if 'train_mode' in state_dict:
            self.train_mode = state_dict['train_mode']
            print(f"[AdamW8bit_RingBuffer] Restored train_mode={self.train_mode}")

        self._repair_degenerate_schedule_free_state()

    @torch.no_grad()
    def _repair_degenerate_schedule_free_state(self):
        """Re-seed a Schedule-Free ``z`` that was written by the broken kernel.

        Before the Schedule-Free kernels' ``__constant__`` quantization map was
        initialised, every ``dequantize_value()`` in them returned 0, so a run
        wrote out ``z`` codes that are all one value with ``absmax_z`` at ~0.
        Reading that back with a working map decodes z as zero everywhere -- and
        z is not inert: ``y = (1 - ckp1) * y + ckp1 * z`` then pulls the weights
        toward zero on every step. Measured on such a checkpoint, 300
        zero-gradient steps after resume took mean|p| from 1.63e-2 to 5.21e-5.

        The state carries no information (it decodes to a constant), so it is
        re-seeded from the current parameter exactly as a fresh run would --
        ``z_0 = p`` is the Schedule-Free initial condition -- rather than
        refusing the resume and discarding exp_avg_sq and the step counters with
        it. Loudly, because it is a change of state the user did not ask for.

        Both signatures are required, not either: a constant tensor is perfectly
        normal (an all-ones RMSNorm weight at initialisation, a zero-initialised
        LoRA B, a zero bias), and such a tensor's z is a single repeated code with
        a healthy absmax. Requiring the decoded z to also DISAGREE with the
        parameter keeps the repair (and its warning) off those.
        """
        if not self.schedule_free:
            return

        repaired = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p)
                if not state or 'z' not in state or not state.get('is_8bit', False):
                    continue

                absmax_z = state.get('absmax_z')
                codes = state['z']
                if absmax_z is None or codes.numel() == 0:
                    continue

                # Signature 1: the decoded z is a constant. Checked per block,
                # because absmax is per block: a single repeated code still
                # decodes to different values in blocks with different scales.
                blocks = codes.reshape(-1)
                pad = (-blocks.numel()) % 256
                if pad:
                    blocks = torch.cat([blocks, blocks[-1:].expand(pad)])
                blocks = blocks.view(-1, 256)
                constant_codes = bool((blocks == blocks[:, :1]).all())
                zero_scale = not bool((absmax_z != 0).any())
                if not (constant_codes or zero_scale):
                    continue

                # Signature 2: that constant is not what z should hold. A healthy
                # z of a constant-valued parameter decodes back to the parameter.
                z_decoded = self._z_dense(p, state).to(dtype=torch.float32)
                reference = p.detach().float()
                scale = reference.abs().mean().item()
                if (z_decoded - reference).abs().mean().item() <= 0.05 * scale:
                    continue

                z_quantized, absmax_z_init = quantize_blockwise_inplace(p.detach(), 256)
                state['z'].copy_(z_quantized.to(state['z'].device))
                state['absmax_z'] = absmax_z_init.to(
                    device=absmax_z.device, dtype=torch.float32
                )
                repaired += 1

        if repaired:
            print(f"[AdamW8bit_RingBuffer] WARNING: {repaired} Schedule-Free 'z' tensor(s) in this "
                  f"checkpoint decode to a constant -- they were written before the Schedule-Free "
                  f"quantization map was initialised, and resuming from them drives the weights to "
                  f"zero. Re-seeded z from the current parameters (z_0 = p, the Schedule-Free "
                  f"initial condition). The second moment and the step counters are kept.")

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

                # Was a silent `continue`: step() never revisits what it skips,
                # so a CPU-resident parameter went untrained for the whole run.
                # The 8-bit update is a CUDA kernel and raises; the FP32 path
                # below runs on CPU tensors, so it is no longer skipped.
                if use_8bit and not p.is_cuda:
                    raise RuntimeError(
                        f"AdamW8bit_RingBuffer.step() reached a parameter "
                        f"{tuple(p.shape)} on {p.device} with 8-bit state. The update "
                        f"is a CUDA kernel and cannot be applied there, and step() "
                        f"does not revisit parameters, so skipping it would leave "
                        f"this parameter untrained for the whole run. Under Block "
                        f"Swap this optimizer updates through its own "
                        f"post-accumulate-grad hooks (patch_adamw8bit_ringbuffer), "
                        f"which run while the parameter is still resident; reaching "
                        f"step() with a CPU parameter means those hooks were not "
                        f"registered. Options: (1) register them "
                        f"(BaseTrainer._setup_fused_backward_pass), (2) keep the "
                        f"parameters resident (blocks_to_swap=0)."
                    )

                # Initialize state on first use
                if len(self.state[p]) == 0:
                    self._init_param_state(p)

                state = self.state[p]
                grad = p.grad

                # step() drives bias correction from the global counter (below);
                # record it so a run that later takes the fused-hook path -- e.g.
                # resumed with Block Swap toggled on -- continues from here
                # instead of restarting at step 1.
                state['step'] = max(int(state.get('step', 0)), self.step_count)

                # Gradient norm scaling (for gradient clipping, if applied)
                gnorm_scale = 1.0

                # ============================================================
                # Stochastic Rounding for BF16 Parameters
                # ============================================================
                # If stochastic_rounding is enabled and param is BF16:
                # 1. Materialise an FP32 image of the param in scratch memory
                # 2. The update (CUDA kernel or FP32 path) is applied to it
                # 3. It is written back to the BF16 param with stochastic rounding
                # The scratch buffers are shared across parameters, so this costs
                # one buffer the size of the largest parameter -- not an FP32
                # master copy of the model. See stochastic_rounding.py.
                use_stochastic_rounding = should_use_stochastic_rounding(
                    group['stochastic_rounding'], p
                )

                if use_stochastic_rounding:
                    # The 8-bit kernels require param.dtype == grad.dtype, and
                    # autograd hands us a BF16 grad for a BF16 param, so the
                    # gradient has to be lifted to FP32 alongside the master.
                    p_fp32, grad = prepare_master_and_grad(p, grad, self._sr_scratch)
                    p_for_kernel = p_fp32
                else:
                    p_fp32 = None
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

                        stochastic_z = bool(group['stochastic_rounding'])

                        # Call Schedule-Free CUDA kernel.
                        # ``stochastic_z`` is gated on the flag alone, not on the
                        # parameter dtype: z lives in 8-bit codes whatever the
                        # parameter's dtype is, and its quantization step is far
                        # coarser than a BF16 ULP, so round-to-nearest pins the
                        # codes of the optimization sequence for the whole run.
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
                            bias_correction2_sf,
                            stochastic_z,
                            self._next_rounding_seed() if stochastic_z else 0,
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

                        # z carries the Schedule-Free optimization sequence and is
                        # allocated with clone(p), so for a BF16 parameter it is BF16
                        # storage. ``z.sub_()`` below then rounds to nearest, which
                        # discards every update under half a ULP -- deterministically,
                        # so those elements of z are frozen for the whole run. That is
                        # the same defect stochastic rounding fixes for the parameter,
                        # on the tensor that actually drives the trajectory. Apply the
                        # update to a pooled FP32 image of z and round it back
                        # stochastically (scratch, so no persistent 4-byte-per-element
                        # master; see stochastic_rounding.py).
                        #
                        # It also makes y's ``lerp_`` well typed: with stochastic
                        # rounding on, y is the FP32 image of p, and ``y.lerp_(end=z)``
                        # against a BF16 z raised "expected dtype float for `end`" --
                        # i.e. Schedule-Free + stochastic rounding could not complete a
                        # single step.
                        if use_stochastic_rounding and z.dtype == torch.bfloat16:
                            z_master = self._sr_scratch.copy_of('z', z)
                        else:
                            z_master = None
                        z_for_update = z if z_master is None else z_master

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
                        y.lerp_(end=z_for_update, weight=ckp1)
                        y.add_(grad_normalized, alpha=scheduled_lr * (beta1 * (1 - ckp1) - 1))

                        # Update z (main sequence)
                        z_for_update.sub_(grad_normalized, alpha=scheduled_lr)

                        # Stochastic rounding: FP32 images → BF16 storage
                        if z_master is not None:
                            self._copy_stochastic_bf16(z, z_master)
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

                        # AdamW update. step_size is the plain LR: bias
                        # correction 1 is already in corrected_exp_avg above, and
                        # dividing by it again here applied it TWICE -- a step
                        # oversized by 1/bias_correction1, which is 10.0x at step
                        # 1 and still 2.44x at step 5. Measured over 5 CPU steps
                        # at lr=1e-2: max|diff| against torch.optim.AdamW was
                        # 1.874e-01 at a parameter scale of 2.97, and 1.19e-07
                        # against a closed form that divides twice. The 8-bit
                        # CUDA kernel this optimizer normally runs never had it
                        # (adamw8bit_kernel.cu:230-241 divides once), so the two
                        # paths of the same optimizer disagreed.
                        denom = corrected_exp_avg_sq.sqrt().add_(eps)
                        step_size = scheduled_lr

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

                # Every branch above has applied this parameter's update (G-RB3).
                record_param_update(self, p)

        # Schedule-Free: Increment k after all parameter updates
        if self.schedule_free:
            self.k += 1

        return loss

    def _signed_qmap(self, device: torch.device) -> torch.Tensor:
        """The signed dynamic quantization map, cached per device."""
        cache = getattr(self, '_qmap_signed_by_device', None)
        if cache is None:
            cache = {}
            self._qmap_signed_by_device = cache
        qmap = cache.get(device)
        if qmap is None:
            qmap = create_quantization_map(signed=True).to(device)
            cache[device] = qmap
        return qmap

    def _z_dense(self, p: nn.Parameter, state: dict) -> torch.Tensor:
        """z for this parameter as a dense tensor shaped like ``p``.

        In 8-bit mode ``state['z']`` holds blockwise-quantized UINT8 CODES, not
        values, so the raw tensor cannot be used as the ``end`` of a lerp -- doing
        so raised ``expected dtype struct c10::BFloat16 for 'end' but got dtype
        unsigned char``, i.e. train()/eval() could not run at all once the state
        existed. Dequantize with the same signed map the CUDA kernel uses.

        Materialises a dense FP32 copy (plus an int64 index temporary), so it is
        for the train()/eval() mode switches, which happen outside the step loop
        -- not for the per-step update, which stays inside the kernel.
        """
        z = state['z']
        if not state.get('is_8bit', False):
            return z

        blocksize = 256
        qmap = self._signed_qmap(z.device)
        absmax = state['absmax_z'].to(device=z.device, dtype=torch.float32)
        values = qmap[z.reshape(-1).long()]
        scales = absmax.repeat_interleave(blocksize)[: values.numel()]
        return (values * scales).view(p.shape)

    def _lerp_param_toward_z(self, p: nn.Parameter, z_dense: torch.Tensor,
                             weight: float, group: dict) -> None:
        """``p = (1 - weight) * p + weight * z``, stochastically rounded when asked.

        These writes land in the parameter's own storage exactly like the ones in
        step(), so under round-to-nearest they drop every sub-half-ULP move of the
        train/eval sequence.
        """
        if should_use_stochastic_rounding(group.get('stochastic_rounding', False), p):
            master = self._sr_scratch.copy_of('master', p.data)
            master.lerp_(end=self._sr_scratch.copy_of('z', z_dense), weight=weight)
            copy_stochastic_bf16(p.data, master)
        else:
            p.data.lerp_(end=z_dense.to(p.dtype), weight=weight)

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
                self._lerp_param_toward_z(p, self._z_dense(p, state), 1 - beta1, group)

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
                self._lerp_param_toward_z(p, self._z_dense(p, state), 1 - 1 / beta1, group)

        self.train_mode = False


def patch_adamw8bit_ringbuffer(model: Optional[nn.Module], optimizer: AdamW8bit_RingBuffer):
    """
    Patch model to use per-parameter fused updates via post_accumulate_grad_hook.

    This allows optimizer updates to happen immediately after each parameter's
    gradient is computed, enabling pipelined execution and reduced peak VRAM.

    Hooks are registered on every parameter ``optimizer.param_groups`` holds, not
    on the parameters of ``model``: the optimizer is what has to update them, and
    the trainer adds text-encoder / vision-encoder groups to the same optimizer
    while passing only the transformer here (see fused_backward_registration).
    ``model`` still supplies parameter names and the check that none of ITS
    trainable parameters is missing from the optimizer.

    Args:
        model: Module the hooks are checked against (may be None)
        optimizer: AdamW8bit_RingBuffer optimizer instance
    """

    # The hooks below run the STANDARD 8-bit update: they read state['exp_avg'] /
    # state['absmax1'], which _init_param_state does not allocate in Schedule-Free
    # mode (it allocates z / absmax_z instead), so the first backward would raise
    # KeyError('exp_avg') from inside the autograd engine. BaseTrainer already
    # refuses this combination before registration; this is the second gate, for
    # the direct callers of a module-level public function.
    if getattr(optimizer, 'schedule_free', False):
        raise RuntimeError(
            "patch_adamw8bit_ringbuffer does not support a Schedule-Free optimizer: the "
            "per-parameter hooks implement the standard AdamW update and read exp_avg / "
            "absmax1, which are not allocated in Schedule-Free mode. Use "
            "schedule_free=False for the fused-backward (Block Swap) path, or call "
            "optimizer.step() instead of registering these hooks."
        )

    def create_update_hook(p: nn.Parameter, group: dict):
        """Create a hook that updates this parameter immediately after grad accumulation."""

        # The group is resolved once at registration (it comes from the iteration
        # over param_groups) instead of being searched for on every backward: the
        # previous per-hook scan was O(P) per parameter, i.e. O(P^2) per step.

        def hook(param: nn.Parameter):
            # This used to `return` with a comment promising the update would be
            # applied "when the layer returns to GPU". Nothing applies it: under
            # fused backward the trainer never calls optimizer.step(), so the
            # parameter would be skipped on every step of the run, with its grad
            # left in place. Block Swap is built so this cannot happen -- a block
            # is evicted from a LATER block's full_backward_hook, i.e. after its
            # own backward and its AccumulateGrad leaves have run, and
            # wait_for_block() makes a block resident before its backward -- so
            # reaching here means that ordering broke, which is worth a crash
            # rather than a silently untrained tensor.
            if not param.is_cuda:
                raise RuntimeError(
                    f"AdamW8bit_RingBuffer's fused-backward hook fired for a parameter "
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

            # Initialize state if needed
            if len(optimizer.state[param]) == 0:
                optimizer._init_param_state(param)

            state = optimizer.state[param]
            if not state.get('is_8bit', False):
                # Registration refuses use_8bit=False groups, so this can only be
                # reached through state loaded from elsewhere. The old `return`
                # here claimed step() would apply the update; it is never called.
                raise RuntimeError(
                    f"AdamW8bit_RingBuffer's fused-backward hook fired for a parameter "
                    f"{tuple(param.shape)} whose optimizer state is not 8-bit. The hook "
                    f"performs the 8-bit CUDA update only, and under the fused backward "
                    f"pass there is no optimizer.step() to apply an FP32 update instead, "
                    f"so skipping would leave this parameter untrained for the whole run."
                )

            step = optimizer._advance_param_step(state)

            # Perform 8-bit update
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            gnorm_scale = 1.0

            # Stochastic rounding: update an FP32 image of the param, then round
            # back into BF16. Without this the fused-backward path silently
            # ignores the stochastic_rounding setting that optimizer.step() honours.
            grad = param.grad
            if should_use_stochastic_rounding(group['stochastic_rounding'], param):
                p_fp32, grad = prepare_master_and_grad(param, grad, optimizer._sr_scratch)
            else:
                p_fp32 = None

            optimizer.ext.adamw_8bit_update(
                p_fp32 if p_fp32 is not None else param,
                grad,
                state['exp_avg'],
                state['exp_avg_sq'],
                state['absmax1'],
                state['absmax2'],
                beta1, beta2, eps, lr, weight_decay, gnorm_scale,
                step,                       # per-parameter, see _advance_param_step
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
        optimizer, model, "patch_adamw8bit_ringbuffer", create_update_hook
    )

    print(f"[AdamW8bit_RingBuffer] Registered post_accumulate_grad hooks for {hooked} "
          f"parameters (every parameter in optimizer.param_groups)")
    if frozen:
        print(f"[AdamW8bit_RingBuffer] {len(frozen)} parameter(s) in param_groups have "
              f"requires_grad=False and get no hook (they receive no gradient), "
              f"e.g. {frozen[:3]}")
