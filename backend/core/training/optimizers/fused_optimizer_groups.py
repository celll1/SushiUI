"""
Fused Optimizer Groups for Block Swap compatibility

This implementation allows ANY optimizer (AdamW, AdamW8bit, Lion8bit, etc.) to work
with Block Swap by dividing parameters into groups and updating each group when
all its gradients are ready.

Based on sd-scripts implementation:
https://github.com/kohya-ss/sd-scripts/blob/main/sdxl_train.py (Lines 521-553)

Advantages:
- Works with any optimizer (not limited to Adafactor)
- Same memory savings as Fused Backward Pass
- Implementation only in training script (no optimizer patching needed)

Trade-off:
- Slightly higher memory than per-parameter fused (Adafactor)
- Requires tuning num_groups (recommended: 4-10)

Requirements:
- PyTorch 2.1+ (for register_post_accumulate_grad_hook)
"""

import math
from typing import List, Dict, Any
import torch
from torch.optim import Optimizer


class FusedOptimizerGroups:
    """
    Manages multiple optimizer instances for parameter groups.

    Each optimizer updates when all its parameters have gradients ready,
    reducing peak memory usage by avoiding storing all gradients simultaneously.
    """

    def __init__(
        self,
        optimizers: List[Optimizer],
        max_grad_norm: float = 0.0
    ):
        """
        Initialize fused optimizer groups.

        Args:
            optimizers: List of optimizer instances (one per group)
            max_grad_norm: Maximum gradient norm for clipping (0 to disable)
        """
        self.optimizers = optimizers
        self.max_grad_norm = max_grad_norm

        # Counters and mappings
        self.optimizer_hooked_count: Dict[int, int] = {}
        self.num_parameters_per_group: List[int] = [0] * len(optimizers)
        self.parameter_optimizer_map: Dict[torch.Tensor, int] = {}

        # Hook handles (for cleanup)
        self.hook_handles = []

    def register_hooks(self):
        """
        Register post-accumulate-grad hooks for all trainable parameters.

        Each hook updates the optimizer when all parameters in its group are ready.
        """
        hooks_registered = 0

        for opt_idx, optimizer in enumerate(self.optimizers):
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def optimizer_hook(tensor: torch.Tensor, idx=opt_idx):
                            """Hook called when gradient is ready for this parameter"""
                            # Gradient clipping (per parameter)
                            if self.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(tensor, self.max_grad_norm)

                            # Get optimizer index for this parameter
                            i = self.parameter_optimizer_map[tensor]

                            # Increment counter
                            self.optimizer_hooked_count[i] += 1

                            # If all parameters in this group are done, step the optimizer
                            if self.optimizer_hooked_count[i] == self.num_parameters_per_group[i]:
                                # Step optimizer
                                self.optimizers[i].step()
                                self.optimizers[i].zero_grad(set_to_none=True)

                        # Register hook
                        handle = parameter.register_post_accumulate_grad_hook(optimizer_hook)
                        self.hook_handles.append(handle)

                        # Map parameter to optimizer
                        self.parameter_optimizer_map[parameter] = opt_idx
                        self.num_parameters_per_group[opt_idx] += 1
                        hooks_registered += 1

        print(f"[FusedOptimizerGroups] Registered {hooks_registered} hooks for {len(self.optimizers)} optimizer groups")
        print(f"[FusedOptimizerGroups] Parameters per group: {self.num_parameters_per_group}")

        # Initialize counters after hooks are registered
        self.reset_counters()

    def reset_counters(self):
        """
        Reset counters for next training step.

        Must be called at the start of each training step.
        """
        self.optimizer_hooked_count = {i: 0 for i in range(len(self.optimizers))}

    def remove_hooks(self):
        """
        Remove all registered hooks.

        Call this when switching from training to inference mode.
        """
        for handle in self.hook_handles:
            handle.remove()

        num_removed = len(self.hook_handles)
        self.hook_handles = []
        print(f"[FusedOptimizerGroups] Removed {num_removed} hooks")


def create_optimizer_groups(
    params: List[torch.nn.Parameter],
    optimizer_type: str,
    num_groups: int,
    learning_rate: float,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs
) -> List[Optimizer]:
    """
    Create multiple optimizer instances by dividing parameters into groups.

    Args:
        params: List of parameters to optimize
        optimizer_type: Optimizer type (adamw, adamw8bit, etc.)
        num_groups: Number of parameter groups (recommended: 4-10)
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Betas for Adam-based optimizers
        eps: Epsilon for numerical stability
        **kwargs: Additional optimizer-specific arguments

    Returns:
        List of optimizer instances
    """
    # Flatten parameters if nested
    flat_params = []
    for p in params:
        if isinstance(p, dict):
            flat_params.extend(p['params'])
        else:
            flat_params.append(p)

    n_total_params = len(flat_params)
    params_per_group = math.ceil(n_total_params / num_groups)

    print(f"[FusedOptimizerGroups] Creating {num_groups} optimizer groups")
    print(f"[FusedOptimizerGroups] Total parameters: {n_total_params}")
    print(f"[FusedOptimizerGroups] Parameters per group: ~{params_per_group}")

    # Divide parameters into groups
    optimizers = []
    for i in range(num_groups):
        start_idx = i * params_per_group
        end_idx = min((i + 1) * params_per_group, n_total_params)
        group_params = flat_params[start_idx:end_idx]

        if not group_params:
            break

        # Create optimizer for this group using factory
        from ..optimizer_factory import OptimizerFactory
        optimizer = OptimizerFactory.create_optimizer(
            optimizer_type=optimizer_type,
            params=group_params,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs
        )
        optimizers.append(optimizer)

    print(f"[FusedOptimizerGroups] Created {len(optimizers)} optimizers")
    return optimizers
