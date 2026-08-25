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

from .fused_grad_norm import record_fused_grad_norm
from .update_census import note_update_applied


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
            max_grad_norm: must be 0 -- this class cannot clip (see below)
        """
        if max_grad_norm and max_grad_norm > 0:
            raise ValueError(
                f"FusedOptimizerGroups cannot clip by global norm (max_grad_norm="
                f"{max_grad_norm}): each group applies its update from a "
                f"post-accumulate-grad hook, before the remaining gradients exist, "
                f"so no global norm is ever available. A per-parameter clip under "
                f"that name would be a different algorithm. BaseTrainer passes 0.0 "
                f"and reports the ignored setting once through "
                f"_warn_grad_clipping_ignored_under_fused."
            )
        self.optimizers = optimizers
        self.max_grad_norm = 0.0

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
                            # No clipping: a per-parameter clip is not the global-norm
                            # clip max_grad_norm names, so BaseTrainer constructs this
                            # class with max_grad_norm=0.0 and says so once through
                            # _warn_grad_clipping_ignored_under_fused.

                            # Before the group's zero_grad(set_to_none=True) below,
                            # which is what leaves the trainer nothing to measure.
                            record_fused_grad_norm(self.optimizers[idx], tensor)

                            # Get optimizer index for this parameter
                            i = self.parameter_optimizer_map[tensor]

                            # Increment counter
                            self.optimizer_hooked_count[i] += 1

                            # If all parameters in this group are done, step the optimizer
                            if self.optimizer_hooked_count[i] == self.num_parameters_per_group[i]:
                                # Step optimizer
                                self.optimizers[i].step()
                                self.optimizers[i].zero_grad(set_to_none=True)
                                # This group's weights now carry an update the
                                # groups after it do not (see the ledger in
                                # update_census).
                                note_update_applied(self.num_parameters_per_group[i])

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
        Reset counters. MUST be called before EVERY backward pass, not per batch.

        The counters count gradients, and gradients arrive one per parameter per
        BACKWARD. A batch runs more than one backward whenever MNT > 1, or the
        batch is micro-split, or an OOM retry splits it: resetting once per batch
        leaves the count above the group size from the second backward on, so
        ``== num_parameters_per_group`` never holds again and every step after
        the first is dropped, its gradient never freed, and the leftover summed
        into the next batch's first step.

        Resetting per backward (rather than inside the hook when a group steps)
        also keeps the group's condition meaning "all of MY parameters got a
        gradient in THIS backward" -- an incomplete group stays put instead of
        drifting into a step on a mixture of this backward's and an older one's
        gradients.
        """
        self.optimizer_hooked_count = {i: 0 for i in range(len(self.optimizers))}

    def step_incomplete_groups(self) -> List[int]:
        """Step the groups that got SOME, but not all, of their gradients.

        MUST be called right after every backward returns. The hook's
        ``== num_parameters_per_group`` condition only fires for a group whose
        parameters ALL received a gradient in this backward, and some never do:
        the Vision Encoder on a reference-free batch, a block that stochastic
        depth dropped this step, any conditionally-executed module. Their group
        would otherwise never step -- freezing the parameters that DID get a
        gradient and happen to share the group, which
        ``create_optimizer_groups`` assigns by nothing but index order -- and
        would hold those gradients live into the next backward, which then sums
        into them.

        Count 0 means nothing to apply, count == the group size means the hook
        already stepped it; both are excluded, so no group steps twice and no
        group steps on an empty gradient set.

        Returns the indices stepped.
        """
        stepped = []
        for i, optimizer in enumerate(self.optimizers):
            count = self.optimizer_hooked_count.get(i, 0)
            if 0 < count < self.num_parameters_per_group[i]:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                note_update_applied(count)
                stepped.append(i)
        self.reset_counters()
        return stepped

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
