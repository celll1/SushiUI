"""
ReLoRA Learning Rate Scheduler with multiple warmup restarts.

Implements a cosine annealing schedule with periodic warmup phases
that restart at each ReLoRA merge-reinit cycle.

LR curve visualization:
    |  /\      /\      /\
    | /  \    /  \    /  \
    |/    \  /    \  /    \
    |      \/      \/      \
    +-------|-------|---------> steps
      cycle1   cycle2   cycle3

Each cycle:
    1. Linear warmup from min_lr to base_lr (restart_warmup_steps)
    2. Cosine decay from base_lr to min_lr (remaining steps in cycle)

Reference:
    "Stack More Layers Differently: High-Rank Training Through Low-Rank Updates"
    (arXiv:2307.05695) by Guitaricet et al.

Author: Claude (2026-01-29)
"""

import math
from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWithMultipleWarmups(_LRScheduler):
    """
    Cosine LR scheduler with periodic warmup restarts.

    During the initial phase, performs linear warmup for `initial_warmup_steps`.
    After each merge-reinit cycle (signaled via `add_restart(step)`),
    performs linear warmup for `restart_warmup_steps`, then cosine decay.

    Between restarts, follows cosine annealing from base_lr to min_lr.

    Args:
        optimizer: Wrapped optimizer
        total_steps: Total number of training steps
        initial_warmup_steps: Number of warmup steps at the beginning of training
        restart_warmup_steps: Number of warmup steps after each restart
        min_lr_ratio: Minimum LR as fraction of base LR (default: 0.0)
        last_epoch: Used for resuming (default: -1)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        initial_warmup_steps: int = 0,
        restart_warmup_steps: int = 100,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.initial_warmup_steps = initial_warmup_steps
        self.restart_warmup_steps = restart_warmup_steps
        self.min_lr_ratio = min_lr_ratio

        # Restart points (steps at which merge-reinit occurred)
        # Populated dynamically by add_restart()
        self.restart_steps: List[int] = []

        # Track current step independently for get_lr()
        # Starts at -1 because _LRScheduler.__init__ calls _initial_step()
        # which calls self.step(), incrementing to 0 before training begins.
        self._relora_step = -1

        super().__init__(optimizer, last_epoch=last_epoch)

    def add_restart(self, step: int) -> None:
        """
        Register a restart point at the given step.

        Called by ReLoRATrainer when a merge-reinit cycle occurs.
        The LR will warmup from min_lr starting at this step.

        Args:
            step: The global training step at which the restart occurs
        """
        self.restart_steps.append(step)
        self.restart_steps.sort()

    def get_lr(self) -> List[float]:
        """
        Compute learning rate for each parameter group at the current step.

        Returns:
            List of learning rates, one per optimizer parameter group
        """
        step = self._relora_step

        # Find which cycle we're in (between which restart points)
        cycle_start = 0
        warmup_steps = self.initial_warmup_steps

        for restart_step in self.restart_steps:
            if step >= restart_step:
                cycle_start = restart_step
                warmup_steps = self.restart_warmup_steps
            else:
                break

        # Steps elapsed since the start of this cycle
        steps_in_cycle = step - cycle_start

        # Determine cycle length (from current restart to next restart or end)
        next_restart = self.total_steps  # Default: until end of training
        for restart_step in self.restart_steps:
            if restart_step > cycle_start:
                next_restart = restart_step
                break

        cycle_length = next_restart - cycle_start

        # Compute LR multiplier
        if steps_in_cycle < warmup_steps:
            # Linear warmup phase
            if warmup_steps > 0:
                lr_mult = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (steps_in_cycle / warmup_steps)
            else:
                lr_mult = 1.0
        else:
            # Cosine decay phase (after warmup within this cycle)
            decay_steps = cycle_length - warmup_steps
            if decay_steps > 0:
                progress = (steps_in_cycle - warmup_steps) / decay_steps
                progress = min(progress, 1.0)
                lr_mult = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                lr_mult = 1.0

        return [base_lr * lr_mult for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """
        Step the scheduler forward by one step.

        Overrides _LRScheduler.step() to use our own step counter
        that is independent of last_epoch (which has confusing semantics
        in PyTorch's base class).
        """
        self._relora_step += 1

        # Update optimizer param group learning rates
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        # Update internal state for compatibility
        self._last_lr = values
