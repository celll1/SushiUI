"""
ReLoRA (Reinitialized Low-Rank Adaptation) Trainer.

Extends LoRATrainer with periodic merge-reinit-reset-restart cycles,
enabling effective high-rank updates through cumulative low-rank training.

Algorithm:
    1. Train LoRA for N steps/epochs
    2. MERGE: W_base += (lora_up @ lora_down) * scale
    3. REINIT: lora_down = kaiming_uniform, lora_up = zeros
    4. RESET: optimizer state reset (configurable strategy)
    5. RESTART: LR warmup restart
    6. Repeat from 1
    -> After K merges, effective rank = K * r

Reference:
    "Stack More Layers Differently: High-Rank Training Through Low-Rank Updates"
    (arXiv:2307.05695) by Guitaricet et al.
    https://github.com/Guitaricet/relora

Author: Claude (2026-01-29)
"""

from typing import Dict, Optional, Set

import torch
import torch.nn as nn

from .lora_trainer import LoRATrainer
from .relora_utils import merge_lora_into_base, reinitialize_lora, reset_optimizer_state
from .relora_scheduler import CosineWithMultipleWarmups


class ReLoRATrainer(LoRATrainer):
    """
    ReLoRA Trainer: LoRA with periodic merge-reinit cycles.

    Inherits all LoRA functionality (adapter system, checkpoint, parameter groups)
    and adds the periodic merge-reinit-reset-restart cycle.

    Class hierarchy: BaseTrainer -> LoRATrainer -> ReLoRATrainer
    Supports SD1.5 / SDXL / Z-Image / DEUS / FLUX.2 via adapter system.
    """

    def __init__(
        self,
        relora_merge_every: int = 500,
        relora_merge_unit: str = "steps",
        restart_warmup_steps: int = 100,
        optimizer_reset_strategy: str = "full_reset",
        optimizer_pruning_ratio: float = 0.9,
        **kwargs,
    ):
        """
        Initialize ReLoRA Trainer.

        Args:
            relora_merge_every: Interval between merge-reinit cycles
            relora_merge_unit: Unit for merge interval ("steps" or "epochs")
            restart_warmup_steps: Number of warmup steps after each merge
            optimizer_reset_strategy: Strategy for optimizer state reset
                ("full_reset", "magnitude_pruning", "random_pruning")
            optimizer_pruning_ratio: Fraction to prune (for pruning strategies)
            **kwargs: Arguments forwarded to LoRATrainer
        """
        # ReLoRA-specific settings (set before super().__init__)
        self.relora_merge_every = relora_merge_every
        self.relora_merge_unit = relora_merge_unit
        self.restart_warmup_steps = restart_warmup_steps
        self.optimizer_reset_strategy = optimizer_reset_strategy
        self.optimizer_pruning_ratio = optimizer_pruning_ratio

        # Merge tracking
        self.merge_count = 0
        self._last_merge_epoch = -1  # For epoch-based merge tracking

        # Initialize parent (LoRATrainer -> BaseTrainer)
        super().__init__(**kwargs)

        # Override log prefix
        self.log_prefix = "[ReLoRA Trainer]"

        print(f"{self.log_prefix} ReLoRA settings:")
        print(f"{self.log_prefix}   merge_every={self.relora_merge_every} {self.relora_merge_unit}")
        print(f"{self.log_prefix}   restart_warmup_steps={self.restart_warmup_steps}")
        print(f"{self.log_prefix}   optimizer_reset_strategy={self.optimizer_reset_strategy}")
        if self.optimizer_reset_strategy != "full_reset":
            print(f"{self.log_prefix}   optimizer_pruning_ratio={self.optimizer_pruning_ratio}")

    # ============================================================
    # Optimizer & LR Scheduler Setup
    # ============================================================

    def setup_optimizer(
        self,
        optimizer_type: str = "adamw",
        lr_scheduler_type: str = "constant",
        total_steps: int = 1000,
    ):
        """
        Setup optimizer and LR scheduler.

        Calls parent setup_optimizer() for optimizer creation, then replaces
        the LR scheduler with CosineWithMultipleWarmups for jagged warmup support.

        Args:
            optimizer_type: Optimizer type
            lr_scheduler_type: LR scheduler type (used for parent; replaced for ReLoRA)
            total_steps: Total training steps
        """
        # 1. Call parent to create optimizer (and initial scheduler, fused groups, etc.)
        super().setup_optimizer(optimizer_type, lr_scheduler_type, total_steps)

        # 2. Replace LR scheduler with ReLoRA's CosineWithMultipleWarmups
        #    Only replace the main scheduler (not fused optimizer group schedulers)
        if self.fused_optimizer_groups is None:
            self.lr_scheduler = CosineWithMultipleWarmups(
                optimizer=self.optimizer,
                total_steps=total_steps,
                initial_warmup_steps=self.optimizer_warmup_steps,
                restart_warmup_steps=self.restart_warmup_steps,
                min_lr_ratio=0.0,
            )
            print(f"{self.log_prefix} LR scheduler replaced with CosineWithMultipleWarmups")
            print(f"{self.log_prefix}   initial_warmup={self.optimizer_warmup_steps}, restart_warmup={self.restart_warmup_steps}")
        else:
            # Fused optimizer groups: replace all schedulers
            new_schedulers = []
            for i, optimizer in enumerate(self.fused_optimizer_groups.optimizers):
                scheduler = CosineWithMultipleWarmups(
                    optimizer=optimizer,
                    total_steps=total_steps,
                    initial_warmup_steps=self.optimizer_warmup_steps,
                    restart_warmup_steps=self.restart_warmup_steps,
                    min_lr_ratio=0.0,
                )
                new_schedulers.append(scheduler)

            self.lr_schedulers = new_schedulers
            self.lr_scheduler = new_schedulers[0]  # For compatibility
            print(f"{self.log_prefix} Replaced {len(new_schedulers)} LR schedulers with CosineWithMultipleWarmups")

    # ============================================================
    # Merge-Reinit Cycle
    # ============================================================

    def should_merge(self, global_step: int, epoch: int, is_first_batch_in_epoch: bool = False) -> bool:
        """
        Check whether a merge-reinit cycle should occur at the current point.

        For step-based merging: fires at every multiple of merge_every
        (after the initial period).
        For epoch-based merging: fires at the first batch of every
        merge_every-th epoch.

        Args:
            global_step: Current global training step
            epoch: Current epoch (0-indexed)
            is_first_batch_in_epoch: Whether this is the first batch in epoch

        Returns:
            True if merge should occur
        """
        if self.relora_merge_unit == "steps":
            # Step-based: merge at multiples of merge_every
            # Guard: don't merge at step 0 or before first full cycle
            if global_step < self.relora_merge_every:
                return False
            return global_step % self.relora_merge_every == 0
        else:
            # Epoch-based: merge at multiples of merge_every epochs
            # Only trigger on first batch of the epoch
            if epoch == 0 or not is_first_batch_in_epoch:
                return False
            # Prevent double-triggering in same epoch
            if epoch == self._last_merge_epoch:
                return False
            return epoch % self.relora_merge_every == 0

    def perform_merge_reinit_cycle(self, global_step: int, epoch: int):
        """
        Execute a full merge-reinit-reset-restart cycle.

        Steps:
            1. Save pre-merge LoRA checkpoint (for debugging/recovery)
            2. Merge LoRA weights into base model
            3. Reinitialize LoRA layers (A=kaiming, B=zeros)
            4. Reset optimizer state
            5. Register LR warmup restart

        Args:
            global_step: Current global training step
            epoch: Current epoch
        """
        self.merge_count += 1
        print(f"{self.log_prefix} ========================================")
        print(f"{self.log_prefix} Merge-Reinit Cycle #{self.merge_count} at step {global_step} (epoch {epoch})")
        print(f"{self.log_prefix} ========================================")

        # 1. Save pre-merge LoRA checkpoint
        self._save_pre_merge_checkpoint(global_step, epoch)

        # 2. Merge LoRA -> base model
        merged_count = merge_lora_into_base(self.lora_layers)
        print(f"{self.log_prefix} Merged {merged_count}/{len(self.lora_layers)} LoRA layers into base model")

        # 3. Reinitialize LoRA layers
        reinitialize_lora(self.lora_layers)
        print(f"{self.log_prefix} Reinitialized LoRA layers (A=kaiming, B=zeros)")

        # 4. Reset optimizer state
        # Collect trainable parameter IDs for targeted reset
        trainable_param_ids = self._get_trainable_param_ids()
        reset_optimizer_state(
            self.optimizer,
            strategy=self.optimizer_reset_strategy,
            pruning_ratio=self.optimizer_pruning_ratio,
            trainable_param_ids=trainable_param_ids,
        )
        print(f"{self.log_prefix} Reset optimizer state (strategy={self.optimizer_reset_strategy})")

        # If fused optimizer groups, reset all optimizers
        if self.fused_optimizer_groups is not None:
            for i, optimizer in enumerate(self.fused_optimizer_groups.optimizers):
                if optimizer is not self.optimizer:  # Already reset above
                    reset_optimizer_state(
                        optimizer,
                        strategy=self.optimizer_reset_strategy,
                        pruning_ratio=self.optimizer_pruning_ratio,
                        trainable_param_ids=trainable_param_ids,
                    )

        # 5. Register LR warmup restart
        self._add_lr_restart(global_step)

        # Update epoch tracking for epoch-based merging
        if self.relora_merge_unit == "epochs":
            self._last_merge_epoch = epoch

        # Free CUDA cache after merge cycle
        torch.cuda.empty_cache()

        print(f"{self.log_prefix} Merge-Reinit Cycle #{self.merge_count} complete")
        print(f"{self.log_prefix} ========================================")

    def _save_pre_merge_checkpoint(self, global_step: int, epoch: int):
        """
        Save a pre-merge LoRA checkpoint for debugging and recovery.

        The checkpoint is saved with a '_premerge_N' suffix to distinguish
        it from regular checkpoints.

        Args:
            global_step: Current global step
            epoch: Current epoch
        """
        checkpoint_path = self.output_dir / f"{self.run_name}_step_{global_step:06d}_premerge_{self.merge_count}.safetensors"
        try:
            self.adapter.save_checkpoint(
                self.lora_layers, global_step, epoch, checkpoint_path
            )
            print(f"{self.log_prefix} Saved pre-merge checkpoint: {checkpoint_path.name}")
        except Exception as e:
            print(f"{self.log_prefix} WARNING: Failed to save pre-merge checkpoint: {e}")

    def _get_trainable_param_ids(self) -> Set[int]:
        """
        Collect Python IDs of all trainable LoRA parameters.

        Used to target optimizer state reset only for LoRA parameters
        (preserving any base model state if it exists).

        Returns:
            Set of parameter IDs (from id())
        """
        param_ids = set()
        for lora_layer in self.lora_layers.values():
            for param in lora_layer.parameters():
                if param.requires_grad:
                    param_ids.add(id(param))
        return param_ids

    def _add_lr_restart(self, global_step: int):
        """
        Register a warmup restart in all LR schedulers.

        Args:
            global_step: Step at which the restart occurs
        """
        if self.fused_optimizer_groups is not None:
            # Multiple schedulers (fused optimizer groups)
            for scheduler in self.lr_schedulers:
                if hasattr(scheduler, 'add_restart'):
                    scheduler.add_restart(global_step)
        else:
            # Single scheduler
            if hasattr(self.lr_scheduler, 'add_restart'):
                self.lr_scheduler.add_restart(global_step)

        print(f"{self.log_prefix} Added LR warmup restart at step {global_step}")

    # ============================================================
    # Training State Save/Restore
    # ============================================================

    def save_training_state(self, step: int, epoch: int, batch_idx: int, multi_noise_timesteps: int = 1):
        """
        Save training state with ReLoRA-specific fields.

        Adds merge_count and last_merge_epoch to the standard training state.

        Args:
            step: Current global step
            epoch: Current epoch
            batch_idx: Current batch index
            multi_noise_timesteps: MNT value
        """
        # Call parent to save standard state
        super().save_training_state(step, epoch, batch_idx, multi_noise_timesteps)

        # Append ReLoRA state to the saved JSON
        import json
        state_file = self.output_dir / f"{self.run_name}_step_{step:06d}_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)

            state["merge_count"] = self.merge_count
            state["last_merge_epoch"] = self._last_merge_epoch

            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

    def _restore_relora_state(self, state: dict):
        """
        Restore ReLoRA-specific state from training state dict.

        Called during resume to restore merge_count and epoch tracking.

        Args:
            state: Training state dict loaded from JSON
        """
        self.merge_count = state.get('merge_count', 0)
        self._last_merge_epoch = state.get('last_merge_epoch', -1)

        if self.merge_count > 0:
            print(f"{self.log_prefix} Restored ReLoRA state: merge_count={self.merge_count}")
            if self.relora_merge_unit == "epochs":
                print(f"{self.log_prefix}   last_merge_epoch={self._last_merge_epoch}")

            # Re-register past restart points in scheduler
            # so that get_lr() computes correctly for the current cycle
            self._restore_scheduler_restarts()

    def _restore_scheduler_restarts(self):
        """
        Re-register historical restart points in LR scheduler(s) on resume.

        When resuming, the scheduler needs to know about past merge points
        to correctly compute the LR for the current cycle.
        """
        if self.merge_count == 0:
            return

        if self.relora_merge_unit == "steps":
            # Re-register all past merge steps
            for i in range(1, self.merge_count + 1):
                merge_step = i * self.relora_merge_every
                if self.fused_optimizer_groups is not None:
                    for scheduler in self.lr_schedulers:
                        if hasattr(scheduler, 'add_restart'):
                            scheduler.add_restart(merge_step)
                else:
                    if hasattr(self.lr_scheduler, 'add_restart'):
                        self.lr_scheduler.add_restart(merge_step)

            print(f"{self.log_prefix} Restored {self.merge_count} LR restart points")
        else:
            # Epoch-based: restart points are less predictable
            # The scheduler will handle new restarts correctly going forward
            print(f"{self.log_prefix} Epoch-based merge: scheduler restarts will be re-registered on next merge")
