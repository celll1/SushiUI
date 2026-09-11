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

from .base_trainer import (
    live_scheduler_step,
    lr_scheduler_advance_interval,
    reapply_lr_schedule_position,
)
from .lora_trainer import LoRATrainer
from .lr_schedules import to_scheduler_axis
from .relora_utils import merge_lora_into_base, reinitialize_lora, reset_optimizer_state


class ReLoRATrainer(LoRATrainer):
    """
    ReLoRA Trainer: LoRA with periodic merge-reinit cycles.

    Inherits all LoRA functionality (adapter system, checkpoint, parameter groups)
    and adds the periodic merge-reinit-reset-restart cycle.

    Class hierarchy: BaseTrainer -> LoRATrainer -> ReLoRATrainer
    Supports SD1.5 / SDXL / Z-Image / FLUX.2 via adapter system.
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
        # Refused, not folded to 1 or to "never": the merge-reinit cycle IS
        # ReLoRA, so neither reading of 0 is what the caller asked for, and the
        # modulo in should_merge() would divide by zero.
        if int(relora_merge_every or 0) <= 0:
            raise ValueError(
                f"relora_merge_every must be >= 1, got {relora_merge_every}. It is the "
                f"interval between merge-reinit cycles; without them ReLoRA is plain "
                f"LoRA, so use training_method='lora' if that is what you want."
            )

        # ReLoRA-specific settings (set before super().__init__)
        self.relora_merge_every = relora_merge_every
        self.relora_merge_unit = relora_merge_unit
        self.restart_warmup_steps = restart_warmup_steps
        self.optimizer_reset_strategy = optimizer_reset_strategy
        self.optimizer_pruning_ratio = optimizer_pruning_ratio

        # Merge tracking
        self.merge_count = 0
        self._last_merge_epoch = -1  # For epoch-based merge tracking

        self._refuse_unsupported_relora(kwargs.get("model_path"))

        super().__init__(**kwargs)

        from core.adapters import count_quantized_linears

        quantized_targets = sum(
            count_quantized_linears(getattr(layer, "original_module", None))
            for layer in self.lora_layers.values()
        )
        if quantized_targets:
            raise ValueError(
                "ReLoRA cannot merge dense LoRA deltas into a weight-only quantized base "
                f"({quantized_targets} target layer(s)); use training_method='lora'.")

        # Override log prefix
        self.log_prefix = "[ReLoRA Trainer]"

        print(f"{self.log_prefix} ReLoRA settings:")
        print(f"{self.log_prefix}   merge_every={self.relora_merge_every} {self.relora_merge_unit}")
        print(f"{self.log_prefix}   restart_warmup_steps={self.restart_warmup_steps}")
        print(f"{self.log_prefix}   optimizer_reset_strategy={self.optimizer_reset_strategy}")
        if self.optimizer_reset_strategy != "full_reset":
            print(f"{self.log_prefix}   optimizer_pruning_ratio={self.optimizer_pruning_ratio}")

    @staticmethod
    def _refuse_unsupported_relora(model_path):
        if not model_path:
            return
        try:
            from core.model_loader import ModelLoader

            arch = ModelLoader.detect_model_type(model_path)
        except Exception:
            return
        from api.arch_capabilities import TRAINING_UNSUPPORTED

        reason = (TRAINING_UNSUPPORTED.get(arch) or {}).get("relora")
        if reason:
            raise ValueError(
                f"ReLoRA is not supported for architecture '{arch}': {reason}")


    def setup_optimizer(
        self,
        optimizer_type: str = "adamw",
        lr_scheduler_type: str = "constant",
        total_steps: int = 1000,
    ):
        """
        Setup optimizer and LR scheduler.

        ReLoRA's schedule is the registry's ``relora`` curve
        (``lr_schedules.py``, §4.2): the merge restarts shape it, so no other
        name can express it and the run's ``lr_scheduler`` setting is ignored --
        as it always was, when this method discarded the scheduler the parent
        had just built. What it builds now is a ``LambdaLR`` like every other
        schedule, which is what makes the resume fast-forward, the post-reset
        re-warmup and the config-LR re-assertion work for ReLoRA too.

        Args:
            optimizer_type: Optimizer type
            lr_scheduler_type: LR scheduler type (ignored; see above)
            total_steps: Total training steps
        """
        if str(lr_scheduler_type or "").strip().lower() != "relora":
            print(f"{self.log_prefix} lr_scheduler '{lr_scheduler_type}' is ignored: "
                  f"ReLoRA's own restart schedule is used "
                  f"(initial warmup {self.optimizer_warmup_steps}, "
                  f"restart warmup {self.restart_warmup_steps} steps)")
        super().setup_optimizer(optimizer_type, "relora", total_steps)


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

        self._save_pre_merge_checkpoint(global_step, epoch)

        merged_count = merge_lora_into_base(self.lora_layers)
        print(f"{self.log_prefix} Merged {merged_count}/{len(self.lora_layers)} LoRA layers into base model")

        reinitialize_lora(self.lora_layers)
        print(f"{self.log_prefix} Reinitialized LoRA layers (A=kaiming, B=zeros)")

        trainable_param_ids = self._get_trainable_param_ids()
        reset_optimizer_state(
            self.optimizer,
            strategy=self.optimizer_reset_strategy,
            pruning_ratio=self.optimizer_pruning_ratio,
            trainable_param_ids=trainable_param_ids,
        )
        print(f"{self.log_prefix} Reset optimizer state (strategy={self.optimizer_reset_strategy})")

        # The fused clip's running scales describe the gradients of the adapters
        # that were just merged away and re-initialised, so every one of them is
        # about to change scale legitimately. Kept, they would clip the new
        # adapters against the old ones' history.
        clipper = getattr(self, "_fused_grad_clipper", None)
        if clipper is not None:
            clipper.reset_scales()
            print(f"{self.log_prefix} Reset fused gradient-clip scales "
                  f"(warmup restarts for every parameter)")

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

        self._add_lr_restart(global_step)

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
        Record the merge's warmup restart on the schedule timeline.

        Seam (d) of the design's §5.2. One event, not one per scheduler: under
        fused optimizer groups the N schedulers share a single timeline, so they
        cannot disagree about where a restart is.

        Args:
            global_step: Step at which the restart occurs (logged; the event's
                position is on the scheduler axis)
        """
        timeline = getattr(self, "lr_timeline", None)
        if timeline is None:
            print(f"{self.log_prefix} WARNING: no LR schedule timeline; the merge at "
                  f"step {global_step} will not restart the warmup")
            return

        at = live_scheduler_step(self)
        result = timeline.add("restart", at=at)
        if result == "applied":
            # This hook runs AFTER scheduler.step(), so the param groups still
            # hold the pre-restart LR; without rewriting them the reinitialized
            # adapter would take one more step at it.
            reapply_lr_schedule_position(self)
        print(f"{self.log_prefix} Added LR warmup restart at scheduler step {at} "
              f"(global step {global_step}): {result}")


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
        # The restarts themselves come back with the schedule timeline in
        # state.json. Checkpoints that predate that are handled by
        # _restore_legacy_lr_restarts, which install_lr_schedule_events calls
        # after the load -- not from here, which runs before it.

    def _restore_legacy_lr_restarts(self, position: int) -> None:
        """
        Rebuild `restart` events for a checkpoint that carries none.

        Since P4 a merge IS a timeline event and is restored with the rest of
        state.json; this is the back-compat path for checkpoints written before
        that, which record only ``merge_count``. Only step-unit merges can be
        placed from it -- an epoch-unit run's merge steps were never recorded
        anywhere, which is why the old re-registration dropped them outright.

        Args:
            position: the resumed scheduler-axis position; restarts after it
                belong to a future this resume has rewound past.
        """
        timeline = getattr(self, "lr_timeline", None)
        if timeline is None or self.merge_count <= 0:
            return
        if timeline.restarts():
            return

        if self.relora_merge_unit != "steps":
            print(f"{self.log_prefix} WARNING: this checkpoint predates LR restart events "
                  f"and merges by epoch, so the steps of its {self.merge_count} past "
                  f"merge(s) are not recorded anywhere. The schedule resumes as if none "
                  f"had happened; merges from here on are recorded.")
            return

        interval = lr_scheduler_advance_interval(self)
        restored = 0
        for i in range(1, self.merge_count + 1):
            at = to_scheduler_axis(i * self.relora_merge_every, interval)
            if at > position:
                break
            if timeline.add("restart", at=at) == "applied":
                restored += 1
        print(f"{self.log_prefix} Restored {restored} LR restart point(s) from merge_count "
              f"(estimated positions: this checkpoint predates restart events)")
