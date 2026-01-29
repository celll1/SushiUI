"""
ReLoRA (Reinitialized Low-Rank Adaptation) utility functions.

Provides merge, reinitialize, and optimizer state reset operations
for the periodic merge-reinit cycle used in ReLoRA training.

Reference:
    "Stack More Layers Differently: High-Rank Training Through Low-Rank Updates"
    (arXiv:2307.05695) by Guitaricet et al.
    https://github.com/Guitaricet/relora

Author: Claude (2026-01-29)
"""

import math
from typing import Dict, Optional, Set

import torch
import torch.nn as nn


def merge_lora_into_base(lora_layers: Dict[str, nn.Module]) -> int:
    """
    Merge all LoRA weight deltas into the base model weights (in-place).

    For each LoRALinearLayer:
        original_module.weight.data += (lora_up.weight @ lora_down.weight) * scale

    The merge is performed layer-by-layer to minimize peak VRAM usage.
    Computation is done in float32 for numerical stability, then cast back
    to the original weight dtype.

    Args:
        lora_layers: Dictionary of LoRA layer name -> LoRALinearLayer module

    Returns:
        Number of layers successfully merged
    """
    merged_count = 0

    for lora_name, lora_layer in lora_layers.items():
        try:
            original_module = lora_layer.original_module
            lora_up = lora_layer.lora_up
            lora_down = lora_layer.lora_down
            scale = lora_layer.scale

            # Target device and dtype from original weight
            device = original_module.weight.device
            orig_dtype = original_module.weight.dtype

            # Compute LoRA delta in float32 for precision
            # lora_up.weight: (out_features, rank)
            # lora_down.weight: (rank, in_features)
            # delta: (out_features, in_features)
            up_weight = lora_up.weight.data.to(device=device, dtype=torch.float32)
            down_weight = lora_down.weight.data.to(device=device, dtype=torch.float32)
            delta = (up_weight @ down_weight) * scale

            # Merge into base weight (in-place)
            original_module.weight.data.add_(delta.to(dtype=orig_dtype))

            # Free temporary tensors
            del up_weight, down_weight, delta

            merged_count += 1
        except Exception as e:
            print(f"[ReLoRA] WARNING: Failed to merge layer '{lora_name}': {e}")

    return merged_count


def reinitialize_lora(lora_layers: Dict[str, nn.Module]) -> None:
    """
    Reinitialize all LoRA layers to zero contribution.

    For each LoRALinearLayer:
        lora_down.weight = kaiming_uniform_ (a=sqrt(5))  [PyTorch Linear default]
        lora_up.weight = zeros_

    After reinitialization, LoRA output is exactly zero:
        lora_up(lora_down(x)) * scale = 0 @ lora_down(x) * scale = 0

    This ensures continuity with the base model after merge.

    Args:
        lora_layers: Dictionary of LoRA layer name -> LoRALinearLayer module
    """
    for lora_name, lora_layer in lora_layers.items():
        try:
            # Reinitialize lora_down with Kaiming uniform (same as PyTorch Linear default)
            nn.init.kaiming_uniform_(lora_layer.lora_down.weight, a=math.sqrt(5))
            # Reinitialize lora_up to zeros (ensures zero initial contribution)
            nn.init.zeros_(lora_layer.lora_up.weight)
        except Exception as e:
            print(f"[ReLoRA] WARNING: Failed to reinitialize layer '{lora_name}': {e}")


def reset_optimizer_state(
    optimizer: torch.optim.Optimizer,
    strategy: str,
    pruning_ratio: float = 0.9,
    trainable_param_ids: Optional[Set[int]] = None,
) -> None:
    """
    Reset optimizer state using the specified strategy.

    Strategies:
        "full_reset": Clear all optimizer state (momentum, variance, step counts).
            Most aggressive reset, completely fresh start for the optimizer.

        "magnitude_pruning": Zero out optimizer state entries below the
            (pruning_ratio * 100)th percentile by magnitude.
            E.g., pruning_ratio=0.9 keeps only the top 10% of state values.
            Preserves the most important gradient history.

        "random_pruning": Randomly zero out (pruning_ratio * 100)% of
            optimizer state entries. E.g., pruning_ratio=0.9 keeps 10%.

    Args:
        optimizer: The optimizer whose state will be reset
        strategy: Reset strategy ("full_reset", "magnitude_pruning", "random_pruning")
        pruning_ratio: Fraction of state entries to zero out (for pruning strategies)
        trainable_param_ids: If provided, only reset state for these parameter IDs.
            If None, reset all parameters' state.
    """
    if strategy == "full_reset":
        _full_reset(optimizer, trainable_param_ids)
    elif strategy == "magnitude_pruning":
        _selective_reset(optimizer, "magnitude", pruning_ratio, trainable_param_ids)
    elif strategy == "random_pruning":
        _selective_reset(optimizer, "random", pruning_ratio, trainable_param_ids)
    else:
        raise ValueError(f"Unknown optimizer reset strategy: '{strategy}'. "
                         f"Expected 'full_reset', 'magnitude_pruning', or 'random_pruning'.")


def _full_reset(
    optimizer: torch.optim.Optimizer,
    trainable_param_ids: Optional[Set[int]] = None,
) -> None:
    """
    Fully reset optimizer state.

    Clears all accumulated momentum (exp_avg) and variance (exp_avg_sq)
    for Adam-type optimizers. Also resets step counts.
    """
    if trainable_param_ids is None:
        # Reset all parameters
        optimizer.state.clear()
        return

    # Reset only specified parameters
    params_to_clear = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) in trainable_param_ids:
                params_to_clear.append(p)

    for p in params_to_clear:
        if p in optimizer.state:
            del optimizer.state[p]


def _selective_reset(
    optimizer: torch.optim.Optimizer,
    method: str,
    pruning_ratio: float,
    trainable_param_ids: Optional[Set[int]] = None,
) -> None:
    """
    Selectively prune optimizer state using magnitude or random pruning.

    Targets the standard Adam state keys: 'exp_avg' (first moment / momentum)
    and 'exp_avg_sq' (second moment / variance).
    """
    # Standard Adam/AdamW/Lion state keys to prune
    state_keys_to_prune = {"exp_avg", "exp_avg_sq"}

    for group in optimizer.param_groups:
        for p in group["params"]:
            # Skip if not in target set
            if trainable_param_ids is not None and id(p) not in trainable_param_ids:
                continue

            if p not in optimizer.state:
                continue

            state = optimizer.state[p]

            for key in state_keys_to_prune:
                if key not in state:
                    continue

                tensor = state[key]
                if not isinstance(tensor, torch.Tensor):
                    continue

                if method == "magnitude":
                    _magnitude_pruning_(tensor, pruning_ratio)
                elif method == "random":
                    _random_pruning_(tensor, pruning_ratio)


def _magnitude_pruning_(tensor: torch.Tensor, prune_ratio: float) -> None:
    """
    Zero out entries below the prune_ratio percentile by magnitude (in-place).

    Example: prune_ratio=0.9 zeros out the bottom 90% of entries,
    keeping only the top 10% by absolute value.

    Args:
        tensor: Tensor to prune (modified in-place)
        prune_ratio: Fraction of entries to zero out (0.0 to 1.0)
    """
    if prune_ratio <= 0.0:
        return
    if prune_ratio >= 1.0:
        tensor.zero_()
        return

    # Compute the threshold: entries below this magnitude will be zeroed
    abs_values = tensor.abs().flatten()
    threshold = torch.quantile(abs_values, prune_ratio)

    # Create mask: keep entries whose magnitude exceeds threshold
    mask = tensor.abs() > threshold
    tensor.mul_(mask)


def _random_pruning_(tensor: torch.Tensor, prune_ratio: float) -> None:
    """
    Randomly zero out prune_ratio fraction of entries (in-place).

    Example: prune_ratio=0.9 randomly zeros out 90% of entries,
    keeping approximately 10%.

    Args:
        tensor: Tensor to prune (modified in-place)
        prune_ratio: Fraction of entries to zero out (0.0 to 1.0)
    """
    if prune_ratio <= 0.0:
        return
    if prune_ratio >= 1.0:
        tensor.zero_()
        return

    # Create random mask: keep entries where random value > prune_ratio
    mask = torch.rand_like(tensor, dtype=torch.float32) > prune_ratio
    tensor.mul_(mask)
