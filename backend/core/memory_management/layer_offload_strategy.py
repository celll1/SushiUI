"""
Layer Offload Strategy Calculation

Determines optimal loading/offloading schedule for transformer layers.
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OffloadAction:
    """Single offload action."""
    layer_idx: int
    action: str  # 'load' or 'offload'
    step: int    # Forward step index or backward step index
    direction: str  # 'forward' or 'backward'


class LayerOffloadStrategy:
    """
    Calculates optimal layer offloading strategy.

    Strategy:
    - Forward pass: Load layers sequentially, offload after use
    - Backward pass: Load layers in reverse order, offload after use
    - Keep minimum layers on GPU to reduce VRAM usage
    """

    def __init__(
        self,
        num_layers: int,
        blocks_to_swap: int,
        device: torch.device
    ):
        """
        Initialize strategy.

        Args:
            num_layers: Total number of transformer layers
            blocks_to_swap: Number of layers to keep on CPU
            device: GPU device for computation
        """
        self.num_layers = num_layers
        self.blocks_to_swap = blocks_to_swap
        self.device = device

        # Layers that stay on GPU
        self.num_resident_layers = num_layers - blocks_to_swap

        # Validate
        if blocks_to_swap >= num_layers:
            raise ValueError(
                f"blocks_to_swap ({blocks_to_swap}) must be less than num_layers ({num_layers})"
            )

        if blocks_to_swap < 0:
            raise ValueError(f"blocks_to_swap must be >= 0, got {blocks_to_swap}")

    def is_resident(self, layer_idx: int) -> bool:
        """
        Check if layer is resident (stays on GPU).

        Args:
            layer_idx: Layer index

        Returns:
            True if layer is resident
        """
        # First layers are resident
        return layer_idx < self.num_resident_layers

    def is_offloadable(self, layer_idx: int) -> bool:
        """
        Check if layer is offloadable (swapped to CPU).

        Args:
            layer_idx: Layer index

        Returns:
            True if layer is offloadable
        """
        return not self.is_resident(layer_idx)

    def get_forward_schedule(self) -> List[OffloadAction]:
        """
        Get forward pass offload schedule.

        Schedule:
        1. Load layer N before executing
        2. Execute layer N
        3. Offload layer N after executing (if offloadable)

        Returns:
            List of offload actions in execution order
        """
        actions = []

        for layer_idx in range(self.num_layers):
            if self.is_offloadable(layer_idx):
                # Load before execution
                actions.append(OffloadAction(
                    layer_idx=layer_idx,
                    action='load',
                    step=layer_idx,
                    direction='forward'
                ))

                # Offload after execution
                actions.append(OffloadAction(
                    layer_idx=layer_idx,
                    action='offload',
                    step=layer_idx,
                    direction='forward'
                ))

        return actions

    def get_backward_schedule(self) -> List[OffloadAction]:
        """
        Get backward pass offload schedule.

        Schedule (reverse order):
        1. Load layer N before backward
        2. Execute layer N backward
        3. Offload layer N after backward (if offloadable)

        Returns:
            List of offload actions in execution order
        """
        actions = []

        for layer_idx in reversed(range(self.num_layers)):
            if self.is_offloadable(layer_idx):
                # Load before backward
                actions.append(OffloadAction(
                    layer_idx=layer_idx,
                    action='load',
                    step=self.num_layers - 1 - layer_idx,
                    direction='backward'
                ))

                # Offload after backward
                actions.append(OffloadAction(
                    layer_idx=layer_idx,
                    action='offload',
                    step=self.num_layers - 1 - layer_idx,
                    direction='backward'
                ))

        return actions

    def get_initial_device(self, layer_idx: int) -> torch.device:
        """
        Get initial device for layer.

        Args:
            layer_idx: Layer index

        Returns:
            Device (GPU for resident, CPU for offloadable)
        """
        if self.is_resident(layer_idx):
            return self.device
        else:
            return torch.device('cpu')

    def should_prefetch(self, current_layer: int, direction: str) -> Optional[int]:
        """
        Determine if next layer should be prefetched.

        Args:
            current_layer: Currently executing layer
            direction: 'forward' or 'backward'

        Returns:
            Layer index to prefetch, or None
        """
        if direction == 'forward':
            next_layer = current_layer + 1
            if next_layer < self.num_layers and self.is_offloadable(next_layer):
                return next_layer
        elif direction == 'backward':
            next_layer = current_layer - 1
            if next_layer >= 0 and self.is_offloadable(next_layer):
                return next_layer

        return None

    def get_offload_summary(self) -> Dict[str, any]:
        """
        Get summary of offload strategy.

        Returns:
            Dictionary with strategy info
        """
        return {
            "num_layers": self.num_layers,
            "blocks_to_swap": self.blocks_to_swap,
            "num_resident_layers": self.num_resident_layers,
            "resident_layers": list(range(self.num_resident_layers)),
            "offloadable_layers": list(range(self.num_resident_layers, self.num_layers)),
            "device": str(self.device),
        }

    def print_strategy(self):
        """Print offload strategy summary."""
        summary = self.get_offload_summary()

        print("=" * 60)
        print("[LayerOffloadStrategy] Strategy Summary")
        print("=" * 60)
        print(f"  Total Layers:     {summary['num_layers']}")
        print(f"  Blocks to Swap:   {summary['blocks_to_swap']}")
        print(f"  Resident Layers:  {summary['num_resident_layers']} (stay on GPU)")
        print(f"  Offloadable:      {summary['blocks_to_swap']} (swap CPU ↔ GPU)")
        print(f"  Device:           {summary['device']}")
        print()
        print(f"  Resident:         {summary['resident_layers']}")
        print(f"  Offloadable:      {summary['offloadable_layers']}")
        print("=" * 60)
