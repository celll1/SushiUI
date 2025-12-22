"""
Fused Block Swap with Optimizer Groups

Integrates Block Swap with Fused Optimizer Groups for Full Fine-Tuning.
This module provides the complete VRAM-efficient training solution.

Key Concepts:
- Block Swap: Move transformer layers between CPU/GPU to reduce VRAM
- Fused Optimizer: Execute optimizer.step() per parameter group, not per batch
- Combination: Minimal VRAM usage for full fine-tuning large models
"""

import torch
import torch.nn as nn
from typing import List, Optional
from torch.optim import Optimizer

from .block_offloading import TransformerBlockOffloader
from ..training.optimizers.fused_optimizer_groups import FusedOptimizerGroups


class FusedBlockSwapTrainer:
    """
    Complete VRAM-efficient training system

    Combines:
    1. Block Swap: Layer offloading (CPU ↔ GPU)
    2. Fused Optimizer Groups: Per-group optimizer step
    3. Gradient Accumulation: Reduce optimizer memory overhead

    Usage:
        trainer = FusedBlockSwapTrainer(
            transformer=model,
            blocks_to_swap=22,
            optimizer_groups=optimizers,
            device=torch.device('cuda:0')
        )
        trainer.prepare()

        for batch in dataloader:
            loss = trainer.train_step(batch)
            loss.backward()
            trainer.optimizer_step()  # Fused optimizer step
    """

    def __init__(
        self,
        transformer: nn.Module,
        blocks_to_swap: int,
        optimizer_groups: List[Optimizer],
        device: torch.device,
        use_pinned_memory: bool = True,
        max_grad_norm: float = 1.0,
        enable_activation_offload: bool = False
    ):
        """
        Initialize Fused Block Swap Trainer

        Args:
            transformer: Transformer model (with .layers attribute)
            blocks_to_swap: Number of layers to swap to CPU
            optimizer_groups: List of optimizer instances (from create_optimizer_groups)
            device: GPU device
            use_pinned_memory: Use pinned CPU memory
            max_grad_norm: Gradient clipping norm
            enable_activation_offload: Enable activation offloading (experimental)
        """
        self.transformer = transformer
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.use_pinned_memory = use_pinned_memory
        self.max_grad_norm = max_grad_norm
        self.enable_activation_offload = enable_activation_offload

        # Get transformer layers
        if not hasattr(transformer, 'layers'):
            raise ValueError("Transformer must have 'layers' attribute (nn.ModuleList)")

        self.layers = transformer.layers

        # Initialize Block Offloader
        self.block_offloader = TransformerBlockOffloader(
            blocks=self.layers,
            blocks_to_swap=blocks_to_swap,
            device=device,
            target_dtype=torch.bfloat16,  # TODO: make configurable
            use_pinned_memory=use_pinned_memory,
            transformer=transformer,
            supports_backward=True  # Training mode
        )

        # Initialize Fused Optimizer Groups
        self.fused_optimizer = FusedOptimizerGroups(
            optimizers=optimizer_groups,
            max_grad_norm=max_grad_norm
        )

        # Register hooks
        self.fused_optimizer.register_hooks()
        self.block_offloader.register_backward_hooks()

        print(f"[FusedBlockSwapTrainer] Initialized")
        print(f"  Blocks to swap: {blocks_to_swap}")
        print(f"  Optimizer groups: {len(optimizer_groups)}")
        print(f"  Pinned memory: {use_pinned_memory}")
        print(f"  Gradient clipping: {max_grad_norm}")

    def prepare(self):
        """
        Prepare for training

        - Move blocks to correct devices (GPU resident, CPU offloadable)
        - Move auxiliary modules to GPU
        """
        print(f"[FusedBlockSwapTrainer] Preparing for training...")
        self.block_offloader.prepare_block_devices_before_forward()
        print(f"[FusedBlockSwapTrainer] Ready for training")

    def train_step_begin(self):
        """
        Call at the beginning of each training step

        - Reset optimizer counters for fused step
        """
        self.fused_optimizer.reset_counters()

    def train_step_end(self):
        """
        Call at the end of each training step (after backward)

        Note: With Fused Optimizer Groups, optimizer.step() is called
        automatically during backward pass via hooks. This method is
        provided for compatibility and future extensions.
        """
        pass

    def forward_with_block_swap(self, *args, **kwargs):
        """
        Execute forward pass with block swapping

        This is a wrapper around transformer forward that handles
        block swapping automatically.

        Args:
            *args, **kwargs: Arguments to transformer forward

        Returns:
            Transformer output
        """
        # Forward pass (block offloader handles swapping via hooks)
        output = self.transformer(*args, **kwargs)
        return output

    def cleanup(self):
        """
        Cleanup resources

        - Remove optimizer hooks
        - Remove block swap hooks
        - Restore all layers to GPU
        """
        print(f"[FusedBlockSwapTrainer] Cleaning up...")

        # Remove hooks
        self.fused_optimizer.remove_hooks()
        self.block_offloader.cleanup()

        print(f"[FusedBlockSwapTrainer] Cleanup complete")

    def get_memory_stats(self):
        """
        Get current VRAM usage statistics

        Returns:
            Dictionary with memory stats
        """
        stats = {}

        if torch.cuda.is_available():
            stats['allocated_gb'] = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            stats['reserved_gb'] = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            stats['max_allocated_gb'] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        return stats

    def print_memory_stats(self):
        """Print current VRAM usage"""
        stats = self.get_memory_stats()

        print("=" * 60)
        print("[FusedBlockSwapTrainer] Memory Statistics")
        print("=" * 60)

        if stats:
            print(f"  Allocated:     {stats['allocated_gb']:.2f} GB")
            print(f"  Reserved:      {stats['reserved_gb']:.2f} GB")
            print(f"  Peak:          {stats['max_allocated_gb']:.2f} GB")

        print("=" * 60)
