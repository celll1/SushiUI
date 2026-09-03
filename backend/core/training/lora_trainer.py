"""
LoRA (Low-Rank Adaptation) Trainer for Stable Diffusion models.

This is a modular implementation using model-specific adapters. Which adapter
serves an architecture is declared on that architecture's ArchHandler
(``core/training/arch/``), not decided here.

Key improvements:
- Model-specific logic separated into adapters
- Clean separation of concerns
- Adding an architecture is one registry entry, not a branch here

References:
- sd-scripts (Apache-2 license) by kohya-ss
- ai-toolkit (MIT license) by ostris
- musubi-tuner (Apache-2 license) by kohya-ss (Z-Image support)

Author: Claude (2026-01-04)
"""

from pathlib import Path
from typing import Dict, List
import torch.nn as nn

from .base_trainer import BaseTrainer


class LoRATrainer(BaseTrainer):
    """
    LoRA Trainer for SD/SDXL/Z-Image models.

    Uses model-specific adapters for LoRA injection, parameter collection,
    and checkpoint saving.
    """

    def __init__(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dtype: str = 'fp32',
        train_unet: bool = True,
        train_text_encoder: bool = False,
        train_image_encoder: bool = False,  # Image Encoder (future support)
        **kwargs
    ):
        """
        Initialize LoRA Trainer.

        Args:
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha (scaling factor = alpha / rank)
            lora_dtype: Data type for LoRA weights ('fp32', 'fp16', 'bf16')
            train_unet: Whether to train U-Net/Transformer
            train_text_encoder: Whether to train Text Encoder(s)
            train_image_encoder: Whether to train Image Encoder (future support)
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # LoRA-specific settings (set before super().__init__)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / lora_rank
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder
        self.train_image_encoder = train_image_encoder

        # LoRA modules storage
        self.lora_layers: Dict[str, nn.Module] = {}

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Convert lora_dtype string to torch.dtype (after super().__init__ to have access to get_torch_dtype)
        from .base_trainer import get_torch_dtype
        self.lora_dtype = get_torch_dtype(lora_dtype)

        # Override log prefix
        self.log_prefix = "[LoRA Trainer]"

        # Create model-specific adapter
        self._create_adapter()

        # Apply LoRA using adapter
        self._apply_lora()

        self._setup_sensenova_phase_eviction()

        # Block swap deferred until after LoRA wraps Linear modules — the
        # LayerOffloadConductor snapshots layer state_dicts at registration
        # time and post-wrap key changes would break the swap.
        # No-op when blocks_to_swap == 0.
        if hasattr(self, "setup_anima_block_swap"):
            self.setup_anima_block_swap()
        if hasattr(self, "setup_lens_block_swap"):
            self.setup_lens_block_swap()
        if hasattr(self, "setup_ideogram4_block_swap"):
            self.setup_ideogram4_block_swap()
        if hasattr(self, "setup_minit2i_block_swap"):
            self.setup_minit2i_block_swap()
        if hasattr(self, "setup_krea2_block_swap"):
            self.setup_krea2_block_swap()
        if hasattr(self, "setup_ltx2_wrapper"):
            self.setup_ltx2_wrapper()
        if hasattr(self, "setup_ltx2_block_swap"):
            self.setup_ltx2_block_swap()
        if hasattr(self, "setup_acestep_block_swap"):
            self.setup_acestep_block_swap()
        if getattr(self, "is_minimax_h3", False):
            # Block swap is NOT required for this arch (measured: 22.45 GB peak
            # at 384x640x22 and 25.63 GB at the largest registered cell, both
            # unswapped) but the knob exists, and the ordering contract is the
            # same as every other arch's: after the LoRA wrap, never before.
            self.arch.setup_block_swap(self)

        print(f"{self.log_prefix} Initialized (rank={self.lora_rank}, alpha={self.lora_alpha})")
        ve_status = getattr(self, '_train_vision_encoder', False)
        print(f"{self.log_prefix} Training U-Net: {self.train_unet}, Text Encoder: {self.train_text_encoder}, Image Encoder: {self.train_image_encoder}, Vision Encoder: {ve_status}")

    def _create_adapter(self):
        """Create the LoRA adapter the arch registry declares for this model.

        Which adapter class an architecture uses, and the scope arguments it
        takes beyond (trainer, rank, alpha, dtype), are declared on its
        ArchHandler (``lora_adapter_class`` / ``lora_adapter_kwargs``). Reading
        them off ``self.arch`` rather than re-testing ``is_<arch>`` here also
        means the adapter and the training ops can never resolve to different
        architectures.
        """
        from core.training.arch import get_arch_handler

        # BaseTrainer.__init__ binds self.arch once every is_<arch> flag is
        # final; resolving from the flags otherwise keeps _create_adapter
        # callable on a bare trainer. Both go through the same registry.
        arch = getattr(self, "arch", None) or get_arch_handler(self)
        plan = arch.lora_adapter_plan(self)
        self.adapter = plan.build(self, self.lora_rank, self.lora_alpha, self.lora_dtype)
        print(f"{self.log_prefix} Using {plan.adapter_cls.__name__}{plan.log_detail}")

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            evictor = getattr(self, "sensenova_phase_evictor", None)
            if evictor is not None:
                try:
                    evictor.teardown()
                except Exception as exc:
                    print(f"{self.log_prefix} WARNING: SenseNova eviction teardown failed: {exc}")
                finally:
                    self.sensenova_phase_evictor = None

    def _setup_sensenova_phase_eviction(self) -> None:
        if not self.is_sensenova:
            return
        from core.training.ops import sensenova_ops

        # Same reason as the full-FT trainer: refuse the shared window without
        # its split HERE, or the flag falls through the eviction gate below and
        # silently does nothing. (The split itself is full-fine-tune only, which
        # is why LoRA installs no four-phase context.)
        sensenova_ops.assert_shared_prefix_contract(self)
        if not self.sensenova_mot_phase_eviction:
            return
        from .sensenova_phase_eviction import install_training_phase_eviction

        install_training_phase_eviction(self)

    def _apply_lora(self):
        """Apply LoRA to U-Net/Transformer and Text Encoders using adapter."""
        print(f"{self.log_prefix} Applying LoRA layers...")

        # Apply LoRA to U-Net/Transformer
        if self.train_unet:
            unet_count = self.adapter.apply_lora_to_unet(self.lora_layers)
            print(f"{self.log_prefix} Injected {unet_count} LoRA layers into U-Net/Transformer")
        else:
            print(f"{self.log_prefix} U-Net/Transformer LoRA skipped (train_unet=False)")

        # Apply LoRA to Text Encoder(s)
        if self.train_text_encoder:
            te_count = self.adapter.apply_lora_to_text_encoders(self.lora_layers)
            print(f"{self.log_prefix} Injected {te_count} LoRA layers into Text Encoder(s)")
        else:
            print(f"{self.log_prefix} Text Encoder LoRA skipped (train_text_encoder=False)")

        print(f"{self.log_prefix} Total LoRA layers: {len(self.lora_layers)}")

    def setup_trainable_parameters(self) -> List[Dict]:
        """
        Collect trainable parameters with per-component learning rates.

        Uses adapter to handle model-specific parameter grouping.

        Returns:
            List of parameter groups for optimizer
        """
        return self.adapter.setup_trainable_parameters(self.lora_layers)

    def save_checkpoint(self, step: int, epoch: int):
        """
        Save LoRA checkpoint.

        Uses adapter to handle model-specific checkpoint format.

        Args:
            step: Current training step
            epoch: Current training epoch
        """
        checkpoint_path = self.output_dir / f"{self.run_name}_step_{step:06d}.safetensors"
        self.adapter.save_checkpoint(self.lora_layers, step, epoch, checkpoint_path)
        # Save Vision Encoder checkpoint separately (if loaded)
        self._save_vision_encoder_checkpoint(step, epoch)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load LoRA checkpoint for resuming training.

        Args:
            checkpoint_path: Path to LoRA checkpoint (.safetensors)

        Returns:
            Step number from checkpoint
        """
        from safetensors import safe_open
        from safetensors.torch import load_file
        import torch

        print(f"{self.log_prefix} Loading LoRA checkpoint: {checkpoint_path}")

        # Extract metadata using safe_open
        step = 0
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if metadata and 'step' in metadata:
                step = int(metadata['step'])

        # Load checkpoint weights
        checkpoint = load_file(checkpoint_path)

        # Load LoRA weights into existing layers. The branch names its own
        # tensors, so an algebra that carries more than down/up resumes too;
        # `alpha` is deliberately not among them (a spec constant, not state).
        for lora_name, lora_layer in self.lora_layers.items():
            slice_ = {}
            for tensor_name in lora_layer.tensor_names():
                value = checkpoint.get(f"{lora_name}.{tensor_name}")
                if value is not None:
                    slice_[tensor_name] = value
            if slice_:
                lora_layer.load_tensors(slice_)

        print(f"{self.log_prefix} Loaded LoRA checkpoint from step {step}")
        return step
