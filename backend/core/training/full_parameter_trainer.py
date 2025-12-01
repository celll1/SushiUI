"""
Full Parameter Trainer for Stable Diffusion Models

Trains all U-Net parameters (full fine-tuning) instead of LoRA adapters.
Based on LoRATrainer but removes LoRA-specific logic.
"""

from pathlib import Path
from typing import Optional, Callable, List, Dict
import torch
from core.training.lora_trainer import LoRATrainer


class FullParameterTrainer(LoRATrainer):
    """
    Full parameter fine-tuning trainer.
    Inherits from LoRATrainer but trains all U-Net parameters instead of LoRA adapters.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        learning_rate: float = 1e-4,
        weight_dtype: str = "fp16",
        training_dtype: str = "fp16",
        output_dtype: str = "fp32",
        vae_dtype: str = "fp16",
        mixed_precision: bool = True,
        debug_vram: bool = False,
        use_flash_attention: bool = False,
        min_snr_gamma: float = 5.0,
    ):
        """
        Initialize full parameter trainer.

        Args:
            model_path: Path to base model
            output_dir: Output directory for checkpoints
            learning_rate: Learning rate
            weight_dtype: Weight dtype (fp16, fp32, bf16, fp8_e4m3fn, fp8_e5m2)
            training_dtype: Training dtype (activation dtype, fp16, bf16, fp8_e4m3fn, fp8_e5m2)
            output_dtype: Output latent dtype (fp32, fp16, bf16, fp8_e4m3fn, fp8_e5m2)
            vae_dtype: VAE-specific dtype (fp16 recommended for SDXL VAE)
            mixed_precision: Enable mixed precision training
            debug_vram: Enable VRAM profiling
            use_flash_attention: Enable Flash Attention
            min_snr_gamma: Min-SNR gamma weighting (0 to disable)
        """
        # Call parent __init__ with dummy lora_rank/lora_alpha (won't be used)
        super().__init__(
            model_path=model_path,
            output_dir=output_dir,
            lora_rank=1,  # Dummy value, not used
            lora_alpha=1,  # Dummy value, not used
            learning_rate=learning_rate,
            weight_dtype=weight_dtype,
            training_dtype=training_dtype,
            output_dtype=output_dtype,
            vae_dtype=vae_dtype,
            mixed_precision=mixed_precision,
            debug_vram=debug_vram,
            use_flash_attention=use_flash_attention,
            min_snr_gamma=min_snr_gamma,
        )

        # Full parameter training settings
        self.train_unet = True  # Always train U-Net in full parameter mode
        self.train_text_encoder = False  # Default: don't train text encoders (can be overridden)
        self.unet_lr = None  # Use default learning_rate
        self.text_encoder_lr = None
        self.text_encoder_1_lr = None
        self.text_encoder_2_lr = None

        print("[FullParameterTrainer] Initialized for full parameter fine-tuning")
        print(f"[FullParameterTrainer] Trainable parameters: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad):,}")

    def _apply_lora(self):
        """Override: No LoRA application for full parameter training."""
        print("[FullParameterTrainer] Skipping LoRA application (full parameter mode)")

        # Enable gradients for all U-Net parameters
        for param in self.unet.parameters():
            param.requires_grad = True

        trainable_count = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f"[FullParameterTrainer] Enabled gradients for {trainable_count:,} U-Net parameters")

    def setup_lora(self):
        """Override: No LoRA setup needed for full parameter training."""
        pass  # Do nothing - we train all U-Net parameters directly

    def setup_optimizer(self, optimizer_type: str = "adamw8bit", lr_scheduler_type: str = "constant"):
        """
        Setup optimizer for U-Net parameters.

        Args:
            optimizer_type: Optimizer type ("adamw8bit", "adamw", "sgd", "adafactor")
            lr_scheduler_type: LR scheduler type ("constant", "cosine", "polynomial")
        """
        print(f"[FullParameterTrainer] Setting up optimizer: {optimizer_type}, scheduler: {lr_scheduler_type}")

        # Collect all trainable parameters from U-Net (and optionally text encoders)
        trainable_params = []

        # U-Net parameters (always trained in full parameter mode)
        if self.train_unet:
            unet_params = [p for p in self.unet.parameters() if p.requires_grad]
            trainable_params.append({
                "params": unet_params,
                "lr": self.unet_lr if self.unet_lr is not None else self.learning_rate
            })
            print(f"[FullParameterTrainer] U-Net trainable parameters: {sum(p.numel() for p in unet_params):,}")

        # Text encoder parameters (optional)
        if self.train_text_encoder:
            if self.is_sdxl:
                # SDXL: Two text encoders
                if self.text_encoder:
                    te1_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
                    te1_lr = self.text_encoder_1_lr if self.text_encoder_1_lr is not None else (
                        self.text_encoder_lr if self.text_encoder_lr is not None else self.learning_rate
                    )
                    trainable_params.append({"params": te1_params, "lr": te1_lr})
                    print(f"[FullParameterTrainer] Text Encoder 1 trainable parameters: {sum(p.numel() for p in te1_params):,}")

                if self.text_encoder_2:
                    te2_params = [p for p in self.text_encoder_2.parameters() if p.requires_grad]
                    te2_lr = self.text_encoder_2_lr if self.text_encoder_2_lr is not None else (
                        self.text_encoder_lr if self.text_encoder_lr is not None else self.learning_rate
                    )
                    trainable_params.append({"params": te2_params, "lr": te2_lr})
                    print(f"[FullParameterTrainer] Text Encoder 2 trainable parameters: {sum(p.numel() for p in te2_params):,}")
            else:
                # SD1.5: Single text encoder
                if self.text_encoder:
                    te_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
                    te_lr = self.text_encoder_lr if self.text_encoder_lr is not None else self.learning_rate
                    trainable_params.append({"params": te_params, "lr": te_lr})
                    print(f"[FullParameterTrainer] Text Encoder trainable parameters: {sum(p.numel() for p in te_params):,}")

        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found. Enable train_unet or train_text_encoder.")

        # Setup optimizer
        if optimizer_type == "adamw8bit":
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=self.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                    eps=1e-8,
                )
                print("[FullParameterTrainer] Using AdamW8bit optimizer")
            except ImportError:
                print("[FullParameterTrainer] bitsandbytes not available, falling back to AdamW")
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                    eps=1e-8,
                )
        elif optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                eps=1e-8,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Setup LR scheduler (placeholder total_steps, will be updated in train())
        from diffusers.optimization import get_scheduler as get_diffusers_scheduler
        self.lr_scheduler = get_diffusers_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=1000,  # Placeholder, will be updated in train()
        )

        print(f"[FullParameterTrainer] Optimizer setup complete: {optimizer_type}, scheduler: {lr_scheduler_type}")

    def save_checkpoint(self, step: int, checkpoint_path: Path, save_optimizer: bool = True):
        """
        Save full model checkpoint.

        Args:
            step: Current training step
            checkpoint_path: Path to save checkpoint
            save_optimizer: Whether to save optimizer state
        """
        print(f"[FullParameterTrainer] Saving checkpoint to {checkpoint_path}")

        # Save U-Net state dict
        checkpoint_data = {
            "unet": self.unet.state_dict(),
            "step": step,
        }

        # Optionally save text encoder states
        if self.train_text_encoder:
            if self.text_encoder:
                checkpoint_data["text_encoder"] = self.text_encoder.state_dict()
            if self.is_sdxl and self.text_encoder_2:
                checkpoint_data["text_encoder_2"] = self.text_encoder_2.state_dict()

        # Save as safetensors
        from safetensors.torch import save_file
        save_file(checkpoint_data, str(checkpoint_path))

        # Save optimizer state separately (as .pt file)
        if save_optimizer and self.optimizer is not None:
            optimizer_path = checkpoint_path.with_suffix(".pt")
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "step": step,
            }, optimizer_path)
            print(f"[FullParameterTrainer] Optimizer state saved: {optimizer_path}")

        print(f"[FullParameterTrainer] Checkpoint saved: {checkpoint_path}")

    def merge_and_save(self, output_path: str):
        """
        Override: For full parameter training, just save the U-Net directly.

        Args:
            output_path: Output safetensors path
        """
        print(f"[FullParameterTrainer] Saving full model to {output_path}")

        # Convert to float32 for saving (compatibility)
        original_dtype = next(self.unet.parameters()).dtype
        self.unet.to(dtype=torch.float32)

        checkpoint_data = {
            "unet": self.unet.state_dict(),
        }

        # Optionally include text encoders
        if self.train_text_encoder:
            if self.text_encoder:
                self.text_encoder.to(dtype=torch.float32)
                checkpoint_data["text_encoder"] = self.text_encoder.state_dict()
            if self.is_sdxl and self.text_encoder_2:
                self.text_encoder_2.to(dtype=torch.float32)
                checkpoint_data["text_encoder_2"] = self.text_encoder_2.state_dict()

        # Save as safetensors
        from safetensors.torch import save_file
        save_file(checkpoint_data, output_path)

        # Restore original dtype
        self.unet.to(dtype=original_dtype)
        if self.train_text_encoder:
            if self.text_encoder:
                self.text_encoder.to(dtype=original_dtype)
            if self.is_sdxl and self.text_encoder_2:
                self.text_encoder_2.to(dtype=original_dtype)

        print(f"[FullParameterTrainer] Full model saved: {output_path}")
