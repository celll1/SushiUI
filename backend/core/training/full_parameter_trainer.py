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
        self.training_method = "full_finetune"  # Identify as full fine-tuning (not LoRA)
        self.train_unet = True  # Always train U-Net/Transformer in full parameter mode
        self.train_text_encoder = False  # Default: don't train text encoders (can be overridden)
        self.unet_lr = None  # Use default learning_rate
        self.text_encoder_lr = None
        self.text_encoder_1_lr = None
        self.text_encoder_2_lr = None

        # Override log prefix for Full Parameter Trainer
        self.specific_log_prefix = "[FullParameterTrainer]"

        print(f"{self.specific_log_prefix} Initialized for full parameter fine-tuning")

        # Get trainable parameter count (U-Net for SD/SDXL, Transformer for Z-Image)
        if self.is_zimage:
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            print(f"{self.specific_log_prefix} Trainable parameters (Transformer): {trainable_params:,}")
        else:
            trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
            print(f"{self.specific_log_prefix} Trainable parameters (U-Net): {trainable_params:,}")

    def _apply_lora(self):
        """Override: No LoRA application for full parameter training."""
        print(f"{self.specific_log_prefix} Skipping LoRA application (full parameter mode)")

        # Enable gradients for all model parameters (U-Net/Transformer)
        if self.is_zimage:
            # Z-Image: Enable gradients for transformer
            for param in self.transformer.parameters():
                param.requires_grad = True
            trainable_count = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            print(f"{self.specific_log_prefix} Enabled gradients for {trainable_count:,} Transformer parameters")
        else:
            # SD/SDXL: Enable gradients for U-Net
            for param in self.unet.parameters():
                param.requires_grad = True
            trainable_count = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
            print(f"{self.specific_log_prefix} Enabled gradients for {trainable_count:,} U-Net parameters")

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
        print(f"{self.specific_log_prefix} Setting up optimizer: {optimizer_type}, scheduler: {lr_scheduler_type}")

        # Collect all trainable parameters from U-Net/Transformer (and optionally text encoders)
        trainable_params = []

        # Model parameters (always trained in full parameter mode)
        if self.train_unet:
            if self.is_zimage:
                # Z-Image: Collect transformer parameters
                model_params = [p for p in self.transformer.parameters() if p.requires_grad]
                trainable_params.append({
                    "params": model_params,
                    "lr": self.unet_lr if self.unet_lr is not None else self.learning_rate
                })
                print(f"{self.specific_log_prefix} Transformer trainable parameters: {sum(p.numel() for p in model_params):,}")
            else:
                # SD/SDXL: Collect U-Net parameters
                unet_params = [p for p in self.unet.parameters() if p.requires_grad]
                trainable_params.append({
                    "params": unet_params,
                    "lr": self.unet_lr if self.unet_lr is not None else self.learning_rate
                })
                print(f"{self.specific_log_prefix} U-Net trainable parameters: {sum(p.numel() for p in unet_params):,}")

        # Text encoder parameters (optional, SD/SDXL only)
        if self.train_text_encoder and not self.is_zimage:
            if self.is_sdxl:
                # SDXL: Two text encoders
                if self.text_encoder:
                    te1_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
                    te1_lr = self.text_encoder_1_lr if self.text_encoder_1_lr is not None else (
                        self.text_encoder_lr if self.text_encoder_lr is not None else self.learning_rate
                    )
                    trainable_params.append({"params": te1_params, "lr": te1_lr})
                    print(f"{self.specific_log_prefix} Text Encoder 1 trainable parameters: {sum(p.numel() for p in te1_params):,}")

                if self.text_encoder_2:
                    te2_params = [p for p in self.text_encoder_2.parameters() if p.requires_grad]
                    te2_lr = self.text_encoder_2_lr if self.text_encoder_2_lr is not None else (
                        self.text_encoder_lr if self.text_encoder_lr is not None else self.learning_rate
                    )
                    trainable_params.append({"params": te2_params, "lr": te2_lr})
                    print(f"{self.specific_log_prefix} Text Encoder 2 trainable parameters: {sum(p.numel() for p in te2_params):,}")
            else:
                # SD1.5: Single text encoder
                if self.text_encoder:
                    te_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
                    te_lr = self.text_encoder_lr if self.text_encoder_lr is not None else self.learning_rate
                    trainable_params.append({"params": te_params, "lr": te_lr})
                    print(f"{self.specific_log_prefix} Text Encoder trainable parameters: {sum(p.numel() for p in te_params):,}")

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
                print(f"{self.specific_log_prefix} Using AdamW8bit optimizer")
            except ImportError:
                print(f"{self.specific_log_prefix} bitsandbytes not available, falling back to AdamW")
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

        print(f"{self.specific_log_prefix} Optimizer setup complete: {optimizer_type}, scheduler: {lr_scheduler_type}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load full model checkpoint from safetensors file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Step number from checkpoint
        """
        from safetensors.torch import load_file

        print(f"{self.specific_log_prefix} Loading checkpoint from {checkpoint_path}")

        state_dict = load_file(checkpoint_path)

        # Load model weights (U-Net for SD/SDXL, Transformer for Z-Image)
        if self.is_zimage:
            # Z-Image: Load transformer weights (Comfy format with no prefix)
            # Check if keys have "transformer." prefix (old format) or no prefix (new Comfy format)
            has_prefix = any(key.startswith("transformer.") for key in state_dict.keys())

            if has_prefix:
                # Old format: Remove "transformer." prefix
                transformer_state = {}
                for key, value in state_dict.items():
                    if key.startswith("transformer."):
                        new_key = key[len("transformer."):]
                        transformer_state[new_key] = value
                print(f"{self.specific_log_prefix} Loading transformer weights (old format with prefix)")
            else:
                # New Comfy format: Use directly
                transformer_state = state_dict
                print(f"{self.specific_log_prefix} Loading transformer weights (Comfy format without prefix)")

            if len(transformer_state) > 0:
                self.transformer.load_state_dict(transformer_state)
                print(f"{self.specific_log_prefix} Loaded {len(transformer_state)} transformer parameters")
            else:
                print(f"{self.specific_log_prefix} WARNING: No transformer weights found in checkpoint")
        else:
            # SD/SDXL: Load U-Net weights
            unet_state = {}
            for key, value in state_dict.items():
                if key.startswith("unet."):
                    # Remove "unet." prefix
                    new_key = key[len("unet."):]
                    unet_state[new_key] = value

            if len(unet_state) > 0:
                self.unet.load_state_dict(unet_state)
                print(f"{self.specific_log_prefix} Loaded {len(unet_state)} U-Net parameters")
            else:
                print(f"{self.specific_log_prefix} WARNING: No U-Net weights found in checkpoint")

        # Load text encoder weights (SD/SDXL only)
        if self.train_text_encoder and not self.is_zimage:
            if self.text_encoder:
                te_state = {}
                for key, value in state_dict.items():
                    if key.startswith("text_encoder."):
                        new_key = key[len("text_encoder."):]
                        te_state[new_key] = value
                if len(te_state) > 0:
                    self.text_encoder.load_state_dict(te_state)
                    print(f"{self.specific_log_prefix} Loaded {len(te_state)} Text Encoder parameters")

            if self.is_sdxl and self.text_encoder_2:
                te2_state = {}
                for key, value in state_dict.items():
                    if key.startswith("text_encoder_2."):
                        new_key = key[len("text_encoder_2."):]
                        te2_state[new_key] = value
                if len(te2_state) > 0:
                    self.text_encoder_2.load_state_dict(te2_state)
                    print(f"{self.specific_log_prefix} Loaded {len(te2_state)} Text Encoder 2 parameters")

        # Extract step from filename (format: full_step_{step}.safetensors)
        step = 0
        try:
            step_str = Path(checkpoint_path).stem.split("_")[-1]
            step = int(step_str)
        except (ValueError, IndexError):
            print(f"{self.specific_log_prefix} WARNING: Could not extract step from filename, defaulting to 0")

        # Load optimizer state if it exists
        optimizer_path = Path(checkpoint_path).with_suffix('.pt')
        if optimizer_path.exists() and self.optimizer is not None:
            try:
                print(f"{self.specific_log_prefix} Loading optimizer state from {optimizer_path}")
                checkpoint_data = torch.load(optimizer_path, map_location=self.device)
                self.optimizer.load_state_dict(checkpoint_data['optimizer'])
                if self.lr_scheduler and 'lr_scheduler' in checkpoint_data and checkpoint_data['lr_scheduler'] is not None:
                    self.lr_scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
                    print(f"{self.specific_log_prefix} Optimizer and LR scheduler states loaded")
                else:
                    print(f"{self.specific_log_prefix} Optimizer state loaded")
            except Exception as e:
                print(f"{self.specific_log_prefix} WARNING: Failed to load optimizer state: {e}")
                print(f"{self.specific_log_prefix} Training will continue with fresh optimizer state")
        else:
            if not optimizer_path.exists():
                print(f"{self.specific_log_prefix} No optimizer state found at {optimizer_path}")

        print(f"{self.specific_log_prefix} Checkpoint loaded (step {step})")
        return step

    def find_latest_checkpoint(self) -> Optional[tuple[str, int]]:
        """
        Find the latest valid checkpoint in output directory.

        Strategy:
        1. Find all .safetensors files
        2. Validate each checkpoint (can be loaded)
        3. Extract step number
        4. Return the one with highest step number

        Returns:
            Tuple of (checkpoint_path, step) or None if no valid checkpoint found
        """
        from safetensors.torch import load_file

        # Find all safetensors files
        checkpoint_files = list(self.output_dir.glob("*.safetensors"))

        if not checkpoint_files:
            return None

        # Validate checkpoints and extract step numbers
        valid_checkpoints = []
        for ckpt_path in checkpoint_files:
            try:
                # Try to load safetensors file (validation)
                state_dict = load_file(str(ckpt_path))

                # Extract step from filename
                stem = ckpt_path.stem
                parts = stem.split("_")
                step = 0
                if "step" in parts:
                    step_idx = parts.index("step")
                    if step_idx + 1 < len(parts):
                        step = int(parts[step_idx + 1])

                # Check if this checkpoint has model weights (basic validation)
                # For Z-Image: check for transformer weights (no prefix)
                # For SD/SDXL: check for unet weights ("unet." prefix)
                has_model_weights = False
                if self.is_zimage:
                    # Z-Image: Keys should be transformer layers without prefix (Comfy format)
                    has_model_weights = any("layers." in key or "final_layer" in key for key in state_dict.keys())
                else:
                    # SD/SDXL: Keys should have "unet." prefix
                    has_model_weights = any(key.startswith("unet.") for key in state_dict.keys())

                if has_model_weights:
                    valid_checkpoints.append((str(ckpt_path), step))
                    print(f"{self.specific_log_prefix} Found valid checkpoint: {ckpt_path.name} (step {step})")

            except Exception as e:
                print(f"{self.specific_log_prefix} Skipping invalid checkpoint {ckpt_path.name}: {e}")
                continue

        if not valid_checkpoints:
            return None

        # Sort by step and return latest
        valid_checkpoints.sort(key=lambda x: x[1], reverse=True)
        latest_ckpt, latest_step = valid_checkpoints[0]

        # Check for optimizer state
        optimizer_path = Path(latest_ckpt).with_suffix('.pt')
        if optimizer_path.exists():
            print(f"{self.specific_log_prefix} Latest checkpoint: {latest_ckpt} (step {latest_step}, with optimizer state)")
        else:
            print(f"{self.specific_log_prefix} Latest checkpoint: {latest_ckpt} (step {latest_step}, no optimizer state)")

        return latest_ckpt, latest_step

    def save_checkpoint(self, step: int, save_path: Optional[str] = None, save_optimizer: bool = True, max_to_keep: Optional[int] = None, save_every: int = 100, run_id: Optional[int] = None, epoch: Optional[int] = None):
        """
        Save full model checkpoint.

        Args:
            step: Current training step
            save_path: Path to save checkpoint (default: output_dir/full_step_{step}.safetensors)
            save_optimizer: Whether to save optimizer state
            max_to_keep: Maximum number of checkpoints to keep (None = keep all)
            save_every: Save interval (used for checkpoint cleanup)
            run_id: Training run ID for database registration (optional)
            epoch: Current epoch number (optional)
        """
        if save_path is None:
            # Extract short name from run_name (same logic as LoRATrainer)
            import re
            match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
            if match:
                short_name = match.group(1)
            else:
                short_name = self.run_name

            save_path = self.output_dir / f"{short_name}_step_{step}.safetensors"
        else:
            save_path = Path(save_path)

        print(f"{self.specific_log_prefix} Saving checkpoint to {save_path}")

        # Flatten model state dict (safetensors requires flat dict of tensors)
        checkpoint_data = {}

        if self.is_zimage:
            # Z-Image: Save transformer state in Comfy format (no prefix)
            # This allows checkpoints to be used directly for inference
            # IMPORTANT: Use transformer_original if wrapped (avoid "transformer." prefix from wrapper)
            transformer_to_save = getattr(self, 'transformer_original', self.transformer)
            checkpoint_data = transformer_to_save.state_dict()
        else:
            # SD/SDXL: Save U-Net state
            for key, value in self.unet.state_dict().items():
                checkpoint_data[f"unet.{key}"] = value

        # Optionally save text encoder states (SD/SDXL only)
        if self.train_text_encoder and not self.is_zimage:
            if self.text_encoder:
                for key, value in self.text_encoder.state_dict().items():
                    checkpoint_data[f"text_encoder.{key}"] = value
            if self.is_sdxl and self.text_encoder_2:
                for key, value in self.text_encoder_2.state_dict().items():
                    checkpoint_data[f"text_encoder_2.{key}"] = value

        # Save as safetensors (with error handling for disk space issues)
        from safetensors.torch import save_file
        try:
            save_file(checkpoint_data, str(save_path))
        except Exception as e:
            error_msg = str(e)
            # Check if it's a disk space error (os error 112 on Windows)
            if "os error 112" in error_msg or "No space left" in error_msg or "I/O error" in error_msg:
                print(f"{self.specific_log_prefix} WARNING: Checkpoint save failed due to insufficient disk space")
                print(f"{self.specific_log_prefix} Training will continue. Please free up disk space for future checkpoints.")
                print(f"{self.specific_log_prefix} Error details: {error_msg}")
                return  # Skip the rest of checkpoint saving but continue training
            else:
                # For other errors, re-raise
                raise

        # Save optimizer state separately (as .pt file)
        if save_optimizer and self.optimizer is not None:
            optimizer_path = save_path.with_suffix(".pt")
            try:
                torch.save({
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    "step": step,
                }, optimizer_path)
                print(f"{self.specific_log_prefix} Optimizer state saved: {optimizer_path}")
            except Exception as e:
                error_msg = str(e)
                if "os error 112" in error_msg or "No space left" in error_msg or "I/O error" in error_msg:
                    print(f"{self.specific_log_prefix} WARNING: Optimizer state save failed due to insufficient disk space")
                    print(f"{self.specific_log_prefix} Model checkpoint was saved successfully")
                else:
                    print(f"{self.specific_log_prefix} WARNING: Failed to save optimizer state: {error_msg}")

        print(f"{self.specific_log_prefix} Checkpoint saved: {save_path}")

        # Register checkpoint in database
        if run_id is not None:
            try:
                from database import get_training_db
                from database.models import TrainingCheckpoint

                db = next(get_training_db())
                try:
                    file_size = save_path.stat().st_size
                    checkpoint_record = TrainingCheckpoint(
                        run_id=run_id,
                        checkpoint_name=save_path.name,
                        step=step,
                        epoch=epoch,
                        file_path=str(save_path),
                        file_size=file_size,
                        loss=None  # Loss can be added if tracked
                    )
                    db.add(checkpoint_record)
                    db.commit()
                    print(f"{self.specific_log_prefix} Checkpoint registered in database (run_id={run_id}, step={step})")
                finally:
                    db.close()
            except Exception as e:
                print(f"{self.specific_log_prefix} WARNING: Failed to register checkpoint in database: {e}")

        # Cleanup old checkpoints if max_to_keep is set
        if max_to_keep is not None and max_to_keep > 0:
            self._cleanup_old_checkpoints(step, max_to_keep, save_every)

    def _cleanup_old_checkpoints(self, current_step: int, max_to_keep: int, save_every: int):
        """
        Remove old checkpoints to keep only the latest N checkpoints.

        Args:
            current_step: Current training step
            max_to_keep: Maximum number of checkpoints to keep
            save_every: Save interval (used to calculate which checkpoint to delete)
        """
        # Calculate which step to remove
        remove_step = current_step - (save_every * max_to_keep)

        if remove_step < save_every:
            # No checkpoint to remove yet
            return

        # Remove checkpoint at remove_step
        checkpoint_path = self.output_dir / f"full_step_{remove_step}.safetensors"
        optimizer_path = self.output_dir / f"full_step_{remove_step}.pt"

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                print(f"{self.specific_log_prefix} Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"{self.specific_log_prefix} WARNING: Failed to remove old checkpoint {checkpoint_path}: {e}")

        if optimizer_path.exists():
            try:
                optimizer_path.unlink()
                print(f"{self.specific_log_prefix} Removed old optimizer state: {optimizer_path}")
            except Exception as e:
                print(f"{self.specific_log_prefix} WARNING: Failed to remove old optimizer state {optimizer_path}: {e}")

    def merge_and_save(self, output_path: str):
        """
        Override: For full parameter training, just save the model directly.

        Args:
            output_path: Output safetensors path
        """
        print(f"{self.specific_log_prefix} Saving full model to {output_path}")

        checkpoint_data = {}

        # Convert to float32 for saving (compatibility)
        if self.is_zimage:
            # Z-Image: Save transformer in Comfy format (no prefix, flat keys)
            # IMPORTANT: Use transformer_original if wrapped (avoid "transformer." prefix from wrapper)
            transformer_to_save = getattr(self, 'transformer_original', self.transformer)
            original_dtype = next(transformer_to_save.parameters()).dtype
            transformer_to_save.to(dtype=torch.float32)
            # Save without "transformer." prefix (Comfy format for compatibility with inference)
            checkpoint_data = transformer_to_save.state_dict()
        else:
            # SD/SDXL: Save U-Net
            original_dtype = next(self.unet.parameters()).dtype
            self.unet.to(dtype=torch.float32)
            checkpoint_data["unet"] = self.unet.state_dict()

        # Optionally include text encoders (SD/SDXL only)
        if self.train_text_encoder and not self.is_zimage:
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
        if self.is_zimage:
            transformer_to_save.to(dtype=original_dtype)
        else:
            self.unet.to(dtype=original_dtype)

        if self.train_text_encoder and not self.is_zimage:
            if self.text_encoder:
                self.text_encoder.to(dtype=original_dtype)
            if self.is_sdxl and self.text_encoder_2:
                self.text_encoder_2.to(dtype=original_dtype)

        print(f"{self.specific_log_prefix} Full model saved: {output_path}")
