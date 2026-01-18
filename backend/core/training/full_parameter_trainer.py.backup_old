"""
Full Parameter Trainer for Stable Diffusion Models

Trains all U-Net/Transformer parameters (full fine-tuning).
Inherits from BaseTrainer and implements full parameter-specific logic.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import torch
from safetensors.torch import save_file, load_file

from .base_trainer import BaseTrainer


class FullParameterTrainer(BaseTrainer):
    """
    Full parameter fine-tuning trainer.
    Trains all U-Net/Transformer parameters instead of LoRA adapters.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        run_name: str = None,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        weight_dtype: str = "fp16",
        training_dtype: str = "fp16",
        output_dtype: str = "fp32",
        vae_dtype: str = "fp16",
        mixed_precision: bool = True,
        debug_vram: bool = False,
        use_flash_attention: bool = False,
        min_snr_gamma: float = 5.0,
        unet_lr: Optional[float] = None,
        text_encoder_lr: Optional[float] = None,
        text_encoder_1_lr: Optional[float] = None,
        text_encoder_2_lr: Optional[float] = None,
        blocks_to_swap: int = 0,
        use_pinned_memory: bool = False,
        num_optimizer_groups: int = 0,
        # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
        prompt_chunking_mode: str = "a1111",
        max_prompt_chunks: int = 0,
    ):
        """
        Initialize full parameter trainer.

        Args:
            model_path: Path to base model
            output_dir: Output directory for checkpoints
            run_name: Training run name (for checkpoint filename generation)
            learning_rate: Learning rate
            device: Device to use (cuda/cpu)
            weight_dtype: Weight dtype (fp16, fp32, bf16, fp8_e4m3fn, fp8_e5m2)
            training_dtype: Training dtype (activation dtype, fp16, bf16, fp8_e4m3fn, fp8_e5m2)
            output_dtype: Output dtype (fp32, fp16, bf16, fp8_e4m3fn, fp8_e5m2)
            vae_dtype: VAE-specific dtype (fp16 recommended for SDXL VAE)
            mixed_precision: Enable mixed precision training
            debug_vram: Enable VRAM profiling
            use_flash_attention: Enable Flash Attention
            min_snr_gamma: Min-SNR gamma weighting (0 to disable)
            unet_lr: U-Net specific learning rate (defaults to learning_rate)
            text_encoder_lr: Text encoder learning rate (defaults to learning_rate)
            text_encoder_1_lr: Text encoder 1 learning rate (SDXL only)
            text_encoder_2_lr: Text encoder 2 learning rate (SDXL only)
        """
        # Full parameter training settings
        self.training_method = "full_finetune"
        self.train_unet = True
        self.train_text_encoder = False
        self.specific_log_prefix = "[FullParameterTrainer]"

        # Call parent __init__
        super().__init__(
            model_path=model_path,
            output_dir=output_dir,
            run_name=run_name,
            learning_rate=learning_rate,
            device=device,
            weight_dtype=weight_dtype,
            training_dtype=training_dtype,
            output_dtype=output_dtype,
            vae_dtype=vae_dtype,
            mixed_precision=mixed_precision,
            debug_vram=debug_vram,
            use_flash_attention=use_flash_attention,
            min_snr_gamma=min_snr_gamma,
            unet_lr=unet_lr,
            text_encoder_lr=text_encoder_lr,
            text_encoder_1_lr=text_encoder_1_lr,
            text_encoder_2_lr=text_encoder_2_lr,
            blocks_to_swap=blocks_to_swap,
            use_pinned_memory=use_pinned_memory,
            num_optimizer_groups=num_optimizer_groups,
            prompt_chunking_mode=prompt_chunking_mode,
            max_prompt_chunks=max_prompt_chunks,
        )

        # Override log prefix
        self.log_prefix = self.specific_log_prefix

        print(f"{self.specific_log_prefix} Initialized for full parameter fine-tuning")

        # Get trainable parameter count
        if self.is_zimage:
            trainable_params = sum(p.numel() for p in self.transformer_original.parameters() if p.requires_grad)
            print(f"{self.specific_log_prefix} Trainable parameters (Transformer): {trainable_params:,}")
        else:
            trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
            print(f"{self.specific_log_prefix} Trainable parameters (U-Net): {trainable_params:,}")

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        """
        Enable gradients for all model parameters.

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []

        # Model parameters (U-Net/Transformer)
        if self.train_unet:
            if self.is_zimage:
                # Z-Image: Enable gradients for transformer
                for param in self.transformer_original.parameters():
                    param.requires_grad = True

                model_params = [p for p in self.transformer_original.parameters() if p.requires_grad]
                param_groups.append({
                    "params": model_params,
                    "lr": self.unet_lr
                })
                print(f"{self.specific_log_prefix} Enabled gradients for {sum(p.numel() for p in model_params):,} Transformer parameters")
            else:
                # SD/SDXL: Enable gradients for U-Net
                for param in self.unet.parameters():
                    param.requires_grad = True

                unet_params = [p for p in self.unet.parameters() if p.requires_grad]
                param_groups.append({
                    "params": unet_params,
                    "lr": self.unet_lr
                })
                print(f"{self.specific_log_prefix} Enabled gradients for {sum(p.numel() for p in unet_params):,} U-Net parameters")

        # Text encoder parameters (optional, SD/SDXL only)
        if self.train_text_encoder and not self.is_zimage:
            if self.is_sdxl:
                # SDXL: Two text encoders
                if self.text_encoder:
                    for param in self.text_encoder.parameters():
                        param.requires_grad = True

                    te1_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
                    param_groups.append({"params": te1_params, "lr": self.text_encoder_1_lr})
                    print(f"{self.specific_log_prefix} Text Encoder 1 trainable parameters: {sum(p.numel() for p in te1_params):,}")

                if self.text_encoder_2:
                    for param in self.text_encoder_2.parameters():
                        param.requires_grad = True

                    te2_params = [p for p in self.text_encoder_2.parameters() if p.requires_grad]
                    param_groups.append({"params": te2_params, "lr": self.text_encoder_2_lr})
                    print(f"{self.specific_log_prefix} Text Encoder 2 trainable parameters: {sum(p.numel() for p in te2_params):,}")
            else:
                # SD1.5: Single text encoder
                if self.text_encoder:
                    for param in self.text_encoder.parameters():
                        param.requires_grad = True

                    te_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
                    param_groups.append({"params": te_params, "lr": self.text_encoder_lr})
                    print(f"{self.specific_log_prefix} Text Encoder trainable parameters: {sum(p.numel() for p in te_params):,}")

        if len(param_groups) == 0:
            raise ValueError("No trainable parameters found. Enable train_unet or train_text_encoder.")

        # Set model modes
        # VAE is always in eval mode (never trained)
        self.vae.eval()

        if self.is_zimage:
            # Z-Image: Transformer in train mode, Text Encoder in eval mode (frozen)
            self.transformer.train()
            self.text_encoder.eval()
            print(f"{self.specific_log_prefix} Z-Image Transformer set to train mode, Text Encoder to eval mode (frozen)")
        else:
            # SD/SDXL: U-Net and Text Encoders in train mode
            if self.train_unet:
                self.unet.train()
            if self.train_text_encoder:
                self.text_encoder.train()
                if self.text_encoder_2 is not None:
                    self.text_encoder_2.train()
            print(f"{self.specific_log_prefix} Models set to train mode for gradient checkpointing")

        return param_groups

    def save_checkpoint(self, step: int, epoch: int = 0):
        """
        Save full model checkpoint.

        Args:
            step: Current training step
            epoch: Current epoch
        """
        # Extract short name from run_name
        import re
        match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
        if match:
            short_name = match.group(1)
        else:
            short_name = self.run_name

        save_path = self.output_dir / f"{short_name}_step_{step}.safetensors"

        print(f"{self.specific_log_prefix} Saving checkpoint to {save_path}")

        # Flatten model state dict
        checkpoint_data = {}

        if self.is_zimage:
            # Z-Image: Save transformer state in Comfy format (no prefix)
            checkpoint_data = self.transformer_original.state_dict()
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

        # Prepare metadata (ModelSpec 1.0.0)
        metadata = {
            "training_step": str(step),
            "epoch": str(epoch),
        }

        # Add ModelSpec metadata for prediction configuration
        if hasattr(self, 'noise_process') and hasattr(self, 'prediction_target'):
            # ModelSpec standard keys
            if self.is_zimage:
                metadata["modelspec.architecture"] = "z-image-transformer"
            elif self.is_sdxl:
                metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
            else:
                metadata["modelspec.architecture"] = "stable-diffusion-v1"

            metadata["modelspec.implementation"] = "https://github.com/huggingface/diffusers"

            # Prediction configuration (unified training framework)
            metadata["modelspec.noise_process"] = self.noise_process  # "ddpm" or "flow"
            metadata["modelspec.prediction_type"] = self.prediction_target  # "epsilon", "velocity", "sample"

            # Legacy compatibility: add v_pred marker for v-prediction models
            if self.prediction_target == "velocity" and self.noise_process == "ddpm":
                # Add empty v_pred tensor as marker (NoobAI-XL-Vpred style)
                checkpoint_data["v_pred"] = torch.tensor([])

        # Save as safetensors
        try:
            save_file(checkpoint_data, str(save_path), metadata=metadata)
        except Exception as e:
            error_msg = str(e)
            if "os error 112" in error_msg or "No space left" in error_msg or "I/O error" in error_msg:
                print(f"{self.specific_log_prefix} WARNING: Checkpoint save failed due to insufficient disk space")
                print(f"{self.specific_log_prefix} Training will continue. Please free up disk space for future checkpoints.")
                return
            else:
                raise

        # Save optimizer state separately
        if self.optimizer is not None:
            optimizer_path = save_path.with_suffix(".pt")
            try:
                torch.save({
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    "step": step,
                    "epoch": epoch,
                }, optimizer_path)
                print(f"{self.specific_log_prefix} Optimizer state saved: {optimizer_path}")
            except Exception as e:
                error_msg = str(e)
                if "os error 112" in error_msg or "No space left" in error_msg or "I/O error" in error_msg:
                    print(f"{self.specific_log_prefix} WARNING: Optimizer state save failed due to insufficient disk space")
                else:
                    print(f"{self.specific_log_prefix} WARNING: Failed to save optimizer state: {error_msg}")

        print(f"{self.specific_log_prefix} Checkpoint saved: {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load full model checkpoint from safetensors file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Step number from checkpoint
        """
        print(f"{self.specific_log_prefix} Loading checkpoint from {checkpoint_path}")

        state_dict = load_file(checkpoint_path)

        # Load model weights
        if self.is_zimage:
            # Z-Image: Load transformer weights (handle both prefixed and Comfy formats)
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
                self.transformer_original.load_state_dict(transformer_state)
                print(f"{self.specific_log_prefix} Loaded {len(transformer_state)} transformer parameters")
            else:
                print(f"{self.specific_log_prefix} WARNING: No transformer weights found in checkpoint")
        else:
            # SD/SDXL: Load U-Net weights
            unet_state = {}
            for key, value in state_dict.items():
                if key.startswith("unet."):
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

        # Extract step from filename
        step = 0
        epoch = 0
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

                if 'epoch' in checkpoint_data:
                    epoch = checkpoint_data['epoch']

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

        print(f"{self.specific_log_prefix} Checkpoint loaded (step {step}, epoch {epoch})")

        return step

    def find_latest_checkpoint(self) -> Optional[Tuple[str, int]]:
        """
        Find the latest valid checkpoint in output directory.

        Returns:
            Tuple of (checkpoint_path, step) or None if no checkpoints exist
        """
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

                # Check if this checkpoint has model weights
                has_model_weights = False
                if self.is_zimage:
                    # Z-Image: Check for transformer keys
                    has_model_weights = any(
                        "layers." in key or "final_layer" in key
                        for key in state_dict.keys()
                    )
                else:
                    # SD/SDXL: Check for U-Net keys
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

        return (latest_ckpt, latest_step)

    def merge_and_save(self, output_path: str):
        """
        Save full model directly (no merge needed).

        Args:
            output_path: Output safetensors path
        """
        print(f"{self.specific_log_prefix} Saving full model to {output_path}")

        checkpoint_data = {}

        # Convert to float32 for saving (compatibility)
        if self.is_zimage:
            # Z-Image: Save transformer in Comfy format (no prefix)
            original_dtype = next(self.transformer_original.parameters()).dtype
            self.transformer_original.to(dtype=torch.float32)
            checkpoint_data = self.transformer_original.state_dict()
        else:
            # SD/SDXL: Save U-Net
            original_dtype = next(self.unet.parameters()).dtype
            self.unet.to(dtype=torch.float32)

            for key, value in self.unet.state_dict().items():
                checkpoint_data[f"unet.{key}"] = value

        # Optionally include text encoders (SD/SDXL only)
        if self.train_text_encoder and not self.is_zimage:
            if self.text_encoder:
                self.text_encoder.to(dtype=torch.float32)
                for key, value in self.text_encoder.state_dict().items():
                    checkpoint_data[f"text_encoder.{key}"] = value

            if self.is_sdxl and self.text_encoder_2:
                self.text_encoder_2.to(dtype=torch.float32)
                for key, value in self.text_encoder_2.state_dict().items():
                    checkpoint_data[f"text_encoder_2.{key}"] = value

        # Save as safetensors
        save_file(checkpoint_data, output_path)

        # Restore original dtype
        if self.is_zimage:
            self.transformer_original.to(dtype=original_dtype)
        else:
            self.unet.to(dtype=original_dtype)

        if self.train_text_encoder and not self.is_zimage:
            if self.text_encoder:
                self.text_encoder.to(dtype=original_dtype)
            if self.is_sdxl and self.text_encoder_2:
                self.text_encoder_2.to(dtype=original_dtype)

        print(f"{self.specific_log_prefix} Full model saved: {output_path}")
