"""
Full Parameter Trainer for Stable Diffusion models.

This is a complete rewrite from scratch, referencing:
- sd-scripts (Apache-2 license) by kohya-ss
- ai-toolkit (MIT license) by ostris

Key improvements:
- Clean, from-scratch implementation (no legacy bugs)
- NO dtype conversions (all FP32 for debugging)
- Proper checkpoint management
- Compatible with SushiUI BaseTrainer architecture

Author: Claude (2026-01-04)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
from safetensors.torch import save_file, load_file

from .base_trainer import BaseTrainer


# ============================================================
# Full Parameter Trainer
# ============================================================

class FullParameterTrainer(BaseTrainer):
    """
    Full Parameter Trainer for SD/SDXL models.

    Trains all parameters (no LoRA, no adapters).

    References: sd-scripts (Apache-2), ai-toolkit (MIT)
    """

    def __init__(
        self,
        train_unet: bool = True,
        train_text_encoder: bool = False,
        **kwargs
    ):
        # Full fine-tune settings (set before super().__init__)
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Override log prefix
        self.log_prefix = "[Full Parameter Trainer]"

        # Enable gradients for trainable components
        self._enable_gradients()

        print(f"{self.log_prefix} Initialized")
        print(f"{self.log_prefix} Training U-Net: {self.train_unet}, Text Encoder: {self.train_text_encoder}")

    def _enable_gradients(self):
        """Enable gradients for trainable components."""
        print(f"{self.log_prefix} Enabling gradients for trainable components...")

        # U-Net / Transformer (Z-Image)
        if self.train_unet:
            if hasattr(self, 'unet') and self.unet is not None:
                self.unet.requires_grad_(True)
                trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
                print(f"{self.log_prefix} U-Net: {trainable_params:,} trainable parameters")
            elif hasattr(self, 'transformer') and self.transformer is not None:
                self.transformer.requires_grad_(True)
                trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
                print(f"{self.log_prefix} Transformer: {trainable_params:,} trainable parameters")
            else:
                print(f"{self.log_prefix} WARNING: No U-Net or Transformer found")
        else:
            print(f"{self.log_prefix} U-Net training disabled")

        # Text Encoders
        if self.train_text_encoder:
            if hasattr(self, 'text_encoder') and self.text_encoder is not None:
                self.text_encoder.requires_grad_(True)
                trainable_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
                print(f"{self.log_prefix} Text Encoder 1: {trainable_params:,} trainable parameters")

            if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
                self.text_encoder_2.requires_grad_(True)
                trainable_params = sum(p.numel() for p in self.text_encoder_2.parameters() if p.requires_grad)
                print(f"{self.log_prefix} Text Encoder 2: {trainable_params:,} trainable parameters")
        else:
            print(f"{self.log_prefix} Text Encoder training disabled")

    def setup_trainable_parameters(self) -> List[Dict]:
        """
        Collect trainable parameters with per-component learning rates.

        Returns:
            List of parameter groups for optimizer
        """
        params = []

        # U-Net / Transformer parameters
        if self.train_unet:
            if hasattr(self, 'unet') and self.unet is not None:
                unet_params = [p for p in self.unet.parameters() if p.requires_grad]
                if unet_params:
                    params.append({"params": unet_params, "lr": self.unet_lr})
            elif hasattr(self, 'transformer') and self.transformer is not None:
                transformer_params = [p for p in self.transformer.parameters() if p.requires_grad]
                if transformer_params:
                    params.append({"params": transformer_params, "lr": self.unet_lr})

        # Text Encoder parameters
        if self.train_text_encoder:
            if hasattr(self, 'text_encoder') and self.text_encoder is not None:
                te1_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
                if te1_params:
                    params.append({"params": te1_params, "lr": self.text_encoder_1_lr})

            if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
                te2_params = [p for p in self.text_encoder_2.parameters() if p.requires_grad]
                if te2_params:
                    params.append({"params": te2_params, "lr": self.text_encoder_2_lr})

        if not params:
            raise ValueError("No trainable parameters found. Check train_unet and train_text_encoder settings.")

        return params

    def save_checkpoint(self, step: int, epoch: int):
        """
        Save full model checkpoint.

        Args:
            step: Current training step
            epoch: Current training epoch
        """
        # Collect full model state
        state_dict = {}

        # U-Net / Transformer
        if hasattr(self, 'unet') and self.unet is not None:
            for key, param in self.unet.state_dict().items():
                state_dict[f"unet.{key}"] = param
        elif hasattr(self, 'transformer') and self.transformer is not None:
            for key, param in self.transformer.state_dict().items():
                state_dict[f"transformer.{key}"] = param

        # Text Encoder 1
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            for key, param in self.text_encoder.state_dict().items():
                state_dict[f"text_encoder.{key}"] = param

        # Text Encoder 2 (SDXL)
        if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
            for key, param in self.text_encoder_2.state_dict().items():
                state_dict[f"text_encoder_2.{key}"] = param

        # Add metadata
        metadata = {
            "step": str(step),
            "epoch": str(epoch),
            "training_type": "full_parameter",
            "train_unet": str(self.train_unet),
            "train_text_encoder": str(self.train_text_encoder),
        }

        # Save safetensors
        checkpoint_path = self.output_dir / f"{self.run_name}_step_{step:06d}.safetensors"
        save_file(state_dict, checkpoint_path, metadata=metadata)

        print(f"{self.log_prefix} Saved checkpoint: {checkpoint_path}")

        # Save optimizer state (separate .pt file)
        optimizer_state_path = self.output_dir / f"{self.run_name}_step_{step:06d}.pt"
        torch.save({
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
        }, optimizer_state_path)

        print(f"{self.log_prefix} Saved optimizer state: {optimizer_state_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load full model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Step number from checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        print(f"{self.log_prefix} Loading checkpoint: {checkpoint_path}")

        # Load full model state
        state_dict = load_file(str(checkpoint_path))

        # Separate by component prefix
        unet_state = {}
        transformer_state = {}
        te1_state = {}
        te2_state = {}

        for key, param in state_dict.items():
            if key.startswith("unet."):
                unet_state[key[len("unet."):]] = param
            elif key.startswith("transformer."):
                transformer_state[key[len("transformer."):]] = param
            elif key.startswith("text_encoder_2."):
                te2_state[key[len("text_encoder_2."):]] = param
            elif key.startswith("text_encoder."):
                te1_state[key[len("text_encoder."):]] = param

        # Load into model components
        if unet_state and hasattr(self, 'unet') and self.unet is not None:
            self.unet.load_state_dict(unet_state, strict=False)
            print(f"{self.log_prefix} Loaded U-Net weights ({len(unet_state)} keys)")

        if transformer_state and hasattr(self, 'transformer') and self.transformer is not None:
            self.transformer.load_state_dict(transformer_state, strict=False)
            print(f"{self.log_prefix} Loaded Transformer weights ({len(transformer_state)} keys)")

        if te1_state and hasattr(self, 'text_encoder') and self.text_encoder is not None:
            self.text_encoder.load_state_dict(te1_state, strict=False)
            print(f"{self.log_prefix} Loaded Text Encoder 1 weights ({len(te1_state)} keys)")

        if te2_state and hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
            self.text_encoder_2.load_state_dict(te2_state, strict=False)
            print(f"{self.log_prefix} Loaded Text Encoder 2 weights ({len(te2_state)} keys)")

        print(f"{self.log_prefix} Loaded model weights")

        # Load optimizer state if exists (optimizer created by BaseTrainer.train() before calling this)
        optimizer_state_path = checkpoint_path.with_suffix(".pt")
        if optimizer_state_path.exists():
            checkpoint_data = torch.load(optimizer_state_path)
            # Note: self.optimizer is created in BaseTrainer.train() before calling this
            if hasattr(self, "optimizer") and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
                print(f"{self.log_prefix} Loaded optimizer state")
            return checkpoint_data.get("step", 0)

        return 0

    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find latest checkpoint in output directory.

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = list(self.output_dir.glob(f"{self.run_name}_step_*.safetensors"))
        if not checkpoints:
            return None

        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
        return checkpoints[-1]
