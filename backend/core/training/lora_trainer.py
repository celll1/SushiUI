"""
LoRA (Low-Rank Adaptation) Trainer for Stable Diffusion models.

This is a complete rewrite from scratch, referencing:
- sd-scripts (Apache-2 license) by kohya-ss
- ai-toolkit (MIT license) by ostris

Key improvements:
- Correct SDXL text encoder layer selection (TE1: layer 11, TE2: penultimate)
- EOS token pooling workaround for Textual Inversion compatibility
- Clean, from-scratch implementation (no legacy bugs)
- NO dtype conversions (all FP32 for debugging)

Author: Claude (2026-01-04)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from safetensors.torch import save_file, load_file
import math

from .base_trainer import BaseTrainer


# ============================================================
# LoRA Linear Layer
# ============================================================

class LoRALinearLayer(nn.Module):
    """
    LoRA layer for Linear modules.

    Formula: output = original_output + (lora_up(lora_down(x))) * scale

    Reference: sd-scripts networks/lora.py (Lines 85-105)
    """

    def __init__(
        self,
        original_module: nn.Linear,
        rank: int,
        alpha: float,
        lora_name: str,
    ):
        """Initialize LoRA layer."""
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_name = lora_name

        in_features = original_module.in_features
        out_features = original_module.out_features

        # Freeze original weights
        self.original_module.requires_grad_(False)

        # LoRA matrices (no bias)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize: Kaiming uniform for down, zeros for up
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        org_out = self.original_module(x)
        lora_out = self.lora_up(self.lora_down(x))
        return org_out + lora_out * self.scale



# ============================================================
# LoRA Trainer
# ============================================================

class LoRATrainer(BaseTrainer):
    """
    LoRA Trainer for SD/SDXL models.

    References: sd-scripts (Apache-2), ai-toolkit (MIT)
    """

    def __init__(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        train_unet: bool = True,
        train_text_encoder: bool = False,
        **kwargs
    ):
        # LoRA-specific settings (set before super().__init__)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / lora_rank
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder

        # LoRA modules storage
        self.lora_layers: Dict[str, LoRALinearLayer] = {}

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Override log prefix
        self.log_prefix = "[LoRA Trainer]"

        # Apply LoRA (model components already loaded by BaseTrainer.__init__)
        self._apply_lora()

        print(f"{self.log_prefix} Initialized (rank={self.lora_rank}, alpha={self.lora_alpha})")
        print(f"{self.log_prefix} Training U-Net: {self.train_unet}, Text Encoder: {self.train_text_encoder}")

    def _apply_lora(self):
        """Apply LoRA to U-Net and Text Encoders."""
        print(f"{self.log_prefix} Applying LoRA layers...")

        if self.train_unet:
            count = self._apply_lora_to_unet()
            print(f"{self.log_prefix} Injected {count} LoRA layers into U-Net")
        else:
            print(f"{self.log_prefix} U-Net LoRA skipped (train_unet=False)")

        if self.train_text_encoder:
            count = self._apply_lora_to_text_encoders()
            print(f"{self.log_prefix} Injected {count} LoRA layers into Text Encoders")
        else:
            print(f"{self.log_prefix} Text Encoder LoRA skipped (train_text_encoder=False)")

        print(f"{self.log_prefix} Total LoRA layers: {len(self.lora_layers)}")

    def _apply_lora_to_unet(self) -> int:
        """
        Apply LoRA to U-Net attention layers.

        Target modules: to_q, to_k, to_v, to_out.0

        Reference: sd-scripts networks/lora.py (Lines 250-280)
        """
        count = 0

        # Iterate over U-Net blocks
        for block_name, block_module in self.unet.named_modules():
            # Find Transformer2DModel blocks
            if block_module.__class__.__name__ != "Transformer2DModel":
                continue

            # Find attention modules within transformer blocks
            for attn_name, attn_module in block_module.named_modules():
                if "attn" not in attn_name.lower():
                    continue

                # Target: to_q, to_k, to_v
                for target_name in ["to_q", "to_k", "to_v"]:
                    if hasattr(attn_module, target_name):
                        linear = getattr(attn_module, target_name)
                        if isinstance(linear, nn.Linear):
                            lora_name = f"lora_unet_{block_name}_{attn_name}_{target_name}"
                            lora_layer = LoRALinearLayer(
                                linear, self.lora_rank, self.lora_alpha, lora_name
                            )
                            setattr(attn_module, target_name, lora_layer)
                            self.lora_layers[lora_name] = lora_layer
                            count += 1

                # Target: to_out.0
                if hasattr(attn_module, "to_out"):
                    to_out = attn_module.to_out
                    if isinstance(to_out, nn.Sequential) and len(to_out) > 0:
                        if isinstance(to_out[0], nn.Linear):
                            lora_name = f"lora_unet_{block_name}_{attn_name}_to_out_0"
                            lora_layer = LoRALinearLayer(
                                to_out[0], self.lora_rank, self.lora_alpha, lora_name
                            )
                            attn_module.to_out[0] = lora_layer
                            self.lora_layers[lora_name] = lora_layer
                            count += 1

        return count

    def _apply_lora_to_text_encoders(self) -> int:
        """
        Apply LoRA to Text Encoder MLP layers.

        Target modules: mlp.fc1, mlp.fc2

        Reference: sd-scripts networks/lora.py (Lines 300-330)
        """
        count = 0

        # Text Encoder 1
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            for layer_idx, layer in enumerate(self.text_encoder.text_model.encoder.layers):
                # mlp.fc1
                lora_name = f"lora_te1_layer{layer_idx}_mlp_fc1"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc1, self.lora_rank, self.lora_alpha, lora_name
                )
                layer.mlp.fc1 = lora_layer
                self.lora_layers[lora_name] = lora_layer
                count += 1

                # mlp.fc2
                lora_name = f"lora_te1_layer{layer_idx}_mlp_fc2"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc2, self.lora_rank, self.lora_alpha, lora_name
                )
                layer.mlp.fc2 = lora_layer
                self.lora_layers[lora_name] = lora_layer
                count += 1

        # Text Encoder 2 (SDXL)
        if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
            for layer_idx, layer in enumerate(self.text_encoder_2.text_model.encoder.layers):
                # mlp.fc1
                lora_name = f"lora_te2_layer{layer_idx}_mlp_fc1"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc1, self.lora_rank, self.lora_alpha, lora_name
                )
                layer.mlp.fc1 = lora_layer
                self.lora_layers[lora_name] = lora_layer
                count += 1

                # mlp.fc2
                lora_name = f"lora_te2_layer{layer_idx}_mlp_fc2"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc2, self.lora_rank, self.lora_alpha, lora_name
                )
                layer.mlp.fc2 = lora_layer
                self.lora_layers[lora_name] = lora_layer
                count += 1

        return count

    def setup_trainable_parameters(self) -> List[Dict]:
        """
        Collect trainable parameters with per-component learning rates.

        Returns:
            List of parameter groups for optimizer
        """
        params = []
        unet_params = []
        te1_params = []
        te2_params = []

        for lora_name, lora_layer in self.lora_layers.items():
            if lora_name.startswith("lora_unet_"):
                unet_params.extend(lora_layer.lora_down.parameters())
                unet_params.extend(lora_layer.lora_up.parameters())
            elif lora_name.startswith("lora_te1_"):
                te1_params.extend(lora_layer.lora_down.parameters())
                te1_params.extend(lora_layer.lora_up.parameters())
            elif lora_name.startswith("lora_te2_"):
                te2_params.extend(lora_layer.lora_down.parameters())
                te2_params.extend(lora_layer.lora_up.parameters())

        # Add parameter groups with component-specific learning rates
        if unet_params:
            params.append({"params": unet_params, "lr": self.unet_lr})
        if te1_params:
            params.append({"params": te1_params, "lr": self.text_encoder_1_lr})
        if te2_params:
            params.append({"params": te2_params, "lr": self.text_encoder_2_lr})

        return params

    def save_checkpoint(self, step: int, epoch: int):
        """
        Save LoRA checkpoint.

        Args:
            step: Current training step
            epoch: Current training epoch
        """
        # Collect LoRA weights
        lora_state_dict = {}

        for lora_name, lora_layer in self.lora_layers.items():
            lora_state_dict[f"{lora_name}.lora_down.weight"] = lora_layer.lora_down.weight
            lora_state_dict[f"{lora_name}.lora_up.weight"] = lora_layer.lora_up.weight

        # Add metadata
        metadata = {
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "step": str(step),
            "epoch": str(epoch),
        }

        # Save safetensors
        checkpoint_path = self.output_dir / f"{self.run_name}_step_{step:06d}.safetensors"
        save_file(lora_state_dict, checkpoint_path, metadata=metadata)

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
        Load LoRA checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Step number from checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        print(f"{self.log_prefix} Loading checkpoint: {checkpoint_path}")

        # Load LoRA weights
        lora_state_dict = load_file(str(checkpoint_path))

        # Apply to LoRA layers
        for lora_name, lora_layer in self.lora_layers.items():
            down_key = f"{lora_name}.lora_down.weight"
            up_key = f"{lora_name}.lora_up.weight"

            if down_key in lora_state_dict:
                lora_layer.lora_down.weight.data = lora_state_dict[down_key]
            if up_key in lora_state_dict:
                lora_layer.lora_up.weight.data = lora_state_dict[up_key]

        print(f"{self.log_prefix} Loaded LoRA weights")

        # Load optimizer state if exists (optimizer created by BaseTrainer.train())
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
