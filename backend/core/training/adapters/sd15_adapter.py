"""
SD1.5 model adapter for LoRA and Full Parameter training.

Model characteristics:
- Single text encoder (CLIP ViT-L/14)
- U-Net with Transformer2DModel blocks
- Simple text embeddings (no pooled output)

Author: Claude (2026-01-04)
"""

from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
from safetensors.torch import save_file
import math

from .base_adapter import BaseLoRAAdapter, BaseFullParameterAdapter


# ============================================================
# LoRA Linear Layer (shared by all adapters)
# ============================================================

class LoRALinearLayer(nn.Module):
    """
    LoRA layer for Linear modules.

    Formula: output = original_output + (lora_up(lora_down(x))) * scale
    """

    def __init__(
        self,
        original_module: nn.Linear,
        rank: int,
        alpha: float,
        lora_name: str,
        lora_dtype: torch.dtype = torch.float32,
    ):
        """Initialize LoRA layer."""
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_name = lora_name
        self.lora_dtype = lora_dtype

        in_features = original_module.in_features
        out_features = original_module.out_features

        # Freeze original weights
        self.original_module.requires_grad_(False)

        # LoRA matrices (no bias)
        # Use lora_dtype for LoRA weights (can be different from main model dtype)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize: Kaiming uniform for down, zeros for up
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Move to same device as original, but use lora_dtype
        device = original_module.weight.device
        self.lora_down.to(device=device, dtype=lora_dtype)
        self.lora_up.to(device=device, dtype=lora_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Uses autocast to automatically handle mixed precision:
        - LoRA weights (fp32) are automatically converted to training dtype during forward
        - Gradients flow back to fp32 master weights correctly
        - GradScaler handles gradient scaling for fp16/bf16 training
        """
        org_out = self.original_module(x)

        # LoRA computation (autocast will handle dtype conversion automatically)
        # If we're in an autocast context (training_dtype), this will run in that dtype
        # Gradients will still flow back to fp32 master weights correctly
        lora_out = self.lora_up(self.lora_down(x))

        return org_out + lora_out * self.scale


# ============================================================
# SD1.5 LoRA Adapter
# ============================================================

class SD15LoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for SD1.5 models."""

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to all Transformer2DModel modules in U-Net.

        Targets all Linear layers inside Transformer2DModel blocks.

        Args:
            lora_layers: Dictionary to store LoRA layer references

        Returns:
            Number of LoRA layers injected
        """
        count = 0
        unet = self.trainer.unet

        # Find all Transformer2DModel blocks
        for block_name, block_module in unet.named_modules():
            # Find Transformer2DModel blocks
            if block_module.__class__.__name__ != "Transformer2DModel":
                continue

            # Find attention modules within transformer blocks
            for attn_name, attn_module in block_module.named_modules():
                # Target: to_q, to_k, to_v, to_out.0
                for attr_name in ["to_q", "to_k", "to_v"]:
                    if hasattr(attn_module, attr_name):
                        original_linear = getattr(attn_module, attr_name)
                        if isinstance(original_linear, nn.Linear):
                            # Build LoRA name: lora_unet_{block}_{attn}_{attr}
                            lora_name = f"lora_unet_{block_name.replace('.', '_')}_{attn_name.replace('.', '_')}_{attr_name}"
                            lora_layer = LoRALinearLayer(original_linear, self.lora_rank, self.lora_alpha, lora_name
                            , self.lora_dtype)
                            setattr(attn_module, attr_name, lora_layer)
                            lora_layers[lora_name] = lora_layer
                            count += 1

                # Handle to_out.0 (first layer of to_out Sequential)
                if hasattr(attn_module, "to_out") and isinstance(attn_module.to_out, nn.Sequential):
                    if len(attn_module.to_out) > 0 and isinstance(attn_module.to_out[0], nn.Linear):
                        original_linear = attn_module.to_out[0]
                        lora_name = f"lora_unet_{block_name.replace('.', '_')}_{attn_name.replace('.', '_')}_to_out_0"
                        lora_layer = LoRALinearLayer(original_linear, self.lora_rank, self.lora_alpha, lora_name
                        , self.lora_dtype)
                        attn_module.to_out[0] = lora_layer
                        lora_layers[lora_name] = lora_layer
                        count += 1

        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to Text Encoder 1 (CLIP ViT-L).

        Targets all MLP layers (fc1, fc2) in all encoder layers.

        Args:
            lora_layers: Dictionary to store LoRA layer references

        Returns:
            Number of LoRA layers injected
        """
        count = 0
        text_encoder = self.trainer.text_encoder

        if text_encoder is None:
            return 0

        # Text Encoder 1: All layers
        for layer_idx, layer in enumerate(text_encoder.text_model.encoder.layers):
            # mlp.fc1
            lora_name = f"lora_te1_layer{layer_idx}_mlp_fc1"
            lora_layer = LoRALinearLayer(layer.mlp.fc1, self.lora_rank, self.lora_alpha, lora_name
            , self.lora_dtype)
            layer.mlp.fc1 = lora_layer
            lora_layers[lora_name] = lora_layer
            count += 1

            # mlp.fc2
            lora_name = f"lora_te1_layer{layer_idx}_mlp_fc2"
            lora_layer = LoRALinearLayer(layer.mlp.fc2, self.lora_rank, self.lora_alpha, lora_name
            , self.lora_dtype)
            layer.mlp.fc2 = lora_layer
            lora_layers[lora_name] = lora_layer
            count += 1

        return count

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        """
        Collect trainable parameters with per-component learning rates.

        Args:
            lora_layers: Dictionary of LoRA layers

        Returns:
            List of parameter groups for optimizer
        """
        params = []
        unet_params = []
        te1_params = []

        for lora_name, lora_layer in lora_layers.items():
            if lora_name.startswith("lora_unet_"):
                unet_params.extend(lora_layer.lora_down.parameters())
                unet_params.extend(lora_layer.lora_up.parameters())
            elif lora_name.startswith("lora_te1_"):
                te1_params.extend(lora_layer.lora_down.parameters())
                te1_params.extend(lora_layer.lora_up.parameters())

        # Add parameter groups with component-specific learning rates
        if unet_params:
            params.append({"params": unet_params, "lr": self.trainer.unet_lr})
        if te1_params:
            params.append({"params": te1_params, "lr": self.trainer.text_encoder_1_lr})

        return params

    def save_checkpoint(
        self,
        lora_layers: Dict[str, nn.Module],
        step: int,
        epoch: int,
        output_path: Path
    ):
        """
        Save LoRA checkpoint in safetensors format.

        Args:
            lora_layers: Dictionary of LoRA layers
            step: Current training step
            epoch: Current training epoch
            output_path: Path to save checkpoint
        """
        # Collect LoRA weights
        lora_state_dict = {}

        for lora_name, lora_layer in lora_layers.items():
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
        save_file(lora_state_dict, output_path, metadata=metadata)
        print(f"[SD15LoRAAdapter] Saved LoRA checkpoint: {output_path}")


# ============================================================
# SD1.5 Full Parameter Adapter
# ============================================================

class SD15FullParameterAdapter(BaseFullParameterAdapter):
    """Full parameter adapter for SD1.5 models."""

    def prepare_models_for_training(self):
        """Prepare models for full parameter training."""
        trainer = self.trainer

        # Set requires_grad based on configuration
        if trainer.train_unet and trainer.unet is not None:
            trainer.unet.requires_grad_(True)
            trainer.unet.train()

        if trainer.train_text_encoder and trainer.text_encoder is not None:
            trainer.text_encoder.requires_grad_(True)
            trainer.text_encoder.train()

        # VAE is always frozen
        if trainer.vae is not None:
            trainer.vae.requires_grad_(False)
            trainer.vae.eval()

        print(f"[SD15FullParameterAdapter] Models prepared for training")
        print(f"  U-Net trainable: {trainer.train_unet}")
        print(f"  Text Encoder trainable: {trainer.train_text_encoder}")

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        """
        Collect trainable parameters with per-component learning rates.

        Returns:
            List of parameter groups for optimizer
        """
        params = []
        trainer = self.trainer

        if trainer.train_unet and trainer.unet is not None:
            unet_params = [p for p in trainer.unet.parameters() if p.requires_grad]
            if unet_params:
                params.append({"params": unet_params, "lr": trainer.unet_lr})

        if trainer.train_text_encoder and trainer.text_encoder is not None:
            te1_params = [p for p in trainer.text_encoder.parameters() if p.requires_grad]
            if te1_params:
                params.append({"params": te1_params, "lr": trainer.text_encoder_1_lr})

        return params

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        """
        Save full parameter checkpoint in diffusers format.

        Args:
            step: Current training step
            epoch: Current training epoch
            output_path: Directory to save checkpoint
        """
        trainer = self.trainer

        # Create checkpoint directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Save U-Net
        if trainer.train_unet and trainer.unet is not None:
            unet_path = output_path / "unet"
            trainer.unet.save_pretrained(unet_path)

        # Save Text Encoder
        if trainer.train_text_encoder and trainer.text_encoder is not None:
            te_path = output_path / "text_encoder"
            trainer.text_encoder.save_pretrained(te_path)

        # Save metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "model_type": "sd15",
        }

        import json
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[SD15FullParameterAdapter] Saved checkpoint: {output_path}")
