"""
DEUS model adapter for training.

Author: Claude (2026-01-08)
"""

from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
from safetensors.torch import save_file
from .base_adapter import BaseLoRAAdapter, BaseFullParameterAdapter


# ============================================================
# DEUS LoRA Adapter
# ============================================================

class DEUSLoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for DEUS models."""

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to DEUS U-Net attention layers.

        Args:
            lora_layers: Dictionary to store LoRA layer references

        Returns:
            Number of LoRA layers injected
        """
        from peft import inject_adapter_in_model, LoraConfig

        # DEUS U-Net uses standard attention layers (same as SD/SDXL)
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=True,
        )

        # Inject LoRA into U-Net
        self.trainer.unet = inject_adapter_in_model(lora_config, self.trainer.unet)

        # Collect LoRA layers
        unet_lora_count = 0
        for name, module in self.trainer.unet.named_modules():
            if "lora_" in name:
                lora_layers[f"unet.{name}"] = module
                unet_lora_count += 1

        print(f"[DEUSLoRAAdapter] Injected {unet_lora_count} LoRA layers into U-Net")
        return unet_lora_count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to SigLIP-2 text encoder.

        Args:
            lora_layers: Dictionary to store LoRA layer references

        Returns:
            Number of LoRA layers injected
        """
        from peft import inject_adapter_in_model, LoraConfig

        # SigLIP-2: Target MLP layers in transformer blocks
        target_modules = ["mlp.fc1", "mlp.fc2"]

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=True,
        )

        # Inject LoRA into SigLIP-2 text encoder
        # Note: self.trainer.text_encoder is SigLIP2Wrapper, access text_model
        self.trainer.text_encoder.text_model = inject_adapter_in_model(
            lora_config,
            self.trainer.text_encoder.text_model
        )

        # Collect LoRA layers
        te_lora_count = 0
        for name, module in self.trainer.text_encoder.text_model.named_modules():
            if "lora_" in name:
                lora_layers[f"text_encoder.{name}"] = module
                te_lora_count += 1

        print(f"[DEUSLoRAAdapter] Injected {te_lora_count} LoRA layers into SigLIP-2 Text Encoder")
        return te_lora_count

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        """
        Collect trainable LoRA parameters with per-component learning rates.

        Args:
            lora_layers: Dictionary of LoRA layers

        Returns:
            List of parameter groups for optimizer
        """
        params = []
        trainer = self.trainer

        # Collect U-Net LoRA parameters
        unet_lora_params = []
        for name, module in lora_layers.items():
            if name.startswith("unet."):
                for param in module.parameters():
                    if param.requires_grad:
                        unet_lora_params.append(param)

        if unet_lora_params:
            params.append({"params": unet_lora_params, "lr": trainer.unet_lr})
            print(f"[DEUSLoRAAdapter] U-Net LoRA params: {len(unet_lora_params)}")

        # Collect Text Encoder LoRA parameters
        te_lora_params = []
        for name, module in lora_layers.items():
            if name.startswith("text_encoder."):
                for param in module.parameters():
                    if param.requires_grad:
                        te_lora_params.append(param)

        if te_lora_params:
            params.append({"params": te_lora_params, "lr": trainer.text_encoder_lr})
            print(f"[DEUSLoRAAdapter] Text Encoder LoRA params: {len(te_lora_params)}")

        return params

    def save_checkpoint(
        self,
        lora_layers: Dict[str, nn.Module],
        step: int,
        epoch: int,
        output_path: Path
    ):
        """
        Save DEUS LoRA checkpoint in safetensors format.

        Args:
            lora_layers: Dictionary of LoRA layers
            step: Current training step
            epoch: Current training epoch
            output_path: Path to save checkpoint (file path, not directory)
        """
        lora_state_dict = {}

        # Collect all LoRA weights
        for lora_name, lora_module in lora_layers.items():
            # Extract lora_A and lora_B weights
            if hasattr(lora_module, "lora_A"):
                lora_state_dict[f"{lora_name}.lora_A.weight"] = lora_module.lora_A.weight.detach().cpu().to(self.trainer.output_dtype)
            if hasattr(lora_module, "lora_B"):
                lora_state_dict[f"{lora_name}.lora_B.weight"] = lora_module.lora_B.weight.detach().cpu().to(self.trainer.output_dtype)

            # Add alpha parameter (scaling factor)
            lora_state_dict[f"{lora_name}.alpha"] = torch.tensor(self.lora_alpha, dtype=torch.float32)

        # Add metadata
        metadata = {
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "deus",
        }

        # Save safetensors
        save_file(lora_state_dict, output_path, metadata=metadata)
        print(f"[DEUSLoRAAdapter] Saved LoRA checkpoint: {output_path}")


# ============================================================
# DEUS Full Parameter Adapter
# ============================================================

class DEUSFullParameterAdapter(BaseFullParameterAdapter):
    """Full parameter adapter for DEUS models."""

    def prepare_models_for_training(self):
        """Prepare models for full parameter training."""
        trainer = self.trainer

        # Set requires_grad based on configuration
        if trainer.train_unet and trainer.unet is not None:
            trainer.unet.requires_grad_(True)
            trainer.unet.train()

        if trainer.train_text_encoder and trainer.text_encoder is not None:
            # SigLIP2Wrapper: enable gradients for text_model
            trainer.text_encoder.text_model.requires_grad_(True)
            trainer.text_encoder.text_model.train()

        # VAE is always frozen
        if trainer.vae is not None:
            trainer.vae.requires_grad_(False)
            trainer.vae.eval()

        print(f"[DEUSFullParameterAdapter] Models prepared for training")
        print(f"  U-Net trainable: {trainer.train_unet}")
        print(f"  SigLIP-2 Text Encoder trainable: {trainer.train_text_encoder}")

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
                print(f"[DEUSFullParameterAdapter] U-Net params: {len(unet_params)}")

        if trainer.train_text_encoder and trainer.text_encoder is not None:
            # SigLIP2Wrapper: collect text_model parameters
            te_params = [p for p in trainer.text_encoder.text_model.parameters() if p.requires_grad]
            if te_params:
                params.append({"params": te_params, "lr": trainer.text_encoder_lr})
                print(f"[DEUSFullParameterAdapter] Text Encoder params: {len(te_params)}")

        return params

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        """
        Save full parameter checkpoint in safetensors format.

        Args:
            step: Current training step
            epoch: Current training epoch
            output_path: Path to save checkpoint (file path, not directory)
        """
        trainer = self.trainer

        # Collect state dict from trainable components
        state_dict = {}

        # U-Net state dict
        if trainer.train_unet and trainer.unet is not None:
            for key, param in trainer.unet.state_dict().items():
                state_dict[f"unet.{key}"] = param.detach().cpu().to(trainer.output_dtype)

        # Text Encoder state dict (SigLIP-2)
        if trainer.train_text_encoder and trainer.text_encoder is not None:
            for key, param in trainer.text_encoder.text_model.state_dict().items():
                state_dict[f"text_encoder.{key}"] = param.detach().cpu().to(trainer.output_dtype)

        # VAE state dict (always include for complete checkpoint)
        if trainer.vae is not None:
            for key, param in trainer.vae.state_dict().items():
                state_dict[f"vae.{key}"] = param.detach().cpu().to(trainer.output_dtype)

        # Add metadata
        metadata = {
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "deus",
            "train_unet": str(trainer.train_unet),
            "train_text_encoder": str(trainer.train_text_encoder),
        }

        # Save safetensors
        save_file(state_dict, output_path, metadata=metadata)
        print(f"[DEUSFullParameterAdapter] Saved full parameter checkpoint: {output_path}")
        print(f"  Total keys: {len(state_dict)}")
