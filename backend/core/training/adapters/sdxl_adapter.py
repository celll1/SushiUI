"""
SDXL model adapter for LoRA and Full Parameter training.

Model characteristics:
- Dual text encoders (TE1: CLIP ViT-L, TE2: OpenCLIP ViT-bigG)
- U-Net with 11 Transformer2DModel blocks
- Pooled embeddings from TE2
- Time IDs for micro-conditioning

Critical fixes from rewrite:
1. Text Encoder layer selection: All layers (not specific layers)
2. TE2 penultimate layer: hidden_states[-2] (NOT final layer)
3. EOS token pooling workaround: Manually pool last token for TI compatibility
4. Component-specific learning rates: te1_lr, te2_lr, unet_lr

Author: Claude (2026-01-04)
"""

from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
from safetensors.torch import save_file
import math

from .base_adapter import BaseLoRAAdapter, BaseFullParameterAdapter
from .sd15_adapter import LoRALinearLayer  # Reuse LoRA layer implementation


# ============================================================
# SDXL LoRA Adapter
# ============================================================

class SDXLLoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for SDXL models."""

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to all Linear layers in Transformer2DModel modules (diffusers style).

        SDXL has 11 transformer blocks:
        - down_blocks.1.attentions.0-1 (IN04, IN05)
        - down_blocks.2.attentions.0-1 (IN07, IN08)
        - mid_block.attentions.0 (MID)
        - up_blocks.0.attentions.0-2 (OUT00-OUT02)
        - up_blocks.1.attentions.0-2 (OUT03-OUT05)

        Following sd-scripts approach: iterate ALL Linear layers within Transformer2DModel.
        This includes:
        - Attention: to_q, to_k, to_v, to_out.0
        - Projection: proj_in, proj_out
        - FeedForward: ff.net.0.proj, ff.net.2

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

            # Iterate ALL child Linear modules within this Transformer2DModel (sd-scripts approach)
            for child_name, child_module in block_module.named_modules():
                if child_module.__class__.__name__ == "Linear":
                    # Build LoRA name: lora_unet_{block_name}_{child_name}
                    # Replace '.' with '_' for diffusers style naming
                    lora_name = f"lora_unet_{block_name}_{child_name}".replace(".", "_")

                    # Create LoRA layer
                    lora_layer = LoRALinearLayer(
                        child_module, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype
                    )

                    # Replace original Linear with LoRA layer
                    # Navigate to parent module and set attribute
                    if "." in child_name:
                        # Child is nested (e.g., "to_out.0", "ff.net.0.proj")
                        path_parts = child_name.split(".")
                        parent_module = block_module
                        for part in path_parts[:-1]:
                            if part.isdigit():
                                parent_module = parent_module[int(part)]
                            else:
                                parent_module = getattr(parent_module, part)

                        # Set the final attribute
                        attr_name = path_parts[-1]
                        if attr_name.isdigit():
                            parent_module[int(attr_name)] = lora_layer
                        else:
                            setattr(parent_module, attr_name, lora_layer)
                    else:
                        # Child is direct attribute (e.g., "proj_in", "proj_out")
                        setattr(block_module, child_name, lora_layer)

                    lora_layers[lora_name] = lora_layer
                    count += 1

        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to both Text Encoders (TE1: CLIP ViT-L, TE2: OpenCLIP ViT-bigG).

        Critical fix: Apply to ALL layers (not specific layers).

        TE1 (CLIP ViT-L): 12 layers × 2 (fc1, fc2) = 24 LoRA layers
        TE2 (OpenCLIP ViT-bigG): 32 layers × 2 (fc1, fc2) = 64 LoRA layers

        Args:
            lora_layers: Dictionary to store LoRA layer references

        Returns:
            Number of LoRA layers injected
        """
        count = 0

        # Text Encoder 1 (CLIP ViT-L): All layers
        if hasattr(self.trainer, "text_encoder") and self.trainer.text_encoder is not None:
            for layer_idx, layer in enumerate(self.trainer.text_encoder.text_model.encoder.layers):
                # mlp.fc1
                lora_name = f"lora_te1_layer{layer_idx}_mlp_fc1"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc1, self.lora_rank, self.lora_alpha, lora_name
                )
                layer.mlp.fc1 = lora_layer
                lora_layers[lora_name] = lora_layer
                count += 1

                # mlp.fc2
                lora_name = f"lora_te1_layer{layer_idx}_mlp_fc2"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc2, self.lora_rank, self.lora_alpha, lora_name
                )
                layer.mlp.fc2 = lora_layer
                lora_layers[lora_name] = lora_layer
                count += 1

        # Text Encoder 2 (OpenCLIP ViT-bigG): All layers
        if hasattr(self.trainer, "text_encoder_2") and self.trainer.text_encoder_2 is not None:
            for layer_idx, layer in enumerate(self.trainer.text_encoder_2.text_model.encoder.layers):
                # mlp.fc1
                lora_name = f"lora_te2_layer{layer_idx}_mlp_fc1"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc1, self.lora_rank, self.lora_alpha, lora_name
                )
                layer.mlp.fc1 = lora_layer
                lora_layers[lora_name] = lora_layer
                count += 1

                # mlp.fc2
                lora_name = f"lora_te2_layer{layer_idx}_mlp_fc2"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc2, self.lora_rank, self.lora_alpha, lora_name
                )
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
        te2_params = []

        for lora_name, lora_layer in lora_layers.items():
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
            params.append({"params": unet_params, "lr": self.trainer.unet_lr})
        if te1_params:
            params.append({"params": te1_params, "lr": self.trainer.text_encoder_1_lr})
        if te2_params:
            params.append({"params": te2_params, "lr": self.trainer.text_encoder_2_lr})

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
            # Add alpha for diffusers compatibility (required by _create_lora_config)
            lora_state_dict[f"{lora_name}.alpha"] = torch.tensor(self.lora_alpha, dtype=torch.float32)

        # Add metadata
        metadata = {
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "sdxl",
        }

        # Save safetensors
        save_file(lora_state_dict, output_path, metadata=metadata)
        print(f"[SDXLLoRAAdapter] Saved LoRA checkpoint: {output_path}")


# ============================================================
# SDXL Full Parameter Adapter
# ============================================================

class SDXLFullParameterAdapter(BaseFullParameterAdapter):
    """Full parameter adapter for SDXL models."""

    def prepare_models_for_training(self):
        """Prepare models for full parameter training."""
        trainer = self.trainer

        # Set requires_grad based on configuration
        if trainer.train_unet and trainer.unet is not None:
            trainer.unet.requires_grad_(True)
            trainer.unet.train()

        if trainer.train_text_encoder:
            if trainer.text_encoder is not None:
                trainer.text_encoder.requires_grad_(True)
                trainer.text_encoder.train()
            if trainer.text_encoder_2 is not None:
                trainer.text_encoder_2.requires_grad_(True)
                trainer.text_encoder_2.train()

        # VAE is always frozen
        if trainer.vae is not None:
            trainer.vae.requires_grad_(False)
            trainer.vae.eval()

        print(f"[SDXLFullParameterAdapter] Models prepared for training")
        print(f"  U-Net trainable: {trainer.train_unet}")
        print(f"  Text Encoder 1 trainable: {trainer.train_text_encoder}")
        print(f"  Text Encoder 2 trainable: {trainer.train_text_encoder}")

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

        if trainer.train_text_encoder:
            if trainer.text_encoder is not None:
                te1_params = [p for p in trainer.text_encoder.parameters() if p.requires_grad]
                if te1_params:
                    params.append({"params": te1_params, "lr": trainer.text_encoder_1_lr})

            if trainer.text_encoder_2 is not None:
                te2_params = [p for p in trainer.text_encoder_2.parameters() if p.requires_grad]
                if te2_params:
                    params.append({"params": te2_params, "lr": trainer.text_encoder_2_lr})

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

        # Save Text Encoders
        if trainer.train_text_encoder:
            if trainer.text_encoder is not None:
                te1_path = output_path / "text_encoder"
                trainer.text_encoder.save_pretrained(te1_path)
            if trainer.text_encoder_2 is not None:
                te2_path = output_path / "text_encoder_2"
                trainer.text_encoder_2.save_pretrained(te2_path)

        # Save metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "model_type": "sdxl",
        }

        import json
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[SDXLFullParameterAdapter] Saved checkpoint: {output_path}")
