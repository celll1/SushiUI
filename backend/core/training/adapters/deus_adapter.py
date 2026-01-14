"""
DEUS model adapter for LoRA and Full Parameter training.

Model characteristics:
- Single SigLIP-2 text encoder (1152d output, variable sequence length)
- U-Net with Transformer2DModel blocks (similar to SDXL)
- No pooled embeddings (unlike SDXL)
- No time_ids / added_cond_kwargs (unlike SDXL)
- SDXL VAE (same scaling factor 0.13025)
- DDPM epsilon prediction (same as SDXL)

Key differences from SDXL:
1. Single text encoder (SigLIP-2) vs dual CLIP
2. No pooled_embeddings in forward pass
3. No time_ids / added_cond_kwargs
4. U-Net forward: unet(latents, timesteps, encoder_hidden_states)

Author: Claude (2026-01-15)
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
# DEUS LoRA Adapter
# ============================================================

class DEUSLoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for DEUS models."""

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to all Linear layers in Transformer2DModel modules.

        DEUS uses a U-Net architecture similar to SDXL with Transformer2DModel blocks.
        The block structure is the same as SDXL, but DEUS uses:
        - DeusCrossAttnDownBlock2D / DeusCrossAttnUpBlock2D
        - DeusMidBlock2DCrossAttn

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

        # Find all Transformer2DModel blocks (or DEUS-specific transformer blocks)
        for block_name, block_module in unet.named_modules():
            # Find Transformer2DModel blocks
            # DEUS may use "Transformer2DModel" or a custom variant
            if block_module.__class__.__name__ not in ["Transformer2DModel", "DeusTransformer2DModel"]:
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
        Apply LoRA to SigLIP-2 Text Encoder.

        SigLIP-2 structure (same as CLIP):
        - text_model.encoder.layers[N].mlp.fc1
        - text_model.encoder.layers[N].mlp.fc2

        DEUS uses a single text encoder (SigLIP-2), unlike SDXL's dual CLIP.

        Args:
            lora_layers: Dictionary to store LoRA layer references

        Returns:
            Number of LoRA layers injected
        """
        count = 0

        # SigLIP-2 Text Encoder: All layers
        if hasattr(self.trainer, "text_encoder") and self.trainer.text_encoder is not None:
            text_encoder = self.trainer.text_encoder

            # Navigate to encoder layers
            # SigLIP structure: text_encoder.text_model.encoder.layers
            # Or directly: text_encoder.encoder.layers (depends on how it's loaded)
            encoder_layers = None

            if hasattr(text_encoder, "text_model") and hasattr(text_encoder.text_model, "encoder"):
                encoder_layers = text_encoder.text_model.encoder.layers
            elif hasattr(text_encoder, "encoder"):
                encoder_layers = text_encoder.encoder.layers

            if encoder_layers is None:
                print("[DEUSLoRAAdapter] Warning: Could not find text encoder layers")
                return count

            for layer_idx, layer in enumerate(encoder_layers):
                # mlp.fc1
                # Use naming: lora_te_text_model_encoder_layers_{N}_mlp_fc1
                lora_name = f"lora_te_text_model_encoder_layers_{layer_idx}_mlp_fc1"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc1, self.lora_rank, self.lora_alpha, lora_name
                )
                layer.mlp.fc1 = lora_layer
                lora_layers[lora_name] = lora_layer
                count += 1

                # mlp.fc2
                lora_name = f"lora_te_text_model_encoder_layers_{layer_idx}_mlp_fc2"
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
        te_params = []

        for lora_name, lora_layer in lora_layers.items():
            if lora_name.startswith("lora_unet_"):
                unet_params.extend(lora_layer.lora_down.parameters())
                unet_params.extend(lora_layer.lora_up.parameters())
            elif lora_name.startswith("lora_te_"):
                te_params.extend(lora_layer.lora_down.parameters())
                te_params.extend(lora_layer.lora_up.parameters())

        # Add parameter groups with component-specific learning rates
        if unet_params:
            params.append({"params": unet_params, "lr": self.trainer.unet_lr})
        if te_params:
            # Use text_encoder_lr for single text encoder
            params.append({"params": te_params, "lr": self.trainer.text_encoder_lr})

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

        if trainer.train_text_encoder:
            if trainer.text_encoder is not None:
                trainer.text_encoder.requires_grad_(True)
                trainer.text_encoder.train()

        # VAE is always frozen
        if trainer.vae is not None:
            trainer.vae.requires_grad_(False)
            trainer.vae.eval()

        print(f"[DEUSFullParameterAdapter] Models prepared for training")
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

        if trainer.train_text_encoder:
            if trainer.text_encoder is not None:
                te_params = [p for p in trainer.text_encoder.parameters() if p.requires_grad]
                if te_params:
                    params.append({"params": te_params, "lr": trainer.text_encoder_lr})

        return params

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        """
        Save full parameter checkpoint in single safetensors format.

        Uses DEUS-compatible key prefixes (same as DeusPipeline.save_to_single_file):
        - UNet: "model.diffusion_model.*"
        - VAE: "first_stage_model.*"
        - Text Encoder: "conditioner.embedders.0.model.*"

        Args:
            step: Current training step
            epoch: Current training epoch
            output_path: Path to save checkpoint (should be .safetensors file)
        """
        trainer = self.trainer

        # Ensure output_path is a file path, not directory
        if output_path.is_dir():
            output_path = output_path / f"model_step_{step}.safetensors"
        elif not str(output_path).endswith(".safetensors"):
            output_path = Path(str(output_path) + ".safetensors")

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined_state_dict = {}

        # Save U-Net weights with DEUS prefix
        if trainer.train_unet and trainer.unet is not None:
            print(f"[DEUSFullParameterAdapter] Collecting U-Net weights...")
            unet_state = trainer.unet.state_dict()
            for key, value in unet_state.items():
                combined_state_dict[f"model.diffusion_model.{key}"] = value.cpu()

        # Save VAE weights with DEUS prefix
        if trainer.vae is not None:
            print(f"[DEUSFullParameterAdapter] Collecting VAE weights...")
            vae_state = trainer.vae.state_dict()
            for key, value in vae_state.items():
                combined_state_dict[f"first_stage_model.{key}"] = value.cpu()

        # Save Text Encoder weights with DEUS prefix (SigLIP-2)
        if trainer.train_text_encoder and trainer.text_encoder is not None:
            print(f"[DEUSFullParameterAdapter] Collecting Text Encoder (SigLIP-2) weights...")
            te_state = trainer.text_encoder.state_dict()
            for key, value in te_state.items():
                combined_state_dict[f"conditioner.embedders.0.model.{key}"] = value.cpu()

        # Save to safetensors with metadata
        metadata = {
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "deus",
        }

        print(f"[DEUSFullParameterAdapter] Saving to {output_path}...")
        save_file(combined_state_dict, output_path, metadata=metadata)

        total_params = sum(p.numel() for p in combined_state_dict.values())
        print(f"[DEUSFullParameterAdapter] Saved {len(combined_state_dict)} tensors ({total_params:,} params) to {output_path}")
