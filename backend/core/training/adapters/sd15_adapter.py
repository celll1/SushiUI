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

# Temporary Phase 1 shim: ``LoRALinearLayer`` moved to ``core.adapters.layers``,
# outside the training package, and is re-exported here so existing importers
# (twelve of them in generation) keep working. Removed at the end of Phase 1.
from core.adapters.layers import LoRALinearLayer  # noqa: F401

from .base_adapter import (
    BaseLoRAAdapter,
    BaseFullParameterAdapter,
    reject_quantized_base,
    LORA_COMPONENT_UNET,
    LORA_COMPONENT_TEXT_ENCODER_1,
)
from .state_dict_converter import (
    convert_unet_state_dict_to_original,
    convert_vae_state_dict_to_original,
    convert_openai_text_enc_to_original,
)


def sd15_modelspec_metadata(trainer) -> Dict[str, str]:
    """Unified SushiUI prediction metadata for a saved SD1.5 model.

    Mirrors sdxl_adapter.sushi_modelspec_metadata for the fields that apply to
    SD1.5: it records the resolved noise_process / prediction_target so the loader
    (ModelLoader.detect_prediction_config) can reproduce the training objective
    (epsilon / velocity / sample) on reload instead of always assuming epsilon.
    Resolved "auto" values are skipped (the loader then infers as before). SD1.5
    has no custom VAE / text-encoder swap, so those sushi.* markers are omitted.
    Keyspace is unchanged — only the safetensors __metadata__ block gains entries.
    """
    md: Dict[str, str] = {"modelspec.architecture": "stable-diffusion-v1"}
    np = str(getattr(trainer, "noise_process", "") or "").strip().lower()
    pt = str(getattr(trainer, "prediction_target", "") or "").strip().lower()
    if np and np != "auto":
        md["modelspec.noise_process"] = np
    if pt and pt != "auto":
        md["modelspec.prediction_type"] = pt
    return md


# ============================================================
# SD1.5 LoRA Adapter
# ============================================================

class SD15LoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for SD1.5 models."""

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to all Linear layers in Transformer2DModel modules (diffusers style).

        Following sd-scripts approach: iterate ALL Linear layers within Transformer2DModel.
        This includes:
        - Attention: to_q, to_k, to_v, to_out.0
        - Projection: proj_in, proj_out (if exists)
        - FeedForward: ff.net.0.proj, ff.net.2 (if exists)

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

                    self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
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
            # Use sd-scripts compatible naming: lora_te1_text_model_encoder_layers_{N}_mlp_fc1
            lora_name = f"lora_te1_text_model_encoder_layers_{layer_idx}_mlp_fc1"
            lora_layer = LoRALinearLayer(layer.mlp.fc1, self.lora_rank, self.lora_alpha, lora_name
            , self.lora_dtype)
            layer.mlp.fc1 = lora_layer
            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_TEXT_ENCODER_1)
            count += 1

            # mlp.fc2
            lora_name = f"lora_te1_text_model_encoder_layers_{layer_idx}_mlp_fc2"
            lora_layer = LoRALinearLayer(layer.mlp.fc2, self.lora_rank, self.lora_alpha, lora_name
            , self.lora_dtype)
            layer.mlp.fc2 = lora_layer
            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_TEXT_ENCODER_1)
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
            # Add alpha for diffusers compatibility (required by _create_lora_config)
            lora_state_dict[f"{lora_name}.alpha"] = torch.tensor(self.lora_alpha, dtype=torch.float32)

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
        reject_quantized_base(trainer.unet, model_label="SD1.5")

        # Set requires_grad based on configuration
        if trainer.train_unet and trainer.unet is not None:
            trainer.unet.requires_grad_(True)
            trainer.unet.train()

        if trainer.train_text_encoder and trainer.text_encoder is not None:
            trainer.text_encoder.requires_grad_(True)
            trainer.text_encoder.train()
        else:
            if trainer.text_encoder is not None:
                trainer.text_encoder.requires_grad_(False)
                trainer.text_encoder.eval()

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
        # Second gate, not a duplicate: a caller that builds the optimizer without
        # going through prepare_models_for_training() would otherwise still get
        # the silently-truncated parameter list this guard exists to prevent.
        reject_quantized_base(trainer.unet, model_label="SD1.5")

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
        Save full parameter checkpoint in single safetensors format.

        Uses ComfyUI-compatible key prefixes:
        - UNet: "model.diffusion_model.*"
        - VAE: "first_stage_model.*"
        - Text Encoder: "cond_stage_model.transformer.*"

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

        # Save U-Net weights: convert diffusers -> CompVis/LDM format
        if trainer.train_unet and trainer.unet is not None:
            print(f"[SD15FullParameterAdapter] Collecting U-Net weights (diffusers -> CompVis)...")
            unet_state = trainer.unet.state_dict()
            converted_unet = convert_unet_state_dict_to_original(unet_state)
            for key, value in converted_unet.items():
                combined_state_dict[f"model.diffusion_model.{key}"] = value.cpu()

        # Save VAE weights: convert diffusers -> CompVis/LDM format (gated on
        # bundle_vae; per-arch default sd15=True for A1111/ComfyUI compatibility).
        from api.param_defaults import resolve_bundle_vae
        bundle_vae = resolve_bundle_vae(getattr(trainer, "bundle_vae", None), "sd15")
        vae_embedded = bundle_vae and trainer.vae is not None
        if vae_embedded:
            print(f"[SD15FullParameterAdapter] Collecting VAE weights (diffusers -> CompVis, bundle_vae)...")
            vae_state = trainer.vae.state_dict()
            converted_vae = convert_vae_state_dict_to_original(vae_state)
            for key, value in converted_vae.items():
                combined_state_dict[f"first_stage_model.{key}"] = value.cpu()

        # Save Text Encoder weights (CLIP ViT-L, no conversion needed)
        if trainer.train_text_encoder and trainer.text_encoder is not None:
            print(f"[SD15FullParameterAdapter] Collecting Text Encoder weights...")
            te_state = trainer.text_encoder.state_dict()
            converted_te = convert_openai_text_enc_to_original(te_state)
            for key, value in converted_te.items():
                combined_state_dict[f"cond_stage_model.transformer.{key}"] = value.cpu()

        # Save to safetensors with metadata
        metadata = {
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "sd15",
            "component.vae.embedded": "1" if vae_embedded else "0",
            **sd15_modelspec_metadata(trainer),
        }

        print(f"[SD15FullParameterAdapter] Saving to {output_path}...")
        save_file(combined_state_dict, output_path, metadata=metadata)

        total_params = sum(p.numel() for p in combined_state_dict.values())
        print(f"[SD15FullParameterAdapter] Saved {len(combined_state_dict)} tensors ({total_params:,} params) to {output_path}")
