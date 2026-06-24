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


def sushi_modelspec_metadata(trainer) -> Dict[str, str]:
    """Unified SushiUI prediction metadata for a saved model.

    Records the resolved noise_process / prediction_target so the loader can
    reproduce the objective (epsilon / v / x / flow) without guessing — read by
    ModelLoader.detect_prediction_config (modelspec.* has top priority). Resolved
    "auto" values are skipped (the loader then infers as before). Architecture is
    tagged for the model browser.
    """
    md: Dict[str, str] = {"modelspec.architecture": "stable-diffusion-xl-v1-base"}
    np = str(getattr(trainer, "noise_process", "") or "").strip().lower()
    pt = str(getattr(trainer, "prediction_target", "") or "").strip().lower()
    if np and np != "auto":
        md["modelspec.noise_process"] = np
    if pt and pt != "auto":
        md["modelspec.prediction_type"] = pt
    # Custom-arch markers (SushiUI): non-standard VAE / latent channels so the loader
    # can reconstruct (swap VAE + resize conv) on load. Only written when non-default.
    vae_type = str(getattr(trainer, "sdxl_vae_type", "") or "").strip().lower()
    if vae_type and vae_type not in ("none", "sdxl"):
        md["sushi.vae_type"] = vae_type
        md["sushi.in_channels"] = str(int(getattr(trainer, "vae_latent_channels", 4) or 4))
        md["modelspec.architecture"] = "sdxl-custom"
    # Custom text encoder (swapped): record how to rebuild + whether the body is embedded.
    te_type = str(getattr(trainer, "sdxl_te_type", "") or "").strip().lower()
    if te_type and te_type not in ("none", "clip"):
        md["sushi.te_type"] = te_type
        md["sushi.te_hidden_layer"] = str(int(getattr(trainer, "te_hidden_layer", -2)))
        md["sushi.te_max_len"] = str(int(getattr(trainer, "te_max_len", 256)))
        md["sushi.te_dim"] = str(int(getattr(trainer, "te_dim", 0) or 0))
        md["sushi.te_embedded"] = "1" if bool(getattr(trainer, "sdxl_te_train_encoder", False)) else "0"
        md["modelspec.architecture"] = "sdxl-custom"
    return md
from .state_dict_converter import (
    convert_unet_state_dict_to_original,
    convert_vae_state_dict_to_original,
    convert_openclip_text_enc_to_original,
    convert_openai_text_enc_to_original,
)


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
                # Use sd-scripts compatible naming: lora_te1_text_model_encoder_layers_{N}_mlp_fc1
                lora_name = f"lora_te1_text_model_encoder_layers_{layer_idx}_mlp_fc1"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc1, self.lora_rank, self.lora_alpha, lora_name
                )
                layer.mlp.fc1 = lora_layer
                lora_layers[lora_name] = lora_layer
                count += 1

                # mlp.fc2
                lora_name = f"lora_te1_text_model_encoder_layers_{layer_idx}_mlp_fc2"
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
                # Use sd-scripts compatible naming: lora_te2_text_model_encoder_layers_{N}_mlp_fc1
                lora_name = f"lora_te2_text_model_encoder_layers_{layer_idx}_mlp_fc1"
                lora_layer = LoRALinearLayer(
                    layer.mlp.fc1, self.lora_rank, self.lora_alpha, lora_name
                )
                layer.mlp.fc1 = lora_layer
                lora_layers[lora_name] = lora_layer
                count += 1

                # mlp.fc2
                lora_name = f"lora_te2_text_model_encoder_layers_{layer_idx}_mlp_fc2"
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
            **sushi_modelspec_metadata(self.trainer),
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
        else:
            # Explicitly freeze text encoders when not training them.
            # PyTorch defaults requires_grad=True for float parameters, so this
            # must be set explicitly to avoid spurious gradients.
            if trainer.text_encoder is not None:
                trainer.text_encoder.requires_grad_(False)
                trainer.text_encoder.eval()
            if trainer.text_encoder_2 is not None:
                trainer.text_encoder_2.requires_grad_(False)
                trainer.text_encoder_2.eval()

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

        # Custom SDXL TE: bridge adapters (always trainable) + optionally the encoder body.
        if getattr(trainer, "sdxl_te_type", "none") not in ("none", "clip", "", None) \
                and getattr(trainer, "te_adapters", None) is not None:
            ad_params = [p for p in trainer.te_adapters.parameters() if p.requires_grad]
            if ad_params:
                ad_lr = getattr(trainer, "text_encoder_lr", None) or trainer.unet_lr
                print(f"[SDXLFullParameterAdapter] {sum(p.numel() for p in ad_params):,} trainable "
                      f"params (custom-TE bridge adapters), lr={ad_lr}")
                params.append({"params": ad_params, "lr": ad_lr})
            if getattr(trainer, "sdxl_te_train_encoder", False) and getattr(trainer, "te_custom", None) is not None:
                te_params = [p for p in trainer.te_custom.parameters() if p.requires_grad]
                if te_params:
                    te_lr = getattr(trainer, "text_encoder_1_lr", None) or getattr(trainer, "text_encoder_lr", None) or trainer.unet_lr
                    params.append({"params": te_params, "lr": te_lr})

        return params

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        """
        Save full parameter checkpoint in single safetensors format.

        Uses ComfyUI-compatible key prefixes:
        - UNet: "model.diffusion_model.*"
        - VAE: "first_stage_model.*"
        - Text Encoder 1: "conditioner.embedders.0.transformer.*"
        - Text Encoder 2: "conditioner.embedders.1.model.*"

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
            print(f"[SDXLFullParameterAdapter] Collecting U-Net weights (diffusers -> CompVis)...")
            unet_state = trainer.unet.state_dict()
            converted_unet = convert_unet_state_dict_to_original(unet_state)
            for key, value in converted_unet.items():
                combined_state_dict[f"model.diffusion_model.{key}"] = value.cpu()

        # Save VAE weights: convert diffusers -> CompVis/LDM format.
        # Custom high-spec VAE (e.g. FLUX.1 16ch) is NOT embedded — it is referenced by
        # registry (metadata sushi.vae_type) and reloaded on load; the SDXL VAE converter
        # assumes the 4ch SDXL VAE structure and would mishandle a different VAE.
        _custom_vae = str(getattr(trainer, "sdxl_vae_type", "") or "").strip().lower() not in ("", "none", "sdxl")
        if trainer.vae is not None and not _custom_vae:
            print(f"[SDXLFullParameterAdapter] Collecting VAE weights (diffusers -> CompVis)...")
            vae_state = trainer.vae.state_dict()
            converted_vae = convert_vae_state_dict_to_original(vae_state)
            for key, value in converted_vae.items():
                combined_state_dict[f"first_stage_model.{key}"] = value.cpu()
        elif _custom_vae:
            print(f"[SDXLFullParameterAdapter] Custom VAE ({trainer.sdxl_vae_type}) not embedded "
                  f"(referenced by metadata sushi.vae_type, reloaded on load).")

        # Save Text Encoder 1 weights (CLIP ViT-L, no conversion needed)
        if trainer.train_text_encoder and trainer.text_encoder is not None:
            print(f"[SDXLFullParameterAdapter] Collecting Text Encoder 1 weights...")
            te1_state = trainer.text_encoder.state_dict()
            converted_te1 = convert_openai_text_enc_to_original(te1_state)
            for key, value in converted_te1.items():
                combined_state_dict[f"conditioner.embedders.0.transformer.{key}"] = value.cpu()

        # Save Text Encoder 2 weights: convert HF -> OpenCLIP format
        if trainer.train_text_encoder and trainer.text_encoder_2 is not None:
            print(f"[SDXLFullParameterAdapter] Collecting Text Encoder 2 weights (HF -> OpenCLIP)...")
            te2_state = trainer.text_encoder_2.state_dict()
            converted_te2 = convert_openclip_text_enc_to_original(te2_state)
            for key, value in converted_te2.items():
                combined_state_dict[f"conditioner.embedders.1.model.{key}"] = value.cpu()
            # text_projection: remove .weight suffix and transpose
            tp_key = "conditioner.embedders.1.model.text_projection.weight"
            if tp_key in combined_state_dict:
                combined_state_dict["conditioner.embedders.1.model.text_projection"] = (
                    combined_state_dict.pop(tp_key).T.contiguous()
                )

        # Custom SDXL Text Encoder: always embed the trained bridge adapters; embed the
        # encoder body only when it was fine-tuned (train_encoder), else it is reloaded
        # from the registry repo on load.
        _custom_te = str(getattr(trainer, "sdxl_te_type", "") or "").strip().lower() not in ("", "none", "clip")
        if _custom_te and getattr(trainer, "te_adapters", None) is not None:
            print(f"[SDXLFullParameterAdapter] Collecting custom-TE bridge adapters...")
            for key, value in trainer.te_adapters.state_dict().items():
                combined_state_dict[f"sushi.te_adapter.{key}"] = value.detach().cpu()
            if bool(getattr(trainer, "sdxl_te_train_encoder", False)) and getattr(trainer, "te_custom", None) is not None:
                print(f"[SDXLFullParameterAdapter] Collecting fine-tuned custom-TE encoder body...")
                for key, value in trainer.te_custom.state_dict().items():
                    combined_state_dict[f"sushi.te_encoder.{key}"] = value.detach().cpu()

        # Save to safetensors with metadata
        metadata = {
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "sdxl",
            **sushi_modelspec_metadata(self.trainer),
        }

        print(f"[SDXLFullParameterAdapter] Saving to {output_path}...")
        save_file(combined_state_dict, output_path, metadata=metadata)

        total_params = sum(p.numel() for p in combined_state_dict.values())
        print(f"[SDXLFullParameterAdapter] Saved {len(combined_state_dict)} tensors ({total_params:,} params) to {output_path}")
