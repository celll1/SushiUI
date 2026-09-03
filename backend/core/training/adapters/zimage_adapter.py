"""
Z-Image model adapter for LoRA and Full Parameter training.

Model characteristics:
- Qwen3 text encoder (AutoModelForCausalLM)
- ZImageTransformer2DModel (flow matching, not DDPM)
- Chat template for text encoding
- BatchedZImageWrapper for batching
- Frame dimension: [B, C, H, W] → [B, C, 1, H, W]

LoRA targets:
- Transformer: ZImageAttention modules (to_q, to_k, to_v, to_out[0])
- Text Encoder: Frozen (no LoRA)

Key implementation details:
- Text encoding uses Qwen chat template
- Penultimate layer extraction (like SDXL TE2)
- Flow matching training (noise_process="flow", prediction_target="velocity")
- Frame dimension handling in train_step_zimage (BaseTrainer)

Restored from old implementation (commit 729ee38).

Author: Claude (2026-01-04)
"""

from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
from safetensors.torch import save_file
import math

from core.adapters import LoRALinearLayer, is_lora_wrappable_linear
from .base_adapter import (
    BaseLoRAAdapter, BaseFullParameterAdapter,
    reject_quantized_base, LORA_COMPONENT_UNET,
)


# ============================================================
# Z-Image LoRA Adapter
# ============================================================

class ZImageLoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for Z-Image models."""

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to Z-Image Transformer attention layers.

        Targets ZImageAttention modules: to_q, to_k, to_v, to_out[0] (ModuleList)

        Based on musubi-tuner's lora_zimage.py implementation:
        - ZIMAGE_TARGET_REPLACE_MODULES = ["ZImageTransformerBlock"]
        - Attention layers: qkv_proj, out_proj (musubi splits into to_q/k/v internally)

        Args:
            lora_layers: Dictionary to store LoRA layer references

        Returns:
            Number of LoRA layers injected
        """
        count = 0

        print(f"[ZImageLoRAAdapter] Applying LoRA to Z-Image Transformer (ZImageAttention modules)")

        # Access the original transformer inside the wrapper
        # self.transformer is BatchedZImageWrapper, self.transformer.transformer is the original model
        transformer = self.trainer.transformer
        target_transformer = (
            transformer.transformer
            if hasattr(transformer, 'transformer')
            else transformer
        )

        # Find all ZImageAttention modules in the Transformer
        attention_modules = []
        for name, module in target_transformer.named_modules():
            if module.__class__.__name__ == "ZImageAttention":
                attention_modules.append((name, module))

        print(f"[ZImageLoRAAdapter] Found {len(attention_modules)} ZImageAttention modules")

        # Target layers: to_q, to_k, to_v, to_out[0]
        target_attrs = ["to_q", "to_k", "to_v"]

        for attn_name, attn_module in attention_modules:
            # Handle to_q, to_k, to_v
            for attr_name in target_attrs:
                if hasattr(attn_module, attr_name):
                    original_linear = getattr(attn_module, attr_name)

                    # NOT ``isinstance(x, torch.nn.Linear)``: a Z-Image base can
                    # now be weight-only quantized (Int8Linear / Fp8Linear), and
                    # those are nn.Modules but not nn.Linear subclasses, so the
                    # naive test drops every quantized target silently.
                    if is_lora_wrappable_linear(original_linear):
                        # Create LoRA layer
                        lora_name = f"lora_transformer_{attn_name.replace('.', '_')}_{attr_name}"
                        lora_layer = LoRALinearLayer(
                            original_linear, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype
                        )

                        # Replace in attention module
                        setattr(attn_module, attr_name, lora_layer)

                        # Store reference
                        self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
                        count += 1

            # Handle to_out (ModuleList in Z-Image, first element is Linear projection)
            if hasattr(attn_module, "to_out") and isinstance(attn_module.to_out, torch.nn.ModuleList):
                if len(attn_module.to_out) > 0 and is_lora_wrappable_linear(attn_module.to_out[0]):
                    original_linear = attn_module.to_out[0]

                    # Create LoRA layer
                    lora_name = f"lora_transformer_{attn_name.replace('.', '_')}_to_out_0"
                    lora_layer = LoRALinearLayer(
                        original_linear, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype
                    )

                    # Replace in ModuleList
                    attn_module.to_out[0] = lora_layer

                    # Store reference
                    self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
                    count += 1

        print(f"[ZImageLoRAAdapter] Injected {count} LoRA layers into Z-Image Transformer")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to Text Encoder (Qwen3).

        Note: For Z-Image, the text encoder (Qwen3) is typically kept frozen.
        This method returns 0 by default.

        Args:
            lora_layers: Dictionary to store LoRA layer references

        Returns:
            Number of LoRA layers injected (0 for Z-Image)
        """
        print(f"[ZImageLoRAAdapter] Text Encoder (Qwen3) is frozen (no LoRA)")
        return 0

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        """
        Collect trainable parameters with per-component learning rates.

        For Z-Image, only Transformer LoRA parameters are trainable.

        Args:
            lora_layers: Dictionary of LoRA layers

        Returns:
            List of parameter groups for optimizer
        """
        params = []
        transformer_params = []

        for lora_name, lora_layer in lora_layers.items():
            if lora_name.startswith("lora_transformer_"):
                transformer_params.extend(lora_layer.lora_down.parameters())
                transformer_params.extend(lora_layer.lora_up.parameters())

        # Add parameter group with Transformer learning rate
        if transformer_params:
            params.append({"params": transformer_params, "lr": self.trainer.unet_lr})

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

        Key stem is ``lora_name`` verbatim (``lora_transformer_<module path with
        dots flattened>``), which is also the resume key in
        ``LoRATrainer.load_checkpoint``; the generation loader
        (``pipeline_backends/zimage._zimage_lora_key_stems``) reconstructs it
        from the module path. Renaming it here would strand resume for every
        checkpoint already on disk, so a spelling mismatch is repaired on the
        load side instead. Alpha is metadata-only, and that loader reads it
        from there -- an ``alpha != rank`` LoRA would otherwise apply at scale 1.

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
            "model_type": "zimage",
        }

        # Save safetensors
        save_file(lora_state_dict, output_path, metadata=metadata)
        print(f"[ZImageLoRAAdapter] Saved LoRA checkpoint: {output_path}")


# ============================================================
# Z-Image Full Parameter Adapter
# ============================================================

class ZImageFullParameterAdapter(BaseFullParameterAdapter):
    """Full parameter adapter for Z-Image models."""

    def prepare_models_for_training(self):
        """Prepare models for full parameter training."""
        trainer = self.trainer
        # Unwrap BatchedZImageWrapper (if present) before checking for
        # weight-only quantized Linears -- the wrapper itself never owns them.
        _target_for_guard = (
            trainer.transformer.transformer
            if trainer.transformer is not None and hasattr(trainer.transformer, 'transformer')
            else trainer.transformer
        )
        reject_quantized_base(_target_for_guard, model_label="Z-Image")

        # Set requires_grad based on configuration
        # Note: For Z-Image, we train the Transformer (not U-Net)
        if trainer.train_unet and trainer.transformer is not None:
            # Access the original transformer inside the wrapper
            target_transformer = (
                trainer.transformer.transformer
                if hasattr(trainer.transformer, 'transformer')
                else trainer.transformer
            )
            target_transformer.requires_grad_(True)
            target_transformer.train()
            print(f"[ZImageFullParameterAdapter] Z-Image Transformer set to train mode")

        if trainer.train_text_encoder and trainer.text_encoder is not None:
            trainer.text_encoder.requires_grad_(True)
            trainer.text_encoder.train()
            print(f"[ZImageFullParameterAdapter] Text Encoder (Qwen3) set to train mode")
        else:
            if trainer.text_encoder is not None:
                trainer.text_encoder.requires_grad_(False)
                trainer.text_encoder.eval()
            print(f"[ZImageFullParameterAdapter] Text Encoder (Qwen3) is frozen")

        # VAE is always frozen
        if trainer.vae is not None:
            trainer.vae.requires_grad_(False)
            trainer.vae.eval()

        print(f"[ZImageFullParameterAdapter] Models prepared for training")
        print(f"  Transformer trainable: {trainer.train_unet}")
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
        _target_for_guard = (
            trainer.transformer.transformer
            if trainer.transformer is not None and hasattr(trainer.transformer, 'transformer')
            else trainer.transformer
        )
        reject_quantized_base(_target_for_guard, model_label="Z-Image")

        if trainer.train_unet and trainer.transformer is not None:
            # Access the original transformer inside the wrapper
            target_transformer = (
                trainer.transformer.transformer
                if hasattr(trainer.transformer, 'transformer')
                else trainer.transformer
            )
            transformer_params = [p for p in target_transformer.parameters() if p.requires_grad]
            if transformer_params:
                params.append({"params": transformer_params, "lr": trainer.unet_lr})

        if trainer.train_text_encoder and trainer.text_encoder is not None:
            te_params = [p for p in trainer.text_encoder.parameters() if p.requires_grad]
            if te_params:
                params.append({"params": te_params, "lr": trainer.text_encoder_1_lr})

        return params

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        """
        Save full parameter checkpoint in single safetensors format.

        Uses ComfyUI-compatible key prefixes:
        - Transformer: "model.diffusion_model.*"
        - VAE: "first_stage_model.*"
        - Text Encoder: "text_encoders.qwen3.*" (Qwen3 for Z-Image)

        Args:
            step: Current training step
            epoch: Current training epoch
            output_path: Path to save checkpoint (should be .safetensors file)
        """
        from safetensors.torch import save_file

        trainer = self.trainer

        # Ensure output_path is a file path, not directory
        if output_path.is_dir():
            output_path = output_path / f"model_step_{step}.safetensors"
        elif not str(output_path).endswith(".safetensors"):
            output_path = Path(str(output_path) + ".safetensors")

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined_state_dict = {}

        # Save Transformer weights with ComfyUI prefix
        if trainer.train_unet and trainer.transformer is not None:
            print(f"[ZImageFullParameterAdapter] Collecting Transformer weights...")
            # Access the original transformer inside the wrapper
            target_transformer = (
                trainer.transformer.transformer
                if hasattr(trainer.transformer, 'transformer')
                else trainer.transformer
            )
            transformer_state = target_transformer.state_dict()
            for key, value in transformer_state.items():
                combined_state_dict[f"model.diffusion_model.{key}"] = value.cpu()

        # Save VAE weights with ComfyUI prefix (only when bundle_vae is enabled;
        # default off -> loader falls back to the default VAE resolution).
        from api.param_defaults import resolve_bundle_vae
        bundle_vae = resolve_bundle_vae(getattr(trainer, "bundle_vae", None), "zimage")
        vae_embedded = bundle_vae and trainer.vae is not None
        if vae_embedded:
            print(f"[ZImageFullParameterAdapter] Collecting VAE weights (bundle_vae)...")
            vae_state = trainer.vae.state_dict()
            for key, value in vae_state.items():
                combined_state_dict[f"first_stage_model.{key}"] = value.cpu()

        # Save Text Encoder weights with Z-Image prefix
        if trainer.train_text_encoder and trainer.text_encoder is not None:
            print(f"[ZImageFullParameterAdapter] Collecting Text Encoder (Qwen3) weights...")
            te_state = trainer.text_encoder.state_dict()
            for key, value in te_state.items():
                # Z-Image uses text_encoders.qwen3 prefix
                combined_state_dict[f"text_encoders.qwen3.{key}"] = value.cpu()

        # Save to safetensors with metadata
        te_embedded = bool(trainer.train_text_encoder and trainer.text_encoder is not None)
        metadata = {
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "zimage",
            "format": "pt",
        }
        # ZImageTransformer2DModel is a plain nn.Module without a serializable config
        # object; the loader reconstructs config from the base repo + shape probes, so
        # no transformer_config JSON is written here. Component hints are declarative.
        try:
            from core.models.common.single_file_format import build_component_metadata
            metadata.update(build_component_metadata(
                te_type="qwen3", te_embedded=te_embedded,
                vae_type="flux1", vae_embedded=vae_embedded,
            ))
        except Exception as _e:
            print(f"[ZImageFullParameterAdapter] component metadata skipped: {_e}")

        print(f"[ZImageFullParameterAdapter] Saving to {output_path}...")
        # Route through the shared single-file writer so >10 GB full-FT saves
        # auto-shard (diffusers convention + <stem>.safetensors.index.json). For
        # sub-threshold saves this writes an identical single .safetensors with
        # the same keys+metadata as the previous direct save_file call.
        from core.models.common.single_file_format import (
            save_single_file_state, dedup_tensors,
        )
        dedup_state, dropped_tied = dedup_tensors(combined_state_dict.items())
        if dropped_tied:
            metadata["tied_weights_dropped"] = ",".join(dropped_tied)
        written_path = save_single_file_state(dedup_state, metadata, str(output_path))

        total_params = sum(p.numel() for p in dedup_state.values())
        print(f"[ZImageFullParameterAdapter] Saved {len(dedup_state)} tensors ({total_params:,} params) to {written_path}")
