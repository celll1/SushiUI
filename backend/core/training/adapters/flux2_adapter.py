"""
FLUX.2 Klein model adapter for LoRA and Full Parameter training.

Model characteristics:
- Qwen3 text encoder (Qwen3ForCausalLM)
- Flux2Transformer2DModel (8 dual stream + 48 single stream blocks)
- AutoencoderKLFlux2 (32ch latent with BatchNorm)
- Flow matching with velocity prediction
- 4D position coordinates for RoPE (T, H, W, L)

Key architecture differences from FLUX.1:
- Single stream blocks use parallel attention+MLP (fused projections)
- VAE uses BatchNorm for latent normalization
- Text encoder extracts hidden states from layers 9, 18, 27

LoRA targets:
- Transformer: Flux2Attention and Flux2ParallelSelfAttention modules
  - Dual stream: to_q, to_k, to_v, to_out[0], add_q_proj, add_k_proj, add_v_proj, to_add_out
  - Single stream: to_qkv_mlp_proj, to_out (fused projections)
- Text Encoder: Qwen3 layers (mlp.gate_proj, mlp.up_proj, mlp.down_proj)

Author: Claude (2026-01-16)
"""

from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
import math

from core.adapters import LoRALinearLayer, is_lora_wrappable_linear
from .base_adapter import (
    BaseLoRAAdapter, BaseFullParameterAdapter,
    reject_quantized_base, LORA_COMPONENT_UNET, LORA_COMPONENT_TEXT_ENCODER,
)


# ============================================================
# FLUX.2 Klein LoRA Adapter
# ============================================================

class FLUX2LoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for FLUX.2 Klein models."""

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to FLUX.2 Transformer attention and feedforward layers.

        FLUX.2 has two block types:
        1. Dual stream blocks (Flux2TransformerBlock):
           - Flux2Attention: to_q, to_k, to_v, to_out[0], add_q_proj, add_k_proj, add_v_proj, to_add_out
           - Flux2FeedForward: linear_in, linear_out

        2. Single stream blocks (Flux2SingleTransformerBlock):
           - Flux2ParallelSelfAttention: to_qkv_mlp_proj (fused), to_out

        Args:
            lora_layers: Dictionary to store LoRA layer references

        Returns:
            Number of LoRA layers injected
        """
        count = 0

        print(f"[FLUX2LoRAAdapter] Applying LoRA to FLUX.2 Transformer")

        transformer = self.trainer.transformer
        if transformer is None:
            print(f"[FLUX2LoRAAdapter] Warning: Transformer is None")
            return 0

        # Target attention modules in FLUX.2
        # Flux2Attention - dual stream blocks (layers 0-7)
        # Flux2ParallelSelfAttention - single stream blocks (layers 8-55)
        for name, module in transformer.named_modules():
            # Flux2Attention (dual stream blocks)
            if module.__class__.__name__ == "Flux2Attention":
                # Standard QKV projections
                for attr_name in ["to_q", "to_k", "to_v"]:
                    if hasattr(module, attr_name):
                        original_linear = getattr(module, attr_name)
                        if is_lora_wrappable_linear(original_linear):
                            lora_name = f"lora_transformer_{name.replace('.', '_')}_{attr_name}"
                            lora_layer = self.build_branch(original_linear, lora_name)
                            setattr(module, attr_name, lora_layer)
                            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
                            count += 1

                # to_out (ModuleList)
                if hasattr(module, "to_out") and isinstance(module.to_out, torch.nn.ModuleList):
                    if len(module.to_out) > 0 and is_lora_wrappable_linear(module.to_out[0]):
                        lora_name = f"lora_transformer_{name.replace('.', '_')}_to_out_0"
                        lora_layer = self.build_branch(module.to_out[0], lora_name)
                        module.to_out[0] = lora_layer
                        self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
                        count += 1

                # Additional projections for encoder cross attention (dual stream specific)
                for attr_name in ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]:
                    if hasattr(module, attr_name):
                        original_linear = getattr(module, attr_name)
                        if is_lora_wrappable_linear(original_linear):
                            lora_name = f"lora_transformer_{name.replace('.', '_')}_{attr_name}"
                            lora_layer = self.build_branch(original_linear, lora_name)
                            setattr(module, attr_name, lora_layer)
                            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
                            count += 1

            # Flux2ParallelSelfAttention (single stream blocks)
            elif module.__class__.__name__ == "Flux2ParallelSelfAttention":
                # Fused QKV + MLP projection
                if hasattr(module, "to_qkv_mlp_proj"):
                    original_linear = module.to_qkv_mlp_proj
                    if is_lora_wrappable_linear(original_linear):
                        lora_name = f"lora_transformer_{name.replace('.', '_')}_to_qkv_mlp_proj"
                        lora_layer = self.build_branch(original_linear, lora_name)
                        module.to_qkv_mlp_proj = lora_layer
                        self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
                        count += 1

                # Output projection (attention out + MLP out fused)
                if hasattr(module, "to_out") and is_lora_wrappable_linear(module.to_out):
                    lora_name = f"lora_transformer_{name.replace('.', '_')}_to_out"
                    lora_layer = self.build_branch(module.to_out, lora_name)
                    module.to_out = lora_layer
                    self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
                    count += 1

            # Flux2FeedForward (dual stream blocks only)
            elif module.__class__.__name__ == "Flux2FeedForward":
                for attr_name in ["linear_in", "linear_out"]:
                    if hasattr(module, attr_name):
                        original_linear = getattr(module, attr_name)
                        if is_lora_wrappable_linear(original_linear):
                            lora_name = f"lora_transformer_{name.replace('.', '_')}_{attr_name}"
                            lora_layer = self.build_branch(original_linear, lora_name)
                            setattr(module, attr_name, lora_layer)
                            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
                            count += 1

        print(f"[FLUX2LoRAAdapter] Injected {count} LoRA layers into FLUX.2 Transformer")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to Qwen3 Text Encoder.

        Qwen3 architecture (QWen2ForCausalLM):
        - model.layers[N].mlp.gate_proj (Linear)
        - model.layers[N].mlp.up_proj (Linear)
        - model.layers[N].mlp.down_proj (Linear)
        - model.layers[N].self_attn.q_proj (Linear)
        - model.layers[N].self_attn.k_proj (Linear)
        - model.layers[N].self_attn.v_proj (Linear)
        - model.layers[N].self_attn.o_proj (Linear)

        Note: For FLUX.2, the text encoder is typically kept frozen.
        This method returns 0 by default (can be enabled via config).

        Args:
            lora_layers: Dictionary to store LoRA layer references

        Returns:
            Number of LoRA layers injected (0 if text encoder is frozen)
        """
        # Check if text encoder training is enabled
        if not getattr(self.trainer, 'train_text_encoder', False):
            print(f"[FLUX2LoRAAdapter] Text Encoder (Qwen3) is frozen (no LoRA)")
            return 0

        count = 0
        text_encoder = self.trainer.text_encoder

        if text_encoder is None:
            print(f"[FLUX2LoRAAdapter] Warning: Text Encoder is None")
            return 0

        # Find Qwen layers
        # Qwen3 structure: model.layers[N].{mlp, self_attn}
        layers = None
        if hasattr(text_encoder, "model") and hasattr(text_encoder.model, "layers"):
            layers = text_encoder.model.layers
        elif hasattr(text_encoder, "layers"):
            layers = text_encoder.layers

        if layers is None:
            print(f"[FLUX2LoRAAdapter] Warning: Could not find Qwen3 layers")
            return 0

        for layer_idx, layer in enumerate(layers):
            # MLP layers
            if hasattr(layer, "mlp"):
                for mlp_attr in ["gate_proj", "up_proj", "down_proj"]:
                    if hasattr(layer.mlp, mlp_attr):
                        original_linear = getattr(layer.mlp, mlp_attr)
                        if is_lora_wrappable_linear(original_linear):
                            lora_name = f"lora_te_model_layers_{layer_idx}_mlp_{mlp_attr}"
                            lora_layer = self.build_branch(original_linear, lora_name)
                            setattr(layer.mlp, mlp_attr, lora_layer)
                            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_TEXT_ENCODER)
                            count += 1

            # Self-attention layers
            if hasattr(layer, "self_attn"):
                for attn_attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if hasattr(layer.self_attn, attn_attr):
                        original_linear = getattr(layer.self_attn, attn_attr)
                        if is_lora_wrappable_linear(original_linear):
                            lora_name = f"lora_te_model_layers_{layer_idx}_self_attn_{attn_attr}"
                            lora_layer = self.build_branch(original_linear, lora_name)
                            setattr(layer.self_attn, attn_attr, lora_layer)
                            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_TEXT_ENCODER)
                            count += 1

        print(f"[FLUX2LoRAAdapter] Injected {count} LoRA layers into Qwen3 Text Encoder")
        return count

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        """Collect trainable parameters with per-component learning rates."""
        return self.component_param_groups(lora_layers, {
            LORA_COMPONENT_UNET: lambda: self.trainer.unet_lr,
            LORA_COMPONENT_TEXT_ENCODER: lambda: self.trainer.text_encoder_1_lr,
        })

    CHECKPOINT_LOG_FORMAT = "[{adapter}] Saved LoRA checkpoint: {path}"

    def checkpoint_metadata(
        self, lora_layers: Dict[str, nn.Module], step: int, epoch: int
    ) -> Dict[str, str]:
        return {
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "flux2",
        }


# ============================================================
# FLUX.2 Klein Full Parameter Adapter
# ============================================================

class FLUX2FullParameterAdapter(BaseFullParameterAdapter):
    """Full parameter adapter for FLUX.2 Klein models."""

    def prepare_models_for_training(self):
        """Prepare models for full parameter training."""
        trainer = self.trainer
        reject_quantized_base(trainer.transformer, model_label="FLUX.2 Klein")

        # Set requires_grad based on configuration
        if trainer.train_unet and trainer.transformer is not None:
            trainer.transformer.requires_grad_(True)
            trainer.transformer.train()
            print(f"[FLUX2FullParameterAdapter] FLUX.2 Transformer set to train mode")

        if trainer.train_text_encoder and trainer.text_encoder is not None:
            trainer.text_encoder.requires_grad_(True)
            trainer.text_encoder.train()
            print(f"[FLUX2FullParameterAdapter] Text Encoder (Qwen3) set to train mode")
        else:
            if trainer.text_encoder is not None:
                trainer.text_encoder.requires_grad_(False)
                trainer.text_encoder.eval()
            print(f"[FLUX2FullParameterAdapter] Text Encoder (Qwen3) is frozen")

        # VAE is always frozen
        if trainer.vae is not None:
            trainer.vae.requires_grad_(False)
            trainer.vae.eval()

        print(f"[FLUX2FullParameterAdapter] Models prepared for training")
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
        reject_quantized_base(trainer.transformer, model_label="FLUX.2 Klein")

        if trainer.train_unet and trainer.transformer is not None:
            transformer_params = [p for p in trainer.transformer.parameters() if p.requires_grad]
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

        Uses FLUX.2-compatible key prefixes:
        - Transformer: "model.diffusion_model.*"
        - VAE: "first_stage_model.*"
        - Text Encoder: "text_encoders.qwen3.*"

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

        # Save Transformer weights
        if trainer.train_unet and trainer.transformer is not None:
            print(f"[FLUX2FullParameterAdapter] Collecting Transformer weights...")
            transformer_state = trainer.transformer.state_dict()

            # DEBUG: Check if weights have changed from base model
            # Compare a few key tensors to verify training is affecting the saved weights
            debug_keys = [
                "transformer_blocks.0.attn.to_q.weight",
                "single_transformer_blocks.0.attn.to_qkv.weight",
                "proj_out.weight",
            ]
            print(f"[FLUX2FullParameterAdapter] DEBUG: Checking weight statistics before save...")
            for debug_key in debug_keys:
                if debug_key in transformer_state:
                    t = transformer_state[debug_key]
                    print(f"[FLUX2FullParameterAdapter]   {debug_key}: "
                          f"mean={t.float().mean().item():.8f}, "
                          f"std={t.float().std().item():.8f}, "
                          f"min={t.float().min().item():.6f}, "
                          f"max={t.float().max().item():.6f}, "
                          f"dtype={t.dtype}, device={t.device}")

            # DEBUG: Also check optimizer state to verify gradients are being accumulated
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                opt_state = trainer.optimizer.state_dict()
                num_params_with_state = len(opt_state.get('state', {}))
                print(f"[FLUX2FullParameterAdapter] DEBUG: Optimizer has state for {num_params_with_state} parameters")
                # Check first param's step count
                if opt_state.get('state'):
                    first_param_state = next(iter(opt_state['state'].values()))
                    if 'step' in first_param_state:
                        print(f"[FLUX2FullParameterAdapter] DEBUG: Optimizer step count: {first_param_state['step']}")

            for key, value in transformer_state.items():
                combined_state_dict[f"model.diffusion_model.{key}"] = value.cpu()

        # Save VAE weights (only when bundle_vae is enabled; default off -> the loader
        # falls back to the default FLUX.2 VAE resolution when the section is absent).
        from api.param_defaults import resolve_bundle_vae
        bundle_vae = resolve_bundle_vae(getattr(trainer, "bundle_vae", None), "flux2")
        vae_embedded = bundle_vae and trainer.vae is not None
        if vae_embedded:
            print(f"[FLUX2FullParameterAdapter] Collecting VAE weights (bundle_vae)...")
            vae_state = trainer.vae.state_dict()
            for key, value in vae_state.items():
                combined_state_dict[f"first_stage_model.{key}"] = value.cpu()

        # Save Text Encoder weights
        if trainer.train_text_encoder and trainer.text_encoder is not None:
            print(f"[FLUX2FullParameterAdapter] Collecting Text Encoder (Qwen3) weights...")
            te_state = trainer.text_encoder.state_dict()
            for key, value in te_state.items():
                combined_state_dict[f"text_encoders.qwen3.{key}"] = value.cpu()

        # Save to safetensors with metadata
        # Include base_model_repo and is_distilled for inference
        base_model_repo = getattr(trainer, 'base_model_repo', None)
        is_distilled = getattr(trainer, 'is_distilled', False)

        te_embedded = bool(trainer.train_text_encoder and trainer.text_encoder is not None)
        metadata = {
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "flux2",
            "format": "pt",
        }

        # Add base model info if available
        if base_model_repo:
            metadata["base_model_repo"] = base_model_repo
        if is_distilled is not None:
            metadata["is_distilled"] = str(is_distilled).lower()

        # Transformer config JSON (declarative; Flux2Transformer2DModel is a ConfigMixin).
        try:
            import json as _json
            cfg = getattr(trainer.transformer, "config", None)
            if cfg is not None:
                metadata["transformer_config"] = _json.dumps(dict(cfg))
        except Exception as _e:
            print(f"[FLUX2FullParameterAdapter] transformer_config not serialized: {_e}")

        # Component hints: VAE embedded only when bundle_vae; TE embedded only when trained.
        try:
            from core.models.common.single_file_format import build_component_metadata
            metadata.update(build_component_metadata(
                te_type="qwen3", te_embedded=te_embedded,
                vae_type="flux2", vae_embedded=vae_embedded,
            ))
        except Exception as _e:
            print(f"[FLUX2FullParameterAdapter] component metadata skipped: {_e}")

        print(f"[FLUX2FullParameterAdapter] Saving to {output_path}...")
        # Route through the shared single-file writer so >10 GB full-FT saves
        # auto-shard (diffusers convention + <stem>.safetensors.index.json). For
        # sub-threshold saves this writes an identical single .safetensors with
        # the same keys+metadata as the previous direct save_file call. dedup
        # drops any tied tensors (safetensors would otherwise reject them; the
        # loader re-ties on read); the flux2 combined dict normally has none.
        from core.models.common.single_file_format import (
            save_single_file_state, dedup_tensors,
        )
        dedup_state, dropped_tied = dedup_tensors(combined_state_dict.items())
        if dropped_tied:
            metadata["tied_weights_dropped"] = ",".join(dropped_tied)
        written_path = save_single_file_state(dedup_state, metadata, str(output_path))

        total_params = sum(p.numel() for p in dedup_state.values())
        print(f"[FLUX2FullParameterAdapter] Saved {len(dedup_state)} tensors ({total_params:,} params) to {written_path}")
