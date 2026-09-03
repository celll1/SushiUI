"""Krea 2 (single-stream flow-matching MMDiT) training adapters.

Model characteristics:
  - Single-stream MMDiT (28 ``transformer_blocks`` + internal ``text_fusion``),
    latent-space, flow matching (velocity ``v = noise - x0``).
  - Qwen3-VL-4B text encoder — ALWAYS FROZEN (no TE LoRA / no TE full-FT). This
    mirrors the ideogram4 Qwen3-VL policy; ``train_text_encoder`` is rejected.
  - AutoencoderKLQwenImage VAE (16ch, latents_mean/std) — frozen.

LoRA target scope (core/models/krea2/krea2_lora.iter_krea2_lora_targets):
  attn:        transformer_blocks.{N}.attn.{to_q,to_k,to_v,to_gate,to_out.0}
  mlp:         transformer_blocks.{N}.ff.{gate,up,down}
  text_fusion: text_fusion.{layerwise,refiner}_blocks + projector (default OFF)
  proj:        img_in / txt_in / final_layer / time embeds        (default OFF)

Save format:
  - LoRA: sd-scripts-style ``lora_unet_<flat>.lora_down/up.weight`` / ``.alpha``
    (flat uses the ``.``<->``__`` reversible encoding from krea2_lora).
  - Full-FT: a sushiUI Krea 2 single-file (transformer [+ optional TE] + metadata)
    via core.models.krea2.vendor.single_file.save_single_file. The Qwen3-VL TE is
    frozen, so it is NOT bundled (has_text_encoder stays 0).
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .base_adapter import (
    BaseLoRAAdapter, BaseFullParameterAdapter, reject_quantized_base,
    resolve_component_lr, LORA_COMPONENT_UNET,
)
from core.adapters import LoRALinearLayer

from core.models.krea2.krea2_lora import (
    iter_krea2_lora_targets, DEFAULT_SCOPE, flatten_to_key,
)


class Krea2LoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for the Krea 2 DiT (transformer only; TE frozen)."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_SCOPE) if scope is None else dict(scope)

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        transformer = self.trainer.transformer
        if transformer is None:
            print("[Krea2LoRAAdapter] WARNING: trainer.transformer is None - skipping")
            return 0
        print(f"[Krea2LoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")
        count = 0
        for module_path, parent, attr, current in iter_krea2_lora_targets(transformer, self.scope):
            if isinstance(current, LoRALinearLayer):
                continue
            lora_name = flatten_to_key(module_path)  # "lora_unet_<flat>"
            lora_layer = LoRALinearLayer(current, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype)
            if isinstance(attr, int):
                parent[attr] = lora_layer
            else:
                setattr(parent, attr, lora_layer)
            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
            count += 1
        print(f"[Krea2LoRAAdapter] Injected {count} LoRA layer(s)")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """Qwen3-VL text encoder is frozen — no LoRA applied."""
        print("[Krea2LoRAAdapter] Qwen3-VL text encoder is frozen - no LoRA on TE")
        return 0

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        params: List[nn.Parameter] = []
        for lora_layer in lora_layers.values():
            params.extend(lora_layer.lora_down.parameters())
            params.extend(lora_layer.lora_up.parameters())
        if not params:
            return []
        base_lr = resolve_component_lr(self.trainer, "unet_lr", label="Krea 2 LoRA")
        lr_factor = float(self.trainer.config.get("krea2_lr_factor", 1.0))
        return [{"params": params, "lr": base_lr * lr_factor}]

    def save_checkpoint(self, lora_layers: Dict[str, nn.Module], step: int, epoch: int, output_path: Path):
        state_dict: Dict[str, torch.Tensor] = {}
        alpha_value = float(self.lora_alpha)
        for lora_name, lora_layer in lora_layers.items():
            state_dict[f"{lora_name}.lora_down.weight"] = lora_layer.lora_down.weight.detach().cpu()
            state_dict[f"{lora_name}.lora_up.weight"] = lora_layer.lora_up.weight.detach().cpu()
            state_dict[f"{lora_name}.alpha"] = torch.tensor(alpha_value, dtype=torch.float32)
        active_scopes = ",".join(k for k, v in self.scope.items() if v)
        metadata = {
            "model_type":             "krea2",
            "modelspec.architecture": "krea2",
            "variant":                "turbo" if bool(getattr(self.trainer, "krea2_is_distilled", False)) else "raw",
            "lora_rank":              str(self.lora_rank),
            "lora_alpha":             str(self.lora_alpha),
            "lora_targets":           active_scopes,
            "step":                   str(step),
            "epoch":                  str(epoch),
            "format":                 "pt",
        }
        save_file(state_dict, str(output_path), metadata=metadata)
        print(f"[Krea2LoRAAdapter] Saved LoRA checkpoint ({len(lora_layers)} layers) -> {output_path}")


class Krea2FullParameterAdapter(BaseFullParameterAdapter):
    """Full-parameter adapter for Krea 2 (transformer; Qwen3-VL TE stays frozen).

    Full fine-tuning of the 12.9B transformer needs a bf16 (non-FP8) base — the
    train_runner forces bf16 for krea2. The frozen Qwen3-VL text encoder is NOT
    bundled into the saved single-file (has_text_encoder=0)."""

    def prepare_models_for_training(self):
        trainer = self.trainer
        reject_quantized_base(trainer.transformer, model_label="Krea 2")
        if bool(getattr(trainer, "train_text_encoder", False)):
            raise ValueError(
                "[Krea2FullParameterAdapter] Qwen3-VL text encoder training is not "
                "supported for Krea 2 - set train_text_encoder=False."
            )
        if getattr(trainer, "train_unet", True) and trainer.transformer is not None:
            trainer.transformer.requires_grad_(True)
            trainer.transformer.train()
            print("[Krea2FullParameterAdapter] Krea 2 transformer set to train mode")
        if trainer.text_encoder is not None:
            trainer.text_encoder.requires_grad_(False)
            trainer.text_encoder.eval()
            print("[Krea2FullParameterAdapter] Qwen3-VL text encoder is frozen")
        if getattr(trainer, "vae", None) is not None:
            trainer.vae.requires_grad_(False)

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        trainer = self.trainer
        # Second gate, not a duplicate: a caller that builds the optimizer without
        # going through prepare_models_for_training() would otherwise still get
        # the silently-truncated parameter list this guard exists to prevent.
        reject_quantized_base(trainer.transformer, model_label="Krea 2")
        groups: List[Dict[str, Any]] = []
        if trainer.transformer is not None:
            t_params = [p for p in trainer.transformer.parameters() if p.requires_grad]
            if t_params:
                base_lr = resolve_component_lr(trainer, "unet_lr", label="Krea 2 transformer")
                print(f"[Krea2FullParameterAdapter] {sum(p.numel() for p in t_params):,} trainable params (transformer)")
                groups.append({"params": t_params, "lr": base_lr})
        return groups

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        from core.models.krea2.vendor.single_file import save_single_file
        trainer = self.trainer
        if trainer.transformer is None:
            print("[Krea2FullParameterAdapter] WARNING: no transformer to save")
            return
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"krea2_step_{step}.safetensors"
        elif not str(output_path).endswith(".safetensors"):
            output_path = Path(str(output_path) + ".safetensors")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        is_distilled = bool(getattr(trainer, "krea2_is_distilled", False))
        # Qwen3-VL TE is frozen -> not bundled. VAE bundled only when bundle_vae is set
        # (default off -> loader resolves the default Qwen-Image VAE). Krea 2 uses the
        # sushiUI-v2 common ``vae.`` prefix (VAE_PREFIX).
        from api.param_defaults import resolve_bundle_vae
        bundle_vae = resolve_bundle_vae(getattr(trainer, "bundle_vae", None), "krea2")
        vae_to_bundle = trainer.vae if (bundle_vae and getattr(trainer, "vae", None) is not None) else None
        save_single_file(
            str(output_path), trainer.transformer, is_distilled=is_distilled,
            text_encoder=None, vae=vae_to_bundle,
            extra_metadata={"step": str(step), "epoch": str(epoch)},
        )
        print(f"[Krea2FullParameterAdapter] Saved single-file (transformer) -> {output_path}")
