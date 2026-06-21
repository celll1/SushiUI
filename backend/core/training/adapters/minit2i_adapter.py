"""MiniT2I (pixel-space MM-JiT) training adapters.

Model characteristics:
  - Pixel-space MM-DiT (double_blocks + txt_preamble_blocks), no VAE
  - Frozen FLAN-T5-Large text encoder (Phase C may unfreeze it)
  - Flow matching, x0 prediction (loss in base_trainer.train_step_minit2i)

LoRA target scope (minit2i_lora.iter_minit2i_lora_targets):
  attn:      double/preamble blocks {img_qkv,txt_qkv,img_attn_proj,txt_attn_proj,qkv,attn_proj}
  mlp:       {img_mlp,txt_mlp,mlp}.{w1,w2,w3}
  txt_embed: txt_embedder, pooled_embedder

Save format: sd-scripts-style `lora_unet_<flat>.lora_down/up.weight` / `alpha`
(flat uses the "."<->"__" reversible encoding from minit2i_lora). Full-parameter
saves a MiniT2I single-file (transformer + variant metadata) via vendor.save_single_file.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .base_adapter import BaseLoRAAdapter, BaseFullParameterAdapter
from .sd15_adapter import LoRALinearLayer

from core.models.minit2i.minit2i_lora import (
    iter_minit2i_lora_targets, DEFAULT_SCOPE, flatten_to_key,
)


class MiniT2ILoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for the MiniT2I MM-JiT transformer."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_SCOPE) if scope is None else dict(scope)

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        transformer = self.trainer.transformer
        if transformer is None:
            print("[MiniT2ILoRAAdapter] WARNING: trainer.transformer is None — skipping")
            return 0
        print(f"[MiniT2ILoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")
        count = 0
        for module_path, parent, attr, current in iter_minit2i_lora_targets(transformer, self.scope):
            if isinstance(current, LoRALinearLayer):
                continue
            lora_name = flatten_to_key(module_path)  # "lora_unet_<flat>"
            lora_layer = LoRALinearLayer(current, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype)
            if isinstance(attr, int):
                parent[attr] = lora_layer
            else:
                setattr(parent, attr, lora_layer)
            lora_layers[lora_name] = lora_layer
            count += 1
        print(f"[MiniT2ILoRAAdapter] Injected {count} LoRA layer(s)")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        print("[MiniT2ILoRAAdapter] FLAN-T5 text encoder is frozen — no LoRA on TE")
        return 0

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        params: List[nn.Parameter] = []
        for lora_layer in lora_layers.values():
            params.extend(lora_layer.lora_down.parameters())
            params.extend(lora_layer.lora_up.parameters())
        if not params:
            return []
        base_lr = getattr(self.trainer, "unet_lr", None) or 1e-4
        lr_factor = float(self.trainer.config.get("minit2i_lr_factor", 1.0))
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
            "model_type": "minit2i",
            "modelspec.architecture": "minit2i",
            "variant": str(getattr(self.trainer, "minit2i_variant", "") or ""),
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_targets": active_scopes,
            "step": str(step),
            "epoch": str(epoch),
            "format": "pt",
        }
        save_file(state_dict, str(output_path), metadata=metadata)
        print(f"[MiniT2ILoRAAdapter] Saved LoRA checkpoint ({len(lora_layers)} layers) -> {output_path}")


class MiniT2IFullParameterAdapter(BaseFullParameterAdapter):
    """Full-parameter adapter for MiniT2I (transformer; FLAN-T5 training is Phase C).

    MiniT2I is small (B/16 ~0.3B, L/16 ~1.8B), so full fine-tuning is practical.
    The frozen FLAN-T5 is left frozen here; `train_text_encoder` support is Phase C.
    """

    def prepare_models_for_training(self):
        trainer = self.trainer
        if getattr(trainer, "train_unet", True) and trainer.transformer is not None:
            trainer.transformer.requires_grad_(True)
            trainer.transformer.train()
            print("[MiniT2IFullParameterAdapter] MM-JiT transformer set to train mode")
        if trainer.text_encoder is not None:
            trainer.text_encoder.requires_grad_(False)
            trainer.text_encoder.eval()
            print("[MiniT2IFullParameterAdapter] FLAN-T5 text encoder is frozen (TE training: Phase C)")

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        trainer = self.trainer
        if trainer.transformer is None:
            return []
        params = [p for p in trainer.transformer.parameters() if p.requires_grad]
        if not params:
            return []
        base_lr = getattr(trainer, "unet_lr", None) or getattr(trainer, "learning_rate", 1e-5)
        total = sum(p.numel() for p in params)
        print(f"[MiniT2IFullParameterAdapter] {total:,} trainable params (transformer)")
        return [{"params": params, "lr": base_lr}]

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        from core.models.minit2i.vendor.single_file import save_single_file
        trainer = self.trainer
        if trainer.transformer is None:
            print("[MiniT2IFullParameterAdapter] WARNING: no transformer to save")
            return
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"minit2i_step_{step}.safetensors"
        elif not str(output_path).endswith(".safetensors"):
            output_path = Path(str(output_path) + ".safetensors")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        variant = getattr(trainer, "minit2i_variant", None) or "b16"
        save_single_file(str(output_path), trainer.transformer, variant=variant,
                         extra_metadata={"step": str(step), "epoch": str(epoch)})
        print(f"[MiniT2IFullParameterAdapter] Saved single-file -> {output_path}")
