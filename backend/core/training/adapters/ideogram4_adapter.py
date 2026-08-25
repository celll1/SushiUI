"""Ideogram 4 (single-stream flow-matching DiT) training adapters.

Model characteristics:
  - Single-stream DiT (34 blocks, RMSNorm, AdaLN, SwiGLU), `transformer.layers`
  - Qwen3-VL text encoder (frozen), 13-layer features -> 53248-dim conditioning
  - AutoencoderKLFlux2 VAE (32ch, 128ch packed flat-sequence latents) — frozen
  - Flow Matching (FlowMatchEulerDiscreteScheduler)
  - Asymmetric CFG with a separate `unconditional_transformer`

LoRA target scope (controlled by the `scope` dict):
  - attn: layers.{N}.attention.{to_q, to_k, to_v, to_out.0}
  - mlp:  layers.{N}.feed_forward.{w1, w2, w3}  (SwiGLU)
  - mod:  layers.{N}.adaln_modulation            (default OFF)

The base transformer is loaded weight-only-FP8 (Fp8Linear); LoRA wraps those
frozen Fp8Linear modules (LoRALinearLayer calls original_module(x) which
dequantizes). Optionally also trains the unconditional transformer when
`trainer.ideogram4_train_uncond` is set (LoRA keys use a distinct `lora_uncond_`
prefix in the same checkpoint).

Save format: sd-scripts native — `lora_unet_<flat>.lora_down/up.weight` / `alpha`
(cond) and `lora_uncond_<flat>.*` (uncond). The inference loader (ideogram4_lora.py)
reads both branches.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .base_adapter import (
    BaseLoRAAdapter, BaseFullParameterAdapter, resolve_component_lr, LORA_COMPONENT_UNET
)
from .sd15_adapter import LoRALinearLayer

from core.models.ideogram4.ideogram4_lora import (
    iter_ideogram4_lora_targets, DEFAULT_SCOPE, _flatten_to_sdscripts,
)


class Ideogram4LoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for the Ideogram 4 DiT (conditional + optional unconditional)."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_SCOPE) if scope is None else dict(scope)

    def _wrap_transformer(self, transformer, lora_layers: Dict[str, nn.Module],
                          key_prefix: str) -> int:
        count = 0
        for module_path, parent, attr, current in iter_ideogram4_lora_targets(transformer, self.scope):
            if isinstance(current, LoRALinearLayer):
                continue
            lora_name = f"{key_prefix}{_flatten_to_sdscripts(module_path)}"
            lora_layer = LoRALinearLayer(
                current, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype,
            )
            if isinstance(attr, int):
                parent[attr] = lora_layer
            else:
                setattr(parent, attr, lora_layer)
            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
            count += 1
        return count

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """Wrap target Linear/Fp8Linear modules of the conditional (and optionally
        unconditional) Ideogram 4 transformer with LoRALinearLayer."""
        transformer = self.trainer.transformer
        if transformer is None:
            print("[Ideogram4LoRAAdapter] WARNING: trainer.transformer is None - skipping")
            return 0

        print(f"[Ideogram4LoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")
        count = self._wrap_transformer(transformer, lora_layers, "lora_unet_")

        uncond = getattr(self.trainer, "transformer_uncond", None)
        if getattr(self.trainer, "ideogram4_train_uncond", False) and uncond is not None:
            n_uncond = self._wrap_transformer(uncond, lora_layers, "lora_uncond_")
            print(f"[Ideogram4LoRAAdapter] Injected {n_uncond} LoRA layer(s) into uncond transformer")
            count += n_uncond

        print(f"[Ideogram4LoRAAdapter] Injected {count} LoRA layer(s) total")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """Qwen3-VL text encoder is frozen — no LoRA applied."""
        print("[Ideogram4LoRAAdapter] Qwen3-VL text encoder is frozen - no LoRA on TE")
        return 0

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]
                                   ) -> List[Dict[str, Any]]:
        params: List[nn.Parameter] = []
        for lora_layer in lora_layers.values():
            params.extend(lora_layer.lora_down.parameters())
            params.extend(lora_layer.lora_up.parameters())
        if not params:
            return []
        base_lr = resolve_component_lr(self.trainer, "unet_lr", label="Ideogram 4 LoRA")
        lr_factor = float(self.trainer.config.get("ideogram4_lr_factor", 1.0))
        return [{"params": params, "lr": base_lr * lr_factor}]

    def save_checkpoint(self, lora_layers: Dict[str, nn.Module],
                        step: int, epoch: int, output_path: Path):
        state_dict: Dict[str, torch.Tensor] = {}
        alpha_value = float(self.lora_alpha)
        for lora_name, lora_layer in lora_layers.items():
            state_dict[f"{lora_name}.lora_down.weight"] = lora_layer.lora_down.weight.detach().cpu()
            state_dict[f"{lora_name}.lora_up.weight"] = lora_layer.lora_up.weight.detach().cpu()
            state_dict[f"{lora_name}.alpha"] = torch.tensor(alpha_value, dtype=torch.float32)

        active_scopes = ",".join(k for k, v in self.scope.items() if v)
        metadata = {
            "model_type":             "ideogram4",
            "modelspec.architecture": "ideogram4",
            "lora_rank":              str(self.lora_rank),
            "lora_alpha":             str(self.lora_alpha),
            "lora_targets":           active_scopes,
            "train_uncond":           str(bool(getattr(self.trainer, "ideogram4_train_uncond", False))),
            "step":                   str(step),
            "epoch":                  str(epoch),
            "format":                 "pt",
        }
        save_file(state_dict, str(output_path), metadata=metadata)
        print(f"[Ideogram4LoRAAdapter] Saved LoRA checkpoint ({len(lora_layers)} layers) -> {output_path}")


class Ideogram4FullParameterAdapter(BaseFullParameterAdapter):
    """Full-parameter adapter for Ideogram 4 (Phase 2b — requires a bf16 base).

    The shipped fp8 checkpoint stores Linear weights as non-trainable buffers
    (Fp8Linear), so full fine-tuning needs an unquantized (bf16) transformer.
    Left as an explicit guard until the bf16 load path is added.
    """

    def prepare_models_for_training(self):
        raise NotImplementedError(
            "Ideogram 4 full fine-tuning requires a bf16 base transformer (the fp8 "
            "checkpoint stores weights as buffers and cannot be trained directly). "
            "Use LoRA, or provide/dequantize a bf16 base (Phase 2b)."
        )

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("Ideogram 4 full fine-tuning is not yet supported (Phase 2b).")

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        raise NotImplementedError("Ideogram 4 full fine-tuning is not yet supported (Phase 2b).")
