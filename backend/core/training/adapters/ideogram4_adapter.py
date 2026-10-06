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

from .base_adapter import (
    BaseLoRAAdapter, BaseFullParameterAdapter, resolve_component_lr, reject_quantized_base, LORA_COMPONENT_UNET
)
from core.adapters import LoRALinearLayer, is_adapter_covered

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
            # A CompositeAdapterLayer is yielded too and exposes
            # in_features/out_features, so wrapping it would NEST, not fail.
            if is_adapter_covered(current):
                continue
            lora_name = f"{key_prefix}{_flatten_to_sdscripts(module_path)}"
            lora_layer = self.build_branch(current, lora_name)
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

    def arch_param_groups(self, lora_layers: Dict[str, nn.Module]
                                   ) -> List[Dict[str, Any]]:
        return self.component_param_groups(lora_layers, {
            LORA_COMPONENT_UNET: lambda: (
                resolve_component_lr(self.trainer, "unet_lr", label="Ideogram 4 LoRA")
                * float(self.trainer.config.get("ideogram4_lr_factor", 1.0))),
        })

    def checkpoint_metadata(self, lora_layers: Dict[str, nn.Module],
                            step: int, epoch: int) -> Dict[str, str]:
        from core.models.ideogram4.residual_unet import pair_topology_metadata
        topology_metadata = pair_topology_metadata(self.trainer.transformer,
            getattr(self.trainer,"transformer_uncond",None))
        return {
            **topology_metadata,
            "model_type":             "ideogram4",
            "modelspec.architecture": "ideogram4",
            "lora_rank":              str(self.lora_rank),
            "lora_alpha":             str(self.lora_alpha),
            "lora_targets":           ",".join(k for k, v in self.scope.items() if v),
            "train_uncond":           str(bool(getattr(self.trainer, "ideogram4_train_uncond", False))),
            "step":                   str(step),
            "epoch":                  str(epoch),
            "format":                 "pt",
        }


class Ideogram4FullParameterAdapter(BaseFullParameterAdapter):
    """Train plain DiT branches; preserve the frozen asymmetric-CFG twin."""

    def _validate(self):
        trainer = self.trainer
        if trainer.transformer is None or getattr(trainer, "transformer_uncond", None) is None:
            raise ValueError("Ideogram 4 full artifacts require both conditional and unconditional transformers")
        if bool(getattr(trainer, "train_text_encoder", False)):
            raise ValueError("Ideogram 4 full fine-tuning keeps its Qwen3-VL encoder frozen")
        if getattr(trainer, "train_unet", True):
            reject_quantized_base(trainer.transformer, model_label="Ideogram 4 conditional DiT")
            if getattr(trainer, "ideogram4_train_uncond", False):
                reject_quantized_base(trainer.transformer_uncond, model_label="Ideogram 4 unconditional DiT")

    def prepare_models_for_training(self):
        self._validate()
        trainer = self.trainer
        enabled = bool(getattr(trainer, "train_unet", True))
        for module, trainable in ((trainer.transformer, enabled),
                (trainer.transformer_uncond, enabled and getattr(trainer, "ideogram4_train_uncond", False))):
            module.requires_grad_(trainable)
            module.train(trainable)
        for module in (trainer.text_encoder, trainer.vae):
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def arch_param_groups(self) -> List[Dict[str, Any]]:
        self._validate()
        trainer = self.trainer
        if not getattr(trainer, "train_unet", True):
            return []
        modules = [trainer.transformer]
        if getattr(trainer, "ideogram4_train_uncond", False):
            modules.append(trainer.transformer_uncond)
        params = list(dict.fromkeys(p for module in modules for p in module.parameters() if p.requires_grad))
        if not params:
            return []
        from api.param_defaults import TRAINING_DEFAULTS
        rate = resolve_component_lr(trainer, "unet_lr", label="Ideogram 4 full DiT")
        rate *= float(trainer.config.get("ideogram4_lr_factor", TRAINING_DEFAULTS["ideogram4_lr_factor"]))
        return [{"params":params, "lr":rate, "name":"unet", "component":"unet"}]

    def write_checkpoint(self, step: int, epoch: int, output_path: Path):
        from core.models.ideogram4.residual_unet import save_checkpoint
        trainer = self.trainer
        self._validate()
        base_dir = getattr(trainer, "ideogram4_base_dir", None)
        if not base_dir:
            raise ValueError("Ideogram 4 full export requires a restorable companion directory")
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"ideogram4_step_{step}.safetensors"
        elif output_path.suffix != ".safetensors":
            output_path = Path(str(output_path) + ".safetensors")
        from core.training.vae_swap import swap_metadata
        swapped,bundled,metadata = swap_metadata(trainer)
        metadata.update({"component.base_dir":str(base_dir), "step":str(step), "epoch":str(epoch),
            "training_method":"full_finetune"})
        return save_checkpoint(trainer.transformer, trainer.transformer_uncond, output_path,
            metadata=metadata,vae=trainer.vae if swapped is not None and bundled else None)
