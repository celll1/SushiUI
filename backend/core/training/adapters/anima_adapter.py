"""Anima (Cosmos-Predict2 DiT) training adapters.

Model characteristics:
  - Cosmos-Predict2 DiT (28 blocks, AdaLN-LoRA modulation, 3D RoPE)
  - Qwen3-0.6B text encoder + 6-layer LLM Adapter
  - Qwen-Image VAE (Wan VAE 2.1 latent space, 16ch)
  - Rectified Flow / Flow Matching (predicts velocity v = noise - x_0)

LoRA targets (controlled by the `scope` dict):
  - attention:   blocks.<N>.{self_attn,cross_attn}.{q,k,v,output}_proj
  - mlp:         blocks.<N>.mlp.{layer1, layer2}
  - mod:         blocks.<N>.adaln_modulation_*.{1, 2}      (default OFF)
  - llm_adapter: llm_adapter.blocks.<N>.{self_attn,cross_attn}.{q,k,v,o}_proj
                 + llm_adapter.blocks.<N>.mlp.{0, 2}
                 + llm_adapter.{in_proj, out_proj}

The Qwen3 text encoder body (transformers Qwen3Model) and the VAE are
kept frozen — fine-tuning text-encoder weights for a small DiT model is
brittle and rarely improves quality.

Save format: sd-scripts native — `lora_unet_<flattened>.lora_down.weight` /
`lora_up.weight` / `alpha`. The Phase B.3 inference loader (anima_lora.py)
accepts this format directly and also accepts the interchange format
(diffusion_model.*.lora_A/B.weight) for files produced by other tools.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .base_adapter import BaseLoRAAdapter, BaseFullParameterAdapter
from .sd15_adapter import LoRALinearLayer

# Reuse Phase B.3 iteration + flatten helpers.
from core.models.anima.anima_lora import (
    iter_anima_lora_targets, DEFAULT_TRAINING_SCOPE, _flatten_to_sdscripts,
)


# ----------------------------------------------------------------------
# LoRA adapter
# ----------------------------------------------------------------------

class AnimaLoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for Anima DiT models."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_TRAINING_SCOPE) if scope is None else dict(scope)

    # -- LoRA injection -------------------------------------------------

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """Wrap target Linear modules of the Anima DiT with LoRALinearLayer.

        The 'unet' in the interface name is historical — Anima's DiT
        (`trainer.transformer`) plays that role here. LoRA wrappers replace
        the originals in-place so the parent module's forward()
        automatically dispatches through them.
        """
        transformer = self.trainer.transformer
        if transformer is None:
            print("[AnimaLoRAAdapter] WARNING: trainer.transformer is None — skipping LoRA injection")
            return 0

        print(f"[AnimaLoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")

        count = 0
        for module_path, parent, attr, current in iter_anima_lora_targets(transformer, self.scope):
            # Skip if this slot was already wrapped (idempotent / stacking-safe).
            if isinstance(current, LoRALinearLayer):
                continue

            lora_name = f"lora_unet_{_flatten_to_sdscripts(module_path)}"
            lora_layer = LoRALinearLayer(
                current, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype,
            )

            # parent.attr might be a normal attribute (str) or a Sequential /
            # ModuleList index (int) — handle both.
            if isinstance(attr, int):
                parent[attr] = lora_layer
            else:
                setattr(parent, attr, lora_layer)

            lora_layers[lora_name] = lora_layer
            count += 1

        print(f"[AnimaLoRAAdapter] Injected {count} LoRA layer(s) into Anima DiT")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """Anima keeps the Qwen3 text encoder frozen.

        The LLM Adapter (the 6-layer transformer that re-projects Qwen3
        hidden states into the DiT cross-attention input space) lives inside
        the DiT module, so LoRA on it is applied via apply_lora_to_unet()
        when scope["llm_adapter"] is enabled.
        """
        print("[AnimaLoRAAdapter] Qwen3 text encoder is frozen — no LoRA applied to TE")
        return 0

    # -- Optimizer parameters ------------------------------------------

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]
                                    ) -> List[Dict[str, Any]]:
        """Single optimizer parameter group for the LoRA weights.

        Per-component LR groups (self_attn / cross_attn / mlp / mod /
        llm_adapter) can be added in a follow-up; for the initial release
        we expose a single `unet_lr` knob.
        """
        params: List[nn.Parameter] = []
        for lora_layer in lora_layers.values():
            params.extend(lora_layer.lora_down.parameters())
            params.extend(lora_layer.lora_up.parameters())
        if not params:
            return []
        return [{"params": params, "lr": getattr(self.trainer, "unet_lr", 1e-4)}]

    # -- Checkpoint --------------------------------------------------

    def save_checkpoint(self, lora_layers: Dict[str, nn.Module],
                         step: int, epoch: int, output_path: Path):
        """Save LoRA weights in sd-scripts native format."""
        state_dict: Dict[str, torch.Tensor] = {}
        alpha_value = float(self.lora_alpha)

        for lora_name, lora_layer in lora_layers.items():
            # lora_name already has the lora_unet_ prefix from injection.
            state_dict[f"{lora_name}.lora_down.weight"] = lora_layer.lora_down.weight.detach().cpu()
            state_dict[f"{lora_name}.lora_up.weight"] = lora_layer.lora_up.weight.detach().cpu()
            # Per-layer alpha — sd-scripts convention. We store the same alpha
            # for every layer (matches our rank/alpha config).
            state_dict[f"{lora_name}.alpha"] = torch.tensor(alpha_value, dtype=torch.float32)

        active_scopes = ",".join(k for k, v in self.scope.items() if v)
        metadata = {
            "model_type": "anima",
            "modelspec.architecture": "anima",
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_targets": active_scopes,
            "step": str(step),
            "epoch": str(epoch),
            "format": "pt",
        }

        save_file(state_dict, str(output_path), metadata=metadata)
        print(f"[AnimaLoRAAdapter] Saved LoRA checkpoint ({len(lora_layers)} layers) -> {output_path}")


# ----------------------------------------------------------------------
# Full-parameter adapter (skeleton; full implementation in Phase C.2)
# ----------------------------------------------------------------------

class AnimaFullParameterAdapter(BaseFullParameterAdapter):
    """Full-parameter training adapter for Anima DiT models.

    Phase C.1 focuses on LoRA; this class is a placeholder that fails fast
    so the dispatch path is wired but Full FT is gated until Phase C.2.
    """

    def prepare_models_for_training(self):
        raise NotImplementedError(
            "Anima full-parameter training is not implemented yet (Phase C.2). "
            "Use training_method='lora' for now."
        )

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("Anima full-parameter training is not implemented yet (Phase C.2).")

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        raise NotImplementedError("Anima full-parameter training is not implemented yet (Phase C.2).")
