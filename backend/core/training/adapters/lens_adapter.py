"""Lens (Microsoft/Lens DiT) training adapters.

Model characteristics:
  - MMDiT double-stream architecture (48 blocks, RMSNorm, GateMLP)
  - GPT-OSS MoE text encoder (LensGptOssEncoder, 24-layer, multi-layer features)
  - AutoencoderKLFlux2 VAE (32ch, flat-sequence latent format)
  - Flow Matching (FlowMatchEulerDiscreteScheduler, velocity target v = noise - x0)

LoRA targets (controlled by the `scope` dict):
  - img_attn:  transformer_blocks.{N}.attn.{img_qkv, to_out[0]}
  - txt_attn:  transformer_blocks.{N}.attn.{txt_qkv, to_add_out}
  - img_mlp:   transformer_blocks.{N}.img_mlp.{w1, w2, w3}  (GateMLP)
  - txt_mlp:   transformer_blocks.{N}.txt_mlp.{w1, w2, w3}  (GateMLP)
  - mod:       transformer_blocks.{N}.{img_mod, txt_mod}[1]   (AdaLN, default OFF)

The GPT-OSS text encoder is kept frozen — fine-tuning it is outside scope.
The AutoencoderKLFlux2 VAE is always frozen.

Save format: sd-scripts native — `lora_unet_<flattened>.lora_down/up.weight` /
`alpha`. The Phase B.3 inference loader (lens_lora.py) accepts this format
directly.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .base_adapter import BaseLoRAAdapter, BaseFullParameterAdapter
from .sd15_adapter import LoRALinearLayer

from core.models.lens.lens_lora import (
    iter_lens_lora_targets, DEFAULT_SCOPE, _flatten_to_sdscripts,
)


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class LensLoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for Lens DiT models."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_SCOPE) if scope is None else dict(scope)

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """Wrap target Linear modules of the Lens transformer with LoRALinearLayer."""
        transformer = self.trainer.transformer
        if transformer is None:
            print("[LensLoRAAdapter] WARNING: trainer.transformer is None — skipping LoRA injection")
            return 0

        print(f"[LensLoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")

        count = 0
        for module_path, parent, attr, current in iter_lens_lora_targets(transformer, self.scope):
            if isinstance(current, LoRALinearLayer):
                continue

            lora_name = f"lora_unet_{_flatten_to_sdscripts(module_path)}"
            lora_layer = LoRALinearLayer(
                current, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype,
            )

            if isinstance(attr, int):
                parent[attr] = lora_layer
            else:
                setattr(parent, attr, lora_layer)

            lora_layers[lora_name] = lora_layer
            count += 1

        print(f"[LensLoRAAdapter] Injected {count} LoRA layer(s) into Lens transformer")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """The GPT-OSS text encoder is frozen — no LoRA applied."""
        print("[LensLoRAAdapter] GPT-OSS text encoder is frozen — no LoRA applied to TE")
        return 0

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]
                                   ) -> List[Dict[str, Any]]:
        """Single optimizer parameter group for all LoRA weights."""
        params: List[nn.Parameter] = []
        for lora_layer in lora_layers.values():
            params.extend(lora_layer.lora_down.parameters())
            params.extend(lora_layer.lora_up.parameters())
        if not params:
            return []
        return [{"params": params, "lr": getattr(self.trainer, "unet_lr", 1e-4)}]

    def save_checkpoint(self, lora_layers: Dict[str, nn.Module],
                        step: int, epoch: int, output_path: Path):
        """Save LoRA weights in sd-scripts native format compatible with Phase B.3 loader."""
        state_dict: Dict[str, torch.Tensor] = {}
        alpha_value = float(self.lora_alpha)

        for lora_name, lora_layer in lora_layers.items():
            state_dict[f"{lora_name}.lora_down.weight"] = (
                lora_layer.lora_down.weight.detach().cpu()
            )
            state_dict[f"{lora_name}.lora_up.weight"] = (
                lora_layer.lora_up.weight.detach().cpu()
            )
            state_dict[f"{lora_name}.alpha"] = torch.tensor(alpha_value, dtype=torch.float32)

        active_scopes = ",".join(k for k, v in self.scope.items() if v)
        metadata = {
            "model_type":              "lens",
            "modelspec.architecture":  "lens",
            "lora_rank":               str(self.lora_rank),
            "lora_alpha":              str(self.lora_alpha),
            "lora_targets":            active_scopes,
            "step":                    str(step),
            "epoch":                   str(epoch),
            "format":                  "pt",
        }

        save_file(state_dict, str(output_path), metadata=metadata)
        print(f"[LensLoRAAdapter] Saved LoRA checkpoint ({len(lora_layers)} layers) -> {output_path}")


# ---------------------------------------------------------------------------
# Full-parameter adapter (placeholder — implemented in Phase C.2)
# ---------------------------------------------------------------------------

class LensFullParameterAdapter(BaseFullParameterAdapter):
    """Full-parameter training adapter for Lens DiT models.

    Phase C.2: trainable surface, 3-group LR schedule (img_stream / txt_stream
    / other), safetensors checkpoint.  Not yet implemented.
    """

    def prepare_models_for_training(self):
        raise NotImplementedError(
            "LensFullParameterAdapter.prepare_models_for_training is implemented in Phase C.2"
        )

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "LensFullParameterAdapter.setup_trainable_parameters is implemented in Phase C.2"
        )

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        raise NotImplementedError(
            "LensFullParameterAdapter.save_checkpoint is implemented in Phase C.2"
        )
