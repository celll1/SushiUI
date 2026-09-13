"""Ordinary LoRA training adapter for YuE2's stage-scoped MoT halves."""
from __future__ import annotations

from typing import Dict, List, Any

import torch
import torch.nn as nn

from core.adapters import is_adapter_covered
from core.models.yue2.yue2_lora import iter_yue2_lora_targets, normalize_yue2_stages
from .base_adapter import BaseLoRAAdapter, LORA_COMPONENT_UNET, resolve_component_lr


class YuE2LoRAAdapter(BaseLoRAAdapter):
    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32, *,
                 objective: str = "abc_ar", scope: Dict[str, bool] | None = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        if objective not in {"abc_ar", "semantic_ar", "acoustic_nar"}:
            raise ValueError(f"Unsupported YuE2 training objective: {objective}")
        self.objective = objective
        self.half = "nar" if objective == "acoustic_nar" else "ar"
        self.apply_stages = normalize_yue2_stages(
            "nar" if self.half == "nar" else ("abc" if objective == "abc_ar" else "semantic")
        )
        self.scope = {"attention": True, "mlp": False, **(scope or {})}

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        transformer = self.trainer.transformer
        if transformer is None:
            return 0
        count = 0
        for target in iter_yue2_lora_targets(transformer, half=self.half, scope=self.scope):
            if is_adapter_covered(target.module):
                continue
            name = "lora_unet_" + target.path.replace(".", "_")
            branch = self.build_branch(target.module, name)
            setattr(target.parent, target.attr, branch)
            self.register_lora_layer(lora_layers, name, branch, LORA_COMPONENT_UNET)
            count += 1
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        return 0

    def arch_param_groups(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        return self.component_param_groups(lora_layers, {
            LORA_COMPONENT_UNET: lambda: resolve_component_lr(
                self.trainer, "unet_lr", label="YuE2 LoRA"
            ),
        })

    def checkpoint_metadata(self, lora_layers, step: int, epoch: int) -> Dict[str, str]:
        selected = ",".join(key for key, enabled in self.scope.items() if enabled)
        identity = getattr(self.trainer, "yue2_model_identity", {}) or {}
        from core.training.ops.yue2_ops import YUE2_TRAINING_PROTOCOL_VERSION
        return {
            "model_type": "yue2",
            "modelspec.architecture": "yue2",
            "modelspec.license": "CC-BY-NC-4.0",
            "modelspec.license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "modelspec.source": "https://huggingface.co/m-a-p/YuE2-3B",
            "yue2_objective": self.objective,
            "yue2_apply_stages": ",".join(self.apply_stages),
            "yue2_lora_half": self.half,
            "yue2_training_protocol": YUE2_TRAINING_PROTOCOL_VERSION,
            "yue2_base_checkpoint": str(identity.get("checkpoint", "unknown")),
            "yue2_upstream_revision": str(identity.get("upstream_revision", "unknown")),
            "lora_targets": selected,
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "step": str(step),
            "epoch": str(epoch),
            "format": "pt",
        }
