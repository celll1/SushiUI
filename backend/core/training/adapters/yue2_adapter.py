"""YuE2 LoRA and dense ABC-planner full-parameter adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn

from core.adapters import is_adapter_covered
from core.models.yue2.yue2_lora import iter_yue2_lora_targets, normalize_yue2_stages
from .base_adapter import (BaseFullParameterAdapter, BaseLoRAAdapter,
                           LORA_COMPONENT_UNET, reject_quantized_base,
                           resolve_component_lr)


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


class YuE2FullParameterAdapter(BaseFullParameterAdapter):
    """Train exactly the dense AR planner; the acoustic NAR half stays frozen."""

    def prepare_models_for_training(self):
        trainer = self.trainer
        reject_quantized_base(trainer.transformer, model_label="YuE2")
        if not getattr(trainer, "gradient_checkpointing", False):
            raise ValueError("YuE2 full_finetune requires gradient_checkpointing=true")
        if not getattr(trainer, "optimizer_stochastic_rounding", False):
            raise ValueError(
                "YuE2 full_finetune requires optimizer_stochastic_rounding=true"
            )
        trainer.transformer.requires_grad_(False)
        from core.models.yue2.pipeline import ar_modules

        for module in ar_modules(trainer.transformer):
            module.requires_grad_(True)
        trainer.transformer.train()

    def arch_param_groups(self) -> List[Dict[str, Any]]:
        trainer = self.trainer
        reject_quantized_base(trainer.transformer, model_label="YuE2")
        from core.models.yue2.pipeline import ar_modules

        parameters, seen = [], set()
        for module in ar_modules(trainer.transformer):
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        if not parameters:
            return []
        count = sum(parameter.numel() for parameter in parameters)
        print(f"[YuE2FullParameterAdapter] {count:,} trainable AR planner parameters")
        return [{
            "params": parameters,
            "lr": resolve_component_lr(trainer, "unet_lr", label="YuE2 AR planner"),
            "name": "unet",
            "component": "unet",
        }]

    def write_checkpoint(self, step: int, epoch: int, output_path: Path):
        from core.models.yue2.single_file import save_yue2_single_file
        from core.training.ops.yue2_ops import YUE2_TRAINING_PROTOCOL_VERSION

        trainer = self.trainer
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"yue2_step_{step}.safetensors"
        elif output_path.suffix.lower() != ".safetensors":
            output_path = Path(str(output_path) + ".safetensors")
        vae = getattr(trainer, "yue2_frozen_vae", None)
        if vae is None:
            raise ValueError("YuE2 full checkpoint save requires its frozen bundled VAE")
        identity = getattr(trainer, "yue2_model_identity", {}) or {}
        save_yue2_single_file(
            output_path, trainer.transformer, vae, trainer.tokenizer,
            extra_metadata={
                "modelspec.architecture": "yue2",
                "modelspec.license": "CC-BY-NC-4.0",
                "modelspec.source": "https://huggingface.co/m-a-p/YuE2-3B",
                "yue2_training_scope": "abc_ar_full",
                "yue2_training_protocol": YUE2_TRAINING_PROTOCOL_VERSION,
                "yue2_base_checkpoint": identity.get("checkpoint", "unknown"),
                "step": step,
                "epoch": epoch,
            },
        )
        print(f"[YuE2FullParameterAdapter] Saved complete dense checkpoint -> {output_path}")
        return output_path
