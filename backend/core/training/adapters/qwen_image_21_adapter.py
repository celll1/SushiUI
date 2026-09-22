"""LoRA and full-parameter adapters for Qwen-Image 2.1."""

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from core.adapters import is_adapter_covered
from core.models.common.convrot_int8_linear import ConvRotInt8Linear
from core.models.common.quantized_export import DEFAULT_EXPORT_SHARD_BYTES, ShardWriter
from core.models.qwen_image_21.artifact import artifact_metadata

from .base_adapter import (
    BaseFullParameterAdapter,
    BaseLoRAAdapter,
    LORA_COMPONENT_UNET,
    reject_quantized_base,
    resolve_component_lr,
)


_DEFAULT_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def _targets(transformer):
    for path, module in transformer.named_modules():
        if not isinstance(module, (nn.Linear, ConvRotInt8Linear)):
            continue
        if not any(path.endswith(suffix) for suffix in _DEFAULT_TARGETS):
            continue
        parent_path, attr = path.rsplit(".", 1)
        parent = transformer.get_submodule(parent_path)
        if attr.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            yield path, parent, int(attr), module
        else:
            yield path, parent, attr, module


class QwenImage21LoRAAdapter(BaseLoRAAdapter):
    def __init__(self, trainer, lora_rank, lora_alpha, lora_dtype=torch.float32):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        from core.models.qwen_image_21.branch_lora import (
            BRANCH_MODE_COND_BASE, BRANCH_MODE_SHARED, QwenCondLoRALinearLayer,
        )

        self.branch_mode = str(getattr(trainer, "config", {}).get(
            "qwen_lora_branch_mode", BRANCH_MODE_SHARED))
        if self.branch_mode not in {BRANCH_MODE_SHARED, BRANCH_MODE_COND_BASE}:
            raise ValueError(f"Unknown Qwen LoRA branch mode {self.branch_mode!r}")
        if self.branch_mode == BRANCH_MODE_COND_BASE:
            if not self.adapter_spec.is_ordinary_lora:
                raise ValueError("Qwen cond/base branch separation requires ordinary LoRA")
            if float(getattr(trainer, "config", {}).get("cfg_uncond_drop_rate") or 0):
                raise ValueError("Qwen cond/base branch separation requires cfg_uncond_drop_rate=0")
            self.LORA_LAYER_CLS = QwenCondLoRALinearLayer

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        count = 0
        for path, parent, attr, current in _targets(self.trainer.transformer):
            if is_adapter_covered(current):
                continue
            from core.models.qwen_image_21.lora import flatten_to_key
            name = flatten_to_key(path)
            if self.branch_mode == "cond_base_v1":
                name = name.replace("lora_unet_", "lora_cond_unet_", 1)
            layer = self.build_branch(current, name)
            if isinstance(attr, int):
                parent[attr] = layer
            else:
                setattr(parent, attr, layer)
            self.register_lora_layer(lora_layers, name, layer, LORA_COMPONENT_UNET)
            count += 1
        config = getattr(self.trainer, "config", {})
        global_enabled = bool(config.get(
            "dit_partition_global_adapter_enabled",
            config.get("qwen_partition_global_adapter_enabled", False),
        ))
        partition_enabled = bool(config.get(
            "dit_partition_training_enabled",
            config.get("qwen_partition_training_enabled", False),
        ))
        if global_enabled:
            if not partition_enabled:
                raise ValueError(
                    "Qwen partition global adapter requires dit_partition_training_enabled"
                )
            from core.training.qwen_partition import QwenPartitionGlobalAdapter

            global_adapter = QwenPartitionGlobalAdapter(
                int(self.trainer.transformer.config.in_channels),
                int(self.trainer.transformer.inner_dim),
                rank=int(config.get(
                    "dit_partition_global_rank",
                    config.get("qwen_partition_global_rank", 64),
                )),
                summary_tokens=int(
                    config.get(
                        "dit_partition_global_tokens",
                        config.get("qwen_partition_global_tokens", 16),
                    )
                ),
                dtype=self.lora_dtype,
            ).to(next(self.trainer.transformer.parameters()).device)
            self.trainer.transformer.qwen_partition_global_adapter = global_adapter
            self.register_lora_layer(
                lora_layers,
                "qwen_partition_global_adapter",
                global_adapter,
                LORA_COMPONENT_UNET,
            )
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        return 0

    def arch_param_groups(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        return self.component_param_groups(lora_layers, {
            LORA_COMPONENT_UNET: lambda: resolve_component_lr(
                self.trainer, "unet_lr", label="Qwen-Image 2.1 LoRA"
            )
        })

    def checkpoint_metadata(self, lora_layers, step, epoch):
        metadata = {
            "model_type": "qwen_image_21",
            "modelspec.architecture": "qwen_image_21",
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "step": str(step),
            "epoch": str(epoch),
            "format": "pt",
        }
        if self.branch_mode == "cond_base_v1":
            metadata.update(
                qwen_lora_branch_mode="cond_base_v1",
                qwen_lora_uncond="base",
            )
        forward = getattr(self.trainer, "qwen_convrot_training_forward", "dequant")
        if forward in {"transient_bf16", "cached_bf16", "prefetch_bf16"}:
            metadata.update(
                qwen_base_variant="int8_convrot",
                qwen_base_forward="convrot_int8_bf16_backward_v1",
            )
        global_adapter = getattr(
            getattr(self.trainer, "transformer", None),
            "qwen_partition_global_adapter",
            None,
        )
        if global_adapter is not None:
            metadata.update(
                qwen_partition_global_adapter="latent_summary_v1",
                qwen_partition_global_rank=str(global_adapter.rank),
                qwen_partition_global_tokens=str(global_adapter.summary_tokens),
            )
        return metadata


class QwenImage21FullParameterAdapter(BaseFullParameterAdapter):
    def prepare_models_for_training(self):
        reject_quantized_base(self.trainer.transformer, model_label="Qwen-Image 2.1")
        if bool(getattr(self.trainer, "train_text_encoder", False)):
            raise ValueError("Qwen-Image 2.1 text-encoder training is not supported")
        self.trainer.transformer.requires_grad_(True).train()
        self.trainer.text_encoder.requires_grad_(False).eval()
        self.trainer.vae.requires_grad_(False).eval()

    def arch_param_groups(self):
        reject_quantized_base(self.trainer.transformer, model_label="Qwen-Image 2.1")
        params = [p for p in self.trainer.transformer.parameters() if p.requires_grad]
        return [{
            "params": params,
            "lr": resolve_component_lr(self.trainer, "unet_lr", label="Qwen-Image 2.1 transformer"),
            "name": "unet",
            "component": "unet",
        }]

    def write_checkpoint(self, step: int, epoch: int, output_path: Path):
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"qwen_image_21_step_{step}.safetensors"
        elif output_path.suffix != ".safetensors":
            output_path = Path(str(output_path) + ".safetensors")
        config = dict(self.trainer.transformer.config)
        metadata = artifact_metadata("transformer", config)
        companion_path = str(getattr(
            self.trainer, "qwen_image_21_companion_path", self.trainer.model_path
        ))
        metadata.update(step=str(step), epoch=str(epoch), companion_path=companion_path)
        writer = ShardWriter(str(output_path), metadata, DEFAULT_EXPORT_SHARD_BYTES)
        try:
            for key, tensor in self.trainer.transformer.state_dict().items():
                writer.add(key, tensor.detach().cpu().contiguous())
            result = writer.close()
        except BaseException:
            writer.abort()
            raise
        return Path(result)
