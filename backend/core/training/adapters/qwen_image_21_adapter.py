"""LoRA and full-parameter adapters for Qwen-Image 2.1."""

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from core.adapters import is_adapter_covered
from core.models.common.convrot_int8_linear import ConvRotInt8Linear, materialize_convrot_linears
from core.models.common.quantized_export import DEFAULT_EXPORT_SHARD_BYTES, ShardWriter
from core.models.qwen_image_21.artifact import artifact_metadata

from .base_adapter import (
    BaseFullParameterAdapter,
    BaseLoRAAdapter,
    LORA_COMPONENT_UNET, LORA_COMPONENT_ADAPTER,
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
    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        from core.models.qwen_image_21.lora import flatten_to_key

        count = 0
        for path, parent, attr, current in (
            _targets(self.trainer.transformer)
            if getattr(self.trainer, "train_unet", True) else ()
        ):
            if is_adapter_covered(current):
                continue
            name = flatten_to_key(path)
            layer = self.build_branch(current, name)
            if isinstance(attr, int):
                parent[attr] = layer
            else:
                setattr(parent, attr, layer)
            self.register_lora_layer(lora_layers, name, layer, LORA_COMPONENT_UNET)
            count += 1
        config = getattr(self.trainer, "config", {})
        if config.get("train_adapter") is True:
            projection = self.trainer.transformer.txt_in
            for attr in ("in_layer", "out_layer"):
                current = getattr(projection, attr)
                if is_adapter_covered(current):
                    continue
                name = flatten_to_key(f"txt_in.{attr}")
                layer = self.build_branch(current, name)
                setattr(projection, attr, layer)
                self.register_lora_layer(
                    lora_layers, name, layer, LORA_COMPONENT_ADAPTER
                )
                count += 1
        global_enabled = bool(config.get(
            "dit_partition_global_adapter_enabled",
            config.get("qwen_partition_global_adapter_enabled", False),
        ))
        partition_enabled = bool(config.get(
            "dit_partition_training_enabled",
            config.get("qwen_partition_training_enabled", False),
        ))
        if global_enabled:
            if not getattr(self.trainer, "train_unet", True):
                raise ValueError("Qwen partition global adapter requires train_unet=true")
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
            ),
            LORA_COMPONENT_ADAPTER: lambda: (
                float(self.trainer.config["adapter_lr"])
                if self.trainer.config.get("adapter_lr") is not None
                else resolve_component_lr(
                    self.trainer, "unet_lr", label="Qwen-Image 2.1 txt_in LoRA"
                )
            ),
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
        train_te = bool(getattr(self.trainer, "train_text_encoder", False))
        if bool(self.trainer.config.get("full_finetune_dequantize_int8_base", False)):
            dtype = self.trainer.weight_dtype
            dit_count = materialize_convrot_linears(self.trainer.transformer, dtype)
            te_count = (
                materialize_convrot_linears(self.trainer.text_encoder, dtype)
                if train_te else 0
            )
            self.trainer.qwen_full_dequantized_from_int8 = bool(dit_count or te_count)
            if dit_count:
                self.trainer.qwen_image_21_transformer_variant = "bf16"
            if te_count:
                self.trainer.qwen_image_21_text_encoder_variant = "bf16"
            self.trainer.transformer.to(self.trainer.device)
        reject_quantized_base(self.trainer.transformer, model_label="Qwen-Image 2.1")
        if train_te:
            reject_quantized_base(
                self.trainer.text_encoder, model_label="Qwen-Image 2.1 text encoder"
            )
        train_dit = bool(getattr(self.trainer, "train_unet", True))
        adapter_choice = self.trainer.config.get("train_adapter")
        self.trainer.transformer.requires_grad_(train_dit).train()
        self.trainer.transformer.txt_in.requires_grad_(
            train_dit if adapter_choice is None else bool(adapter_choice)
        )
        self.trainer.text_encoder.requires_grad_(train_te)
        self.trainer.text_encoder.train(train_te)
        if train_te and bool(getattr(self.trainer, "gradient_checkpointing", False)):
            self.trainer.text_encoder.gradient_checkpointing_enable()
        self.trainer.vae.requires_grad_(False).eval()

    def arch_param_groups(self):
        reject_quantized_base(self.trainer.transformer, model_label="Qwen-Image 2.1")
        adapter_lr = self.trainer.config.get("adapter_lr")
        if adapter_lr is None:
            params = [p for p in self.trainer.transformer.parameters() if p.requires_grad]
            groups = [{
                "params": params,
                "lr": resolve_component_lr(
                    self.trainer, "unet_lr", label="Qwen-Image 2.1 transformer"
                ),
                "name": "unet", "component": "unet",
            }] if params else []
            return self._with_text_encoder_group(groups)
        projection = self.trainer.transformer.txt_in
        adapter_ids = {id(p) for p in projection.parameters() if p.requires_grad}
        base_params = [p for p in self.trainer.transformer.parameters()
                       if p.requires_grad and id(p) not in adapter_ids]
        groups = []
        if base_params:
            groups.append({
                "params": base_params,
                "lr": resolve_component_lr(self.trainer, "unet_lr", label="Qwen-Image 2.1 transformer"),
                "name": "unet", "component": "unet",
            })
        if adapter_ids:
            groups.append({
                "params": [p for p in projection.parameters() if p.requires_grad],
                "lr": float(adapter_lr),
                "name": "adapter", "component": "adapter",
            })
        return self._with_text_encoder_group(groups)

    def _with_text_encoder_group(self, groups):
        te_params = [p for p in self.trainer.text_encoder.parameters() if p.requires_grad]
        if te_params:
            groups.append({
                "params": te_params,
                "lr": resolve_component_lr(
                    self.trainer, "text_encoder_lr", label="Qwen-Image 2.1 text encoder"
                ),
                "name": "text_encoder_1", "component": "text_encoder_1",
            })
        return groups

    def write_checkpoint(self, step: int, epoch: int, output_path: Path):
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"qwen_image_21_step_{step}.safetensors"
        elif output_path.suffix != ".safetensors":
            output_path = Path(str(output_path) + ".safetensors")
        config = dict(self.trainer.transformer.config)
        train_te = bool(getattr(self.trainer, "train_text_encoder", False))
        metadata = artifact_metadata(
            "training_bundle" if train_te else "transformer", config
        )
        companion_path = str(getattr(
            self.trainer, "qwen_image_21_companion_path", self.trainer.model_path
        ))
        metadata.update(step=str(step), epoch=str(epoch), companion_path=companion_path)
        if bool(getattr(self.trainer, "qwen_full_dequantized_from_int8", False)):
            metadata["full_finetune_initialization"] = "dequantized_int8_convrot"
        writer = ShardWriter(str(output_path), metadata, DEFAULT_EXPORT_SHARD_BYTES)
        try:
            for key, tensor in self.trainer.transformer.state_dict().items():
                writer.add(
                    f"transformer.{key}" if train_te else key,
                    tensor.detach().cpu().contiguous(),
                )
            if train_te:
                for key, tensor in self.trainer.text_encoder.state_dict().items():
                    writer.add(
                        f"text_encoder.{key}",
                        tensor.detach().to("cpu", copy=True).contiguous(),
                    )
            result = writer.close()
        except BaseException:
            writer.abort()
            raise
        return Path(result)
