"""LoRA and full-parameter adapters for Qwen-Image 2.1."""

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from core.adapters import is_adapter_covered
from core.models.common.convrot_int8_linear import ConvRotInt8Linear, materialize_convrot_linears
from core.models.common.quantized_export import DEFAULT_EXPORT_SHARD_BYTES, ShardWriter
from core.models.qwen_image_21.artifact import artifact_metadata
from core.models.qwen_image_21.residual_unet import model_topology_metadata
from core.models.qwen_image_21.partition_global import (
    global_adapter_metadata, prepare_training_global_adapter,
)

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
        global_adapter = prepare_training_global_adapter(self.trainer, dtype=self.lora_dtype)
        if global_adapter is not None:
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
        metadata.update(global_adapter_metadata(global_adapter, getattr(self.trainer, "config", {})))
        metadata.update(model_topology_metadata(self.trainer.transformer))
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
        prepare_training_global_adapter(
            self.trainer, dtype=next(self.trainer.transformer.parameters()).dtype
        )

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
        aligned = getattr(self.trainer.transformer, "vae_latent_bridge", None)
        bundle = train_te or aligned is not None
        bundle_target_vae = False
        if aligned is not None and not aligned.ready:
            raise ValueError("Cannot save an uncalibrated Qwen VAE swap")
        metadata = artifact_metadata(
            "training_bundle" if bundle else "transformer", config
        )
        if aligned is not None:
            from api.param_defaults import resolve_bundle_vae
            from core.models.common.single_file_format import build_component_metadata
            inner = self.trainer.vae.inner
            d = aligned.declaration
            identity = getattr(self.trainer, "vae_identity", None)
            bundle_target_vae = resolve_bundle_vae(getattr(self.trainer,"bundle_vae",None),"qwen_image_21",vae_swapped=True)
            locator = getattr(identity,"locator",None)
            if not bundle_target_vae and not locator:
                raise ValueError("Qwen target VAE has no portable locator; bundle_vae must remain enabled")
            metadata.update(build_component_metadata(vae_type="custom", vae_channels=d["new_channels"],
                vae_embedded=bundle_target_vae, vae_prefix="vae." if bundle_target_vae else None,
                vae_locator=None if bundle_target_vae else locator,
                vae_class=type(inner).__name__, vae_config=dict(inner.config),
                vae_scale_factor=d["new_scale"], vae_norm=d["norm"], vae_norm_pack=d["norm_pack"],
                vae_struct_native=False, vae_identity_native=False,
                vae_hash=getattr(identity,"content_hash",None) or d["target_identity"]))
            metadata["train_text_encoder"] = "1" if train_te else "0"
        metadata.update(model_topology_metadata(self.trainer.transformer))
        companion_path = str(getattr(
            self.trainer, "qwen_image_21_companion_path", self.trainer.model_path
        ))
        metadata.update(step=str(step), epoch=str(epoch), companion_path=companion_path)
        metadata.update(global_adapter_metadata(getattr(
            self.trainer.transformer, "qwen_partition_global_adapter", None
        ), getattr(self.trainer, "config", {})))
        if bool(getattr(self.trainer, "qwen_full_dequantized_from_int8", False)):
            metadata["full_finetune_initialization"] = "dequantized_int8_convrot"
        writer = ShardWriter(str(output_path), metadata, DEFAULT_EXPORT_SHARD_BYTES)
        try:
            for key, tensor in self.trainer.transformer.state_dict().items():
                writer.add(
                    f"transformer.{key}" if bundle else key,
                    tensor.detach().cpu().contiguous(),
                )
            if train_te:
                for key, tensor in self.trainer.text_encoder.state_dict().items():
                    writer.add(
                        f"text_encoder.{key}",
                        tensor.detach().to("cpu", copy=True).contiguous(),
                    )
            if bundle_target_vae:
                for key, tensor in self.trainer.vae.inner.state_dict().items():
                    writer.add(f"vae.{key}", tensor.detach().to("cpu", copy=True).contiguous())
            result = writer.close()
        except BaseException:
            writer.abort()
            raise
        return Path(result)
