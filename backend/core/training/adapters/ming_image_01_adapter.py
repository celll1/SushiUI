"""Ming full/LoRA ownership, component rates and native checkpoint exports."""

from pathlib import Path
from contextlib import nullcontext
from core.models.ming_image_01.block_offload import execution_device, move_component

from core.models.ming_image_01.checkpoint import save_bundle
from core.models.ming_image_01.training import optimizer_groups
from .base_adapter import (BaseFullParameterAdapter, BaseLoRAAdapter, LORA_COMPONENT_UNET,
    LORA_COMPONENT_ADAPTER, reject_quantized_base, resolve_component_lr)


class MingImage01FullParameterAdapter(BaseFullParameterAdapter):
    def __init__(self, trainer, *, conditioner_groups, conditioner_lrs):
        super().__init__(trainer)
        self.conditioner_groups = set(conditioner_groups)
        self.conditioner_lrs = dict(conditioner_lrs)

    def prepare_models_for_training(self):
        trainer = self.trainer
        components = trainer.ming_image_01_components
        transformer, conditioner = components.transformer, components.conditioner
        unknown = self.conditioner_groups - conditioner.component_groups().keys()
        if unknown:
            raise ValueError(f"Unknown Ming trainable groups: {sorted(unknown)}")
        from core.models.ming_image_01.quantization import materialize_training_linears
        from api.param_defaults import TRAINING_DEFAULTS
        materialize = trainer.config.get("full_finetune_dequantize_int8_base", TRAINING_DEFAULTS["full_finetune_dequantize_int8_base"])
        if type(materialize) is not bool:
            raise ValueError("Ming full dequantization requires a boolean opt-in")
        if materialize:
            targets = [("ming_full_initialization",transformer)]
            if self.conditioner_groups:
                targets.append(("ming_conditioner_full_initialization",conditioner))
            for attribute,module in targets:
                device = execution_device(module)
                move_component(module,"cpu")
                try:
                    report = materialize_training_linears(module,trainer.weight_dtype)
                    if report is not None:
                        setattr(trainer,attribute,report)
                finally:
                    move_component(module,device)
        reject_quantized_base(transformer, model_label="Ming DiT")
        if self.conditioner_groups:
            from core.models.ming_image_01.quantization import configure_frozen_quantized_training
            trainer.ming_conditioner_quantized_base_training = configure_frozen_quantized_training(conditioner)
            self._check_conditioner_bases(conditioner)
        choice = trainer.config.get("train_adapter")
        caption_trainable = trainer.train_unet if choice is None else choice
        transformer.requires_grad_(trainer.train_unet).train(trainer.train_unet or caption_trainable)
        transformer.cap_embedder.requires_grad_(caption_trainable).train(caption_trainable)
        conditioner.set_trainable(self.conditioner_groups)
        conditioner.eval()
        for name in self.conditioner_groups:
            for module in conditioner.component_groups()[name]:
                module.train()
        components.vae.requires_grad_(False).eval()
        if trainer.gradient_checkpointing:
            if any(parameter.requires_grad for parameter in transformer.parameters()):
                transformer.enable_gradient_checkpointing()
            for module in (conditioner.native.model, conditioner.native.vision, conditioner.native.connector):
                if any(parameter.requires_grad for parameter in module.parameters()):
                    module.gradient_checkpointing_enable()
        # Validate optimizer ownership before any training batch is assembled.
        self.arch_param_groups()

    def _check_conditioner_bases(self, conditioner):
        for name in self.conditioner_groups:
            for module in conditioner.component_groups()[name]:
                reject_quantized_base(module, model_label=f"Ming conditioner {name}")

    def arch_param_groups(self):
        trainer = self.trainer
        components = trainer.ming_image_01_components
        reject_quantized_base(components.transformer, model_label="Ming DiT")
        self._check_conditioner_bases(components.conditioner)
        groups = optimizer_groups(components.transformer, components.conditioner,
            dit_lr=resolve_component_lr(trainer, "unet_lr", label="Ming DiT"),
            conditioner_lrs=self.conditioner_lrs, caption_lr=trainer.config.get("adapter_lr"))
        for group in groups:
            name = group["name"]
            group["component"] = ("unet" if name == "dit" else "vision_encoder" if name == "vision"
                                  else "text_encoder_1" if name == "language" else "adapter")
        return groups

    def grad_norm_components(self):
        return {id(parameter): group["component"] for group in self.arch_param_groups() for parameter in group["params"]}

    def write_checkpoint(self, step, epoch, output_path):
        buffer = getattr(self.trainer, "ming_batch_buffer", None)
        with buffer.suspend_prefetch() if buffer is not None else nullcontext():
            return self._write_checkpoint(step, epoch, output_path)

    def _write_checkpoint(self, step, epoch, output_path):
        components = self.trainer.ming_image_01_components
        groups = self.arch_param_groups()
        metadata = {"schema_version": 1, "kind": "full_parameter_weights", "step": step, "epoch": epoch,
                    "profile": components.artifact.profile,
                    "conditioner_groups": sorted(self.conditioner_groups),
                    "train_adapter": self.trainer.config.get("train_adapter"),
                    "adapter_lr": self.trainer.config.get("adapter_lr"),
                    "optimizer_groups": [{"name": group["name"], "lr": group["lr"], "component": group["component"]}
                                         for group in groups]}
        if getattr(self.trainer,"ming_full_initialization",None) is not None:
            metadata["ming_full_initialization"] = self.trainer.ming_full_initialization
        for key in ("ming_conditioner_full_initialization","ming_conditioner_quantized_base_training"):
            if getattr(self.trainer,key,None) is not None:
                metadata[key] = getattr(self.trainer,key)
        modules = (components.transformer, components.conditioner, components.vae)
        devices = []
        for module in modules:
            devices.append(execution_device(module))
        try:
            for module in modules:
                move_component(module, "cpu")
            written = save_bundle(components, Path(output_path), training_metadata=metadata,
                cancelled=getattr(self.trainer, "ming_checkpoint_cancelled", None))
        finally:
            failures = []
            for module, device in zip(modules, devices):
                try:
                    move_component(module, device)
                except Exception as exc:
                    failures.append(exc)
            if failures:
                raise RuntimeError("Ming checkpoint component residency restoration failed") from failures[0]
        return written


class MingImage01LoRAAdapter(BaseLoRAAdapter):
    def apply_lora_to_unet(self, lora_layers):
        from core.models.ming_image_01.lora import iter_lora_slots, flatten_to_key
        from core.adapters import is_adapter_covered
        owner = self.trainer
        components = owner.ming_image_01_components
        if owner.ming_conditioner_trainable_groups:
            raise ValueError("Ming conditioner LoRA is not yet connected; freeze native conditioner groups")
        from core.models.ming_image_01.quantization import configure_frozen_quantized_training
        owner.ming_quantized_base_training = configure_frozen_quantized_training(components.transformer)
        components.transformer.requires_grad_(False).train()
        components.conditioner.set_trainable(set())
        components.conditioner.eval()
        components.vae.requires_grad_(False).eval()
        slots = list(iter_lora_slots(components.transformer, attention=owner.train_unet,
            projection=owner.config.get("train_adapter") is True))
        count = 0
        for parent, attr, path in slots:
            current = parent[attr] if isinstance(attr, int) else getattr(parent, attr)
            if is_adapter_covered(current):
                continue
            name = flatten_to_key(path)
            layer = self.build_branch(current, name)
            if isinstance(attr, int): parent[attr] = layer
            else: setattr(parent, attr, layer)
            component = LORA_COMPONENT_ADAPTER if path == "cap_embedder.1" else LORA_COMPONENT_UNET
            self.register_lora_layer(lora_layers, name, layer, component)
            count += 1
        if not count:
            raise ValueError("Ming LoRA requires native DiT attention or caption projection targets")
        if owner.gradient_checkpointing:
            components.transformer.enable_gradient_checkpointing()
        return count

    def apply_lora_to_text_encoders(self, lora_layers):
        raise ValueError("Ming conditioner LoRA is not yet connected")

    def arch_param_groups(self, lora_layers):
        owner = self.trainer
        return self.component_param_groups(lora_layers, {
            LORA_COMPONENT_UNET: lambda: resolve_component_lr(owner, "unet_lr", label="Ming DiT LoRA"),
            LORA_COMPONENT_ADAPTER: lambda: (float(owner.config["adapter_lr"])
                if owner.config.get("adapter_lr") is not None
                else resolve_component_lr(owner, "unet_lr", label="Ming caption LoRA")),
        })

    def checkpoint_metadata(self, lora_layers, step, epoch):
        import json
        from core.models.ming_image_01.residual_unet import model_topology_metadata
        quantized = getattr(self.trainer,"ming_quantized_base_training",None)
        quantized_metadata = {} if quantized is None else {"ming_quantized_base_training":json.dumps(quantized,sort_keys=True,allow_nan=False)}
        return {"model_type": "ming_image_01", "modelspec.architecture": "ming_image_01",
                "ming_image_01_profile": self.trainer.ming_image_01_components.artifact.profile,
                "lora_rank": str(self.lora_rank), "lora_alpha": str(self.lora_alpha),
                "step": str(step), "epoch": str(epoch), "format": "pt",
                **model_topology_metadata(self.trainer.transformer), **quantized_metadata}
