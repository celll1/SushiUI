"""Ming full-parameter ownership, component rates and native bundle export."""

from pathlib import Path
from contextlib import nullcontext
from core.models.ming_image_01.block_offload import execution_device, move_component

from core.models.ming_image_01.checkpoint import save_bundle
from core.models.ming_image_01.training import optimizer_groups
from .base_adapter import BaseFullParameterAdapter, reject_quantized_base, resolve_component_lr


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
        reject_quantized_base(transformer, model_label="Ming DiT")
        if self.conditioner_groups:
            reject_quantized_base(conditioner, model_label="Ming conditioner")
        transformer.requires_grad_(trainer.train_unet).train(trainer.train_unet)
        conditioner.set_trainable(self.conditioner_groups)
        conditioner.eval()
        for name in self.conditioner_groups:
            for module in conditioner.component_groups()[name]:
                module.train()
        components.vae.requires_grad_(False).eval()
        if trainer.gradient_checkpointing:
            if trainer.train_unet:
                transformer.enable_gradient_checkpointing()
            for module in (conditioner.native.model, conditioner.native.vision, conditioner.native.connector):
                if any(parameter.requires_grad for parameter in module.parameters()):
                    module.gradient_checkpointing_enable()
        # Validate optimizer ownership before any training batch is assembled.
        self.arch_param_groups()

    def arch_param_groups(self):
        trainer = self.trainer
        components = trainer.ming_image_01_components
        groups = optimizer_groups(components.transformer, components.conditioner,
            dit_lr=resolve_component_lr(trainer, "unet_lr", label="Ming DiT"),
            conditioner_lrs=self.conditioner_lrs)
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
                    "optimizer_groups": [{"name": group["name"], "lr": group["lr"], "component": group["component"]}
                                         for group in groups]}
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
