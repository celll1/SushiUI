"""Stage-exact full-parameter adapter for SenseNova SDXL Chimera."""

from __future__ import annotations

from pathlib import Path

from .base_adapter import BaseFullParameterAdapter, resolve_component_lr


class SenseNovaSDXLChimeraFullParameterAdapter(BaseFullParameterAdapter):
    def prepare_models_for_training(self):
        from core.training.ops.sensenova_sdxl_chimera_ops import training_stage

        trainer = self.trainer
        stage = training_stage(trainer)
        trainer.chimera_understanding.requires_grad_(False).eval()
        trainer.vae.requires_grad_(False).eval()
        trainer.unet.requires_grad_(stage in {"unet", "joint"})
        trainer.condition_bridge.requires_grad_(stage in {"bridge_align", "joint"})
        trainer.unet.train(stage in {"unet", "joint"})
        trainer.condition_bridge.train(stage in {"bridge_align", "joint"})
        teacher = getattr(trainer, "chimera_teacher", None)
        if teacher is not None:
            for module in (teacher.text_encoder, teacher.text_encoder_2):
                module.requires_grad_(False).eval()

    def arch_param_groups(self):
        from core.training.ops.sensenova_sdxl_chimera_ops import training_stage

        trainer = self.trainer
        stage = training_stage(trainer)
        groups = []
        if stage in {"unet", "joint"}:
            parameters = [parameter for parameter in trainer.unet.parameters() if parameter.requires_grad]
            if parameters:
                groups.append({
                    "params": parameters,
                    "lr": resolve_component_lr(trainer, "unet_lr", label="Chimera U-Net"),
                    "name": "unet",
                    "component": "unet",
                })
        if stage in {"bridge_align", "joint"}:
            parameters = [
                parameter for parameter in trainer.condition_bridge.parameters()
                if parameter.requires_grad
            ]
            if parameters:
                groups.append({
                    "params": parameters,
                    "lr": resolve_component_lr(
                        trainer, "chimera_bridge_lr", label="Chimera conditioning bridge"
                    ),
                    "name": "condition_bridge",
                    "component": "text_encoder",
                })
        if not groups:
            raise RuntimeError(f"Chimera stage {stage!r} produced no trainable parameter groups")
        return groups

    def write_checkpoint(self, step: int, epoch: int, output_path: Path):
        from core.models.sensenova_sdxl_chimera.artifact import save_chimera_checkpoint
        from core.training.ops.sensenova_sdxl_chimera_ops import training_stage

        trainer = self.trainer
        target = Path(output_path)
        if target.suffix:
            target = target.with_suffix("")
        return save_chimera_checkpoint(
            target,
            base_manifest=trainer.chimera_manifest,
            runtime=trainer.chimera_runtime_config,
            bridge=trainer.condition_bridge,
            unet=trainer.unet,
            vae=trainer.vae,
            stage=training_stage(trainer),
            step=step,
            epoch=epoch,
            alignment_metrics=getattr(trainer, "chimera_alignment_metrics", None),
            alignment_passed=bool(getattr(trainer, "chimera_alignment_passed", False)),
            max_shard_bytes=int((getattr(trainer, "config", None) or {}).get(
                "chimera_max_shard_bytes", 10 * 1024**3
            )),
        )
