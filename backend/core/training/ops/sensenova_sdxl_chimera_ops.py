"""Training operations for SenseNova SDXL Chimera."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from core.models.sensenova_sdxl_chimera.attention_processor import (
    ChimeraAttentionContext,
    clear_chimera_attention_caches,
    install_chimera_attention_processors,
    set_chimera_attention_context,
)
from core.models.sensenova_sdxl_chimera.flow import (
    flow_noising,
    flow_velocity_target,
)
from core.models.sensenova_sdxl_chimera.prefix import encode_chimera_conditioning


STAGES = ("bridge_align", "unet", "joint")


def training_stage(trainer) -> str:
    stage = str((getattr(trainer, "config", None) or {}).get(
        "chimera_training_stage", "unet"
    )).strip().lower()
    if stage not in STAGES:
        raise ValueError(f"chimera_training_stage must be one of {STAGES}, got {stage!r}")
    return stage


def _load_alignment_teacher(trainer, donor: str) -> None:
    from diffusers import StableDiffusionXLPipeline

    source = donor[len("model:") :] if donor.startswith("model:") else donor
    path = Path(source)
    if path.is_dir():
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            str(path), torch_dtype=trainer.weight_dtype
        )
    else:
        pipeline = StableDiffusionXLPipeline.from_single_file(
            str(path), torch_dtype=trainer.weight_dtype, use_safetensors=True
        )
    pipeline.unet = None
    pipeline.vae = None
    pipeline.requires_safety_checker = False
    for module in (pipeline.text_encoder, pipeline.text_encoder_2):
        module.requires_grad_(False)
        module.eval()
    trainer.chimera_teacher = pipeline


def load_components(trainer) -> None:
    from core.attention import AttentionMode
    from core.models.sensenova_sdxl_chimera.loader import load_chimera_artifact

    components = load_chimera_artifact(
        trainer.model_path,
        torch_dtype=trainer.weight_dtype,
        load_understanding=True,
    )
    trainer.unet = components["unet"]
    trainer.vae = components["vae"].to(dtype=trainer.vae_dtype)
    trainer.condition_bridge = components["condition_bridge"]
    trainer.chimera_understanding = components["understanding"]["transformer"]
    trainer.text_encoder = trainer.chimera_understanding
    trainer.tokenizer = components["understanding"]["tokenizer"]
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.transformer = None
    trainer.transformer_original = None
    trainer.noise_scheduler = None
    trainer.original_scheduler = None
    trainer.chimera_manifest = components["manifest"]
    trainer.chimera_runtime_config = components["config"]
    trainer.chimera_model_path = components["model_path"]
    trainer.noise_process = "flow"
    trainer.prediction_target = "flow"

    stage = training_stage(trainer)
    manifest = trainer.chimera_manifest
    if stage in {"unet", "joint"}:
        initialization = manifest["unet"]["initialization"]
        aligned = manifest["conditioning"]["bridge_state"] == "aligned"
        allow = bool((getattr(trainer, "config", None) or {}).get(
            "chimera_allow_unaligned_scratch", False
        ))
        if not aligned and not (initialization == "scratch" and allow):
            raise ValueError(
                f"Chimera {stage} training requires bridge_state='aligned'; "
                "an unaligned scratch artifact needs chimera_allow_unaligned_scratch=true"
            )
        if initialization == "sdxl_transplant" and not aligned:
            raise ValueError("sdxl_transplant diffusion training requires an aligned bridge")
        if not aligned and initialization == "scratch" and allow:
            print(
                f"{trainer.log_prefix} WARNING: training a scratch Chimera U-Net with an "
                "unaligned bridge by explicit override; conditioning has not passed the "
                "held-out alignment gate"
            )
    if stage == "bridge_align":
        hidden_weight = (getattr(trainer, "config", None) or {}).get(
            "chimera_clip_hidden_weight"
        )
        pooled_weight = (getattr(trainer, "config", None) or {}).get(
            "chimera_clip_pooled_weight"
        )
        if hidden_weight is None or pooled_weight is None:
            raise ValueError(
                "bridge_align requires explicit chimera_clip_hidden_weight and "
                "chimera_clip_pooled_weight; no unmeasured numerical default is defined"
            )
        _load_alignment_teacher(trainer, manifest["sdxl_donor"]["provenance"])

    trainer.chimera_understanding.requires_grad_(False).eval()
    trainer.vae.requires_grad_(False).eval()
    install_chimera_attention_processors(
        trainer.unet,
        backend=getattr(trainer, "attention_backend", "normal"),
        mode=AttentionMode.TRAINING,
    )
    if getattr(trainer, "gradient_checkpointing", False):
        trainer.unet.enable_gradient_checkpointing()


def setup_attention_backend(trainer, backend: str) -> None:
    from core.attention import AttentionMode

    install_chimera_attention_processors(
        trainer.unet, backend=backend, mode=AttentionMode.TRAINING
    )


def encode_prompt(trainer, prompt: str, *, requires_grad: bool = False) -> tuple[torch.Tensor, dict]:
    device = torch.device(trainer.device)
    trainer.chimera_understanding.to(device)
    trainer.condition_bridge.to(device=device, dtype=trainer.weight_dtype)
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        output = encode_chimera_conditioning(
            trainer.chimera_understanding,
            trainer.tokenizer,
            trainer.condition_bridge,
            prompt,
        )
    auxiliary: dict[str, Any] = {
        "pooled_text_embeds": output.pooled_text_embeds,
        "context_positions": output.context_positions,
    }
    if training_stage(trainer) == "bridge_align":
        teacher = trainer.chimera_teacher.to(device)
        with torch.no_grad():
            teacher_hidden, _negative, teacher_pooled, _negative_pooled = teacher.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        auxiliary.update({
            "teacher_hidden": teacher_hidden,
            "teacher_pooled": teacher_pooled,
        })
    return output.encoder_hidden_states, auxiliary


def collate_aux(batch: list[dict]) -> dict:
    keys = set.intersection(*(set(item) for item in batch)) if batch else set()
    return {key: torch.cat([item[key] for item in batch], dim=0) for key in keys}


def bridge_alignment_loss(trainer, student: torch.Tensor, auxiliary: dict) -> tuple[torch.Tensor, dict]:
    teacher = auxiliary["teacher_hidden"].to(student)
    student_pooled = auxiliary["pooled_text_embeds"].to(student)
    teacher_pooled = auxiliary["teacher_pooled"].to(student)
    hidden_mse = F.mse_loss(F.normalize(student.float(), dim=-1), F.normalize(teacher.float(), dim=-1))
    student_rms = student.float().square().mean(dim=-1).sqrt()
    teacher_rms = teacher.float().square().mean(dim=-1).sqrt()
    rms_mse = F.mse_loss(student_rms, teacher_rms)
    pooled_mse = F.mse_loss(student_pooled.float(), teacher_pooled.float())
    config = getattr(trainer, "config", None) or {}
    loss = (
        hidden_mse
        + float(config["chimera_clip_hidden_weight"]) * rms_mse
        + float(config["chimera_clip_pooled_weight"]) * pooled_mse
    )
    metrics = {
        "hidden_mse": hidden_mse.detach(),
        "hidden_rms_mse": rms_mse.detach(),
        "pooled_mse": pooled_mse.detach(),
        "hidden_cosine": F.cosine_similarity(student.float(), teacher.float(), dim=-1).mean().detach(),
        "pooled_cosine": F.cosine_similarity(student_pooled.float(), teacher_pooled.float(), dim=-1).mean().detach(),
    }
    return loss, metrics


def train_step(trainer, ctx) -> tuple[torch.Tensor, float, float]:
    conditioning = ctx.text_embeddings.to(device=trainer.device, dtype=trainer.training_dtype)
    auxiliary = ctx.attention_mask or {}
    auxiliary = {
        key: value.to(device=trainer.device) if isinstance(value, torch.Tensor) else value
        for key, value in auxiliary.items()
    }
    if training_stage(trainer) == "bridge_align":
        loss, metrics = bridge_alignment_loss(trainer, conditioning, auxiliary)
        trainer.chimera_alignment_metrics = {
            key: float(value.cpu()) for key, value in metrics.items()
        }
        if hasattr(trainer, "log_extra_metric"):
            for key, value in trainer.chimera_alignment_metrics.items():
                trainer.log_extra_metric(f"chimera_clip_{key}", value)
        value = float(loss.detach().cpu())
        return loss, value, 0.0

    latents = ctx.latents.to(device=trainer.device, dtype=trainer.training_dtype)
    noise = torch.randn_like(latents)
    timesteps = ctx.timesteps
    if timesteps is None:
        timesteps = torch.rand(latents.shape[0], device=latents.device)
    timesteps = timesteps.to(device=latents.device, dtype=latents.dtype)
    noisy = flow_noising(latents, noise, timesteps)
    target = flow_velocity_target(latents, noise)
    pooled = auxiliary["pooled_text_embeds"].to(conditioning)
    positions = auxiliary["context_positions"].to(device=latents.device)
    height, width = latents.shape[-2] * 8, latents.shape[-1] * 8
    time_ids = ctx.time_ids
    if time_ids is None:
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]],
            device=latents.device,
            dtype=pooled.dtype,
        ).repeat(latents.shape[0], 1)
    else:
        time_ids = time_ids.to(device=latents.device, dtype=pooled.dtype)
    set_chimera_attention_context(
        trainer.unet,
        ChimeraAttentionContext(
            context_positions=positions,
            target_height=height,
            target_width=width,
        ),
    )
    try:
        prediction = trainer.unet(
            noisy,
            timesteps,
            encoder_hidden_states=conditioning,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
            return_dict=False,
        )[0]
        loss = F.mse_loss(prediction.float(), target.float())
        value = float(loss.detach().cpu())
        if hasattr(trainer, "log_extra_metric"):
            trainer.log_extra_metric("chimera_velocity_loss", value)
        return loss, value, 0.0
    finally:
        clear_chimera_attention_caches(trainer.unet)


def vae_encode(trainer, image_tensor: torch.Tensor, **_kwargs) -> torch.Tensor:
    latent = trainer.vae.encode(image_tensor).latent_dist.sample()
    shift = float(getattr(trainer.vae.config, "shift_factor", 0.0) or 0.0)
    scale = float(getattr(trainer.vae.config, "scaling_factor", 1.0))
    return (latent - shift) * scale


def vae_decode(trainer, latents: torch.Tensor, **_kwargs) -> torch.Tensor:
    shift = float(getattr(trainer.vae.config, "shift_factor", 0.0) or 0.0)
    scale = float(getattr(trainer.vae.config, "scaling_factor", 1.0))
    return trainer.vae.decode(latents / scale + shift, return_dict=False)[0]
