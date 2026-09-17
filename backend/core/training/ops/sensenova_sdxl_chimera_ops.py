"""Training operations for SenseNova SDXL Chimera."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from core.models.sensenova_sdxl_chimera.attention_processor import (
    ChimeraAttentionContext,
    install_chimera_attention_processors,
    set_chimera_attention_context,
)
from core.models.sensenova_sdxl_chimera.flow import (
    flow_noising,
    flow_velocity_target,
)
from core.models.sensenova_sdxl_chimera.prefix import encode_chimera_conditioning


STAGES = ("bridge_align", "unet", "joint")


def configured_training_stage(trainer) -> str:
    stage = str((getattr(trainer, "config", None) or {}).get(
        "chimera_training_stage", "unet"
    )).strip().lower()
    if stage not in STAGES:
        raise ValueError(f"chimera_training_stage must be one of {STAGES}, got {stage!r}")
    return stage


def bridge_align_steps(trainer) -> int:
    return int((getattr(trainer, "config", None) or {}).get(
        "chimera_bridge_align_steps", 0
    ) or 0)


def training_stage_for_step(trainer, completed_steps: int) -> str:
    target = configured_training_stage(trainer)
    warmup = bridge_align_steps(trainer)
    if warmup > 0 and target in {"unet", "joint"} and int(completed_steps) < warmup:
        return "bridge_align"
    return target


def training_stage(trainer) -> str:
    active = getattr(trainer, "chimera_active_training_stage", None)
    if active is not None:
        return str(active)
    return training_stage_for_step(
        trainer, int(getattr(trainer, "chimera_completed_steps", 0) or 0)
    )


def training_stage_plan(trainer) -> tuple[str, ...]:
    target = configured_training_stage(trainer)
    if bridge_align_steps(trainer) > 0 and target in {"unet", "joint"}:
        return ("bridge_align", target)
    return (target,)


def sync_training_stage(trainer, completed_steps: int) -> str:
    """Apply the exact staged graph before encoding the next training step."""
    completed_steps = int(completed_steps)
    desired = training_stage_for_step(trainer, completed_steps)
    previous = getattr(trainer, "chimera_active_training_stage", None)
    trainer.chimera_completed_steps = completed_steps
    if previous == desired:
        return desired

    trainer.chimera_active_training_stage = desired
    trainer.unet.requires_grad_(desired in {"unet", "joint"})
    trainer.condition_bridge.requires_grad_(desired in {"bridge_align", "joint"})
    trainer.unet.train(desired in {"unet", "joint"})
    trainer.condition_bridge.train(desired in {"bridge_align", "joint"})
    if previous == "bridge_align" and desired != "bridge_align":
        teacher = getattr(trainer, "chimera_teacher", None)
        if teacher is not None:
            teacher.to("cpu")
            trainer.chimera_teacher = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"{getattr(trainer, 'log_prefix', '[Chimera]')} Chimera stage transition at completed step "
            f"{completed_steps}: bridge_align -> {desired}"
        )
    return desired


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
        vae_dtype=trainer.vae_dtype,
        load_understanding=True,
    )
    trainer.unet = components["unet"]
    trainer.vae = components["vae"].to(dtype=trainer.vae_dtype)
    trainer.chimera_frozen_vae_state = components["frozen_vae_state"]
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
    if bridge_align_steps(trainer) > 0:
        initialization = manifest["unet"]["initialization"]
        aligned = manifest["conditioning"]["bridge_state"] == "aligned"
        if initialization == "sdxl_transplant" and not aligned:
            raise ValueError(
                "A staged bridge warmup cannot replace the held-out alignment gate for "
                "sdxl_transplant; start from bridge_state='aligned' or use a scratch U-Net"
            )
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
    trainer.condition_bridge.to(device=device, dtype=trainer.weight_dtype)
    prefix = None
    prefetcher = getattr(trainer, "chimera_prefix_prefetcher", None)
    if prefetcher is not None:
        prefix = prefetcher.take(prompt, device)
        if prefix is None:
            prefix = prefetcher.capture_sync(prompt, device)
    else:
        trainer.chimera_understanding.to(device)
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        output = encode_chimera_conditioning(
            trainer.chimera_understanding,
            trainer.tokenizer,
            trainer.condition_bridge,
            prompt,
            prefix=prefix,
        )
    auxiliary: dict[str, Any] = {
        "pooled_text_embeds": output.pooled_text_embeds,
        "context_positions": output.context_positions,
        "context_attention_mask": output.attention_mask,
        "alignment_hidden_states": output.alignment_hidden_states,
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
    result = {}
    sequence_keys = {"context_positions", "context_attention_mask"}
    max_length = max(
        (int(item["context_attention_mask"].shape[1]) for item in batch),
        default=0,
    )
    for key in keys:
        values = [item[key] for item in batch]
        if key in sequence_keys:
            padded = []
            for value in values:
                pad = max_length - int(value.shape[1])
                if pad:
                    if key == "context_positions":
                        value = F.pad(value, (0, 0, 0, pad))
                    else:
                        value = F.pad(value, (0, pad), value=False)
                padded.append(value)
            values = padded
        result[key] = torch.cat(values, dim=0)
    return result


def bridge_alignment_loss(trainer, student: torch.Tensor, auxiliary: dict) -> tuple[torch.Tensor, dict]:
    student = auxiliary["alignment_hidden_states"].to(student)
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


def repa_tap(trainer):
    """Expose the SDXL-shaped U-Net map only when that U-Net is trainable."""
    stages = training_stage_plan(trainer)
    if stages == ("bridge_align",):
        raise ValueError(
            "repa_enable is not supported for Chimera bridge_align: that stage "
            "freezes the U-Net, so representation alignment would update only "
            "the projector and could not align the denoiser"
        )
    from core.training.ops import sd_sdxl_ops

    return sd_sdxl_ops.repa_tap(trainer, "sensenova_sdxl_chimera")


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
    context_mask = auxiliary["context_attention_mask"].to(device=latents.device)
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
    repa_pixels = getattr(ctx, "repa_pixels", None)
    repa_armed = bool(getattr(trainer, "repa_enable", False)) and repa_pixels is not None
    repa_handle = None
    if repa_armed:
        from core.training.repa import arm_spatial_tap, spatial_tap_sites

        site = spatial_tap_sites(trainer.unet)[trainer.repa_align_depth][1]
        repa_handle = arm_spatial_tap(trainer._repa_tap_module, site)
    try:
        prediction = trainer.unet(
            noisy,
            timesteps,
            encoder_hidden_states=conditioning,
            encoder_attention_mask=context_mask,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
            return_dict=False,
        )[0]
    finally:
        if repa_handle is not None:
            repa_handle.remove()
    loss = F.mse_loss(prediction.float(), target.float())
    value = float(loss.detach().cpu())
    with torch.no_grad():
        predicted_clean = (
            noisy.detach()
            + (1.0 - timesteps[:, None, None, None]) * prediction.detach()
        )
        recon_value = float(F.mse_loss(
            predicted_clean.float(), latents.float()
        ).cpu())
    if hasattr(trainer, "log_extra_metric"):
        trainer.log_extra_metric("chimera_velocity_loss", value)
    if repa_armed:
        from core.training.repa import apply_repa_loss_spatial, take_repa_tap

        trainer._ensure_repa_on_device()
        tap = take_repa_tap(trainer)
        if tap is None:
            raise RuntimeError(
                "REPA is enabled but the Chimera U-Net forward produced no spatial tap"
            )
        loss = apply_repa_loss_spatial(trainer, loss, tap, repa_pixels)
    debug_save_path = getattr(ctx, "debug_save_path", None)
    if debug_save_path is not None:
        try:
            debug_save_path.mkdir(parents=True, exist_ok=True)
            t_value = float(timesteps[0].detach().float().cpu())
            debug_data = {
                "timestep": t_value,
                "model_type": "sensenova_sdxl_chimera",
                "is_latent": True,
                "prediction_type": "flow_velocity",
                "loss": float(loss.detach().cpu()),
                "recon_loss": recon_value,
                "batch_size": int(latents.shape[0]),
                "latents": latents[:1].detach().cpu(),
                "noisy_latents": noisy[:1].detach().cpu(),
                "predicted_latent": predicted_clean[:1].detach().cpu(),
            }
            captions = getattr(ctx, "debug_captions", None)
            if captions:
                debug_data["caption"] = captions[0]
            reference_paths = getattr(ctx, "debug_reference_image_paths", None)
            if reference_paths:
                first_ref = next((path for path in reference_paths if path), None)
                if first_ref:
                    debug_data["reference_image_path"] = first_ref
            torch.save(debug_data, debug_save_path / f"latents_t{t_value:.4f}.pt")
        except Exception as debug_error:
            print(f"{trainer.log_prefix} [debug_latents] save failed: {debug_error}")
    # Gradient checkpointing replays the U-Net during backward, after this
    # function returns. The next step overwrites this small context in place.
    return loss, value, recon_value


def vae_encode(trainer, image_tensor: torch.Tensor, **_kwargs) -> torch.Tensor:
    latent = trainer.vae.encode(image_tensor).latent_dist.sample()
    shift = float(getattr(trainer.vae.config, "shift_factor", 0.0) or 0.0)
    scale = float(getattr(trainer.vae.config, "scaling_factor", 1.0))
    return (latent - shift) * scale


def vae_decode(trainer, latents: torch.Tensor, **_kwargs) -> torch.Tensor:
    shift = float(getattr(trainer.vae.config, "shift_factor", 0.0) or 0.0)
    scale = float(getattr(trainer.vae.config, "scaling_factor", 1.0))
    return trainer.vae.decode(latents / scale + shift, return_dict=False)[0]
