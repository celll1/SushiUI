"""Image inference operations for SenseNova SDXL Chimera."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image

from .attention_processor import (
    ChimeraAttentionContext,
    clear_chimera_attention_caches,
    install_chimera_attention_processors,
    set_chimera_attention_context,
)
from .flow import flow_euler_step, flow_noising
from .prefix import encode_chimera_conditioning


@dataclass(frozen=True)
class ChimeraConditioning:
    encoder_hidden_states: torch.Tensor
    pooled_text_embeds: torch.Tensor
    context_positions: torch.Tensor
    attention_mask: torch.Tensor
    fingerprint: str


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().float().cpu().contiguous().numpy()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_conditioning(transformer, tokenizer, bridge, prompt: str) -> ChimeraConditioning:
    output = encode_chimera_conditioning(transformer, tokenizer, bridge, prompt)
    return ChimeraConditioning(
        encoder_hidden_states=output.encoder_hidden_states.detach(),
        pooled_text_embeds=output.pooled_text_embeds.detach(),
        context_positions=output.context_positions.detach(),
        attention_mask=output.attention_mask.detach(),
        fingerprint=_tensor_digest(
            output.encoder_hidden_states,
            output.pooled_text_embeds,
            output.context_positions,
            output.attention_mask,
        ),
    )


def _pad_conditioning(conditioning: ChimeraConditioning, length: int) -> ChimeraConditioning:
    current = int(conditioning.encoder_hidden_states.shape[1])
    if current == length:
        return conditioning
    if current > length:
        raise ValueError(f"cannot pad Chimera conditioning from {current} down to {length}")
    pad = length - current
    hidden = torch.nn.functional.pad(conditioning.encoder_hidden_states, (0, 0, 0, pad))
    positions = torch.nn.functional.pad(conditioning.context_positions, (0, 0, 0, pad))
    mask = torch.nn.functional.pad(conditioning.attention_mask, (0, pad), value=False)
    return ChimeraConditioning(
        encoder_hidden_states=hidden,
        pooled_text_embeds=conditioning.pooled_text_embeds,
        context_positions=positions,
        attention_mask=mask,
        fingerprint=conditioning.fingerprint,
    )


def shifted_timesteps(steps: int, shift: float, *, device: torch.device | str) -> torch.Tensor:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if shift <= 0:
        raise ValueError(f"timestep shift must be positive, got {shift}")
    time = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
    sigma = 1.0 - time
    sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    return 1.0 - sigma


def sdxl_time_ids(
    batch: int,
    *,
    original_height: int,
    original_width: int,
    crop_top: int,
    crop_left: int,
    target_height: int,
    target_width: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = [
        original_height,
        original_width,
        crop_top,
        crop_left,
        target_height,
        target_width,
    ]
    return torch.tensor([values], device=device, dtype=dtype).repeat(batch, 1)


def _unet_velocity(
    unet,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    conditioning: ChimeraConditioning,
    time_ids: torch.Tensor,
    *,
    cache_metadata: tuple,
) -> torch.Tensor:
    device = sample.device
    dtype = sample.dtype
    context = ChimeraAttentionContext(
        context_positions=conditioning.context_positions.to(device=device),
        target_height=int(cache_metadata[0]),
        target_width=int(cache_metadata[1]),
        crop_top=int(cache_metadata[2]),
        crop_left=int(cache_metadata[3]),
        prefix_terminal_t=float(cache_metadata[4]),
        cache_key=(conditioning.fingerprint, *cache_metadata),
    )
    set_chimera_attention_context(unet, context)
    added = {
        "text_embeds": conditioning.pooled_text_embeds.to(device=device, dtype=dtype),
        "time_ids": time_ids.to(device=device, dtype=dtype),
    }
    return unet(
        sample,
        timestep.to(device=device).expand(sample.shape[0]),
        encoder_hidden_states=conditioning.encoder_hidden_states.to(
            device=device, dtype=dtype
        ),
        encoder_attention_mask=conditioning.attention_mask.to(device=device),
        added_cond_kwargs=added,
        return_dict=False,
    )[0]


def encode_image_latents(
    vae,
    image: Image.Image,
    *,
    height: int,
    width: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode one RGB image with the artifact VAE's deterministic posterior mode."""
    image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
    vae_parameter = next(vae.parameters(), None)
    vae_device = vae_parameter.device if vae_parameter is not None else torch.device(device)
    vae_dtype = vae_parameter.dtype if vae_parameter is not None else dtype
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(
        device=vae_device, dtype=vae_dtype
    )
    pixels = pixels.mul(2.0).sub(1.0)
    encoded = vae.encode(pixels).latent_dist.mode()
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    return ((encoded - shift) * scaling).to(device=device, dtype=dtype)


def prepare_generate_mask(
    mask_image: Image.Image,
    *,
    latent_height: int,
    latent_width: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a latent mask where one means generate and zero means preserve."""
    mask = mask_image.convert("L").resize(
        (latent_width, latent_height), Image.Resampling.NEAREST
    )
    values = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy())
    return values.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype).div(255.0)


def _combine_cfg_velocity(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    cfg_scale: float,
    cfg_norm: str,
) -> torch.Tensor:
    """Blend CFG and optionally cap its norm at the conditional branch norm."""
    guided = unconditional + float(cfg_scale) * (conditional - unconditional)
    if float(cfg_scale) <= 1.0 or cfg_norm == "none":
        return guided
    if cfg_norm == "global":
        dims = tuple(range(1, guided.ndim))
    elif cfg_norm == "channel":
        dims = tuple(range(2, guided.ndim))
    else:
        raise ValueError(f"unsupported Chimera CFG norm: {cfg_norm}")
    conditional_norm = torch.linalg.vector_norm(conditional.float(), dim=dims, keepdim=True)
    guided_norm = torch.linalg.vector_norm(guided.float(), dim=dims, keepdim=True)
    shrink = (conditional_norm / guided_norm.clamp_min(1e-8)).clamp(max=1.0)
    return guided * shrink.to(dtype=guided.dtype)


def _cfg_probe_record(
    *,
    step: int,
    total_steps: int,
    timestep: torch.Tensor,
    next_timestep: torch.Tensor,
    sample_before: torch.Tensor,
    sample_after: torch.Tensor,
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    guided_raw: torch.Tensor,
    guided_post: torch.Tensor,
    cfg_scale: float,
) -> dict[str, float | int]:
    """Reduce one CFG step to bounded scalar diagnostics.

    The probe is explicit and sampling already stalls training, so synchronizing
    these small reductions is acceptable. No prompt or tensor payload leaves the
    trainer process.
    """

    eps = 1e-12

    def rms(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().square().mean().sqrt()

    def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        return float((numerator / denominator.clamp_min(eps)).item())

    def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        left_flat = left.detach().float().flatten()
        right_flat = right.detach().float().flatten()
        denom = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
        if float(denom.item()) <= eps:
            return 0.0
        return float(torch.dot(left_flat, right_flat).div(denom).item())

    def abs_p99(value: torch.Tensor) -> float:
        return float(torch.quantile(value.detach().float().abs().flatten(), 0.99).item())

    t = timestep.detach().float()
    t_next = next_timestep.detach().float()
    one_minus_t = 1.0 - t
    delta = conditional - unconditional
    x0_cond = sample_before + one_minus_t * conditional
    x0_uncond = sample_before + one_minus_t * unconditional
    x0_raw = sample_before + one_minus_t * guided_raw

    cond_rms = rms(conditional)
    uncond_rms = rms(unconditional)
    delta_rms = rms(delta)
    raw_rms = rms(guided_raw)
    post_rms = rms(guided_post)
    sample_rms = rms(sample_before)
    x0_cond_rms = rms(x0_cond)
    x0_uncond_rms = rms(x0_uncond)

    return {
        "step": int(step),
        "total_steps": int(total_steps),
        "timestep": float(t.item()),
        "next_timestep": float(t_next.item()),
        "delta_t": float((t_next - t).item()),
        "cfg_scale": float(cfg_scale),
        "velocity_cond_rms": float(cond_rms.item()),
        "velocity_uncond_rms": float(uncond_rms.item()),
        "velocity_delta_rms": float(delta_rms.item()),
        "velocity_guided_raw_rms": float(raw_rms.item()),
        "velocity_guided_post_rms": float(post_rms.item()),
        "guidance_rel": ratio(delta_rms, uncond_rms),
        "cond_uncond_cosine": cosine(conditional, unconditional),
        "delta_cond_cosine": cosine(delta, conditional),
        "raw_cond_norm_ratio": ratio(raw_rms, cond_rms),
        "post_cond_norm_ratio": ratio(post_rms, cond_rms),
        "clamp_norm_ratio": ratio(post_rms, raw_rms),
        "euler_update_rel": ratio((t_next - t).abs() * post_rms, sample_rms),
        "x0_guidance_rel": ratio(rms(x0_cond - x0_uncond), x0_uncond_rms),
        "x0_raw_cond_norm_ratio": ratio(rms(x0_raw), x0_cond_rms),
        "latent_rms_before": float(sample_rms.item()),
        "latent_rms_after": float(rms(sample_after).item()),
        "latent_abs_p99_before": abs_p99(sample_before),
        "latent_abs_p99_after": abs_p99(sample_after),
        "latent_abs_max_after": float(sample_after.detach().float().abs().max().item()),
    }


def _sample_flow_latents(
    unet,
    positive: ChimeraConditioning,
    negative: ChimeraConditioning | None,
    *,
    sample: torch.Tensor,
    times: torch.Tensor,
    start_index: int,
    source_latents: torch.Tensor | None,
    source_noise: torch.Tensor | None,
    generate_mask: torch.Tensor | None,
    height: int,
    width: int,
    cfg_scale: float,
    cfg_schedule_type: str,
    cfg_schedule_min: float,
    cfg_schedule_max: float | None,
    cfg_schedule_power: float,
    cfg_mode: str,
    cfg_norm: str,
    original_height: int,
    original_width: int,
    crop_top: int,
    crop_left: int,
    attention_backend: str,
    progress_callback: Callable[[int, int, torch.Tensor], None] | None,
    step_progress_callback: Callable[[int, int], None] | None = None,
    cfg_probe_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> torch.Tensor:
    device = sample.device
    dtype = sample.dtype
    steps = len(times) - 1
    time_ids = sdxl_time_ids(
        1,
        original_height=original_height,
        original_width=original_width,
        crop_top=crop_top,
        crop_left=crop_left,
        target_height=height,
        target_width=width,
        device=device,
        dtype=dtype,
    )
    cache_metadata = (height, width, crop_top, crop_left, 0.0, "native-prefix-v2")
    scheduled_peak = max(
        float(cfg_scale),
        float(cfg_schedule_min),
        float(cfg_schedule_max) if cfg_schedule_max is not None else float(cfg_scale),
    ) if cfg_schedule_type != "constant" else float(cfg_scale)
    needs_cfg = negative is not None and scheduled_peak > 1.0
    if cfg_probe_callback is not None and not needs_cfg:
        raise ValueError("Chimera CFG probe requires a negative branch and cfg_scale > 1")
    install_chimera_attention_processors(unet, backend=attention_backend)
    try:
        with torch.inference_mode():
            for index in range(start_index, steps):
                timestep = times[index].to(dtype=dtype)
                from core.inference.custom_sampling import calculate_dynamic_cfg
                cfg_now = calculate_dynamic_cfg(
                    sigma=float(1.0 - times[index].item()),
                    sigma_max=1.0,
                    cfg_base=float(cfg_scale),
                    cfg_schedule_type=cfg_schedule_type,
                    cfg_schedule_min=cfg_schedule_min,
                    cfg_schedule_max=cfg_schedule_max,
                    cfg_schedule_power=cfg_schedule_power,
                    denoise_progress=float(times[index].item()),
                )
                if not needs_cfg:
                    velocity = _unet_velocity(
                        unet, sample, timestep, positive, time_ids, cache_metadata=cache_metadata
                    )
                elif cfg_mode == "batched":
                    pair_length = max(
                        negative.encoder_hidden_states.shape[1],
                        positive.encoder_hidden_states.shape[1],
                    )
                    negative_padded = _pad_conditioning(negative, pair_length)
                    positive_padded = _pad_conditioning(positive, pair_length)
                    combined = ChimeraConditioning(
                        encoder_hidden_states=torch.cat(
                            (negative_padded.encoder_hidden_states,
                             positive_padded.encoder_hidden_states)
                        ),
                        pooled_text_embeds=torch.cat(
                            (negative.pooled_text_embeds, positive.pooled_text_embeds)
                        ),
                        context_positions=torch.cat(
                            (negative_padded.context_positions,
                             positive_padded.context_positions)
                        ),
                        attention_mask=torch.cat(
                            (negative_padded.attention_mask,
                             positive_padded.attention_mask)
                        ),
                        fingerprint=f"{negative.fingerprint}:{positive.fingerprint}",
                    )
                    pair = _unet_velocity(
                        unet,
                        sample.repeat(2, 1, 1, 1),
                        timestep,
                        combined,
                        time_ids.repeat(2, 1),
                        cache_metadata=cache_metadata,
                    )
                    uncond, cond = pair.chunk(2)
                    guided_raw = uncond + float(cfg_now) * (cond - uncond)
                    velocity = _combine_cfg_velocity(cond, uncond, cfg_now, cfg_norm)
                else:
                    uncond = _unet_velocity(
                        unet, sample, timestep, negative, time_ids, cache_metadata=cache_metadata
                    )
                    cond = _unet_velocity(
                        unet, sample, timestep, positive, time_ids, cache_metadata=cache_metadata
                    )
                    guided_raw = uncond + float(cfg_now) * (cond - uncond)
                    velocity = _combine_cfg_velocity(cond, uncond, cfg_now, cfg_norm)
                sample_before = sample
                sample = flow_euler_step(sample, velocity, times[index], times[index + 1])
                if generate_mask is not None:
                    source_at_next = flow_noising(
                        source_latents, source_noise, times[index + 1]
                    )
                    sample = generate_mask * sample + (1.0 - generate_mask) * source_at_next
                if cfg_probe_callback is not None:
                    cfg_probe_callback(_cfg_probe_record(
                        step=index + 1,
                        total_steps=steps,
                        timestep=times[index],
                        next_timestep=times[index + 1],
                        sample_before=sample_before,
                        sample_after=sample,
                        conditional=cond,
                        unconditional=uncond,
                        guided_raw=guided_raw,
                        guided_post=velocity,
                        cfg_scale=cfg_now,
                    ))
                if progress_callback is not None:
                    progress_callback(index + 1, steps, sample)
                if step_progress_callback is not None:
                    step_progress_callback(index + 1, steps)
        return sample
    finally:
        clear_chimera_attention_caches(unet)


def sample_txt2img_latents(
    unet,
    positive: ChimeraConditioning,
    negative: ChimeraConditioning | None,
    *,
    height: int,
    width: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    timestep_shift: float = 1.0,
    cfg_mode: str = "sequential",
    cfg_norm: str = "none",
    cfg_schedule_type: str = "constant",
    cfg_schedule_min: float = 1.0,
    cfg_schedule_max: float | None = None,
    cfg_schedule_power: float = 2.0,
    original_height: int | None = None,
    original_width: int | None = None,
    crop_top: int = 0,
    crop_left: int = 0,
    attention_backend: str = "normal",
    progress_callback: Callable[[int, int, torch.Tensor], None] | None = None,
    step_progress_callback: Callable[[int, int], None] | None = None,
    cfg_probe_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> torch.Tensor:
    if height % 8 or width % 8:
        raise ValueError("Chimera width and height must be divisible by 8")
    if cfg_mode not in {"sequential", "batched"}:
        raise ValueError(f"unsupported Chimera CFG mode: {cfg_mode}")
    if cfg_norm not in {"none", "global", "channel"}:
        raise ValueError(f"unsupported Chimera CFG norm: {cfg_norm}")
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(int(seed))
    sample = torch.randn((1, 4, height // 8, width // 8), generator=generator, device=device, dtype=dtype)
    times = shifted_timesteps(steps, timestep_shift, device=device)
    original_height = int(original_height or height)
    original_width = int(original_width or width)
    return _sample_flow_latents(
        unet,
        positive,
        negative,
        sample=sample,
        times=times,
        start_index=0,
        source_latents=None,
        source_noise=None,
        generate_mask=None,
        height=height,
        width=width,
        cfg_scale=cfg_scale,
        cfg_schedule_type=cfg_schedule_type,
        cfg_schedule_min=cfg_schedule_min,
        cfg_schedule_max=cfg_schedule_max,
        cfg_schedule_power=cfg_schedule_power,
        cfg_mode=cfg_mode,
        cfg_norm=cfg_norm,
        original_height=original_height,
        original_width=original_width,
        crop_top=crop_top,
        crop_left=crop_left,
        attention_backend=attention_backend,
        progress_callback=progress_callback,
        step_progress_callback=step_progress_callback,
        cfg_probe_callback=cfg_probe_callback,
    )


def sample_img2img_latents(
    unet,
    positive: ChimeraConditioning,
    negative: ChimeraConditioning | None,
    source_latents: torch.Tensor,
    *,
    steps: int,
    denoising_strength: float,
    cfg_scale: float,
    seed: int,
    generate_mask: torch.Tensor | None = None,
    timestep_shift: float = 1.0,
    cfg_mode: str = "sequential",
    cfg_norm: str = "none",
    cfg_schedule_type: str = "constant",
    cfg_schedule_min: float = 1.0,
    cfg_schedule_max: float | None = None,
    cfg_schedule_power: float = 2.0,
    original_height: int | None = None,
    original_width: int | None = None,
    crop_top: int = 0,
    crop_left: int = 0,
    attention_backend: str = "normal",
    progress_callback: Callable[[int, int, torch.Tensor], None] | None = None,
    step_progress_callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Run deterministic SDEdit, optionally pinning the mask's preserve region."""
    if not 0.0 <= float(denoising_strength) <= 1.0:
        raise ValueError("Chimera denoising_strength must be between 0 and 1")
    if cfg_mode not in {"sequential", "batched"}:
        raise ValueError(f"unsupported Chimera CFG mode: {cfg_mode}")
    if cfg_norm not in {"none", "global", "channel"}:
        raise ValueError(f"unsupported Chimera CFG norm: {cfg_norm}")
    if source_latents.ndim != 4 or source_latents.shape[0] != 1:
        raise ValueError("Chimera source latents must have shape [1,C,H,W]")
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    source_latents = source_latents.to(device=device, dtype=dtype)
    height = int(source_latents.shape[-2] * 8)
    width = int(source_latents.shape[-1] * 8)
    if generate_mask is not None:
        expected = (1, 1, source_latents.shape[-2], source_latents.shape[-1])
        if tuple(generate_mask.shape) != expected:
            raise ValueError(f"Chimera generate mask must have shape {expected}")
        generate_mask = generate_mask.to(device=device, dtype=dtype).clamp(0, 1)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(
        source_latents.shape, generator=generator, device=device, dtype=dtype
    )
    times = shifted_timesteps(steps, timestep_shift, device=device)
    start_index = min(steps, max(0, int(round((1.0 - float(denoising_strength)) * steps))))
    if start_index == steps:
        return source_latents
    sample = flow_noising(source_latents, noise, times[start_index])
    return _sample_flow_latents(
        unet,
        positive,
        negative,
        sample=sample,
        times=times,
        start_index=start_index,
        source_latents=source_latents,
        source_noise=noise,
        generate_mask=generate_mask,
        height=height,
        width=width,
        cfg_scale=cfg_scale,
        cfg_schedule_type=cfg_schedule_type,
        cfg_schedule_min=cfg_schedule_min,
        cfg_schedule_max=cfg_schedule_max,
        cfg_schedule_power=cfg_schedule_power,
        cfg_mode=cfg_mode,
        cfg_norm=cfg_norm,
        original_height=int(original_height or height),
        original_width=int(original_width or width),
        crop_top=crop_top,
        crop_left=crop_left,
        attention_backend=attention_backend,
        progress_callback=progress_callback,
        step_progress_callback=step_progress_callback,
    )


def decode_latents(
    vae,
    latents: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    restore_device: bool = False,
) -> Image.Image:
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    vae_parameter = next(vae.parameters(), None)
    original_device = vae_parameter.device if vae_parameter is not None else latents.device
    vae_dtype = vae_parameter.dtype if vae_parameter is not None else latents.dtype
    decode_device = torch.device(device) if device is not None else original_device
    try:
        if vae_parameter is not None and original_device != decode_device:
            vae.to(device=decode_device)
        latents = latents.to(device=decode_device, dtype=vae_dtype)
        with torch.inference_mode():
            decoded = vae.decode(latents / scaling + shift, return_dict=False)[0]
            pixels = (decoded.float() / 2.0 + 0.5).clamp(0, 1)
            array = (
                pixels[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
            )
        return Image.fromarray(np.asarray(array))
    finally:
        if restore_device and vae_parameter is not None and original_device != decode_device:
            vae.to(device=original_device)
            if decode_device.type == "cuda":
                torch.cuda.empty_cache()
