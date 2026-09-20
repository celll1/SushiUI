"""Image inference operations for SenseNova SDXL Chimera."""

from __future__ import annotations

import hashlib
import math
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
from .flow import (
    FLOW_V1_PREDICTION,
    FLOW_V2_PREDICTION,
    FLOW_V2_VELOCITY_PREDICTION,
    FLOW_V3_CONFIDENCE_ANGULAR_SCHEDULE,
    FLOW_V3_PREDICTION,
    FLOW_V4_PREDICTION,
    conditioning_reliability,
    destruction_coordinate,
    destruction_coordinate_polar_flow_target,
    destruction_coordinate_recover_clean,
    endpoint_observable_noising,
    endpoint_observable_preconditioning,
    endpoint_observable_recover_clean,
    flow_euler_step,
    flow_noising,
    gate_conditioning_tensor,
    mixed_coordinate_polar_exp_euler_step,
    polar_compose_velocity,
    polar_exp_euler_step,
    polar_flow_target,
    polar_recover_clean,
    polar_state,
    polar_tangent_projection,
)
from .prefix import encode_chimera_conditioning
from .unet import polar_unet_forward


@dataclass(frozen=True)
class ChimeraConditioning:
    encoder_hidden_states: torch.Tensor
    pooled_text_embeds: torch.Tensor
    context_positions: torch.Tensor
    attention_mask: torch.Tensor
    fingerprint: str
    key_lengths: tuple[int, ...] | None = None


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
        key_lengths=(int(output.attention_mask.shape[1]),),
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
        key_lengths=conditioning.key_lengths,
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


def endpoint_observable_logsnr_timesteps(
    steps: int,
    latent_centered_second_moment: float,
    *,
    log_snr_min: float = -12.0,
    log_snr_max: float = 12.0,
    device: torch.device | str,
) -> torch.Tensor:
    """Place v2 interior Euler evaluations uniformly in bounded log-SNR."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    q = float(latent_centered_second_moment)
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("latent_centered_second_moment must be finite and positive")
    if not float(log_snr_min) < float(log_snr_max):
        raise ValueError("log_snr_min must be smaller than log_snr_max")
    if steps == 1:
        return torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)
    targets = torch.linspace(
        float(log_snr_min),
        float(log_snr_max),
        steps + 1,
        device=device,
        dtype=torch.float64,
    )[1:-1]
    low = torch.zeros_like(targets)
    high = torch.ones_like(targets)
    log_q = math.log(q)
    for _ in range(64):
        middle = (low + high) * 0.5
        alpha = 2.0 * middle.square() - middle.pow(3)
        sigma = 1.0 - middle - middle.square() + middle.pow(3)
        actual = log_q + 2.0 * (
            alpha.clamp_min(1e-30).log() - sigma.clamp_min(1e-30).log()
        )
        low = torch.where(actual < targets, middle, low)
        high = torch.where(actual < targets, high, middle)
    return torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.float64),
            (low + high) * 0.5,
            torch.ones(1, device=device, dtype=torch.float64),
        )
    ).to(torch.float32)


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
        key_lengths=conditioning.key_lengths,
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


def _unet_polar(
    unet,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    conditioning: ChimeraConditioning,
    time_ids: torch.Tensor,
    *,
    cache_metadata: tuple,
    prediction: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = sample.device
    dtype = sample.dtype
    _centered, radius, direction = polar_state(
        sample,
        timestep,
        latent_mean=prediction["latent_mean"],
        radius_floor=prediction["radius_floor"],
    )
    is_v4 = prediction.get("type") == FLOW_V4_PREDICTION
    encoder_hidden_states = conditioning.encoder_hidden_states.to(
        device=device, dtype=dtype
    )
    pooled_text_embeds = conditioning.pooled_text_embeds.to(
        device=device, dtype=dtype
    )
    reliability_key = None
    if is_v4:
        encoder_hidden_states = gate_conditioning_tensor(
            encoder_hidden_states, timestep, sample
        )
        pooled_text_embeds = gate_conditioning_tensor(
            pooled_text_embeds, timestep, sample
        )
        reliability = conditioning_reliability(timestep, sample)
        reliability_key = tuple(float(value) for value in reliability.flatten().tolist())
    context = ChimeraAttentionContext(
        context_positions=conditioning.context_positions.to(device=device),
        target_height=int(cache_metadata[0]),
        target_width=int(cache_metadata[1]),
        crop_top=int(cache_metadata[2]),
        crop_left=int(cache_metadata[3]),
        prefix_terminal_t=float(cache_metadata[4]),
        cache_key=(conditioning.fingerprint, reliability_key, *cache_metadata),
        key_lengths=conditioning.key_lengths,
    )
    set_chimera_attention_context(unet, context)
    expanded_timestep = timestep.to(device=device).expand(sample.shape[0])
    raw_tangent, radial = polar_unet_forward(
        unet,
        direction.to(dtype=dtype),
        expanded_timestep,
        log_radius=radius.clamp_min(prediction["radius_floor"]).log(),
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=conditioning.attention_mask.to(device=device),
        added_cond_kwargs={
            "text_embeds": pooled_text_embeds,
            "time_ids": time_ids.to(device=device, dtype=dtype),
        },
        return_dict=False,
    )
    tangent = polar_tangent_projection(raw_tangent, direction)
    return radial, tangent, direction, radius


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
    prediction_type: str = FLOW_V1_PREDICTION,
    analytic: torch.Tensor | None = None,
    prediction_conditional: torch.Tensor | None = None,
    prediction_unconditional: torch.Tensor | None = None,
    prediction_guided_raw: torch.Tensor | None = None,
    prediction_guided_post: torch.Tensor | None = None,
    bypassed_unet: bool = False,
    latent_mean: list[float] | None = None,
    x0_estimates: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    extra_metrics: dict[str, float] | None = None,
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
    delta = conditional - unconditional
    if x0_estimates is not None:
        x0_cond, x0_uncond, x0_raw = x0_estimates
    elif prediction_type in {FLOW_V2_PREDICTION, FLOW_V2_VELOCITY_PREDICTION}:
        mean = latent_mean or [0.0] * 4
        x0_cond = endpoint_observable_recover_clean(
            sample_before, conditional, t, latent_mean=mean
        )
        x0_uncond = endpoint_observable_recover_clean(
            sample_before, unconditional, t, latent_mean=mean
        )
        x0_raw = endpoint_observable_recover_clean(
            sample_before, guided_raw, t, latent_mean=mean
        )
    else:
        one_minus_t = 1.0 - t
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

    record = {
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
        "bypassed_unet": int(bypassed_unet),
    }
    if analytic is not None:
        record["analytic_velocity_rms"] = float(rms(analytic).item())
    prediction_values = {
        "prediction_cond_rms": prediction_conditional,
        "prediction_uncond_rms": prediction_unconditional,
        "prediction_guided_raw_rms": prediction_guided_raw,
        "prediction_guided_post_rms": prediction_guided_post,
    }
    for name, value in prediction_values.items():
        if value is not None:
            record[name] = float(rms(value).item())
    if prediction_conditional is not None and prediction_unconditional is not None:
        record["prediction_delta_rms"] = float(
            rms(prediction_conditional - prediction_unconditional).item()
        )
    if extra_metrics:
        record.update({key: float(value) for key, value in extra_metrics.items()})
    return record


def _polar_cfg_step(
    unet,
    positive: ChimeraConditioning,
    negative: ChimeraConditioning | None,
    *,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    next_timestep: torch.Tensor,
    time_ids: torch.Tensor,
    cache_metadata: tuple,
    prediction: dict,
    needs_cfg: bool,
    cfg_mode: str,
    cfg_scale: float,
    cfg_norm: str,
    collect_probe: bool,
) -> tuple[torch.Tensor, dict]:
    is_v4 = prediction.get("type") == FLOW_V4_PREDICTION
    analytic_noise_step = (
        (is_v4 or prediction.get("angular_schedule")
        == FLOW_V3_CONFIDENCE_ANGULAR_SCHEDULE)
        and bool(torch.all(timestep == 0).item())
    )
    if analytic_noise_step:
        _centered, radius, direction = polar_state(
            sample,
            timestep,
            latent_mean=prediction["latent_mean"],
            radius_floor=prediction["radius_floor"],
        )
        radial = -radius
        tangent = torch.zeros_like(sample)
        analytic = polar_compose_velocity(
            sample,
            radial,
            tangent,
            timestep,
            latent_mean=prediction["latent_mean"],
            radius_floor=prediction["radius_floor"],
        )
        solver = (
            mixed_coordinate_polar_exp_euler_step if is_v4 else polar_exp_euler_step
        )
        result = solver(
            sample,
            radial,
            tangent,
            timestep,
            next_timestep,
            latent_mean=prediction["latent_mean"],
            radius_floor=prediction["radius_floor"],
            angular_step_limit=prediction.get("angular_step_limit"),
        )
        if not collect_probe:
            return result.sample, {}
        if is_v4:
            recovered = destruction_coordinate_recover_clean(
                sample,
                radial,
                tangent,
                timestep,
                latent_mean=prediction["latent_mean"],
                latent_centered_second_moment=prediction[
                    "latent_centered_second_moment"
                ],
                radius_floor=prediction["radius_floor"],
            )
        else:
            recovered = polar_recover_clean(
                sample,
                radial,
                tangent,
                timestep,
                latent_mean=prediction["latent_mean"],
                latent_centered_second_moment=prediction[
                    "latent_centered_second_moment"
                ],
                angular_schedule=prediction["angular_schedule"],
                angular_endpoint_slope=prediction["angular_endpoint_slope"],
                radius_floor=prediction["radius_floor"],
            )
        radial_ratio = result.radius.float() / radius.float().clamp_min(
            float(prediction["radius_floor"])
        )
        return result.sample, {
            "conditional": analytic,
            "unconditional": analytic,
            "guided_raw": analytic,
            "guided_post": analytic,
            "prediction_conditional": tangent,
            "prediction_unconditional": tangent,
            "prediction_guided_raw": tangent,
            "prediction_guided_post": tangent,
            "x0_estimates": (recovered, recovered, recovered),
            "analytic": analytic,
            "bypassed_unet": True,
            "extra_metrics": {
                "radial_cond": float(radial.float().mean().item()),
                "radial_uncond": float(radial.float().mean().item()),
                "radial_anchor": float(radial.float().mean().item()),
                "radial_cfg_delta": 0.0,
                "tangent_orthogonality_abs_max": 0.0,
                "polar_radius_before": float(radius.float().mean().item()),
                "polar_radius_after": float(result.radius.float().mean().item()),
                "polar_radius_min_after": float(result.radius.float().min().item()),
                "radial_exponential_ratio_abs_max": float(
                    radial_ratio.abs().max().item()
                ),
                "angular_displacement_abs_max": 0.0,
                "angular_cap_scale_min": 1.0,
            },
        }
    if not needs_cfg:
        radial_cond, tangent_cond, direction, radius = _unet_polar(
            unet,
            sample,
            timestep,
            positive,
            time_ids,
            cache_metadata=cache_metadata,
            prediction=prediction,
        )
        radial_uncond = radial_cond
        tangent_uncond = tangent_cond
        tangent_raw = tangent_cond
        tangent_post = tangent_cond
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
            key_lengths=(
                *(negative_padded.key_lengths or (int(negative.encoder_hidden_states.shape[1]),)),
                *(positive_padded.key_lengths or (int(positive.encoder_hidden_states.shape[1]),)),
            ),
        )
        radial_pair, tangent_pair, direction_pair, radius_pair = _unet_polar(
            unet,
            sample.repeat(2, 1, 1, 1),
            timestep,
            combined,
            time_ids.repeat(2, 1),
            cache_metadata=cache_metadata,
            prediction=prediction,
        )
        radial_uncond, radial_cond = radial_pair.chunk(2)
        tangent_uncond, tangent_cond = tangent_pair.chunk(2)
        direction = direction_pair[: sample.shape[0]]
        radius = radius_pair[: sample.shape[0]]
        tangent_raw = tangent_uncond + float(cfg_scale) * (
            tangent_cond - tangent_uncond
        )
        tangent_post = _combine_cfg_velocity(
            tangent_cond, tangent_uncond, cfg_scale, cfg_norm
        )
    else:
        radial_uncond, tangent_uncond, direction, radius = _unet_polar(
            unet,
            sample,
            timestep,
            negative,
            time_ids,
            cache_metadata=cache_metadata,
            prediction=prediction,
        )
        radial_cond, tangent_cond, _direction_cond, _radius_cond = _unet_polar(
            unet,
            sample,
            timestep,
            positive,
            time_ids,
            cache_metadata=cache_metadata,
            prediction=prediction,
        )
        tangent_raw = tangent_uncond + float(cfg_scale) * (
            tangent_cond - tangent_uncond
        )
        tangent_post = _combine_cfg_velocity(
            tangent_cond, tangent_uncond, cfg_scale, cfg_norm
        )

    tangent_raw = polar_tangent_projection(tangent_raw, direction)
    tangent_post = polar_tangent_projection(tangent_post, direction)
    radial_anchor = radial_cond
    solver = mixed_coordinate_polar_exp_euler_step if is_v4 else polar_exp_euler_step
    result = solver(
        sample,
        radial_anchor,
        tangent_post,
        timestep,
        next_timestep,
        latent_mean=prediction["latent_mean"],
        radius_floor=prediction["radius_floor"],
        angular_step_limit=prediction.get("angular_step_limit"),
    )
    if not collect_probe:
        return result.sample, {}

    def full(radial, tangent):
        tangent_velocity = tangent
        if is_v4:
            _coordinate, coordinate_prime = destruction_coordinate(timestep, sample)
            tangent_velocity = coordinate_prime * tangent
        return polar_compose_velocity(
            sample,
            radial,
            tangent_velocity,
            timestep,
            latent_mean=prediction["latent_mean"],
            radius_floor=prediction["radius_floor"],
        )

    clean_kwargs = {
        "latent_mean": prediction["latent_mean"],
        "latent_centered_second_moment": prediction[
            "latent_centered_second_moment"
        ],
        "radius_floor": prediction["radius_floor"],
    }
    recover_clean = (
        destruction_coordinate_recover_clean if is_v4 else polar_recover_clean
    )
    if not is_v4:
        clean_kwargs.update({
            "angular_schedule": prediction["angular_schedule"],
            "angular_endpoint_slope": prediction["angular_endpoint_slope"],
        })
    x0_cond = recover_clean(
        sample, radial_cond, tangent_cond, timestep, **clean_kwargs
    )
    x0_uncond = recover_clean(
        sample, radial_uncond, tangent_uncond, timestep, **clean_kwargs
    )
    x0_raw = recover_clean(
        sample, radial_anchor, tangent_raw, timestep, **clean_kwargs
    )
    radial_ratio = result.radius.float() / radius.float().clamp_min(
        float(prediction["radius_floor"])
    )
    tangent_inner = (
        direction.float() * tangent_post.float()
    ).flatten(1).mean(1).abs()
    info = {
        "conditional": full(radial_cond, tangent_cond),
        "unconditional": full(radial_uncond, tangent_uncond),
        "guided_raw": full(radial_anchor, tangent_raw),
        "guided_post": full(radial_anchor, tangent_post),
        "prediction_conditional": tangent_cond,
        "prediction_unconditional": tangent_uncond,
        "prediction_guided_raw": tangent_raw,
        "prediction_guided_post": tangent_post,
        "x0_estimates": (x0_cond, x0_uncond, x0_raw),
        "analytic": None,
        "bypassed_unet": False,
        "extra_metrics": {
            "radial_cond": float(radial_cond.float().mean().item()),
            "radial_uncond": float(radial_uncond.float().mean().item()),
            "radial_anchor": float(radial_anchor.float().mean().item()),
            "radial_cfg_delta": 0.0,
            "tangent_orthogonality_abs_max": float(tangent_inner.max().item()),
            "polar_radius_before": float(radius.float().mean().item()),
            "polar_radius_after": float(result.radius.float().mean().item()),
            "polar_radius_min_after": float(result.radius.float().min().item()),
            "radial_exponential_ratio_abs_max": float(
                radial_ratio.abs().max().item()
            ),
            "angular_displacement_abs_max": float(
                result.angular_displacement.float().abs().max().item()
            ),
            "angular_cap_scale_min": float(
                result.angular_cap_scale.float().min().item()
            ),
        },
    }
    if is_v4:
        coordinate, _coordinate_prime = destruction_coordinate(timestep, sample)
        next_coordinate, _next_coordinate_prime = destruction_coordinate(
            next_timestep, sample
        )
        reliability = conditioning_reliability(timestep, sample)
        info["extra_metrics"].update({
            "destruction_coordinate": float(coordinate.float().mean().item()),
            "delta_d": float((next_coordinate - coordinate).float().mean().item()),
            "conditioning_reliability": float(reliability.float().mean().item()),
            "conditioned_tangent_delta_rms": float(
                (tangent_cond - tangent_uncond).float().square().mean().sqrt().item()
            ),
        })
    return result.sample, info


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
    prediction: dict | None = None,
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
    prediction = dict(prediction or {"type": FLOW_V1_PREDICTION})
    prediction_type = str(prediction.get("type") or FLOW_V1_PREDICTION)
    is_residual_v2 = prediction_type == FLOW_V2_PREDICTION
    is_direct_v2 = prediction_type == FLOW_V2_VELOCITY_PREDICTION
    is_v2 = is_residual_v2 or is_direct_v2
    is_v3 = prediction_type == FLOW_V3_PREDICTION
    is_v4 = prediction_type == FLOW_V4_PREDICTION
    is_polar = is_v3 or is_v4
    latent_mean = prediction.get("latent_mean")
    latent_moment = prediction.get("latent_centered_second_moment")
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
                if is_polar:
                    sample_before = sample
                    sample, polar_info = _polar_cfg_step(
                        unet,
                        positive,
                        negative,
                        sample=sample,
                        timestep=timestep,
                        next_timestep=times[index + 1],
                        time_ids=time_ids,
                        cache_metadata=cache_metadata,
                        prediction=prediction,
                        needs_cfg=needs_cfg,
                        cfg_mode=cfg_mode,
                        cfg_scale=cfg_now,
                        cfg_norm=cfg_norm,
                        collect_probe=cfg_probe_callback is not None,
                    )
                    if generate_mask is not None:
                        if is_v4:
                            source_at_next = destruction_coordinate_polar_flow_target(
                                source_latents,
                                source_noise,
                                times[index + 1],
                                latent_mean=latent_mean,
                                radius_floor=prediction["radius_floor"],
                                angular_singularity_threshold=prediction[
                                    "angular_singularity_threshold"
                                ],
                            ).sample
                        else:
                            source_at_next = polar_flow_target(
                                source_latents,
                                source_noise,
                                times[index + 1],
                                latent_mean=latent_mean,
                                angular_schedule=prediction["angular_schedule"],
                                angular_endpoint_slope=prediction[
                                    "angular_endpoint_slope"
                                ],
                                radius_floor=prediction["radius_floor"],
                                angular_singularity_threshold=prediction[
                                    "angular_singularity_threshold"
                                ],
                            ).sample
                        sample = (
                            generate_mask * sample
                            + (1.0 - generate_mask) * source_at_next
                        )
                    if cfg_probe_callback is not None:
                        cfg_probe_callback(_cfg_probe_record(
                            step=index + 1,
                            total_steps=steps,
                            timestep=times[index],
                            next_timestep=times[index + 1],
                            sample_before=sample_before,
                            sample_after=sample,
                            conditional=polar_info["conditional"],
                            unconditional=polar_info["unconditional"],
                            guided_raw=polar_info["guided_raw"],
                            guided_post=polar_info["guided_post"],
                            cfg_scale=cfg_now,
                            prediction_type=prediction_type,
                            prediction_conditional=polar_info[
                                "prediction_conditional"
                            ],
                            prediction_unconditional=polar_info[
                                "prediction_unconditional"
                            ],
                            prediction_guided_raw=polar_info[
                                "prediction_guided_raw"
                            ],
                            prediction_guided_post=polar_info[
                                "prediction_guided_post"
                            ],
                            analytic=polar_info["analytic"],
                            bypassed_unet=polar_info["bypassed_unet"],
                            latent_mean=latent_mean,
                            x0_estimates=polar_info["x0_estimates"],
                            extra_metrics=polar_info["extra_metrics"],
                        ))
                    if progress_callback is not None:
                        progress_callback(index + 1, steps, sample)
                    if step_progress_callback is not None:
                        step_progress_callback(index + 1, steps)
                    continue
                analytic_noise_step = (
                    index == 0
                    and float(times[index].item()) == 0.0
                    and not is_direct_v2
                    and not is_v3
                )
                analytic = None
                if analytic_noise_step:
                    # At the pure-noise endpoint the paired x0 is unobservable.
                    # Advance only the known -epsilon term before asking the U-Net.
                    velocity = -sample
                    if is_residual_v2:
                        analytic = velocity
                    prediction_cond = prediction_uncond = prediction_raw = prediction_post = (
                        torch.zeros_like(sample) if is_residual_v2 else velocity
                    )
                    if cfg_probe_callback is not None:
                        cond = uncond = guided_raw = velocity
                elif not needs_cfg:
                    prediction_post = _unet_velocity(
                        unet, sample, timestep, positive, time_ids, cache_metadata=cache_metadata
                    )
                    if is_residual_v2:
                        analytic, _ = endpoint_observable_preconditioning(
                            sample,
                            timestep,
                            latent_mean=latent_mean,
                            latent_centered_second_moment=latent_moment,
                        )
                        velocity = analytic + prediction_post
                    else:
                        velocity = prediction_post
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
                        key_lengths=(
                            *(negative_padded.key_lengths or (int(negative.encoder_hidden_states.shape[1]),)),
                            *(positive_padded.key_lengths or (int(positive.encoder_hidden_states.shape[1]),)),
                        ),
                    )
                    pair = _unet_velocity(
                        unet,
                        sample.repeat(2, 1, 1, 1),
                        timestep,
                        combined,
                        time_ids.repeat(2, 1),
                        cache_metadata=cache_metadata,
                    )
                    prediction_uncond, prediction_cond = pair.chunk(2)
                    prediction_raw = prediction_uncond + float(cfg_now) * (
                        prediction_cond - prediction_uncond
                    )
                    prediction_post = _combine_cfg_velocity(
                        prediction_cond, prediction_uncond, cfg_now, cfg_norm
                    )
                else:
                    prediction_uncond = _unet_velocity(
                        unet, sample, timestep, negative, time_ids, cache_metadata=cache_metadata
                    )
                    prediction_cond = _unet_velocity(
                        unet, sample, timestep, positive, time_ids, cache_metadata=cache_metadata
                    )
                    prediction_raw = prediction_uncond + float(cfg_now) * (
                        prediction_cond - prediction_uncond
                    )
                    prediction_post = _combine_cfg_velocity(
                        prediction_cond, prediction_uncond, cfg_now, cfg_norm
                    )
                if needs_cfg and not analytic_noise_step:
                    if is_residual_v2:
                        analytic, _ = endpoint_observable_preconditioning(
                            sample,
                            timestep,
                            latent_mean=latent_mean,
                            latent_centered_second_moment=latent_moment,
                        )
                        uncond = analytic + prediction_uncond
                        cond = analytic + prediction_cond
                        guided_raw = analytic + prediction_raw
                        velocity = analytic + prediction_post
                    else:
                        uncond = prediction_uncond
                        cond = prediction_cond
                        guided_raw = prediction_raw
                        velocity = prediction_post
                sample_before = sample
                sample = flow_euler_step(sample, velocity, times[index], times[index + 1])
                if generate_mask is not None:
                    source_at_next = (
                        endpoint_observable_noising(
                            source_latents, source_noise, times[index + 1]
                        )
                        if is_v2
                        else flow_noising(
                            source_latents, source_noise, times[index + 1]
                        )
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
                        prediction_type=prediction_type,
                        analytic=analytic,
                        prediction_conditional=prediction_cond,
                        prediction_unconditional=prediction_uncond,
                        prediction_guided_raw=prediction_raw,
                        prediction_guided_post=prediction_post,
                        bypassed_unet=analytic_noise_step,
                        latent_mean=latent_mean,
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
    prediction: dict | None = None,
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
        prediction=prediction,
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
    prediction: dict | None = None,
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
    prediction = dict(prediction or {"type": FLOW_V1_PREDICTION})
    is_v2 = prediction.get("type") in {
        FLOW_V2_PREDICTION,
        FLOW_V2_VELOCITY_PREDICTION,
    }
    is_v3 = prediction.get("type") == FLOW_V3_PREDICTION
    is_v4 = prediction.get("type") == FLOW_V4_PREDICTION
    if is_v4:
        sample = destruction_coordinate_polar_flow_target(
            source_latents,
            noise,
            times[start_index],
            latent_mean=prediction["latent_mean"],
            radius_floor=prediction["radius_floor"],
            angular_singularity_threshold=prediction[
                "angular_singularity_threshold"
            ],
        ).sample
    elif is_v3:
        sample = polar_flow_target(
            source_latents,
            noise,
            times[start_index],
            latent_mean=prediction["latent_mean"],
            angular_schedule=prediction["angular_schedule"],
            angular_endpoint_slope=prediction["angular_endpoint_slope"],
            radius_floor=prediction["radius_floor"],
            angular_singularity_threshold=prediction[
                "angular_singularity_threshold"
            ],
        ).sample
    elif is_v2:
        sample = endpoint_observable_noising(
            source_latents, noise, times[start_index]
        )
    else:
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
        prediction=prediction,
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
