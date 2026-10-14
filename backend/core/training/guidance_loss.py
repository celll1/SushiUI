"""Architecture-independent guidance-target loss (docs/decisions/GUIDANCE_LOSS_GENERALIZATION.md).

The guided target ``u + g * (v - u)`` is linear in the prediction, so it holds in
whatever space an architecture regresses (epsilon, v, flow velocity, x0). Callers
supply the prediction, its ordinary target, the no-gradient null-condition
prediction ``u`` and a noise level in [0, 1] (1 = pure noise).
"""
import math

import torch

from api.param_defaults import TRAINING_DEFAULTS

LEGACY_PREFIX = "qwen_guidance_loss_"
PREFIX = "guidance_loss_"
NAMES = ("weight", "mix_mode", "scale", "extrapolation_max_rms", "schedule",
         "weight_schedule", "high_noise_weight", "ramp_start", "ramp_end")


def setting(config, name: str):
    """Canonical ``guidance_loss_<name>``, else legacy ``qwen_guidance_loss_<name>``, else default."""
    canonical, legacy = PREFIX + name, LEGACY_PREFIX + name
    if config.get(canonical) is not None:
        return config[canonical]
    if config.get(legacy) is not None:
        return config[legacy]
    return TRAINING_DEFAULTS[canonical]


def resolve_weight_schedule(config, low_weight: float):
    kind = str(setting(config, "weight_schedule"))
    high = float(setting(config, "high_noise_weight"))
    start = float(setting(config, "ramp_start"))
    end = float(setting(config, "ramp_end"))
    if kind not in {"constant", "high_noise_smoothstep"}:
        raise ValueError(f"Invalid guidance weight schedule: {kind}")
    if not 0 <= high <= 1 or not 0 <= start < end <= 1:
        raise ValueError("Guidance weight ramp requires 0 <= high <= 1 and 0 <= start < end <= 1")
    if kind == "high_noise_smoothstep" and high < low_weight:
        raise ValueError("Guidance high-noise weight must be at least the low-noise weight")
    return kind, high, start, end


def mix_weight(sigma, low_weight, kind, high, start, end):
    if kind == "constant":
        return torch.full_like(sigma.float(), low_weight)
    t = ((sigma.float() - start) / (end - start)).clamp(0, 1)
    return low_weight + (high - low_weight) * t.square() * (3 - 2 * t)


def weight(config) -> float:
    value = float(setting(config, "weight") or 0.0)
    if not 0 <= value <= 1:
        raise ValueError("guidance_loss_weight must be between 0 and 1")
    return value


def scale_and_schedule(config):
    scale = float(setting(config, "scale"))
    schedule = str(setting(config, "schedule"))
    if not 1 <= scale <= 10 or schedule not in {"constant", "sigma"}:
        raise ValueError("Guidance loss requires scale in [1, 10] and schedule constant/sigma")
    return scale, schedule


def active(config) -> bool:
    """A positive weight, or a positive high-noise weight on the smoothstep ramp."""
    low = weight(config)
    kind, high, _, _ = resolve_weight_schedule(config, low)
    return low > 0 or (kind == "high_noise_smoothstep" and high > 0)


def mix_mode(config) -> str:
    mode = str(setting(config, "mix_mode"))
    if mode not in {"stochastic", "blend"}:
        raise ValueError(f"Invalid guidance loss mix mode: {mode}")
    return mode


def mix_for_condition(sigma, condition):
    _, _, eligible, low, _, _, *weight_schedule = condition
    kind, high, start, end = weight_schedule or ("constant", low, 0.5, 0.8)
    return mix_weight(sigma, low, kind, high, start, end) * eligible.float()


def selection(trainer, sigma, condition):
    probability = mix_for_condition(sigma, condition)
    mode = str(setting(trainer.config, "mix_mode")) if _has_mode(trainer.config) else "blend"
    if mode == "stochastic":
        draw = getattr(trainer, "_guidance_draw", None)
        if draw is None:
            draw = torch.rand_like(probability)
        selected = (draw.to(device=probability.device) < probability) & condition[2]
        return selected.float(), selected
    if mode == "blend":
        return probability, probability > 0
    raise ValueError(f"Invalid guidance loss mix mode: {mode}")


def _has_mode(config) -> bool:
    # A saved YAML without a mix-mode key predates stochastic selection and keeps blend.
    return config.get(PREFIX + "mix_mode") is not None or config.get(LEGACY_PREFIX + "mix_mode") is not None


def condition(config, eligible, null_features=None, null_mask=None):
    """Per-batch guidance condition tuple, or None when nothing is guided.

    Layout (kept from the Qwen implementation): (null_features, null_mask, eligible,
    weight, scale, schedule, kind, high, start, end).
    """
    low = weight(config)
    weight_schedule = resolve_weight_schedule(config, low)
    if not bool(eligible.any()):
        return None
    if low == 0 and not (weight_schedule[0] == "high_noise_smoothstep" and weight_schedule[1] > 0):
        return None
    scale, schedule = scale_and_schedule(config)
    return (null_features, null_mask, eligible, low, scale, schedule, *weight_schedule)


def extrapolation_cap(config) -> float:
    cap = float(setting(config, "extrapolation_max_rms"))
    if not math.isfinite(cap) or cap < 0:
        raise ValueError("Guidance extrapolation RMS cap must be finite and nonnegative")
    return cap


def base_scale(sigma, condition):
    scale, schedule = condition[4:6]
    return (torch.full_like(sigma, scale, dtype=torch.float32) if schedule == "constant"
            else 1 + (scale - 1) * sigma.float())


@torch.no_grad()
def capped_scale(sigma, condition, distance_squared, cap):
    distance = distance_squared.detach().float().clamp_min(0).sqrt()
    return 1 + torch.minimum(base_scale(sigma, condition) - 1,
                             cap / distance.clamp_min(torch.finfo(torch.float32).tiny))


def log_extrapolation_cap(trainer, sigma, condition, selected, distance_squared, effective, prefix):
    base = base_scale(sigma, condition)
    values = torch.stack([
        effective[selected].mean(), effective[selected].min(),
        (effective[selected] < base[selected]).float().mean(),
        distance_squared[selected].clamp_min(0).sqrt().mean(),
    ]).detach().tolist()
    for name, value in zip(("effective_scale_mean", "effective_scale_min", "capped_fraction", "null_clean_rms"), values):
        trainer.log_extra_metric(f"{prefix}_{name}", value)


def objective(prediction, target, uncond_prediction, sigma, condition, mix=None, sample_losses=None,
              guide_scale_override=None):
    """Mixed ordinary/guided per-image MSE; returns (loss, mean ordinary, mean guided over used rows)."""
    _, _, eligible, *_ = condition
    guide_scale = base_scale(sigma, condition) if guide_scale_override is None else guide_scale_override.detach()
    while isinstance(guide_scale, torch.Tensor) and guide_scale.ndim < target.ndim:
        guide_scale = guide_scale.unsqueeze(-1)
    guided_target = uncond_prediction.float() + guide_scale * (target.float() - uncond_prediction.float())
    normal = (prediction.float() - target.float()).square().flatten(1).mean(1)
    guided = (prediction.float() - guided_target).square().flatten(1).mean(1)
    mix = (mix_for_condition(sigma, condition) if mix is None else mix).to(dtype=normal.dtype)
    per_image = (1 - mix) * normal + mix * guided
    if sample_losses is not None:
        sample_losses.append(per_image.detach())
    used = eligible & (mix > 0)
    guided_metric = guided[used].mean() if bool(used.any()) else guided.new_zeros(())
    return per_image.mean(), normal.mean().detach(), guided_metric.detach()
