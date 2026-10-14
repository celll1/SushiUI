"""Architecture-independent guidance-target loss (docs/decisions/GUIDANCE_LOSS_GENERALIZATION.md).

The guided target ``u + g * (v - u)`` is linear in the prediction, so it holds in
whatever space an architecture regresses (epsilon, v, flow velocity, x0). Callers
supply the prediction, its ordinary target, the no-gradient null-condition
prediction ``u`` and a noise level in [0, 1] (1 = pure noise).
"""
import math
from typing import NamedTuple

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


class Plan(NamedTuple):
    condition: tuple
    mix: torch.Tensor
    selected: torch.Tensor


def plan(trainer, sigma, cfg_drop_mask=None):
    """Per-row guidance choice for one step, or None when nothing is guided.

    Rows drawn for the CFG null keep ordinary MSE.
    """
    eligible = torch.ones(sigma.shape[0], dtype=torch.bool, device=sigma.device)
    if cfg_drop_mask is not None:
        eligible &= ~cfg_drop_mask.to(device=sigma.device, dtype=torch.bool)
    guided = condition(trainer.config, eligible)
    if guided is None:
        return None
    mix, selected = selection(trainer, sigma, guided)
    return Plan(guided, mix, selected)


def null_encoding(trainer):
    """The run's cached inference null-condition encoding (CPU)."""
    cached = getattr(trainer, "_guidance_null_encoding", None)
    if cached is None:
        raise RuntimeError("Guidance loss requires the cached null-condition encoding")
    return cached


def null_rows(trainer, count, device, dtype):
    """The cached null encoding repeated for ``count`` rows on ``device``.

    The embedding is cast to ``dtype``; the auxiliary (tensor, dict of tensors
    or None) keeps its dtypes, as the caption batch does. Like the caption
    collation, a 2-D embedding or 1-D auxiliary gains the batch axis and
    anything else already carries a leading batch of one.
    """
    def rows(value, cast, unbatched_ndim):
        if isinstance(value, dict):
            return {key: rows(item, False, unbatched_ndim) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            value = value.to(device=device, dtype=dtype if cast else value.dtype)
            if value.dim() == unbatched_ndim:
                value = value.unsqueeze(0)
            return value.expand(count, *value.shape[1:])
        return value
    embedding, auxiliary = null_encoding(trainer)
    return rows(embedding, True, 2), rows(auxiliary, False, 1)


@torch.no_grad()
def null_prediction(step_plan, like, forward):
    """``forward(indices)`` predicts the null condition for the selected rows only.

    Returns a tensor shaped like ``like`` (zero on unselected rows), or None
    when no row is selected, so an unguided step costs no extra forward.
    """
    if not bool(step_plan.selected.any()):
        return None
    indices = step_plan.selected.nonzero(as_tuple=True)[0]
    selected = forward(indices)
    out = torch.zeros(like.shape, dtype=selected.dtype, device=selected.device)
    out[indices] = selected
    return out


def loss(trainer, step_plan, prediction, target, uncond_prediction, sigma, sample_losses=None):
    """Mixed loss with the optional extrapolation cap, logging guidance metrics.

    Returns None when no row was selected; the caller keeps its ordinary MSE.
    ``sigma`` is the noise level, and the cap's clean-space distance is
    ``sigma * (target - u)``, which holds for flow velocity targets.
    """
    trainer.log_extra_metric(PREFIX + "mix_weight", float(mix_for_condition(sigma, step_plan.condition).mean()))
    trainer.log_extra_metric(PREFIX + "selected_fraction", float(step_plan.selected.float().mean()))
    if uncond_prediction is None:
        return None
    effective = None
    cap = extrapolation_cap(trainer.config)
    if cap > 0:
        level = sigma.float().view(-1, *([1] * (target.ndim - 1)))
        distance_squared = (level * (target.float() - uncond_prediction.float())).square().flatten(1).mean(1)
        effective = capped_scale(sigma, step_plan.condition, distance_squared, cap)
        log_extrapolation_cap(trainer, sigma, step_plan.condition, step_plan.selected, distance_squared,
                              effective, prefix=PREFIX.rstrip("_"))
    value, normal, guided = objective(prediction, target, uncond_prediction, sigma, step_plan.condition,
                                      mix=step_plan.mix, sample_losses=sample_losses,
                                      guide_scale_override=effective)
    trainer.log_extra_metric(PREFIX + "normal", float(normal))
    trainer.log_extra_metric(PREFIX + "guided", float(guided))
    return value


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
