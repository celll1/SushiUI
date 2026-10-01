"""Native Ming flow objective; trainer orchestration owns optimization/state."""

from dataclasses import dataclass

import torch

from .conditioning import MingConditioning
from .flow import flow_training_pair


@dataclass
class MingTrainingResult:
    loss: torch.Tensor
    per_item_loss: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    model_time: torch.Tensor


def optimizer_groups(transformer, conditioner, *, dit_lr: float, conditioner_lrs: dict):
    """Collect each trainable parameter once with explicit component rates."""
    modules = {"dit": (transformer,), **conditioner.component_groups()}
    rates = {"dit": dit_lr, **conditioner_lrs}
    if set(conditioner_lrs) - (modules.keys() - {"dit"}):
        raise ValueError("Unknown Ming conditioner learning-rate group")
    result, seen = [], set()
    for name, parts in modules.items():
        parameters = []
        for part in parts:
            for parameter in part.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        if not parameters:
            continue
        rate = rates.get(name)
        if not isinstance(rate, (float, int)) or isinstance(rate, bool) or not 0 <= rate < float("inf"):
            raise ValueError(f"Ming trainable component requires a nonnegative finite rate: {name}")
        result.append({"name": name, "params": parameters, "lr": rate})
    if not result:
        raise ValueError("Ming training requires at least one trainable component")
    return result


def flow_loss(transformer, *, clean: torch.Tensor, noise: torch.Tensor,
              sigma: torch.Tensor, conditioning: MingConditioning,
              frame_weights: torch.Tensor, reference_latents=None, cfg_drop_mask=None):
    """References condition attention; only explicitly supervised targets incur loss."""
    if clean.ndim != 5 or clean.shape[1] != transformer.in_channels:
        raise ValueError("Ming training targets must be [B, C, F, H, W] native latents")
    batch, _, frames, height, width = clean.shape
    if min(batch, frames, height, width) < 1 or height % 2 or width % 2:
        raise ValueError("Ming targets require nonempty, patch-aligned native grids")
    if not transformer.config.multi_frame_output and frames != 1:
        raise ValueError("Ming Design training requires exactly one target frame")
    if transformer.config.multi_frame_output and frames < 2:
        raise ValueError("Ming Layer training requires auxiliary canvas plus ordered layers")
    if sigma.shape != (batch,) or not torch.isfinite(sigma).all() or ((sigma < 0) | (sigma > 1)).any():
        raise ValueError("Ming training sigma must be finite and within [0, 1]")
    if frame_weights.shape != (batch, frames) or not torch.isfinite(frame_weights).all() or (frame_weights < 0).any():
        raise ValueError("Ming supervision weights must be finite nonnegative [B, F]")
    if (frame_weights.sum(dim=1) <= 0).any():
        raise ValueError("Each Ming training item requires a supervised target frame")
    if (conditioning.queries.ndim != 3 or conditioning.direct.ndim != 3
            or conditioning.queries.shape[0] != batch or conditioning.direct.shape[0] != batch
            or conditioning.queries.shape[2] != transformer.config.cap_feat_dim
            or conditioning.direct.shape[2] != transformer.config.dim):
        raise ValueError("Ming conditioning geometry disagrees with the native checkpoint")
    if cfg_drop_mask is not None:
        if cfg_drop_mask.dtype != torch.bool or cfg_drop_mask.shape != (batch,):
            raise ValueError("Ming CFG dropout mask must be boolean [B]")
        keep = (~cfg_drop_mask).to(device=clean.device).reshape(batch, 1, 1)
        conditioning = MingConditioning(conditioning.queries * keep, conditioning.direct * keep)
    if reference_latents is not None:
        if len(reference_latents) != batch:
            raise ValueError("Ming references require one native tensor per training item")
        for reference in reference_latents:
            if reference is None:
                continue
            if (reference.ndim != 4 or reference.shape[0] != clean.shape[1]
                    or reference.shape[1] < 1 or reference.shape[2:] != (height, width)):
                raise ValueError("Ming references must preserve native channel/frame/grid axes")
    noisy, target, time = flow_training_pair(clean, noise, sigma)
    outputs = transformer(x=list(noisy.to(clean.dtype).unbind()), t=time,
        cap_feats=list(conditioning.queries.unbind()), cap_feats_2=list(conditioning.direct.unbind()),
        ref_x=reference_latents, return_dict=False)[0]
    prediction = torch.stack([value[:, :frames] for value in outputs])
    if prediction.shape != target.shape:
        raise ValueError("Ming prediction failed target-only reconstruction")
    per_frame = (prediction.float() - target.float()).square().mean(dim=(1, 3, 4))
    weights = frame_weights.to(device=per_frame.device, dtype=torch.float32)
    per_item = (per_frame * weights).sum(dim=1) / weights.sum(dim=1)
    return MingTrainingResult(per_item.mean(), per_item, prediction, target, time)
