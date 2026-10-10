"""Private conditional-minus-null field alignment diagnostic."""
import math

import torch


def conditioning_losses(conditional, unconditional, teacher_conditional, teacher_unconditional, *, floor_fraction):
    if type(floor_fraction) not in (int, float) or not math.isfinite(floor_fraction) or not 0 < floor_fraction <= 1:
        raise ValueError("Contrast energy floor must lie in (0, 1]")
    tensors = (conditional, unconditional, teacher_conditional, teacher_unconditional)
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point() for value in tensors):
        raise ValueError("Conditioning alignment requires floating tensors")
    if (conditional.ndim < 2 or not all(conditional.shape)
            or any(value.shape != conditional.shape or value.device != conditional.device for value in tensors)):
        raise ValueError("Conditioning fields must have matching nonempty batch layouts and devices")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("Conditioning fields must be finite")
    sc, su = conditional.float(), unconditional.float()
    tc, tu = teacher_conditional.detach().float(), teacher_unconditional.detach().float()
    axes = tuple(range(1, sc.ndim))
    target_delta = tc-tu
    delta_energy = target_delta.square().mean(axes)
    common_energy = .5*(tc.square().mean(axes)+tu.square().mean(axes))
    denominator = torch.maximum(delta_energy, floor_fraction*common_energy).clamp_min(torch.finfo(torch.float32).eps)
    error = ((sc-su)-target_delta).square().mean(axes)
    return dict(anchor=.5*((sc-tc).square().mean()+(su-tu).square().mean()),
                contrast=(error/denominator).mean(), delta_energy=delta_energy.mean(),
                denominator=denominator.mean())
