"""Transactional added-module fitting; architecture owners supply native inputs."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math

import torch

from .initialization import frozen_native_teacher
from .provenance import validate_record
from .spec import canonical_json


class InitializationCancelled(RuntimeError):
    pass


@dataclass
class FitExample:
    args: tuple
    kwargs: dict
    target: torch.Tensor | None = None


@dataclass
class FitPhase:
    kind: str
    updates: int
    learning_rate: float
    examples: object


def weights_digest(module):
    return state_digest(module.state_dict())


def state_digest(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        header = canonical_json(dict(name=name, dtype=str(value.dtype), shape=list(value.shape)))
        result.update(header.encode("utf-8") + b"\0")
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def fit_teacher_modules(model, phases, predict, optimizer_factory, *, source_sha256,
                        calibration_sha256, sample_count, seed, clip_norm,
                        progress=None, cancelled=None, recipe=None):
    """Fit only new weights; preserve native ownership and roll back failed fits.

    Residency, input/noise RNG, encoding and output publication belong to the caller.
    ``predict`` extracts a tensor from the architecture's native return type.
    Each phase starts a fresh optimizer, matching the tested initialization recipe.
    """
    phases = tuple(phases)
    if any(not isinstance(phase, FitPhase) for phase in phases):
        raise ValueError("Initialization requires FitPhase declarations")
    record = dict(schema_version=1, method="native_teacher_velocity",
        source_sha256=source_sha256, calibration_sha256=calibration_sha256,
        initialized_weights_sha256="0"*64, origin_spec=model._residual_unet_spec,
        sample_count=sample_count, seed=seed,
        phases=[dict(kind=phase.kind, updates=phase.updates, learning_rate=phase.learning_rate)
                for phase in phases])
    if recipe is not None:
        record["recipe"] = deepcopy(recipe)
    record = validate_record(record, model._residual_unet_spec)
    if type(clip_norm) not in (int, float) or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("Initialization gradient clipping must be finite and positive")
    total = sum(phase.updates for phase in phases)
    completed, snapshot = 0, None

    def check_cancel():
        if cancelled is not None and cancelled():
            raise InitializationCancelled("Hybrid initialization cancelled")

    try:
        with frozen_native_teacher(model) as teacher:
            snapshot = {name: value.detach().cpu().clone()
                        for name, value in model.rewiring_modules.state_dict().items()}
            check_cancel()
            for phase in phases:
                optimizer = optimizer_factory(teacher.parameters, phase.learning_rate)
                if not isinstance(optimizer, torch.optim.Optimizer) or optimizer.state:
                    raise ValueError("Initialization requires a fresh optimizer")
                owned = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
                if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in teacher.parameters}:
                    raise ValueError("Initialization optimizer must own exactly the added parameters")
                if any(group["lr"] != phase.learning_rate for group in optimizer.param_groups):
                    raise ValueError("Initialization optimizer disagrees with the phase learning rate")
                examples = iter(phase.examples)
                for index in range(phase.updates):
                    check_cancel()
                    try:
                        example = next(examples)
                    except StopIteration as exc:
                        raise ValueError("Initialization phase ended before its declared updates") from exc
                    if not isinstance(example, FitExample):
                        raise ValueError("Initialization requires native FitExample inputs")
                    optimizer.zero_grad(set_to_none=True)
                    target = example.target
                    if target is None:
                        if phase.kind != "data_noising":
                            raise ValueError("Teacher trajectory requires captured native velocity targets")
                        target = predict(teacher, example.args, example.kwargs)
                    prediction = predict(model, example.args, example.kwargs)
                    if (not isinstance(target, torch.Tensor) or not isinstance(prediction, torch.Tensor)
                            or target.shape != prediction.shape or target.device != prediction.device):
                        raise ValueError("Initialization target and prediction must match exactly in shape and device")
                    loss = (prediction.float() - target.detach().float()).square().mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite hybrid initialization loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(teacher.parameters, clip_norm, error_if_nonfinite=True)
                    optimizer.step()
                    if not torch.stack([torch.isfinite(parameter).all() for parameter in teacher.parameters]).all():
                        raise ValueError("Non-finite fitted hybrid weights")
                    completed += 1
                    if progress is not None:
                        progress(dict(completed=completed, total=total, phase=phase.kind,
                                      phase_update=index+1, loss=float(loss.detach())))
                del optimizer
            check_cancel()
            record["initialized_weights_sha256"] = weights_digest(model.rewiring_modules)
        model._hybrid_initialization = deepcopy(record)
        return deepcopy(record)
    except BaseException:
        if snapshot is not None:
            model.rewiring_modules.load_state_dict(snapshot, strict=True)
        raise
