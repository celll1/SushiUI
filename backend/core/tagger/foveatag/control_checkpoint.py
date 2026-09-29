"""Epoch-boundary checkpoints for the Phase-1 SigLIP2 control."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn


CONTROL_CHECKPOINT_FORMAT_VERSION = 1
_ENCODER_PREFIX = "vision_encoder."


def save_control_checkpoint(
    path: os.PathLike[str] | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    next_epoch: int,
    global_step: int,
    encoder_provenance: Mapping[str, Any],
    scheduler: Optional[Any] = None,
    best_validation_f1: float = 0.0,
    best_threshold: float = 0.5,
) -> Path:
    """Atomically save trainable control state without duplicating the encoder."""
    if next_epoch < 0 or global_step < 0:
        raise ValueError("next_epoch and global_step must be non-negative")
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError("control checkpoint path must end in .pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CONTROL_CHECKPOINT_FORMAT_VERSION,
        "kind": "foveatag_siglip2_control",
        "next_epoch": int(next_epoch),
        "global_step": int(global_step),
        "best_validation_f1": float(best_validation_f1),
        "best_threshold": float(best_threshold),
        "encoder_provenance": dict(encoder_provenance),
        "control_model": _control_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_control_checkpoint(
    path: os.PathLike[str] | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_encoder_provenance: Optional[Mapping[str, Any]] = None,
    scheduler: Optional[Any] = None,
) -> dict[str, Any]:
    """Restore a control at an epoch boundary and verify its external encoder."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CONTROL_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag control checkpoint")
    if payload.get("kind") != "foveatag_siglip2_control":
        raise ValueError("checkpoint is not a FoveaTag SigLIP2 control")
    saved_provenance = payload.get("encoder_provenance")
    if not isinstance(saved_provenance, Mapping):
        raise ValueError("control checkpoint has no encoder provenance")
    if (
        expected_encoder_provenance is not None
        and dict(saved_provenance) != dict(expected_encoder_provenance)
    ):
        raise ValueError("control checkpoint encoder provenance does not match")

    saved_state = payload.get("control_model")
    if not isinstance(saved_state, Mapping):
        raise ValueError("control checkpoint has no model state")
    expected_keys = {
        name for name in model.state_dict() if not name.startswith(_ENCODER_PREFIX)
    }
    if set(saved_state) != expected_keys:
        missing = sorted(expected_keys - set(saved_state))
        unexpected = sorted(set(saved_state) - expected_keys)
        raise ValueError(
            f"control checkpoint state mismatch; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    incompatible = model.load_state_dict(saved_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_encoder_missing = [
        name for name in incompatible.missing_keys if not name.startswith(_ENCODER_PREFIX)
    ]
    if unexpected or non_encoder_missing:
        raise ValueError("control checkpoint could not be loaded strictly")

    optimizer.load_state_dict(payload["optimizer"])
    _move_optimizer_state_to_parameters(optimizer)
    saved_scheduler = payload.get("scheduler")
    if scheduler is None and saved_scheduler is not None:
        raise ValueError("checkpoint contains scheduler state but no scheduler was supplied")
    if scheduler is not None:
        if saved_scheduler is None:
            raise ValueError("scheduler was supplied but checkpoint has no scheduler state")
        scheduler.load_state_dict(saved_scheduler)
    return payload


def _control_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if not name.startswith(_ENCODER_PREFIX)
    }
    if not state:
        raise ValueError("control model has no non-encoder state")
    return state


def _move_optimizer_state_to_parameters(optimizer: torch.optim.Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(parameter.device)
