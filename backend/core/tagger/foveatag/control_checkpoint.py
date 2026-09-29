"""Epoch-boundary checkpoints for the Phase-1 SigLIP2 control."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from safetensors.torch import load_file, save_file
from torch import nn


CONTROL_CHECKPOINT_FORMAT_VERSION = 1
_EXTERNAL_PREFIXES = ("vision_encoder.", "baseline_head.")


def save_control_checkpoint(
    path: os.PathLike[str] | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    next_epoch: int,
    global_step: int,
    encoder_provenance: Mapping[str, Any],
    run_contract: Optional[Mapping[str, Any]] = None,
    scheduler: Optional[Any] = None,
    best_validation_f1: float = 0.0,
    best_threshold: float = 0.5,
) -> Path:
    """Save trainable state without duplicating the external encoder or baseline head."""
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
        "run_contract": dict(run_contract) if run_contract is not None else None,
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
    expected_run_contract: Optional[Mapping[str, Any]] = None,
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
    if expected_run_contract is not None:
        saved_contract = payload.get("run_contract")
        if not isinstance(saved_contract, Mapping):
            raise ValueError("control checkpoint has no run contract")
        if dict(saved_contract) != dict(expected_run_contract):
            raise ValueError("control checkpoint run contract does not match")

    saved_state = payload.get("control_model")
    if not isinstance(saved_state, Mapping):
        raise ValueError("control checkpoint has no model state")
    _load_control_state(model, saved_state)

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


def save_control_weights(
    path: os.PathLike[str] | str,
    model: nn.Module,
    *,
    encoder_provenance: Mapping[str, Any],
    run_contract: Optional[Mapping[str, Any]] = None,
    epoch: int,
    global_step: int,
    validation_f1: float,
    threshold: float,
) -> Path:
    """Save the best inference-time control state without optimizer tensors."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("control weights path must end in .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    save_file(_control_state(model), str(temporary))
    os.replace(temporary, path)
    _write_json(
        _metadata_path(path),
        {
            "format_version": CONTROL_CHECKPOINT_FORMAT_VERSION,
            "kind": "foveatag_siglip2_control_weights",
            "encoder_provenance": dict(encoder_provenance),
            "run_contract": dict(run_contract) if run_contract is not None else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "validation_f1": float(validation_f1),
            "threshold": float(threshold),
        },
    )
    return path


def load_control_weights(
    path: os.PathLike[str] | str,
    model: nn.Module,
    *,
    expected_encoder_provenance: Optional[Mapping[str, Any]] = None,
    expected_run_contract: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    path = Path(path)
    metadata = _read_json(_metadata_path(path))
    if metadata.get("format_version") != CONTROL_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag control weights")
    if metadata.get("kind") != "foveatag_siglip2_control_weights":
        raise ValueError("artifact is not FoveaTag SigLIP2 control weights")
    saved_provenance = metadata.get("encoder_provenance")
    if (
        expected_encoder_provenance is not None
        and dict(saved_provenance or {}) != dict(expected_encoder_provenance)
    ):
        raise ValueError("control weights encoder provenance does not match")
    if expected_run_contract is not None:
        saved_contract = metadata.get("run_contract")
        if not isinstance(saved_contract, Mapping):
            raise ValueError("control weights have no run contract")
        if dict(saved_contract) != dict(expected_run_contract):
            raise ValueError("control weights run contract does not match")
    state = load_file(str(path), device="cpu")
    _load_control_state(model, state)
    return metadata


def _control_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if not name.startswith(_EXTERNAL_PREFIXES)
    }
    if not state:
        raise ValueError("control model has no non-encoder state")
    return state


def _load_control_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected_keys = {
        name for name in model.state_dict() if not name.startswith(_EXTERNAL_PREFIXES)
    }
    if set(state) != expected_keys:
        missing = sorted(expected_keys - set(state))
        unexpected = sorted(set(state) - expected_keys)
        raise ValueError(
            f"control weights state mismatch; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_encoder_missing = [
        name for name in incompatible.missing_keys if not name.startswith(_EXTERNAL_PREFIXES)
    ]
    if unexpected or non_encoder_missing:
        raise ValueError("control weights could not be loaded strictly")


def _move_optimizer_state_to_parameters(optimizer: torch.optim.Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(parameter.device)


def _metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metadata.json")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    import json

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
