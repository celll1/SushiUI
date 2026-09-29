"""Exact-resume checkpoints for FoveaTag stage-4 joint fine-tuning."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional

import torch
from torch import nn


JOINT_CHECKPOINT_FORMAT_VERSION = 1


def save_joint_checkpoint(
    path: str | Path,
    model: nn.Module,
    router: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Optional[Any],
    epoch: int,
    sample_offset: int,
    global_step: int,
    best_validation_f1: float,
    best_threshold: float,
    run_contract: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    epoch_accumulator: Optional[Mapping[str, Any]] = None,
) -> Path:
    if min(epoch, sample_offset, global_step) < 0:
        raise ValueError("joint checkpoint progress must be non-negative")
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError("joint checkpoint path must end in .pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": JOINT_CHECKPOINT_FORMAT_VERSION,
        "kind": "foveatag_joint_training",
        "epoch": int(epoch),
        "sample_offset": int(sample_offset),
        "global_step": int(global_step),
        "best_validation_f1": float(best_validation_f1),
        "best_threshold": float(best_threshold),
        "run_contract": _canonical_json(run_contract),
        "history": list(history),
        "epoch_accumulator": (
            dict(epoch_accumulator) if epoch_accumulator is not None else None
        ),
        "model": model.state_dict(),
        "router": router.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_joint_checkpoint(
    path: str | Path,
    model: nn.Module,
    router: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Optional[Any],
    expected_run_contract: Mapping[str, Any],
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != JOINT_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag joint checkpoint")
    if payload.get("kind") != "foveatag_joint_training":
        raise ValueError("checkpoint is not FoveaTag joint training")
    if payload.get("run_contract") != _canonical_json(expected_run_contract):
        raise ValueError("joint checkpoint run contract does not match")
    model.load_state_dict(payload["model"], strict=True)
    router.load_state_dict(payload["router"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _move_optimizer_state_to_parameters(optimizer)
    saved_scheduler = payload.get("scheduler")
    if (scheduler is None) != (saved_scheduler is None):
        raise ValueError("checkpoint and runtime disagree about scheduler state")
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)
    if restore_rng:
        random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return payload


def _move_optimizer_state_to_parameters(optimizer: torch.optim.Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(parameter.device)


def _canonical_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
