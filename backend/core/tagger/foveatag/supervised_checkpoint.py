"""Shared initialization and resumable Phase-3 FoveaTag artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional

import torch
from safetensors.torch import load_file, save_file
from torch import nn


SUPERVISED_CHECKPOINT_FORMAT_VERSION = 1


def save_supervised_initialization(
    path: str | Path,
    model: nn.Module,
    *,
    initialization_contract: Mapping[str, Any],
) -> Path:
    """Freeze the exact common tensor state from which A/B/C must branch."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("supervised initialization path must end in .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    save_file(state, str(temporary))
    os.replace(temporary, path)
    _write_json_atomic(
        _metadata_path(path),
        {
            "format_version": SUPERVISED_CHECKPOINT_FORMAT_VERSION,
            "kind": "foveatag_supervised_shared_initialization",
            "initialization_contract": dict(initialization_contract),
        },
    )
    return path


def load_supervised_initialization(
    path: str | Path,
    model: nn.Module,
    *,
    expected_initialization_contract: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    metadata = json.loads(_metadata_path(path).read_text(encoding="utf-8"))
    if metadata.get("format_version") != SUPERVISED_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag supervised initialization")
    if metadata.get("kind") != "foveatag_supervised_shared_initialization":
        raise ValueError("artifact is not a FoveaTag supervised initialization")
    if metadata.get("initialization_contract") != dict(expected_initialization_contract):
        raise ValueError("supervised initialization contract does not match")
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return metadata


def save_supervised_checkpoint(
    path: str | Path,
    model: nn.Module,
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
        raise ValueError("checkpoint progress must be non-negative")
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError("supervised checkpoint path must end in .pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": SUPERVISED_CHECKPOINT_FORMAT_VERSION,
        "kind": "foveatag_supervised_training",
        "epoch": int(epoch),
        "sample_offset": int(sample_offset),
        "global_step": int(global_step),
        "best_validation_f1": float(best_validation_f1),
        "best_threshold": float(best_threshold),
        "run_contract": dict(run_contract),
        "history": list(history),
        "epoch_accumulator": dict(epoch_accumulator) if epoch_accumulator is not None else None,
        "model": model.state_dict(),
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


def load_supervised_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Optional[Any],
    expected_run_contract: Mapping[str, Any],
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != SUPERVISED_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag supervised checkpoint")
    if payload.get("kind") != "foveatag_supervised_training":
        raise ValueError("checkpoint is not FoveaTag supervised training")
    if payload.get("run_contract") != dict(expected_run_contract):
        raise ValueError("supervised checkpoint run contract does not match")
    model.load_state_dict(payload["model"], strict=True)
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


def _metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metadata.json")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
