"""Compact resumable and inference-only artifacts for FoveaTag routing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional

import torch
from safetensors.torch import load_file, save_file
from torch import nn


ROUTER_CHECKPOINT_FORMAT_VERSION = 1


def save_router_checkpoint(
    path: str | Path,
    router: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Optional[Any],
    epoch: int,
    sample_offset: int,
    global_step: int,
    best_utility_recovery: float,
    run_contract: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    epoch_accumulator: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save router state without duplicating the frozen stage-2 tagger."""
    if min(epoch, sample_offset, global_step) < 0:
        raise ValueError("checkpoint progress must be non-negative")
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError("router checkpoint path must end in .pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": ROUTER_CHECKPOINT_FORMAT_VERSION,
        "kind": "foveatag_router_training",
        "epoch": int(epoch),
        "sample_offset": int(sample_offset),
        "global_step": int(global_step),
        "best_utility_recovery": float(best_utility_recovery),
        "run_contract": _canonical_json(run_contract),
        "history": list(history),
        "epoch_accumulator": (
            dict(epoch_accumulator) if epoch_accumulator is not None else None
        ),
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


def load_router_checkpoint(
    path: str | Path,
    router: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Optional[Any],
    expected_run_contract: Mapping[str, Any],
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != ROUTER_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag router checkpoint")
    if payload.get("kind") != "foveatag_router_training":
        raise ValueError("checkpoint is not FoveaTag router training")
    if payload.get("run_contract") != _canonical_json(expected_run_contract):
        raise ValueError("router checkpoint run contract does not match")
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


def save_router_weights(
    path: str | Path,
    router: nn.Module,
    *,
    run_contract: Mapping[str, Any],
    epoch: int,
    global_step: int,
    validation: Mapping[str, Any],
) -> Path:
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("router weights path must end in .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in router.state_dict().items()
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    save_file(state, str(temporary))
    os.replace(temporary, path)
    _write_json_atomic(
        path.with_name(f"{path.stem}_metadata.json"),
        {
            "format_version": ROUTER_CHECKPOINT_FORMAT_VERSION,
            "kind": "foveatag_router_weights",
            "run_contract": _canonical_json(run_contract),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "validation": dict(validation),
        },
    )
    return path


def load_router_weights(
    path: str | Path,
    router: nn.Module,
    *,
    expected_run_contract: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    path = Path(path)
    metadata = json.loads(
        path.with_name(f"{path.stem}_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("format_version") != ROUTER_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag router weights")
    if metadata.get("kind") != "foveatag_router_weights":
        raise ValueError("artifact is not FoveaTag router weights")
    if (
        expected_run_contract is not None
        and metadata.get("run_contract") != _canonical_json(expected_run_contract)
    ):
        raise ValueError("router weights run contract does not match")
    router.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return metadata


def _move_optimizer_state_to_parameters(optimizer: torch.optim.Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(parameter.device)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
