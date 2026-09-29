"""Atomic resumable and encoder-only artifacts for FoveaTag Phase 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional

import torch
from safetensors.torch import load_file, save_file
from torch import nn


PRETRAINING_CHECKPOINT_FORMAT_VERSION = 1


def save_pretraining_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Optional[Any],
    epoch: int,
    sample_offset: int,
    global_step: int,
    best_validation_loss: float,
    run_contract: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    epoch_accumulator: Optional[Mapping[str, Any]] = None,
) -> Path:
    if min(epoch, sample_offset, global_step) < 0:
        raise ValueError("checkpoint progress must be non-negative")
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError("pretraining checkpoint path must end in .pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": PRETRAINING_CHECKPOINT_FORMAT_VERSION,
        "kind": "foveatag_scratch_pretraining",
        "epoch": int(epoch),
        "sample_offset": int(sample_offset),
        "global_step": int(global_step),
        "best_validation_loss": float(best_validation_loss),
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


def load_pretraining_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Optional[Any],
    expected_run_contract: Mapping[str, Any],
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != PRETRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag pretraining checkpoint")
    if payload.get("kind") != "foveatag_scratch_pretraining":
        raise ValueError("checkpoint is not FoveaTag scratch pretraining")
    if payload.get("run_contract") != dict(expected_run_contract):
        raise ValueError("pretraining checkpoint run contract does not match")
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


def save_pretrained_encoder(
    path: str | Path,
    encoder: nn.Module,
    *,
    run_contract: Mapping[str, Any],
    epoch: int,
    global_step: int,
    validation: Mapping[str, Any],
) -> Path:
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("pretrained encoder path must end in .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in encoder.state_dict().items()
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    save_file(state, str(temporary))
    os.replace(temporary, path)
    _write_json_atomic(
        path.with_name(f"{path.stem}_metadata.json"),
        {
            "format_version": PRETRAINING_CHECKPOINT_FORMAT_VERSION,
            "kind": "foveatag_scratch_encoder",
            "initialization": "scratch",
            "external_weights": [],
            "run_contract": dict(run_contract),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "validation": dict(validation),
        },
    )
    return path


def load_pretrained_encoder(
    path: str | Path,
    encoder: nn.Module,
    *,
    expected_run_contract: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Load a scratch encoder artifact and reject external-weight lineage."""
    path = Path(path)
    metadata_path = path.with_name(f"{path.stem}_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != PRETRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag pretrained encoder")
    if metadata.get("kind") != "foveatag_scratch_encoder":
        raise ValueError("artifact is not a FoveaTag scratch encoder")
    if metadata.get("initialization") != "scratch" or metadata.get("external_weights") != []:
        raise ValueError("pretrained encoder does not have pure scratch lineage")
    if (
        expected_run_contract is not None
        and metadata.get("run_contract") != dict(expected_run_contract)
    ):
        raise ValueError("pretrained encoder run contract does not match")
    encoder.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return metadata


def _move_optimizer_state_to_parameters(optimizer: torch.optim.Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(parameter.device)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
