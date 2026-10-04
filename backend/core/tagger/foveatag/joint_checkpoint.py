"""Exact-resume checkpoints for FoveaTag stage-4 joint fine-tuning."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn


JOINT_CHECKPOINT_FORMAT_VERSION = 1
EDITABLE_JOINT_FIELDS = ("epochs", "batch_size", "workers", "learning_rate", "weight_decay",
    "warmup_steps", "candidate_count", "selection_budget", "random_count", "oracle_interval",
    "oracle_chunk_size", "router_weight", "checkpoint_every_steps", "log_every", "review_steps", "external_flash",
    "save_tag_metrics", "train_f1_eval_every_n_steps", "train_f1_threshold_search_every_n_steps",
    "train_f1_initial_threshold", "train_f1_buffer_batches")


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
    metrics_runtime: Optional[Mapping[str, Any]] = None,
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
        "metrics_runtime": dict(metrics_runtime) if metrics_runtime is not None else None,
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
    allowed_contract_changes: Sequence[str] = (),
    scheduler_contract_fields: Sequence[str] = (),
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != JOINT_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag joint checkpoint")
    if payload.get("kind") != "foveatag_joint_training":
        raise ValueError("checkpoint is not FoveaTag joint training")
    actual, expected = payload.get("run_contract"), _canonical_json(expected_run_contract)
    if actual is None or {k: v for k, v in actual.items() if k not in allowed_contract_changes} != {
            k: v for k, v in expected.items() if k not in allowed_contract_changes}:
        raise ValueError("joint checkpoint run contract does not match")
    model.load_state_dict(payload["model"], strict=True)
    router.load_state_dict(payload["router"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _move_optimizer_state_to_parameters(optimizer)
    saved_scheduler = payload.get("scheduler")
    if (scheduler is None) != (saved_scheduler is None):
        raise ValueError("checkpoint and runtime disagree about scheduler state")
    if scheduler is not None and not any(actual.get(key) != expected.get(key) for key in scheduler_contract_fields):
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
