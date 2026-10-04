"""Step-resumable epoch mechanics for FoveaTag router training."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Protocol

from torch.utils.data import DataLoader, Dataset, Subset

from .router_checkpoint import save_router_checkpoint
from .router_data import collate_router_samples
from .router_training import FoveaRouterTrainer


class RouterRunControl(Protocol):
    def should_stop(self) -> bool: ...


def train_router_epoch(
    trainer: FoveaRouterTrainer,
    dataset: Dataset,
    *,
    epoch: int,
    sample_offset: int,
    batch_size: int,
    workers: int,
    checkpoint_every_steps: int,
    log_every: int,
    global_step: int,
    output_dir: Path,
    best_utility_recovery: float,
    run_contract: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    initial_accumulator: Optional[Mapping[str, Any]],
    remaining_step_budget: Optional[int],
    progress_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
    run_control: Optional[RouterRunControl] = None,
) -> tuple[dict[str, Any], int, int, int, Optional[str], dict[str, Any]]:
    """Train one epoch segment and checkpoint at exact source offsets."""
    _validate_epoch_arguments(
        epoch=epoch,
        sample_offset=sample_offset,
        batch_size=batch_size,
        workers=workers,
        checkpoint_every_steps=checkpoint_every_steps,
        log_every=log_every,
        global_step=global_step,
        dataset_size=len(dataset),
        remaining_step_budget=remaining_step_budget,
    )
    accumulator = _restore_accumulator(initial_accumulator)
    stopped = _requested_stop(run_control, remaining_step_budget)
    if stopped is not None:
        return (
            _training_summary(accumulator),
            sample_offset,
            global_step,
            0,
            stopped,
            accumulator,
        )
    loader = _loader(
        Subset(dataset, range(sample_offset, len(dataset))),
        batch_size=batch_size,
        workers=workers,
    )
    consumed = steps_added = 0
    segment_started = time.monotonic()
    for batch in loader:
        if run_control is not None and run_control.should_stop():
            accumulator["seconds"] += time.monotonic() - segment_started
            _save_progress(
                trainer,
                output_dir=output_dir,
                epoch=epoch,
                sample_offset=sample_offset + consumed,
                global_step=global_step,
                best_utility_recovery=best_utility_recovery,
                run_contract=run_contract,
                history=history,
                accumulator=accumulator,
            )
            return (
                _training_summary(accumulator),
                sample_offset + consumed,
                global_step,
                steps_added,
                "stopped_by_request",
                accumulator,
            )
        requested = min(batch_size, len(dataset) - sample_offset - consumed)
        consumed += requested
        if batch is None:
            accumulator["skipped"] += requested
            continue
        result = trainer.train_batch(batch)
        if result is None or not math.isfinite(result.loss):
            raise FloatingPointError(f"non-finite router loss at global step {global_step}")
        actual = int(batch.labels.shape[0])
        accumulator["loss_sum"] += result.loss * actual
        accumulator["ranking_sum"] += result.ranking * actual
        accumulator["diversity_sum"] += result.diversity * actual
        accumulator["positive_utility_sum"] += result.mean_positive_utility * actual
        accumulator["samples"] += actual
        accumulator["skipped"] += requested - actual
        accumulator["optimizer_steps"] += 1
        global_step += 1
        steps_added += 1
        current_offset = sample_offset + consumed
        if global_step == 1 or global_step % log_every == 0:
            elapsed = accumulator["seconds"] + time.monotonic() - segment_started
            message = {
                "epoch": epoch,
                "sample_offset": current_offset,
                "epoch_samples": len(dataset),
                "global_step": global_step,
                "loss": result.loss,
                "ranking": result.ranking,
                "diversity": result.diversity,
                "samples_per_second": accumulator["samples"] / max(elapsed, 1e-9),
            }
            print(
                f"[FoveaTagRouter] epoch={epoch} sample={current_offset}/{len(dataset)} "
                f"step={global_step} loss={result.loss:.6f} "
                f"samples_per_second={message['samples_per_second']:.2f}",
                flush=True,
            )
            _emit_progress(progress_callback, "train", message)
        if global_step % checkpoint_every_steps == 0:
            now = time.monotonic()
            accumulator["seconds"] += now - segment_started
            segment_started = now
            _save_progress(
                trainer,
                output_dir=output_dir,
                epoch=epoch,
                sample_offset=current_offset,
                global_step=global_step,
                best_utility_recovery=best_utility_recovery,
                run_contract=run_contract,
                history=history,
                accumulator=accumulator,
            )
        if remaining_step_budget is not None and steps_added >= remaining_step_budget:
            accumulator["seconds"] += time.monotonic() - segment_started
            _save_progress(
                trainer,
                output_dir=output_dir,
                epoch=epoch,
                sample_offset=current_offset,
                global_step=global_step,
                best_utility_recovery=best_utility_recovery,
                run_contract=run_contract,
                history=history,
                accumulator=accumulator,
            )
            return (
                _training_summary(accumulator),
                current_offset,
                global_step,
                steps_added,
                "stopped_after_requested_steps",
                accumulator,
            )
    accumulator["seconds"] += time.monotonic() - segment_started
    return (
        _training_summary(accumulator),
        len(dataset),
        global_step,
        steps_added,
        None,
        accumulator,
    )


def _save_progress(
    trainer: FoveaRouterTrainer,
    *,
    output_dir: Path,
    epoch: int,
    sample_offset: int,
    global_step: int,
    best_utility_recovery: float,
    run_contract: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    accumulator: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_router_checkpoint(
        output_dir / "latest.pt",
        trainer.router,
        trainer.optimizer,
        scheduler=trainer.scheduler,
        epoch=epoch,
        sample_offset=sample_offset,
        global_step=global_step,
        best_utility_recovery=best_utility_recovery,
        run_contract=run_contract,
        history=list(history),
        epoch_accumulator=accumulator,
    )
    _write_json_atomic(
        output_dir / "progress.json",
        {
            "epoch": epoch,
            "sample_offset": sample_offset,
            "global_step": global_step,
            **_training_summary(accumulator),
        },
    )


def _restore_accumulator(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    names = (
        "loss_sum",
        "ranking_sum",
        "diversity_sum",
        "positive_utility_sum",
        "samples",
        "skipped",
        "optimizer_steps",
        "seconds",
    )
    defaults = {name: 0 if name in {"samples", "skipped", "optimizer_steps"} else 0.0 for name in names}
    if value is None:
        return defaults
    return {
        name: int(value.get(name, default)) if isinstance(default, int) else float(value.get(name, default))
        for name, default in defaults.items()
    }


def _training_summary(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    samples = int(accumulator["samples"])
    divisor = max(samples, 1)
    return {
        "mean_loss": float(accumulator["loss_sum"]) / divisor,
        "mean_ranking": float(accumulator["ranking_sum"]) / divisor,
        "mean_diversity": float(accumulator["diversity_sum"]) / divisor,
        "mean_positive_utility": float(accumulator["positive_utility_sum"]) / divisor,
        "samples": samples,
        "skipped": int(accumulator["skipped"]),
        "optimizer_steps": int(accumulator["optimizer_steps"]),
        "seconds": float(accumulator["seconds"]),
    }


def _requested_stop(
    run_control: Optional[RouterRunControl], remaining_step_budget: Optional[int]
) -> Optional[str]:
    if run_control is not None and run_control.should_stop():
        return "stopped_by_request"
    if remaining_step_budget is not None and remaining_step_budget <= 0:
        return "stopped_after_requested_steps"
    return None


def _validate_epoch_arguments(**values: Any) -> None:
    if min(values["epoch"], values["sample_offset"], values["global_step"]) < 0:
        raise ValueError("epoch progress must be non-negative")
    if values["sample_offset"] > values["dataset_size"]:
        raise ValueError("sample_offset exceeds dataset length")
    if min(
        values["batch_size"],
        values["checkpoint_every_steps"],
        values["log_every"],
    ) <= 0:
        raise ValueError("batch and interval settings must be positive")
    if values["workers"] < 0:
        raise ValueError("workers must be non-negative")
    budget = values["remaining_step_budget"]
    if budget is not None and budget < 0:
        raise ValueError("remaining_step_budget must be non-negative")


def _loader(dataset: Dataset, *, batch_size: int, workers: int) -> DataLoader:
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "collate_fn": collate_router_samples,
        "pin_memory": True,
    }
    if workers:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def _emit_progress(
    callback: Optional[Callable[[str, Mapping[str, Any]], None]],
    event: str,
    payload: Mapping[str, Any],
) -> None:
    if callback is not None:
        callback(event, payload)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
