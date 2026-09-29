"""Resumable teacher-free Phase-2 pretraining for the scratch FoveaTag encoder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import functools
import gc
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from .encoder import FoveaEncoderConfig
from .objectives import FoveaSelfSupervisedLoss
from .pretraining import FoveaPretrainer, collate_pretraining_samples
from .pretraining_checkpoint import (
    load_pretraining_checkpoint,
    save_pretrained_encoder,
    save_pretraining_checkpoint,
)
from .pretraining_manifest import (
    DomainBalancedPretrainingDataset,
    ManifestPretrainingDataset,
    build_pretraining_manifests,
    open_domain_manifests,
)
from .run_control import FoveaRunControl
from .views import ViewConfig


def run_pretraining(
    *,
    datasets_db: Path,
    output_dir: Path,
    dataset_ids: Sequence[int],
    domain_weights: Mapping[int, float],
    validation_counts: Mapping[int, int],
    holdout_manifests: Sequence[Path],
    epochs: int,
    epoch_samples: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    checkpoint_every_steps: int,
    log_every: int,
    seed: int,
    minimum_side: int = 256,
    max_source_pixels: Optional[int] = 40_000_000,
    resume: Optional[Path] = None,
    stop_after_steps: Optional[int] = None,
    run_control: Optional[FoveaRunControl] = None,
    progress_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
) -> dict[str, Any]:
    _validate_run_arguments(
        dataset_ids=dataset_ids,
        domain_weights=domain_weights,
        validation_counts=validation_counts,
        epochs=epochs,
        epoch_samples=epoch_samples,
        batch_size=batch_size,
        workers=workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        log_every=log_every,
        minimum_side=minimum_side,
        max_source_pixels=max_source_pixels,
        stop_after_steps=stop_after_steps,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FoveaTag Phase-2 pretraining")
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_paths = _holdout_paths(holdout_manifests)
    manifest_dir = output_dir / "manifests"
    manifest_metadata = build_pretraining_manifests(
        datasets_db,
        manifest_dir,
        dataset_ids=dataset_ids,
        validation_counts=validation_counts,
        excluded_paths=excluded_paths,
        minimum_side=minimum_side,
        max_source_pixels=max_source_pixels,
        seed=seed,
        reuse_existing=resume is not None,
    )
    train_manifests = open_domain_manifests(manifest_dir, manifest_metadata, "train")
    validation_manifests = open_domain_manifests(
        manifest_dir, manifest_metadata, "validation"
    )

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    encoder_config = FoveaEncoderConfig()
    view_config = ViewConfig(overview_size=448, local_size=256, local_views=4)
    objective = FoveaSelfSupervisedLoss(
        correspondence_weight=1.0,
        reconstruction_weight=1.0,
        chroma_weight=0.5,
        max_coordinate_distance=0.02,
    )
    model = FoveaPretrainer(
        encoder_config,
        mask_ratio=0.4,
        objective=objective,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    steps_per_epoch = math.ceil(epoch_samples / batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = _cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
    )
    normalized_weights = _normalized_domain_weights(dataset_ids, domain_weights)
    run_contract = {
        "format_version": 1,
        "kind": "foveatag_teacher_free_pretraining",
        "initialization": "scratch",
        "external_weights": [],
        "datasets_db": str(datasets_db.resolve()),
        "manifest_request": manifest_metadata["request"],
        "manifest_domains": manifest_metadata["domains"],
        "domain_weights": {str(key): normalized_weights[key] for key in sorted(normalized_weights)},
        "holdout_manifests": [str(path.resolve()) for path in holdout_manifests],
        "epochs": epochs,
        "epoch_samples": epoch_samples,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": min(warmup_steps, max(total_steps - 1, 0)),
        "seed": seed,
        "encoder": asdict(encoder_config),
        "views": asdict(view_config),
        "mask_ratio": model.mask_ratio,
        "objective": {
            "correspondence_weight": objective.correspondence_weight,
            "reconstruction_weight": objective.reconstruction_weight,
            "chroma_weight": objective.chroma_weight,
            "max_coordinate_distance": objective.max_coordinate_distance,
        },
        "max_source_pixels": max_source_pixels,
    }
    train_dataset = DomainBalancedPretrainingDataset(
        train_manifests,
        normalized_weights,
        epoch_samples=epoch_samples,
        view_config=view_config,
        seed=seed,
        max_source_pixels=max_source_pixels,
    )
    start_epoch = 0
    sample_offset = 0
    global_step = 0
    best_validation_loss = math.inf
    history: list[dict[str, Any]] = []
    epoch_accumulator = None
    if resume is not None:
        payload = load_pretraining_checkpoint(
            resume,
            model,
            optimizer,
            scheduler=scheduler,
            expected_run_contract=run_contract,
        )
        start_epoch = int(payload["epoch"])
        sample_offset = int(payload["sample_offset"])
        global_step = int(payload["global_step"])
        best_validation_loss = float(payload["best_validation_loss"])
        history = [dict(row) for row in payload.get("history", [])]
        epoch_accumulator = payload.get("epoch_accumulator")
        if start_epoch > epochs or sample_offset > epoch_samples:
            raise ValueError("checkpoint progress is beyond the configured run")

    use_bfloat16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    torch.cuda.reset_peak_memory_stats(device)
    steps_this_run = 0
    run_started = time.monotonic()
    status = "complete"
    resume_epoch = start_epoch
    _emit_progress(progress_callback, "run", {
        "status": "running",
        "epoch": start_epoch,
        "sample_offset": sample_offset,
        "global_step": global_step,
        "total_steps": total_steps,
    })
    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        epoch_offset = sample_offset if epoch == start_epoch else 0
        train_result, epoch_offset, global_step, steps_added, stopped, epoch_accumulator = _train_epoch(
            model,
            optimizer,
            scheduler,
            train_dataset,
            epoch=epoch,
            sample_offset=epoch_offset,
            batch_size=batch_size,
            workers=workers,
            amp_dtype=amp_dtype,
            device=device,
            checkpoint_every_steps=checkpoint_every_steps,
            log_every=log_every,
            global_step=global_step,
            output_dir=output_dir,
            best_validation_loss=best_validation_loss,
            run_contract=run_contract,
            history=history,
            initial_accumulator=(epoch_accumulator if epoch == start_epoch else None),
            remaining_step_budget=(
                None if stop_after_steps is None else stop_after_steps - steps_this_run
            ),
            run_control=run_control,
            progress_callback=progress_callback,
        )
        steps_this_run += steps_added
        if stopped:
            sample_offset = epoch_offset
            resume_epoch = epoch
            status = (
                "stopped_by_request"
                if run_control is not None and run_control.should_stop()
                else "stopped_after_requested_steps"
            )
            break

        validation_started = time.monotonic()
        _emit_progress(progress_callback, "validation", {
            "status": "running",
            "epoch": epoch,
            "global_step": global_step,
        })
        validation = _validate(
            model,
            validation_manifests,
            domain_weights=normalized_weights,
            view_config=view_config,
            seed=seed,
            batch_size=batch_size,
            workers=workers,
            max_source_pixels=max_source_pixels,
            amp_dtype=amp_dtype,
            device=device,
        )
        validation["seconds"] = time.monotonic() - validation_started
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_result,
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        _emit_progress(progress_callback, "validation", {
            "status": "complete",
            **row,
        })
        improved = validation["weighted"]["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation["weighted"]["loss"]
            save_pretrained_encoder(
                output_dir / "best_encoder.safetensors",
                model.encoder,
                run_contract=run_contract,
                epoch=epoch,
                global_step=global_step,
                validation=validation,
            )
        sample_offset = 0
        resume_epoch = epoch + 1
        save_pretraining_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scheduler=scheduler,
            epoch=resume_epoch,
            sample_offset=0,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            run_contract=run_contract,
            history=history,
            epoch_accumulator=None,
        )
        epoch_accumulator = None
        _write_json_atomic(output_dir / "history.json", history)
        print(
            f"[FoveaTagPretrain] epoch={epoch} validation_loss="
            f"{validation['weighted']['loss']:.6f} best={best_validation_loss:.6f}",
            flush=True,
        )
        if stop_after_steps is not None and steps_this_run >= stop_after_steps:
            status = "stopped_after_requested_steps"
            break
    else:
        resume_epoch = epochs
        sample_offset = 0

    summary = {
        "format_version": 1,
        "kind": "foveatag_teacher_free_pretraining_run",
        "status": status,
        "epoch": resume_epoch,
        "sample_offset": sample_offset,
        "global_step": global_step,
        "steps_this_run": steps_this_run,
        "best_validation_loss": (
            best_validation_loss if math.isfinite(best_validation_loss) else None
        ),
        "elapsed_seconds_this_run": time.monotonic() - run_started,
        "dtype": str(amp_dtype).removeprefix("torch."),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "run_contract": run_contract,
        "history": history,
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    if status != "complete":
        save_pretraining_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scheduler=scheduler,
            epoch=resume_epoch,
            sample_offset=sample_offset,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            run_contract=run_contract,
            history=history,
            epoch_accumulator=epoch_accumulator,
        )
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    _emit_progress(progress_callback, "run", {
        "status": status,
        "epoch": resume_epoch,
        "sample_offset": sample_offset,
        "global_step": global_step,
        "steps_this_run": steps_this_run,
    })
    return summary


def _train_epoch(
    model: FoveaPretrainer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    dataset: DomainBalancedPretrainingDataset,
    *,
    epoch: int,
    sample_offset: int,
    batch_size: int,
    workers: int,
    amp_dtype: torch.dtype,
    device: torch.device,
    checkpoint_every_steps: int,
    log_every: int,
    global_step: int,
    output_dir: Path,
    best_validation_loss: float,
    run_contract: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    initial_accumulator: Optional[Mapping[str, Any]],
    remaining_step_budget: Optional[int],
    run_control: Optional[FoveaRunControl],
    progress_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
) -> tuple[dict[str, Any], int, int, int, bool, dict[str, Any]]:
    accumulator = _restore_accumulator(initial_accumulator)
    if (
        remaining_step_budget is not None and remaining_step_budget <= 0
    ) or (run_control is not None and run_control.should_stop()):
        _emit_progress(progress_callback, "train", {
            "status": "stopped",
            "epoch": epoch,
            "sample_offset": sample_offset,
            "global_step": global_step,
        })
        return (
            _training_summary(accumulator),
            sample_offset,
            global_step,
            0,
            True,
            accumulator,
        )
    loader = _loader(
        Subset(dataset, range(sample_offset, len(dataset))),
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
    )
    model.train()
    steps_added = consumed = 0
    segment_started = time.monotonic()
    for batch_index, batch in enumerate(loader, start=1):
        requested = min(batch_size, len(dataset) - sample_offset - consumed)
        consumed += requested
        if batch is None:
            accumulator["skipped"] += requested
            continue
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            output = model(batch)
        if not torch.isfinite(output.losses.loss):
            raise FloatingPointError(f"non-finite pretraining loss at global step {global_step}")
        output.losses.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at global step {global_step}")
        optimizer.step()
        scheduler.step()
        actual = batch.first_images.shape[0]
        _accumulate_losses(accumulator["sums"], output.losses, actual)
        accumulator["samples"] += actual
        accumulator["skipped"] += requested - actual
        accumulator["optimizer_steps"] += 1
        global_step += 1
        steps_added += 1
        current_offset = sample_offset + consumed
        if global_step == 1 or global_step % log_every == 0:
            elapsed = accumulator["seconds"] + time.monotonic() - segment_started
            message = {
                "status": "running",
                "epoch": epoch,
                "sample_offset": current_offset,
                "epoch_samples": len(dataset),
                "global_step": global_step,
                "loss": float(output.losses.loss.detach()),
                "samples_per_second": accumulator["samples"] / max(elapsed, 1e-9),
            }
            print(
                f"[FoveaTagPretrain] epoch={epoch} sample={current_offset}/{len(dataset)} "
                f"step={global_step} loss={message['loss']:.6f} "
                f"samples_per_second={message['samples_per_second']:.2f}",
                flush=True,
            )
            _emit_progress(progress_callback, "train", message)
        if global_step % checkpoint_every_steps == 0:
            now = time.monotonic()
            accumulator["seconds"] += now - segment_started
            segment_started = now
            save_pretraining_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                scheduler=scheduler,
                epoch=epoch,
                sample_offset=current_offset,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                run_contract=run_contract,
                history=list(history),
                epoch_accumulator=accumulator,
            )
            _write_json_atomic(
                output_dir / "progress.json",
                {
                    "epoch": epoch,
                    "sample_offset": current_offset,
                    "global_step": global_step,
                    "elapsed_seconds": accumulator["seconds"],
                    "losses": _mean_losses(accumulator["sums"], accumulator["samples"]),
                },
            )
        if (
            remaining_step_budget is not None and steps_added >= remaining_step_budget
        ) or (run_control is not None and run_control.should_stop()):
            accumulator["seconds"] += time.monotonic() - segment_started
            _emit_progress(progress_callback, "train", {
                "status": "stopped",
                "epoch": epoch,
                "sample_offset": current_offset,
                "epoch_samples": len(dataset),
                "global_step": global_step,
            })
            return (
                _training_summary(accumulator),
                current_offset,
                global_step,
                steps_added,
                True,
                accumulator,
            )
    accumulator["seconds"] += time.monotonic() - segment_started
    return (
        _training_summary(accumulator),
        len(dataset),
        global_step,
        steps_added,
        False,
        accumulator,
    )


@torch.no_grad()
def _validate(
    model: FoveaPretrainer,
    manifests,
    *,
    domain_weights: Mapping[int, float],
    view_config: ViewConfig,
    seed: int,
    batch_size: int,
    workers: int,
    max_source_pixels: Optional[int],
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    domains: dict[str, Any] = {}
    for domain, manifest in sorted(manifests.items()):
        dataset = ManifestPretrainingDataset(
            manifest,
            view_config=view_config,
            seed=seed + domain * 1009,
            max_source_pixels=max_source_pixels,
        )
        loader = _loader(dataset, batch_size=batch_size, workers=workers, shuffle=False)
        generator = torch.Generator(device=device).manual_seed(seed + domain * 65537)
        sums = _loss_sums()
        samples = skipped = consumed = 0
        for batch in loader:
            requested = min(batch_size, len(dataset) - consumed)
            consumed += requested
            if batch is None:
                skipped += requested
                continue
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                output = model(batch, generator=generator)
            actual = batch.first_images.shape[0]
            _accumulate_losses(sums, output.losses, actual)
            samples += actual
            skipped += requested - actual
        domains[str(domain)] = {
            "samples": samples,
            "skipped": skipped,
            **_mean_losses(sums, samples),
        }
    weighted = {
        name: sum(domain_weights[int(domain)] * values[name] for domain, values in domains.items())
        for name in ("loss", "correspondence", "reconstruction", "chroma")
    }
    return {"weighted": weighted, "domains": domains}


def _loader(dataset, *, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": False,
        "collate_fn": collate_pretraining_samples,
        "drop_last": False,
    }
    if workers:
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def _cosine_with_warmup(optimizer, *, total_steps, warmup_steps, learning_rate):
    effective_warmup = min(max(0, warmup_steps), max(0, total_steps - 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - effective_warmup),
        eta_min=learning_rate * 0.01,
    )
    if not effective_warmup:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=effective_warmup,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[effective_warmup],
    )


def _loss_sums() -> dict[str, float]:
    return {name: 0.0 for name in ("loss", "correspondence", "reconstruction", "chroma")}


def _accumulate_losses(sums, losses, samples: int) -> None:
    for name in sums:
        sums[name] += float(getattr(losses, name).detach()) * samples


def _mean_losses(sums, samples: int) -> dict[str, float]:
    if samples <= 0:
        return {name: math.nan for name in sums}
    return {name: value / samples for name, value in sums.items()}


def _restore_accumulator(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if value is None:
        return {
            "sums": _loss_sums(),
            "samples": 0,
            "skipped": 0,
            "optimizer_steps": 0,
            "seconds": 0.0,
        }
    sums = {name: float(value["sums"][name]) for name in _loss_sums()}
    return {
        "sums": sums,
        "samples": int(value["samples"]),
        "skipped": int(value["skipped"]),
        "optimizer_steps": int(value["optimizer_steps"]),
        "seconds": float(value["seconds"]),
    }


def _training_summary(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    samples = int(accumulator["samples"])
    seconds = float(accumulator["seconds"])
    return {
        "samples": samples,
        "skipped": int(accumulator["skipped"]),
        "optimizer_steps": int(accumulator["optimizer_steps"]),
        "seconds": seconds,
        "samples_per_second": samples / max(seconds, 1e-9),
        **_mean_losses(accumulator["sums"], samples),
    }


def _normalized_domain_weights(dataset_ids, weights) -> dict[int, float]:
    domains = set(int(value) for value in dataset_ids)
    if set(weights) != domains or any(float(value) <= 0 for value in weights.values()):
        raise ValueError("domain weights must be positive and specify every dataset")
    total = sum(float(weights[domain]) for domain in domains)
    return {domain: float(weights[domain]) / total for domain in domains}


def _validate_run_arguments(**values) -> None:
    dataset_ids = tuple(values["dataset_ids"])
    if not dataset_ids or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("dataset_ids must be non-empty and unique")
    _normalized_domain_weights(dataset_ids, values["domain_weights"])
    if set(values["validation_counts"]) != set(dataset_ids):
        raise ValueError("validation counts must specify every dataset")
    if any(int(value) <= 0 for value in values["validation_counts"].values()):
        raise ValueError("validation counts must be positive")
    for name in ("epochs", "epoch_samples", "batch_size", "checkpoint_every_steps", "log_every", "minimum_side"):
        if int(values[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(values["workers"]) < 0 or int(values["warmup_steps"]) < 0:
        raise ValueError("workers and warmup_steps must be non-negative")
    if float(values["learning_rate"]) <= 0 or float(values["weight_decay"]) < 0:
        raise ValueError("optimizer settings are invalid")
    if values["max_source_pixels"] is not None and values["max_source_pixels"] <= 0:
        raise ValueError("max_source_pixels must be positive when provided")
    if values["stop_after_steps"] is not None and values["stop_after_steps"] <= 0:
        raise ValueError("stop_after_steps must be positive when provided")


def _holdout_paths(manifests: Sequence[Path]) -> list[str]:
    paths = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        paths.extend(str(item["image_path"]) for item in payload.get("items", []))
    return paths


def _parse_mapping(values: Sequence[str], *, cast, name: str):
    result = {}
    for value in values:
        try:
            raw_domain, raw_amount = value.split("=", 1)
            domain = int(raw_domain)
            amount = cast(raw_amount)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} entries must use DATASET_ID=VALUE") from error
        if domain in result:
            raise ValueError(f"duplicate dataset in {name}: {domain}")
        result[domain] = amount
    return result


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _emit_progress(
    callback: Optional[Callable[[str, Mapping[str, Any]], None]],
    event: str,
    payload: Mapping[str, Any],
) -> None:
    if callback is None:
        return
    try:
        callback(event, payload)
    except Exception:
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, action="append", required=True)
    parser.add_argument("--domain-weight", action="append", required=True)
    parser.add_argument("--validation-count", action="append", required=True)
    parser.add_argument("--holdout-manifest", type=Path, action="append", default=[])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epoch-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--minimum-side", type=int, default=256)
    parser.add_argument("--max-source-megapixels", type=float, default=40.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_pretraining(
        datasets_db=args.datasets_db,
        output_dir=args.output_dir,
        dataset_ids=args.dataset_id,
        domain_weights=_parse_mapping(args.domain_weight, cast=float, name="domain-weight"),
        validation_counts=_parse_mapping(
            args.validation_count, cast=int, name="validation-count"
        ),
        holdout_manifests=args.holdout_manifest,
        epochs=args.epochs,
        epoch_samples=args.epoch_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        log_every=args.log_every,
        seed=args.seed,
        minimum_side=args.minimum_side,
        max_source_pixels=(
            None
            if args.max_source_megapixels == 0
            else round(args.max_source_megapixels * 1_000_000)
        ),
        resume=args.resume,
        stop_after_steps=args.stop_after_steps,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
