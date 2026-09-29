"""Step-resumable Phase-3 A/B/C training for the scratch FoveaTag model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from torch.utils.data import DataLoader, Subset

from ..tag_vocabulary import TagVocabulary
from ..tagger_dataset import TaggerDataset
from ..tagger_loss import AsymmetricLossOptimized
from .artifact import FoveaArtifactConfig, save_inference_artifact
from .control_run import select_domain_splits
from .encoder import FoveaEncoderConfig
from .model import FoveaTagger
from .phase2_gate import load_passed_phase2_gate, require_phase2_encoder
from .pretraining_checkpoint import load_pretrained_encoder
from .run_control import FoveaRunControl
from .supervised_checkpoint import (
    load_supervised_checkpoint,
    load_supervised_initialization,
    save_supervised_checkpoint,
    save_supervised_initialization,
)
from .supervised_data import (
    DomainBalancedSupervisedDataset,
    FoveaInputMode,
    FoveaSupervisedDataset,
    PHASE3_INPUT_MODES,
    SupervisionPolicy,
    collate_supervised_samples,
)
from .supervised_training import FoveaSupervisedTrainer
from .views import ViewConfig


def run_supervised_comparison(
    *,
    comparison_dir: Path,
    branch_runner: Optional[Callable[..., dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
    run_control: Optional[FoveaRunControl] = None,
    **training_kwargs: Any,
) -> dict[str, Any]:
    """Run or resume the pre-registered A/B/C branches in a fixed order."""
    forbidden = {"comparison_dir", "input_mode", "resume"} & set(training_kwargs)
    if forbidden:
        raise ValueError(f"comparison runner owns arguments: {sorted(forbidden)}")
    branch_runner = branch_runner or run_supervised_training
    summaries: dict[str, Any] = {}
    status = "complete"
    active_mode: Optional[str] = None
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for mode in PHASE3_INPUT_MODES:
        if run_control is not None and run_control.should_stop():
            status = "stopped_by_request"
            active_mode = mode
            break
        active_mode = mode
        resume_path = comparison_dir / mode / "latest.pt"
        _emit_progress(
            progress_callback,
            "branch_start",
            {
                "input_mode": mode,
                "resume": str(resume_path) if resume_path.is_file() else None,
            },
        )
        _write_json_atomic(
            comparison_dir / "comparison_summary.json",
            _comparison_summary(
                status="running",
                active_mode=mode,
                summaries=summaries,
            ),
        )
        summary = branch_runner(
            comparison_dir=comparison_dir,
            input_mode=mode,
            resume=resume_path if resume_path.is_file() else None,
            progress_callback=progress_callback,
            run_control=run_control,
            **training_kwargs,
        )
        summaries[mode] = summary
        status = str(summary.get("status", "unknown"))
        experiment = _comparison_summary(
            status=status,
            active_mode=mode,
            summaries=summaries,
        )
        _write_json_atomic(comparison_dir / "comparison_summary.json", experiment)
        if status != "complete":
            break
    else:
        active_mode = None
    result = _comparison_summary(
        status=status,
        active_mode=active_mode,
        summaries=summaries,
    )
    _write_json_atomic(comparison_dir / "comparison_summary.json", result)
    return result


def _comparison_summary(
    *, status: str, active_mode: Optional[str], summaries: Mapping[str, Any]
) -> dict[str, Any]:
    completed = [
        mode for mode in PHASE3_INPUT_MODES
        if summaries.get(mode, {}).get("status") == "complete"
    ]
    return {
        "format_version": 1,
        "kind": "foveatag_phase3_supervised_comparison",
        "status": status,
        "active_mode": active_mode,
        "completed_modes": completed,
        "remaining_modes": [mode for mode in PHASE3_INPUT_MODES if mode not in completed],
        "branches": dict(summaries),
    }


def run_supervised_training(
    *,
    phase2_gate: Path,
    pretrained_encoder: Path,
    vocabulary_path: Path,
    datasets_db: Path,
    comparison_dir: Path,
    input_mode: FoveaInputMode,
    train_counts: Mapping[int, int],
    validation_counts: Mapping[int, int],
    domain_weights: Mapping[int, float],
    holdout_manifests: Sequence[Path],
    photo_dataset_id: Optional[int],
    photo_loss_weight: float,
    epochs: int,
    epoch_samples: int,
    batch_size: int,
    workers: int,
    local_views: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    checkpoint_every_steps: int,
    log_every: int,
    seed: int,
    max_source_pixels: Optional[int] = 40_000_000,
    resume: Optional[Path] = None,
    stop_after_steps: Optional[int] = None,
    progress_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
    run_control: Optional[FoveaRunControl] = None,
) -> dict[str, Any]:
    _validate_arguments(
        input_mode=input_mode,
        train_counts=train_counts,
        validation_counts=validation_counts,
        domain_weights=domain_weights,
        epochs=epochs,
        epoch_samples=epoch_samples,
        batch_size=batch_size,
        workers=workers,
        local_views=local_views,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        log_every=log_every,
        photo_loss_weight=photo_loss_weight,
        max_source_pixels=max_source_pixels,
        stop_after_steps=stop_after_steps,
    )
    passed_phase2_gate = load_passed_phase2_gate(phase2_gate)
    require_phase2_encoder(passed_phase2_gate, pretrained_encoder)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FoveaTag Phase-3 training")
    branch_dir = comparison_dir / input_mode
    branch_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = TagVocabulary.from_dict(
        json.loads(vocabulary_path.read_text(encoding="utf-8"))
    )
    vocabulary_snapshot = comparison_dir / "vocabulary.json"
    serialized_vocabulary = json.dumps(
        vocabulary.to_dict(), indent=2, ensure_ascii=False
    ) + "\n"
    if vocabulary_snapshot.is_file():
        if vocabulary_snapshot.read_text(encoding="utf-8") != serialized_vocabulary:
            raise ValueError("comparison vocabulary snapshot does not match")
    else:
        vocabulary_snapshot.write_text(serialized_vocabulary, encoding="utf-8")

    excluded_paths, fine_tags = _holdout_contract(holdout_manifests)
    raw_dataset = _load_raw_dataset(
        datasets_db,
        dataset_ids=sorted(set(train_counts) | set(validation_counts)),
        vocabulary=vocabulary,
        max_source_pixels=max_source_pixels,
    )
    train_indices, validation_indices, available = select_domain_splits(
        raw_dataset,
        train_counts=train_counts,
        validation_counts=validation_counts,
        excluded_paths=excluded_paths,
        seed=seed,
    )
    if not validation_indices:
        raise ValueError("validation split must not be empty")
    domain_indices = {
        domain: tuple(
            index for index in train_indices
            if raw_dataset.dataset_id_for_index(index) == domain
        )
        for domain in sorted(train_counts)
    }
    fine_indices = tuple(
        sorted(vocabulary.tag_to_idx[tag] for tag in fine_tags if tag in vocabulary.tag_to_idx)
    )
    policies = {}
    if photo_dataset_id is not None:
        policies[photo_dataset_id] = SupervisionPolicy(
            loss_weight=photo_loss_weight,
            positive_only_indices=fine_indices,
        )
    view_config = ViewConfig(overview_size=448, local_size=256, local_views=local_views)
    train_source = FoveaSupervisedDataset(
        raw_dataset,
        view_config=view_config,
        seed=seed,
        training=True,
        supervision_policies=policies,
        input_mode=input_mode,
    )
    train_dataset = DomainBalancedSupervisedDataset(
        train_source,
        domain_indices,
        domain_weights,
        epoch_samples=epoch_samples,
        seed=seed,
    )
    validation_dataset = FoveaSupervisedDataset(
        raw_dataset,
        view_config=view_config,
        seed=seed,
        training=False,
        supervision_policies=policies,
        input_mode=input_mode,
    )

    encoder_metadata = json.loads(
        pretrained_encoder.with_name(
            f"{pretrained_encoder.stem}_metadata.json"
        ).read_text(encoding="utf-8")
    )
    encoder_config = _encoder_config(encoder_metadata)
    artifact_config = FoveaArtifactConfig(
        num_tags=vocabulary.num_tags,
        encoder=encoder_config,
        views=view_config,
    )
    initialization_contract = {
        "format_version": 1,
        "kind": "foveatag_phase3_shared_initialization",
        "initialization": "scratch",
        "external_weights": [],
        "phase2_gate": str(phase2_gate.resolve()),
        "phase2_gate_sha256": _file_sha256(phase2_gate),
        "pretrained_encoder": str(pretrained_encoder.resolve()),
        "pretrained_encoder_sha256": _file_sha256(pretrained_encoder),
        "pretrained_encoder_metadata_sha256": _file_sha256(
            pretrained_encoder.with_name(f"{pretrained_encoder.stem}_metadata.json")
        ),
        "vocabulary_sha256": _file_sha256(vocabulary_path),
        "seed": seed,
        "model": artifact_config.to_dict(),
    }
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _model(artifact_config)
    shared_initialization = comparison_dir / "shared_initialization.safetensors"
    if shared_initialization.is_file():
        load_supervised_initialization(
            shared_initialization,
            model,
            expected_initialization_contract=initialization_contract,
        )
    else:
        load_pretrained_encoder(pretrained_encoder, model.encoder)
        save_supervised_initialization(
            shared_initialization,
            model,
            initialization_contract=initialization_contract,
        )
    shared_initialization_sha256 = _file_sha256(shared_initialization)

    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = epochs * math.ceil(epoch_samples / batch_size)
    scheduler = _cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
    )
    criterion = AsymmetricLossOptimized()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    trainer = FoveaSupervisedTrainer(
        model,
        criterion,
        optimizer,
        device=device,
        mixed_precision=amp_dtype,
        scheduler=scheduler,
    )
    normalized_weights = _normalized_domain_weights(domain_weights, train_counts)
    run_contract = {
        "format_version": 1,
        "kind": "foveatag_phase3_supervised_branch",
        "input_mode": input_mode,
        "shared_initialization_sha256": shared_initialization_sha256,
        "datasets_db": str(datasets_db.resolve()),
        "train_counts": {str(key): int(value) for key, value in sorted(train_counts.items())},
        "validation_counts": {
            str(key): int(value) for key, value in sorted(validation_counts.items())
        },
        "domain_weights": {
            str(key): value for key, value in sorted(normalized_weights.items())
        },
        "holdout_manifests": [str(path.resolve()) for path in holdout_manifests],
        "photo_dataset_id": photo_dataset_id,
        "photo_loss_weight": photo_loss_weight,
        "fine_detail_positive_only_tags": len(fine_indices),
        "epochs": epochs,
        "epoch_samples": epoch_samples,
        "batch_size": batch_size,
        "workers": workers,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": min(warmup_steps, max(total_steps - 1, 0)),
        "seed": seed,
        "views": asdict(view_config),
        "max_source_pixels": max_source_pixels,
    }
    start_epoch = sample_offset = global_step = 0
    best_f1 = -1.0
    best_threshold = 0.5
    history: list[dict[str, Any]] = []
    accumulator = None
    if resume is not None:
        payload = load_supervised_checkpoint(
            resume,
            model,
            optimizer,
            scheduler=scheduler,
            expected_run_contract=run_contract,
        )
        start_epoch = int(payload["epoch"])
        sample_offset = int(payload["sample_offset"])
        global_step = int(payload["global_step"])
        best_f1 = float(payload["best_validation_f1"])
        best_threshold = float(payload["best_threshold"])
        history = [dict(row) for row in payload.get("history", [])]
        accumulator = payload.get("epoch_accumulator")

    torch.cuda.reset_peak_memory_stats(device)
    steps_this_run = 0
    run_started = time.monotonic()
    status = "complete"
    resume_epoch = start_epoch
    try:
        for epoch in range(start_epoch, epochs):
            train_dataset.set_epoch(epoch)
            epoch_offset = sample_offset if epoch == start_epoch else 0
            trained, epoch_offset, global_step, added, stopped, accumulator = _train_epoch(
                trainer,
                train_dataset,
                epoch=epoch,
                sample_offset=epoch_offset,
                batch_size=batch_size,
                workers=workers,
                checkpoint_every_steps=checkpoint_every_steps,
                log_every=log_every,
                global_step=global_step,
                output_dir=branch_dir,
                best_f1=best_f1,
                best_threshold=best_threshold,
                run_contract=run_contract,
                history=history,
                initial_accumulator=(accumulator if epoch == start_epoch else None),
                remaining_step_budget=(
                    None if stop_after_steps is None else stop_after_steps - steps_this_run
                ),
                progress_callback=progress_callback,
                run_control=run_control,
            )
            steps_this_run += added
            if stopped:
                sample_offset = epoch_offset
                resume_epoch = epoch
                status = stopped
                break

            validation_started = time.monotonic()
            validation = trainer.validate_epoch(
                _loader(
                    Subset(validation_dataset, validation_indices),
                    batch_size=batch_size,
                    workers=workers,
                )
            )
            validation_seconds = time.monotonic() - validation_started
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train": trained,
                "validation": vars(validation),
                "validation_seconds": validation_seconds,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            _emit_progress(
                progress_callback,
                "validation",
                {
                    "input_mode": input_mode,
                    "epoch": epoch,
                    "global_step": global_step,
                    **vars(validation),
                },
            )
            improved = validation.macro_f1 > best_f1
            if improved:
                best_f1 = validation.macro_f1
                best_threshold = validation.threshold
                save_inference_artifact(
                    branch_dir / "best_f1.safetensors",
                    model,
                    artifact_config,
                    vocabulary=vocabulary.to_dict(),
                    provenance={
                        "initialization": "scratch",
                        "external_weights": [],
                        "shared_initialization_sha256": shared_initialization_sha256,
                        "input_mode": input_mode,
                        "epoch": epoch,
                        "global_step": global_step,
                        "validation_f1": best_f1,
                        "threshold": best_threshold,
                    },
                )
            sample_offset = 0
            resume_epoch = epoch + 1
            accumulator = None
            save_supervised_checkpoint(
                branch_dir / "latest.pt",
                model,
                optimizer,
                scheduler=scheduler,
                epoch=resume_epoch,
                sample_offset=0,
                global_step=global_step,
                best_validation_f1=best_f1,
                best_threshold=best_threshold,
                run_contract=run_contract,
                history=history,
            )
            _write_json_atomic(branch_dir / "history.json", history)
            print(
                f"[FoveaTagSupervised] mode={input_mode} epoch={epoch} "
                f"validation_f1={validation.macro_f1:.6f} best={best_f1:.6f}",
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
            "kind": "foveatag_phase3_supervised_run",
            "status": status,
            "input_mode": input_mode,
            "epoch": resume_epoch,
            "sample_offset": sample_offset,
            "global_step": global_step,
            "steps_this_run": steps_this_run,
            "best_validation_f1": best_f1,
            "best_threshold": best_threshold,
            "elapsed_seconds_this_run": time.monotonic() - run_started,
            "dtype": str(amp_dtype).removeprefix("torch."),
            "parameter_count": model.parameter_count(),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "available_after_holdout": {str(key): value for key, value in available.items()},
            "run_contract": run_contract,
            "history": history,
        }
        _write_json_atomic(branch_dir / "summary.json", summary)
        if status != "complete":
            save_supervised_checkpoint(
                branch_dir / "latest.pt",
                model,
                optimizer,
                scheduler=scheduler,
                epoch=resume_epoch,
                sample_offset=sample_offset,
                global_step=global_step,
                best_validation_f1=best_f1,
                best_threshold=best_threshold,
                run_contract=run_contract,
                history=history,
                epoch_accumulator=accumulator,
            )
        return summary
    finally:
        del trainer, criterion, optimizer, scheduler, model
        gc.collect()
        torch.cuda.empty_cache()


def _train_epoch(
    trainer: FoveaSupervisedTrainer,
    dataset: DomainBalancedSupervisedDataset,
    *,
    epoch: int,
    sample_offset: int,
    batch_size: int,
    workers: int,
    checkpoint_every_steps: int,
    log_every: int,
    global_step: int,
    output_dir: Path,
    best_f1: float,
    best_threshold: float,
    run_contract: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    initial_accumulator: Optional[Mapping[str, Any]],
    remaining_step_budget: Optional[int],
    progress_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
    run_control: Optional[FoveaRunControl] = None,
) -> tuple[dict[str, Any], int, int, int, Optional[str], dict[str, Any]]:
    accumulator = _restore_accumulator(initial_accumulator)
    if run_control is not None and run_control.should_stop():
        return (
            _training_summary(accumulator), sample_offset, global_step, 0,
            "stopped_by_request", accumulator,
        )
    if remaining_step_budget is not None and remaining_step_budget <= 0:
        return (
            _training_summary(accumulator), sample_offset, global_step, 0,
            "stopped_after_requested_steps", accumulator,
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
            return (
                _training_summary(accumulator), sample_offset + consumed, global_step,
                steps_added, "stopped_by_request", accumulator,
            )
        requested = min(batch_size, len(dataset) - sample_offset - consumed)
        consumed += requested
        if batch is None:
            accumulator["skipped"] += requested
            continue
        loss = trainer.train_batch(batch)
        if loss is None or not math.isfinite(loss):
            raise FloatingPointError(f"non-finite supervised loss at global step {global_step}")
        actual = int(batch.labels.shape[0])
        accumulator["loss_sum"] += loss * actual
        accumulator["samples"] += actual
        accumulator["skipped"] += requested - actual
        accumulator["optimizer_steps"] += 1
        global_step += 1
        steps_added += 1
        current_offset = sample_offset + consumed
        if global_step == 1 or global_step % log_every == 0:
            elapsed = accumulator["seconds"] + time.monotonic() - segment_started
            print(
                f"[FoveaTagSupervised] mode={run_contract['input_mode']} epoch={epoch} "
                f"sample={current_offset}/{len(dataset)} step={global_step} loss={loss:.6f} "
                f"samples_per_second={accumulator['samples'] / max(elapsed, 1e-9):.2f}",
                flush=True,
            )
            _emit_progress(
                progress_callback,
                "train",
                {
                    "input_mode": run_contract["input_mode"],
                    "epoch": epoch,
                    "sample_offset": current_offset,
                    "epoch_samples": len(dataset),
                    "global_step": global_step,
                    "loss": loss,
                    "samples_per_second": accumulator["samples"] / max(elapsed, 1e-9),
                },
            )
        if global_step % checkpoint_every_steps == 0:
            now = time.monotonic()
            accumulator["seconds"] += now - segment_started
            segment_started = now
            save_supervised_checkpoint(
                output_dir / "latest.pt",
                trainer.model,
                trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=epoch,
                sample_offset=current_offset,
                global_step=global_step,
                best_validation_f1=best_f1,
                best_threshold=best_threshold,
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
                    **_training_summary(accumulator),
                },
            )
        if remaining_step_budget is not None and steps_added >= remaining_step_budget:
            accumulator["seconds"] += time.monotonic() - segment_started
            return (
                _training_summary(accumulator), current_offset, global_step,
                steps_added, "stopped_after_requested_steps", accumulator,
            )
    accumulator["seconds"] += time.monotonic() - segment_started
    return (
        _training_summary(accumulator), len(dataset), global_step,
        steps_added, None, accumulator,
    )


def _load_raw_dataset(database_path, *, dataset_ids, vocabulary, max_source_pixels):
    engine = create_engine(
        f"sqlite:///{database_path.resolve().as_posix()}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    session = sessionmaker(bind=engine)()
    try:
        return TaggerDataset(
            list(dataset_ids),
            vocabulary,
            session,
            processor=None,
            max_source_pixels=max_source_pixels,
        )
    finally:
        session.close()
        engine.dispose()


def _model(config: FoveaArtifactConfig) -> FoveaTagger:
    return FoveaTagger(
        config.num_tags,
        encoder_config=config.encoder,
        feature_dim=config.feature_dim,
        evidence_slots_per_view=config.evidence_slots_per_view,
        pool_heads=config.pool_heads,
        temperature=config.temperature,
    )


def _encoder_config(metadata: Mapping[str, Any]) -> FoveaEncoderConfig:
    if metadata.get("initialization") != "scratch" or metadata.get("external_weights") != []:
        raise ValueError("Phase 3 requires a pure-scratch pretrained encoder")
    value = dict(metadata["run_contract"]["encoder"])
    for name in ("widths", "depths", "attention_heads"):
        value[name] = tuple(value[name])
    return FoveaEncoderConfig(**value)


def _loader(dataset, *, batch_size: int, workers: int) -> DataLoader:
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": False,
        "collate_fn": collate_supervised_samples,
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
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=effective_warmup
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[effective_warmup]
    )


def _restore_accumulator(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if value is None:
        return {"loss_sum": 0.0, "samples": 0, "skipped": 0, "optimizer_steps": 0, "seconds": 0.0}
    return {
        "loss_sum": float(value["loss_sum"]),
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
        "loss": float(accumulator["loss_sum"]) / samples if samples else None,
    }


def _normalized_domain_weights(weights, counts) -> dict[int, float]:
    if set(weights) != set(counts) or any(float(value) <= 0 for value in weights.values()):
        raise ValueError("domain weights must be positive and specify every training dataset")
    total = sum(float(value) for value in weights.values())
    return {int(domain): float(value) / total for domain, value in weights.items()}


def _holdout_contract(manifests: Sequence[Path]) -> tuple[list[str], set[str]]:
    paths: list[str] = []
    fine_tags: set[str] = set()
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            paths.append(str(item["image_path"]))
            fine_tags.update(str(tag) for tag in item.get("fine_detail_tags", []))
    return paths, fine_tags


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if callback is not None:
        callback(event, payload)


def _validate_arguments(**values) -> None:
    if values["input_mode"] not in {"overview_only", "overview_placebo", "source_native"}:
        raise ValueError("unsupported input_mode")
    _normalized_domain_weights(values["domain_weights"], values["train_counts"])
    if set(values["validation_counts"]) != set(values["train_counts"]):
        raise ValueError("validation counts must specify every training dataset")
    if any(int(value) <= 0 for value in values["train_counts"].values()):
        raise ValueError("training counts must be positive")
    if any(int(value) <= 0 for value in values["validation_counts"].values()):
        raise ValueError("validation counts must be positive")
    for name in ("epochs", "epoch_samples", "batch_size", "checkpoint_every_steps", "log_every"):
        if int(values[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(values["workers"]) < 0 or int(values["warmup_steps"]) < 0:
        raise ValueError("workers and warmup_steps must be non-negative")
    if int(values["local_views"]) <= 0:
        raise ValueError("local_views must be positive for compute-matched B/C branches")
    if float(values["learning_rate"]) <= 0 or float(values["weight_decay"]) < 0:
        raise ValueError("optimizer settings are invalid")
    if not 0 <= float(values["photo_loss_weight"]) <= 1:
        raise ValueError("photo_loss_weight must be within [0, 1]")
    if values["max_source_pixels"] is not None and values["max_source_pixels"] <= 0:
        raise ValueError("max_source_pixels must be positive when provided")
    if values["stop_after_steps"] is not None and values["stop_after_steps"] <= 0:
        raise ValueError("stop_after_steps must be positive when provided")


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-gate", type=Path, required=True)
    parser.add_argument("--pretrained-encoder", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument(
        "--input-mode",
        choices=(*PHASE3_INPUT_MODES, "all"),
        required=True,
    )
    parser.add_argument("--train-count", action="append", required=True)
    parser.add_argument("--validation-count", action="append", required=True)
    parser.add_argument("--domain-weight", action="append", required=True)
    parser.add_argument("--holdout-manifest", type=Path, action="append", default=[])
    parser.add_argument("--photo-dataset-id", type=int)
    parser.add_argument("--photo-loss-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epoch-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--local-views", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-source-megapixels", type=float, default=40.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    common = dict(
        phase2_gate=args.phase2_gate,
        pretrained_encoder=args.pretrained_encoder,
        vocabulary_path=args.vocabulary,
        datasets_db=args.datasets_db,
        comparison_dir=args.comparison_dir,
        train_counts=_parse_mapping(args.train_count, cast=int, name="train-count"),
        validation_counts=_parse_mapping(
            args.validation_count, cast=int, name="validation-count"
        ),
        domain_weights=_parse_mapping(args.domain_weight, cast=float, name="domain-weight"),
        holdout_manifests=args.holdout_manifest,
        photo_dataset_id=args.photo_dataset_id,
        photo_loss_weight=args.photo_loss_weight,
        epochs=args.epochs,
        epoch_samples=args.epoch_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        local_views=args.local_views,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        log_every=args.log_every,
        seed=args.seed,
        max_source_pixels=(
            None if args.max_source_megapixels == 0
            else round(args.max_source_megapixels * 1_000_000)
        ),
        stop_after_steps=args.stop_after_steps,
    )
    if args.input_mode == "all":
        if args.resume is not None:
            raise ValueError("--resume is selected automatically when --input-mode=all")
        summary = run_supervised_comparison(**common)
    else:
        summary = run_supervised_training(
            **common,
            input_mode=args.input_mode,
            resume=args.resume,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
