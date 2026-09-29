"""Gate-bound, resumable FoveaTag stage-4 joint fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch.utils.data import Subset

from ..tag_vocabulary import TagVocabulary
from ..tagger_loss import AsymmetricLossOptimized
from .artifact import FoveaArtifactConfig, load_inference_artifact, save_inference_artifact
from .control_run import select_domain_splits
from .joint_checkpoint import load_joint_checkpoint, save_joint_checkpoint
from .joint_training import FoveaJointTrainer
from .model import FoveaTagger
from .router import EvidenceRouter
from .router_checkpoint import load_router_weights
from .router_data import FoveaRouterDataset
from .router_experiment import validate_router_teacher
from .router_run import _loader
from .router_training import FoveaRouterTrainer
from .run_control import FoveaRunControl
from .supervised_data import DomainBalancedSupervisedDataset, SupervisionPolicy
from .supervised_run import (
    _cosine_with_warmup,
    _file_sha256,
    _holdout_contract,
    _load_raw_dataset,
    _normalized_domain_weights,
    _parse_mapping,
)


def run_joint_training(
    *,
    phase3_gate: Path,
    source_native_model: Path,
    router_weights: Path,
    datasets_db: Path,
    output_dir: Path,
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
    candidate_count: int,
    selection_budget: int,
    random_count: int,
    oracle_interval: int,
    oracle_chunk_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    checkpoint_every_steps: int,
    log_every: int,
    seed: int,
    max_source_pixels: Optional[int] = 40_000_000,
    router_weight: float = 1.0,
    diversity_weight: float = 0.05,
    resume: Optional[Path] = None,
    stop_after_steps: Optional[int] = None,
    progress_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
    run_control: Optional[FoveaRunControl] = None,
) -> dict[str, Any]:
    _validate_arguments(
        train_counts=train_counts,
        validation_counts=validation_counts,
        domain_weights=domain_weights,
        epochs=epochs,
        epoch_samples=epoch_samples,
        batch_size=batch_size,
        workers=workers,
        candidate_count=candidate_count,
        selection_budget=selection_budget,
        random_count=random_count,
        oracle_interval=oracle_interval,
        oracle_chunk_size=oracle_chunk_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        log_every=log_every,
        photo_loss_weight=photo_loss_weight,
        max_source_pixels=max_source_pixels,
        router_weight=router_weight,
        diversity_weight=diversity_weight,
        stop_after_steps=stop_after_steps,
    )
    gate, source_metadata, artifact_config, router_metadata = validate_joint_sources(
        phase3_gate,
        source_native_model,
        router_weights,
    )
    validate_joint_budget(
        router_metadata,
        candidate_count=candidate_count,
        selection_budget=selection_budget,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FoveaTag joint fine-tuning")
    _prepare_output_dir(output_dir, resume=resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = TagVocabulary.from_dict(source_metadata["vocabulary"])
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
        raise ValueError("joint validation split must not be empty")
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
    candidate_views = replace(artifact_config.views, local_views=candidate_count)
    train_source = FoveaRouterDataset(
        raw_dataset,
        view_config=candidate_views,
        candidate_count=candidate_count,
        seed=seed,
        training=True,
        supervision_policies=policies,
    )
    train_dataset = DomainBalancedSupervisedDataset(
        train_source,
        domain_indices,
        domain_weights,
        epoch_samples=epoch_samples,
        seed=seed,
    )
    validation_dataset = FoveaRouterDataset(
        raw_dataset,
        view_config=candidate_views,
        candidate_count=candidate_count,
        seed=seed,
        training=False,
        supervision_policies=policies,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _model(artifact_config)
    load_inference_artifact(source_native_model, model)
    router = EvidenceRouter(
        feature_dim=artifact_config.encoder.widths[2],
        routing_dim=artifact_config.routing_dim,
        routing_slots=artifact_config.routing_slots,
    )
    load_router_weights(router_weights, router)
    device = torch.device("cuda")
    model.to(device)
    router.to(device)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *router.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
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
    trainer = FoveaJointTrainer(
        model,
        router,
        criterion,
        optimizer,
        device=device,
        selection_budget=selection_budget,
        random_count=random_count,
        oracle_interval=oracle_interval,
        oracle_chunk_size=oracle_chunk_size,
        router_weight=router_weight,
        diversity_weight=diversity_weight,
        mixed_precision=amp_dtype,
        scheduler=scheduler,
    )
    utility_validator = FoveaRouterTrainer(
        model,
        router,
        optimizer,
        device=device,
        mixed_precision=amp_dtype,
        oracle_chunk_size=oracle_chunk_size,
    )
    run_contract = {
        "format_version": 1,
        "kind": "foveatag_phase4_joint",
        "phase3_gate_sha256": _file_sha256(phase3_gate),
        "source_native_model_sha256": _file_sha256(source_native_model),
        "source_native_metadata_sha256": _file_sha256(_sidecar_path(source_native_model)),
        "router_weights_sha256": _file_sha256(router_weights),
        "router_metadata_sha256": _file_sha256(_sidecar_path(router_weights)),
        "datasets_db": str(datasets_db.resolve()),
        "train_counts": {str(key): int(value) for key, value in sorted(train_counts.items())},
        "validation_counts": {
            str(key): int(value) for key, value in sorted(validation_counts.items())
        },
        "domain_weights": {
            str(key): value
            for key, value in sorted(_normalized_domain_weights(domain_weights, train_counts).items())
        },
        "holdout_manifests": [str(path.resolve()) for path in holdout_manifests],
        "photo_dataset_id": photo_dataset_id,
        "photo_loss_weight": photo_loss_weight,
        "fine_detail_positive_only_tags": len(fine_indices),
        "epochs": epochs,
        "epoch_samples": epoch_samples,
        "batch_size": batch_size,
        "workers": workers,
        "candidate_count": candidate_count,
        "selection_budget": selection_budget,
        "random_count": random_count,
        "oracle_interval": oracle_interval,
        "oracle_chunk_size": oracle_chunk_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": min(warmup_steps, max(total_steps - 1, 0)),
        "checkpoint_every_steps": checkpoint_every_steps,
        "seed": seed,
        "views": asdict(candidate_views),
        "router_weight": router_weight,
        "diversity_weight": diversity_weight,
        "max_source_pixels": max_source_pixels,
        "gate_decision": gate["decision"],
        "router_source_validation": dict(router_metadata["validation"]),
    }
    start_epoch = sample_offset = global_step = 0
    best_f1 = -1.0
    best_threshold = 0.5
    history: list[dict[str, Any]] = []
    accumulator = None
    if resume is not None:
        payload = load_joint_checkpoint(
            resume,
            model,
            router,
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
    status = "complete"
    resume_epoch = start_epoch
    steps_this_run = 0
    run_started = time.monotonic()
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
                output_dir=output_dir,
                best_f1=best_f1,
                best_threshold=best_threshold,
                run_contract=run_contract,
                history=history,
                initial_accumulator=(accumulator if epoch == start_epoch else None),
                remaining_step_budget=(
                    None if stop_after_steps is None else stop_after_steps - steps_this_run
                ),
                seed=seed,
                progress_callback=progress_callback,
                run_control=run_control,
            )
            steps_this_run += added
            if stopped:
                sample_offset = epoch_offset
                resume_epoch = epoch
                status = stopped
                break
            validation_subset = Subset(validation_dataset, validation_indices)
            validation_started = time.monotonic()
            validation = trainer.validate_epoch(
                _loader(validation_subset, batch_size=batch_size, workers=workers)
            )
            utility = utility_validator.validate(
                _loader(validation_subset, batch_size=batch_size, workers=workers),
                budget=selection_budget,
                random_seed=seed,
            )
            validation_seconds = time.monotonic() - validation_started
            router_beats_random = (
                utility.mean_utility_recovery > utility.random_mean_utility_recovery
            )
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train": trained,
                "validation": asdict(validation),
                "router_validation": asdict(utility),
                "router_beats_random": router_beats_random,
                "validation_seconds": validation_seconds,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            _emit_progress(progress_callback, "validation", row)
            if router_beats_random and validation.macro_f1 > best_f1:
                best_f1 = validation.macro_f1
                best_threshold = validation.threshold
                save_inference_artifact(
                    output_dir / "best_joint.safetensors",
                    model,
                    artifact_config,
                    router=router,
                    vocabulary=vocabulary.to_dict(),
                    provenance={
                        "initialization": "scratch",
                        "external_weights": [],
                        "stage": "joint_routed_finetune",
                        "phase3_gate_sha256": run_contract["phase3_gate_sha256"],
                        "source_native_model_sha256": run_contract["source_native_model_sha256"],
                        "router_weights_sha256": run_contract["router_weights_sha256"],
                        "epoch": epoch,
                        "global_step": global_step,
                        "validation_f1": best_f1,
                        "threshold": best_threshold,
                        "router_validation": asdict(utility),
                    },
                )
            sample_offset = 0
            resume_epoch = epoch + 1
            accumulator = None
            save_joint_checkpoint(
                output_dir / "latest.pt",
                model,
                router,
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
            _write_json_atomic(output_dir / "history.json", history)
            if stop_after_steps is not None and steps_this_run >= stop_after_steps:
                status = "stopped_after_requested_steps"
                break
        else:
            resume_epoch = epochs
            sample_offset = 0
        accepted_artifact = (output_dir / "best_joint.safetensors").is_file()
        summary = {
            "format_version": 1,
            "kind": "foveatag_phase4_joint_run",
            "status": status,
            "epoch": resume_epoch,
            "sample_offset": sample_offset,
            "global_step": global_step,
            "steps_this_run": steps_this_run,
            "best_validation_f1": best_f1,
            "best_threshold": best_threshold,
            "has_accepted_artifact": accepted_artifact,
            "acceptance_status": "passed" if accepted_artifact else "failed_router_gate",
            "elapsed_seconds_this_run": time.monotonic() - run_started,
            "dtype": str(amp_dtype).removeprefix("torch."),
            "model_parameter_count": model.parameter_count(),
            "router_parameter_count": sum(parameter.numel() for parameter in router.parameters()),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "available_after_holdout": {str(key): value for key, value in available.items()},
            "run_contract": run_contract,
            "history": history,
        }
        _write_json_atomic(output_dir / "summary.json", summary)
        return summary
    finally:
        del utility_validator, trainer, criterion, optimizer, scheduler, router, model
        gc.collect()
        torch.cuda.empty_cache()


def validate_joint_sources(
    phase3_gate: Path | str,
    source_native_model: Path | str,
    router_weights: Path | str,
) -> tuple[dict[str, Any], dict[str, Any], FoveaArtifactConfig, dict[str, Any]]:
    gate, source_metadata, config = validate_router_teacher(
        phase3_gate,
        source_native_model,
    )
    router_path = Path(router_weights)
    router_metadata = json.loads(_sidecar_path(router_path).read_text(encoding="utf-8"))
    if router_metadata.get("format_version") != 1 or router_metadata.get("kind") != "foveatag_router_weights":
        raise ValueError("router artifact is not FoveaTag router weights")
    contract = router_metadata.get("run_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("router artifact has no run contract")
    expected = {
        "phase3_gate_sha256": _file_sha256(Path(phase3_gate)),
        "source_native_model_sha256": _file_sha256(Path(source_native_model)),
        "source_native_metadata_sha256": _file_sha256(_sidecar_path(Path(source_native_model))),
    }
    for name, digest in expected.items():
        if contract.get(name) != digest:
            raise ValueError(f"router artifact {name} does not match the joint source")
    validation = router_metadata.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("router artifact has no fixed-budget validation")
    learned = validation.get("mean_utility_recovery")
    random_baseline = validation.get("random_mean_utility_recovery")
    if not isinstance(learned, (int, float)) or not isinstance(random_baseline, (int, float)):
        raise ValueError("router artifact lacks learned-versus-random utility recovery")
    if learned <= random_baseline:
        raise ValueError("router artifact did not beat random views at the same K")
    if int(validation.get("budget", -1)) != int(contract.get("selection_budget", -2)):
        raise ValueError("router validation budget does not match its run contract")
    if int(validation.get("candidates_per_sample", -1)) != int(contract.get("candidate_count", -2)):
        raise ValueError("router validation candidates do not match its run contract")
    return gate, source_metadata, config, router_metadata


def validate_joint_budget(
    router_metadata: Mapping[str, Any],
    *,
    candidate_count: int,
    selection_budget: int,
) -> None:
    contract = router_metadata.get("run_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("router artifact has no run contract")
    if selection_budget != int(contract.get("selection_budget", -1)):
        raise ValueError("joint selection_budget must match the validated router budget")
    if candidate_count != int(contract.get("candidate_count", -1)):
        raise ValueError("joint candidate_count must match the validated router candidate set")


def _train_epoch(
    trainer: FoveaJointTrainer,
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
    seed: int,
    progress_callback: Optional[Callable[[str, Mapping[str, Any]], None]],
    run_control: Optional[FoveaRunControl],
) -> tuple[dict[str, Any], int, int, int, Optional[str], dict[str, Any]]:
    if min(epoch, sample_offset, global_step) < 0 or sample_offset > len(dataset):
        raise ValueError("joint epoch progress is invalid")
    if min(batch_size, checkpoint_every_steps, log_every) <= 0 or workers < 0:
        raise ValueError("joint loader and interval settings are invalid")
    accumulator = _restore_accumulator(initial_accumulator)
    if run_control is not None and run_control.should_stop():
        _save_progress(trainer, output_dir, epoch, sample_offset, global_step, best_f1, best_threshold, run_contract, history, accumulator)
        return _training_summary(accumulator), sample_offset, global_step, 0, "stopped_by_request", accumulator
    if remaining_step_budget is not None and remaining_step_budget <= 0:
        return _training_summary(accumulator), sample_offset, global_step, 0, "stopped_after_requested_steps", accumulator
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
            _save_progress(trainer, output_dir, epoch, sample_offset + consumed, global_step, best_f1, best_threshold, run_contract, history, accumulator)
            return _training_summary(accumulator), sample_offset + consumed, global_step, steps_added, "stopped_by_request", accumulator
        requested = min(batch_size, len(dataset) - sample_offset - consumed)
        consumed += requested
        if batch is None:
            accumulator["skipped"] += requested
            continue
        result = trainer.train_batch(
            batch,
            global_step=global_step,
            random_seed=seed + global_step,
        )
        if result is None or not math.isfinite(result.loss):
            raise FloatingPointError(f"non-finite joint loss at global step {global_step}")
        actual = int(batch.labels.shape[0])
        for name, value in (
            ("loss_sum", result.loss),
            ("classification_sum", result.classification),
            ("ranking_sum", result.router_ranking),
            ("diversity_sum", result.router_diversity),
        ):
            accumulator[name] += value * actual
        accumulator["samples"] += actual
        accumulator["skipped"] += requested - actual
        accumulator["optimizer_steps"] += 1
        accumulator["oracle_steps"] += int(result.oracle_used)
        accumulator["selected_views"] += result.selected_views
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
                "classification": result.classification,
                "oracle_used": result.oracle_used,
                "samples_per_second": accumulator["samples"] / max(elapsed, 1e-9),
            }
            print(
                f"[FoveaTagJoint] epoch={epoch} sample={current_offset}/{len(dataset)} "
                f"step={global_step} loss={result.loss:.6f} "
                f"samples_per_second={message['samples_per_second']:.2f}",
                flush=True,
            )
            _emit_progress(progress_callback, "train", message)
        should_checkpoint = global_step % checkpoint_every_steps == 0
        should_stop = remaining_step_budget is not None and steps_added >= remaining_step_budget
        if should_checkpoint or should_stop:
            now = time.monotonic()
            accumulator["seconds"] += now - segment_started
            segment_started = now
            _save_progress(trainer, output_dir, epoch, current_offset, global_step, best_f1, best_threshold, run_contract, history, accumulator)
        if should_stop:
            return _training_summary(accumulator), current_offset, global_step, steps_added, "stopped_after_requested_steps", accumulator
    accumulator["seconds"] += time.monotonic() - segment_started
    return _training_summary(accumulator), len(dataset), global_step, steps_added, None, accumulator


def _save_progress(trainer, output_dir, epoch, offset, step, best_f1, threshold, contract, history, accumulator):
    save_joint_checkpoint(
        output_dir / "latest.pt",
        trainer.model,
        trainer.router,
        trainer.optimizer,
        scheduler=trainer.scheduler,
        epoch=epoch,
        sample_offset=offset,
        global_step=step,
        best_validation_f1=best_f1,
        best_threshold=threshold,
        run_contract=contract,
        history=list(history),
        epoch_accumulator=accumulator,
    )


def _restore_accumulator(value):
    defaults = {
        "loss_sum": 0.0, "classification_sum": 0.0, "ranking_sum": 0.0,
        "diversity_sum": 0.0, "samples": 0, "skipped": 0,
        "optimizer_steps": 0, "oracle_steps": 0, "selected_views": 0,
        "seconds": 0.0,
    }
    if value is None:
        return defaults
    return {
        name: int(value.get(name, default)) if isinstance(default, int) else float(value.get(name, default))
        for name, default in defaults.items()
    }


def _training_summary(value):
    samples = int(value["samples"])
    divisor = max(samples, 1)
    return {
        "mean_loss": float(value["loss_sum"]) / divisor,
        "mean_classification": float(value["classification_sum"]) / divisor,
        "mean_router_ranking": float(value["ranking_sum"]) / divisor,
        "mean_router_diversity": float(value["diversity_sum"]) / divisor,
        "samples": samples,
        "skipped": int(value["skipped"]),
        "optimizer_steps": int(value["optimizer_steps"]),
        "oracle_steps": int(value["oracle_steps"]),
        "selected_views": int(value["selected_views"]),
        "seconds": float(value["seconds"]),
    }


def _model(config: FoveaArtifactConfig) -> FoveaTagger:
    return FoveaTagger(
        config.num_tags,
        encoder_config=config.encoder,
        feature_dim=config.feature_dim,
        evidence_slots_per_view=config.evidence_slots_per_view,
        pool_heads=config.pool_heads,
        temperature=config.temperature,
    )


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metadata.json")


def _prepare_output_dir(output_dir: Path, *, resume: Optional[Path]) -> None:
    if resume is not None:
        return
    occupied = [
        path.name
        for path in (output_dir / "latest.pt", output_dir / "best_joint.safetensors")
        if path.exists()
    ]
    if occupied:
        raise ValueError(
            f"joint output directory already contains run artifacts: {occupied}; use --resume"
        )


def _validate_arguments(**values: Any) -> None:
    if not values["train_counts"] or not values["validation_counts"]:
        raise ValueError("train and validation counts must not be empty")
    if set(values["domain_weights"]) != set(values["train_counts"]):
        raise ValueError("domain weights must match train-count domains")
    positive = (
        "epochs", "epoch_samples", "batch_size", "candidate_count",
        "selection_budget", "oracle_interval", "oracle_chunk_size",
        "checkpoint_every_steps", "log_every",
    )
    if any(int(values[name]) <= 0 for name in positive):
        raise ValueError("joint counts, budgets, and intervals must be positive")
    if not 0 < values["random_count"] < values["selection_budget"]:
        raise ValueError("random_count must retain part, but not all, of selection_budget")
    if values["selection_budget"] > values["candidate_count"]:
        raise ValueError("selection_budget cannot exceed candidate_count")
    if values["workers"] < 0 or values["warmup_steps"] < 0:
        raise ValueError("workers and warmup_steps must be non-negative")
    if values["learning_rate"] <= 0 or values["weight_decay"] < 0:
        raise ValueError("optimizer settings are invalid")
    if not 0 <= values["photo_loss_weight"] <= 1:
        raise ValueError("photo_loss_weight must be within [0, 1]")
    if min(values["router_weight"], values["diversity_weight"]) < 0:
        raise ValueError("router loss weights must be non-negative")
    if values["max_source_pixels"] is not None and values["max_source_pixels"] <= 0:
        raise ValueError("max_source_pixels must be positive when supplied")
    if values["stop_after_steps"] is not None and values["stop_after_steps"] <= 0:
        raise ValueError("stop_after_steps must be positive when supplied")


def _emit_progress(callback, event, payload):
    if callback is not None:
        callback(event, payload)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-gate", type=Path, required=True)
    parser.add_argument("--source-native-model", type=Path, required=True)
    parser.add_argument("--router-weights", type=Path, required=True)
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", action="append", required=True)
    parser.add_argument("--validation-count", action="append", required=True)
    parser.add_argument("--domain-weight", action="append", required=True)
    parser.add_argument("--holdout-manifest", type=Path, action="append", default=[])
    parser.add_argument("--photo-dataset-id", type=int)
    parser.add_argument("--photo-loss-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--epoch-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--selection-budget", type=int, default=4)
    parser.add_argument("--random-count", type=int, default=1)
    parser.add_argument("--oracle-interval", type=int, default=8)
    parser.add_argument("--oracle-chunk-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-source-megapixels", type=float, default=40.0)
    parser.add_argument("--router-weight", type=float, default=1.0)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_joint_training(
        phase3_gate=args.phase3_gate,
        source_native_model=args.source_native_model,
        router_weights=args.router_weights,
        datasets_db=args.datasets_db,
        output_dir=args.output_dir,
        train_counts=_parse_mapping(args.train_count, cast=int, name="train-count"),
        validation_counts=_parse_mapping(args.validation_count, cast=int, name="validation-count"),
        domain_weights=_parse_mapping(args.domain_weight, cast=float, name="domain-weight"),
        holdout_manifests=args.holdout_manifest,
        photo_dataset_id=args.photo_dataset_id,
        photo_loss_weight=args.photo_loss_weight,
        epochs=args.epochs,
        epoch_samples=args.epoch_samples,
        batch_size=args.batch_size,
        workers=args.workers,
        candidate_count=args.candidate_count,
        selection_budget=args.selection_budget,
        random_count=args.random_count,
        oracle_interval=args.oracle_interval,
        oracle_chunk_size=args.oracle_chunk_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        log_every=args.log_every,
        seed=args.seed,
        max_source_pixels=(None if args.max_source_megapixels == 0 else round(args.max_source_megapixels * 1_000_000)),
        router_weight=args.router_weight,
        diversity_weight=args.diversity_weight,
        resume=args.resume,
        stop_after_steps=args.stop_after_steps,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
