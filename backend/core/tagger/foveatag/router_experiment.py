"""Gate-bound, resumable Phase-4 router training on project datasets."""

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
from .artifact import FoveaArtifactConfig, load_inference_artifact
from .control_run import select_domain_splits
from .model import FoveaTagger
from .phase3_gate import (
    load_passed_phase3_gate,
    source_native_checkpoint_hashes,
)
from .router import EvidenceRouter
from .router_checkpoint import (
    load_router_checkpoint,
    save_router_checkpoint,
    save_router_weights,
)
from .router_data import FoveaRouterDataset
from .router_run import _loader, train_router_epoch
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


def run_router_training(
    *,
    phase3_gate: Path,
    source_native_model: Path,
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
    oracle_chunk_size: int,
    selection_budget: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    checkpoint_every_steps: int,
    log_every: int,
    seed: int,
    max_source_pixels: Optional[int] = 40_000_000,
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
        oracle_chunk_size=oracle_chunk_size,
        selection_budget=selection_budget,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        log_every=log_every,
        photo_loss_weight=photo_loss_weight,
        max_source_pixels=max_source_pixels,
        diversity_weight=diversity_weight,
        stop_after_steps=stop_after_steps,
    )
    gate, artifact_metadata, artifact_config = validate_router_teacher(
        phase3_gate, source_native_model
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FoveaTag Phase-4 router training")
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = TagVocabulary.from_dict(artifact_metadata["vocabulary"])
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
        raise ValueError("router validation split must not be empty")
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    router = EvidenceRouter(
        feature_dim=artifact_config.encoder.widths[2],
        routing_dim=artifact_config.routing_dim,
        routing_slots=artifact_config.routing_slots,
    )
    device = torch.device("cuda")
    model.to(device).eval()
    router.to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = epochs * math.ceil(epoch_samples / batch_size)
    scheduler = _cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    trainer = FoveaRouterTrainer(
        model,
        router,
        optimizer,
        device=device,
        mixed_precision=amp_dtype,
        oracle_chunk_size=oracle_chunk_size,
        diversity_weight=diversity_weight,
        scheduler=scheduler,
    )
    run_contract = {
        "format_version": 1,
        "kind": "foveatag_phase4_router",
        "phase3_gate": str(phase3_gate.resolve()),
        "phase3_gate_sha256": _file_sha256(phase3_gate),
        "source_native_model": str(source_native_model.resolve()),
        "source_native_model_sha256": _file_sha256(source_native_model),
        "source_native_metadata_sha256": _file_sha256(_metadata_path(source_native_model)),
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
        "oracle_chunk_size": oracle_chunk_size,
        "selection_budget": selection_budget,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": min(warmup_steps, max(total_steps - 1, 0)),
        "checkpoint_every_steps": checkpoint_every_steps,
        "seed": seed,
        "views": asdict(candidate_views),
        "diversity_weight": diversity_weight,
        "max_source_pixels": max_source_pixels,
        "gate_decision": gate["decision"],
    }
    start_epoch = sample_offset = global_step = 0
    best_recovery = -1.0
    history: list[dict[str, Any]] = []
    accumulator = None
    if resume is not None:
        payload = load_router_checkpoint(
            resume,
            router,
            optimizer,
            scheduler=scheduler,
            expected_run_contract=run_contract,
        )
        start_epoch = int(payload["epoch"])
        sample_offset = int(payload["sample_offset"])
        global_step = int(payload["global_step"])
        best_recovery = float(payload["best_utility_recovery"])
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
            trained, epoch_offset, global_step, added, stopped, accumulator = train_router_epoch(
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
                best_utility_recovery=best_recovery,
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
            validation = trainer.validate(
                _loader(
                    Subset(validation_dataset, validation_indices),
                    batch_size=batch_size,
                    workers=workers,
                ),
                budget=selection_budget,
                random_seed=seed,
            )
            validation_seconds = time.monotonic() - validation_started
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train": trained,
                "validation": asdict(validation),
                "validation_seconds": validation_seconds,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            _emit_progress(progress_callback, "validation", row)
            if validation.mean_utility_recovery > best_recovery:
                best_recovery = validation.mean_utility_recovery
                save_router_weights(
                    output_dir / "best_router.safetensors",
                    router,
                    run_contract=run_contract,
                    epoch=epoch,
                    global_step=global_step,
                    validation=asdict(validation),
                )
            sample_offset = 0
            resume_epoch = epoch + 1
            accumulator = None
            save_router_checkpoint(
                output_dir / "latest.pt",
                router,
                optimizer,
                scheduler=scheduler,
                epoch=resume_epoch,
                sample_offset=0,
                global_step=global_step,
                best_utility_recovery=best_recovery,
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
        summary = {
            "format_version": 1,
            "kind": "foveatag_phase4_router_run",
            "status": status,
            "epoch": resume_epoch,
            "sample_offset": sample_offset,
            "global_step": global_step,
            "steps_this_run": steps_this_run,
            "best_utility_recovery": best_recovery,
            "elapsed_seconds_this_run": time.monotonic() - run_started,
            "dtype": str(amp_dtype).removeprefix("torch."),
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
        del trainer, optimizer, scheduler, router, model
        gc.collect()
        torch.cuda.empty_cache()


def validate_router_teacher(
    phase3_gate: Path | str, source_native_model: Path | str
) -> tuple[dict[str, Any], dict[str, Any], FoveaArtifactConfig]:
    gate = load_passed_phase3_gate(phase3_gate)
    expected = source_native_checkpoint_hashes(gate)
    model_path = Path(source_native_model)
    metadata_path = _metadata_path(model_path)
    if _file_sha256(model_path) != expected["checkpoint_sha256"]:
        raise ValueError("source-native model does not match the passed Phase-3 gate")
    if _file_sha256(metadata_path) != expected["checkpoint_metadata_sha256"]:
        raise ValueError("source-native metadata does not match the passed Phase-3 gate")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("source-native artifact has no provenance")
    if provenance.get("initialization") != "scratch" or provenance.get("external_weights") != []:
        raise ValueError("Phase-4 teacher does not have pure-scratch lineage")
    if provenance.get("input_mode") != "source_native":
        raise ValueError("Phase-4 teacher is not the source-native branch")
    config = FoveaArtifactConfig.from_dict(metadata["config"])
    vocabulary = metadata.get("vocabulary")
    if not isinstance(vocabulary, Mapping) or not vocabulary.get("tag_to_idx"):
        raise ValueError("Phase-4 teacher has no vocabulary")
    return gate, metadata, config


def _model(config: FoveaArtifactConfig) -> FoveaTagger:
    return FoveaTagger(
        config.num_tags,
        encoder_config=config.encoder,
        feature_dim=config.feature_dim,
        evidence_slots_per_view=config.evidence_slots_per_view,
        pool_heads=config.pool_heads,
        temperature=config.temperature,
    )


def _metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metadata.json")


def _validate_arguments(**values: Any) -> None:
    if not values["train_counts"] or not values["validation_counts"]:
        raise ValueError("train and validation counts must not be empty")
    if set(values["domain_weights"]) != set(values["train_counts"]):
        raise ValueError("domain weights must match train-count domains")
    positive = (
        "epochs", "epoch_samples", "batch_size", "candidate_count",
        "oracle_chunk_size", "selection_budget", "checkpoint_every_steps", "log_every",
    )
    if any(int(values[name]) <= 0 for name in positive):
        raise ValueError("training counts, budgets, and intervals must be positive")
    if values["selection_budget"] > values["candidate_count"]:
        raise ValueError("selection_budget cannot exceed candidate_count")
    if values["workers"] < 0 or values["warmup_steps"] < 0:
        raise ValueError("workers and warmup_steps must be non-negative")
    if values["learning_rate"] <= 0 or values["weight_decay"] < 0:
        raise ValueError("optimizer settings are invalid")
    if not 0 <= values["photo_loss_weight"] <= 1 or values["diversity_weight"] < 0:
        raise ValueError("loss weights are invalid")
    if values["max_source_pixels"] is not None and values["max_source_pixels"] <= 0:
        raise ValueError("max_source_pixels must be positive when supplied")
    if values["stop_after_steps"] is not None and values["stop_after_steps"] <= 0:
        raise ValueError("stop_after_steps must be positive when supplied")


def _emit_progress(
    callback: Optional[Callable[[str, Mapping[str, Any]], None]],
    event: str,
    payload: Mapping[str, Any],
) -> None:
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
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", action="append", required=True)
    parser.add_argument("--validation-count", action="append", required=True)
    parser.add_argument("--domain-weight", action="append", required=True)
    parser.add_argument("--holdout-manifest", type=Path, action="append", default=[])
    parser.add_argument("--photo-dataset-id", type=int)
    parser.add_argument("--photo-loss-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--epoch-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--oracle-chunk-size", type=int, default=8)
    parser.add_argument("--selection-budget", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-source-megapixels", type=float, default=40.0)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_router_training(
        phase3_gate=args.phase3_gate,
        source_native_model=args.source_native_model,
        datasets_db=args.datasets_db,
        output_dir=args.output_dir,
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
        candidate_count=args.candidate_count,
        oracle_chunk_size=args.oracle_chunk_size,
        selection_budget=args.selection_budget,
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
        diversity_weight=args.diversity_weight,
        resume=args.resume,
        stop_after_steps=args.stop_after_steps,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
