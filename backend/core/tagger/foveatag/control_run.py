"""Reproducible Phase-1 fixed-encoder control training entry point."""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from ..siglip2_inference_manager import SigLIP2InferenceManager
from ..tag_vocabulary import TagVocabulary
from ..tagger_dataset import TaggerDataset
from ..tagger_loss import AsymmetricLossOptimized
from .control_checkpoint import (
    load_control_checkpoint,
    save_control_checkpoint,
    save_control_weights,
)
from .control_training import SigLIP2ControlTrainer
from .siglip2_control import SigLIP2ControlTagger
from .siglip2_control_data import collate_siglip2_control_samples
from .supervised_data import FoveaSupervisedDataset, SupervisionPolicy
from .views import ViewConfig


def select_domain_splits(
    source: TaggerDataset,
    *,
    train_counts: Mapping[int, int],
    validation_counts: Mapping[int, int],
    excluded_paths: Sequence[str] = (),
    seed: int = 0,
) -> tuple[list[int], list[int], dict[int, int]]:
    """Choose stable disjoint per-domain train/validation subsets."""
    domains = set(train_counts) | set(validation_counts)
    if not domains or any(value < 0 for value in (*train_counts.values(), *validation_counts.values())):
        raise ValueError("domain split counts must be non-negative and non-empty")
    excluded = {_path_key(path) for path in excluded_paths}
    candidates: dict[int, list[int]] = {domain: [] for domain in domains}
    for index in range(len(source)):
        domain = source.dataset_id_for_index(index)
        if domain in candidates and _path_key(source.source_path_for_index(index)) not in excluded:
            candidates[domain].append(index)

    available: dict[int, int] = {}
    train: list[int] = []
    validation: list[int] = []
    for domain in sorted(domains):
        ordered = sorted(
            candidates[domain],
            key=lambda index: _stable_rank(
                source.source_path_for_index(index), seed=seed, domain=domain
            ),
        )
        available[domain] = len(ordered)
        validation_count = int(validation_counts.get(domain, 0))
        train_count = int(train_counts.get(domain, 0))
        if validation_count + train_count > len(ordered):
            raise ValueError(
                f"dataset {domain} has {len(ordered)} eligible samples, fewer than "
                f"the requested {validation_count + train_count}"
            )
        validation.extend(ordered[:validation_count])
        train.extend(ordered[validation_count:validation_count + train_count])
    return train, validation, available


def domain_sample_weights(
    source: TaggerDataset,
    indices: Sequence[int],
    target_weights: Mapping[int, float],
) -> torch.Tensor:
    """Return per-index weights whose expected domain mass matches the target."""
    if not indices:
        raise ValueError("training indices must not be empty")
    if not target_weights or any(not math.isfinite(value) or value <= 0 for value in target_weights.values()):
        raise ValueError("target domain weights must be finite and positive")
    counts: dict[int, int] = {}
    domains = []
    for index in indices:
        domain = source.dataset_id_for_index(index)
        domains.append(domain)
        counts[domain] = counts.get(domain, 0) + 1
    missing = sorted(set(counts) - set(target_weights))
    if missing:
        raise ValueError(f"target weights are missing datasets: {missing}")
    total_target = sum(float(target_weights[domain]) for domain in counts)
    return torch.tensor(
        [float(target_weights[domain]) / total_target / counts[domain] for domain in domains],
        dtype=torch.double,
    )


def run_control_training(
    *,
    checkpoint: Path,
    vocabulary_path: Path,
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
    local_views: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    log_every: int,
    seed: int,
    max_source_pixels: Optional[int] = 40_000_000,
    resume: Optional[Path] = None,
) -> dict[str, Any]:
    if min(epochs, epoch_samples, batch_size) <= 0:
        raise ValueError("epochs, epoch_samples, and batch_size must be positive")
    if learning_rate <= 0 or weight_decay < 0 or warmup_steps < 0:
        raise ValueError("optimizer settings must be non-negative with a positive learning rate")
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if max_source_pixels is not None and max_source_pixels <= 0:
        raise ValueError("max_source_pixels must be positive when provided")
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary_path = _checkpoint_vocabulary(checkpoint, vocabulary_path)
    vocabulary = TagVocabulary.from_dict(
        json.loads(vocabulary_path.read_text(encoding="utf-8"))
    )
    (output_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    excluded_paths, fine_tags = _holdout_contract(holdout_manifests)

    engine = create_engine(
        f"sqlite:///{datasets_db.resolve().as_posix()}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    session = sessionmaker(bind=engine)()
    try:
        raw_dataset = TaggerDataset(
            sorted(set(train_counts) | set(validation_counts)),
            vocabulary,
            session,
            processor=None,
            max_source_pixels=max_source_pixels,
        )
    finally:
        session.close()
        engine.dispose()

    train_indices, validation_indices, available = select_domain_splits(
        raw_dataset,
        train_counts=train_counts,
        validation_counts=validation_counts,
        excluded_paths=excluded_paths,
        seed=seed,
    )
    if not validation_indices:
        raise ValueError("validation split must not be empty")
    fine_indices = tuple(
        sorted(
            vocabulary.tag_to_idx[tag]
            for tag in fine_tags
            if tag in vocabulary.tag_to_idx
        )
    )
    policies = {}
    if photo_dataset_id is not None:
        policies[photo_dataset_id] = SupervisionPolicy(
            loss_weight=photo_loss_weight,
            positive_only_indices=fine_indices,
        )
    view_config = ViewConfig(overview_size=448, local_size=256, local_views=local_views)
    train_dataset = FoveaSupervisedDataset(
        raw_dataset,
        view_config=view_config,
        seed=seed,
        training=True,
        supervision_policies=policies,
        baseline_overview=True,
    )
    validation_dataset = FoveaSupervisedDataset(
        raw_dataset,
        view_config=view_config,
        seed=seed,
        training=False,
        supervision_policies=policies,
        baseline_overview=True,
    )

    manager = SigLIP2InferenceManager()
    manager.load_model(str(checkpoint), str(vocabulary_path))
    processor = manager.processor
    is_naflex = manager.is_naflex
    encoder_path = Path(manager.vision_encoder_path) if manager.vision_encoder_path else None
    original = manager.model
    encoder = original.vision_encoder
    baseline_head = original.head
    hidden_size = int(encoder.config.hidden_size)
    manager.model = None
    del original
    gc.collect()
    torch.cuda.empty_cache()

    model = SigLIP2ControlTagger(
        encoder,
        hidden_size=hidden_size,
        feature_dim=512,
        num_tags=vocabulary.num_tags,
        evidence_slots_per_view=2,
        is_naflex=is_naflex,
        freeze_encoder=True,
        baseline_head=baseline_head,
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    steps_per_epoch = math.ceil(epoch_samples / batch_size)
    total_steps = epochs * steps_per_epoch
    effective_warmup = min(max(0, warmup_steps), max(0, total_steps - 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - effective_warmup),
        eta_min=learning_rate * 0.01,
    )
    if effective_warmup:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=effective_warmup,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[effective_warmup],
        )
    else:
        scheduler = cosine
    criterion = AsymmetricLossOptimized()
    trainer = SigLIP2ControlTrainer(
        model,
        criterion,
        optimizer,
        device="cuda",
        mixed_precision=torch.bfloat16,
        scheduler=scheduler,
    )
    provenance = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "vocabulary": str(vocabulary_path.resolve()),
        "vocabulary_sha256": _file_sha256(vocabulary_path),
        "vision_encoder": str(encoder_path.resolve()) if encoder_path is not None else None,
        "vision_encoder_sha256": (
            _file_sha256(encoder_path) if encoder_path is not None and encoder_path.is_file() else None
        ),
        "is_naflex": is_naflex,
        "control_head": "baseline_zero_residual_v1",
    }
    database_stat = datasets_db.stat()
    run_contract = {
        "format_version": 1,
        "kind": "foveatag_phase1_control_run",
        "datasets_db": str(datasets_db.resolve()),
        "datasets_db_size": database_stat.st_size,
        "datasets_db_mtime_ns": database_stat.st_mtime_ns,
        "train_counts": {str(key): int(value) for key, value in sorted(train_counts.items())},
        "validation_counts": {
            str(key): int(value) for key, value in sorted(validation_counts.items())
        },
        "domain_weights": {
            str(key): float(value) for key, value in sorted(domain_weights.items())
        },
        "holdout_manifests": [
            {
                "path": str(path.resolve()),
                "sha256": _file_sha256(path),
            }
            for path in holdout_manifests
        ],
        "photo_dataset_id": photo_dataset_id,
        "photo_loss_weight": float(photo_loss_weight),
        "fine_detail_positive_only_tags": len(fine_indices),
        "epochs": int(epochs),
        "epoch_samples": int(epoch_samples),
        "batch_size": int(batch_size),
        "local_views": int(local_views),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_steps": int(effective_warmup),
        "seed": int(seed),
        "max_source_pixels": max_source_pixels,
        "loss": {
            "kind": "asymmetric_loss",
            "gamma_negative": criterion.gamma_neg,
            "gamma_positive": criterion.gamma_pos,
            "negative_clip": criterion.clip,
            "epsilon": criterion.eps,
            "detach_focal_weight": criterion.disable_torch_grad_focal_loss,
            "reduction": criterion.reduction,
        },
    }
    start_epoch = global_step = 0
    best_f1 = 0.0
    best_threshold = 0.5
    history: list[dict[str, Any]] = []
    if resume is not None:
        payload = load_control_checkpoint(
            resume,
            model,
            optimizer,
            expected_encoder_provenance=provenance,
            expected_run_contract=run_contract,
            scheduler=scheduler,
        )
        start_epoch = int(payload["next_epoch"])
        global_step = int(payload["global_step"])
        best_f1 = float(payload.get("best_validation_f1", 0.0))
        best_threshold = float(payload.get("best_threshold", 0.5))
        saved_history = payload.get("history")
        if not isinstance(saved_history, list) or len(saved_history) != start_epoch:
            raise ValueError("control checkpoint history does not match next_epoch")
        history = [dict(row) for row in saved_history]

    sampler_generator = torch.Generator()
    sampler = WeightedRandomSampler(
        domain_sample_weights(raw_dataset, train_indices, domain_weights),
        num_samples=epoch_samples,
        replacement=True,
        generator=sampler_generator,
    )
    collate = functools.partial(
        collate_siglip2_control_samples,
        processor=processor,
        is_naflex=is_naflex,
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )

    try:
        for epoch in range(start_epoch, epochs):
            train_dataset.set_epoch(epoch)
            sampler_generator.manual_seed(seed + epoch)
            train_started = time.monotonic()
            trained = trainer.train_epoch(
                _logged_batches(train_loader, epoch=epoch, phase="train", every=log_every)
            )
            train_seconds = time.monotonic() - train_started
            validation_started = time.monotonic()
            validated = trainer.validate_epoch(
                _logged_batches(
                    validation_loader,
                    epoch=epoch,
                    phase="validation",
                    every=log_every,
                )
            )
            validation_seconds = time.monotonic() - validation_started
            global_step += trained.optimizer_steps
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train": vars(trained),
                "validation": vars(validated),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "train_samples_per_second": trained.samples / train_seconds,
            }
            history.append(row)
            improved = validated.macro_f1 > best_f1
            if improved:
                best_f1 = validated.macro_f1
                best_threshold = validated.threshold
            checkpoint_kwargs = {
                "next_epoch": epoch + 1,
                "global_step": global_step,
                "encoder_provenance": provenance,
                "run_contract": run_contract,
                "history": history,
                "scheduler": scheduler,
                "best_validation_f1": best_f1,
                "best_threshold": best_threshold,
            }
            save_control_checkpoint(
                output_dir / "latest.pt", model, optimizer, **checkpoint_kwargs
            )
            if improved:
                save_control_weights(
                    output_dir / "best_f1.safetensors",
                    model,
                    encoder_provenance=provenance,
                    run_contract=run_contract,
                    epoch=epoch,
                    global_step=global_step,
                    validation_f1=best_f1,
                    threshold=best_threshold,
                )
            _write_json(output_dir / "history.json", history)
        summary = {
            "format_version": 1,
            "kind": "foveatag_siglip2_control_run",
            "encoder_provenance": provenance,
            "run_contract": run_contract,
            "datasets_db": str(datasets_db.resolve()),
            "train_counts": {str(key): value for key, value in train_counts.items()},
            "validation_counts": {str(key): value for key, value in validation_counts.items()},
            "available_after_holdout": {str(key): value for key, value in available.items()},
            "domain_weights": {str(key): value for key, value in domain_weights.items()},
            "holdout_manifests": [str(path.resolve()) for path in holdout_manifests],
            "fine_detail_positive_only_tags": len(fine_indices),
            "photo_dataset_id": photo_dataset_id,
            "photo_loss_weight": photo_loss_weight,
            "epochs": epochs,
            "epoch_samples": epoch_samples,
            "batch_size": batch_size,
            "local_views": local_views,
            "max_source_pixels": max_source_pixels,
            "warmup_steps": effective_warmup,
            "best_validation_f1": best_f1,
            "best_threshold": best_threshold,
            "history": history,
        }
        _write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        del trainer, optimizer, criterion, model, encoder
        manager.unload()
        gc.collect()
        torch.cuda.empty_cache()


def _holdout_contract(manifests: Sequence[Path]) -> tuple[list[str], set[str]]:
    paths: list[str] = []
    fine_tags: set[str] = set()
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            paths.append(str(item["image_path"]))
            fine_tags.update(str(tag) for tag in item.get("fine_detail_tags", []))
    return paths, fine_tags


def _checkpoint_vocabulary(checkpoint: Path, fallback: Path) -> Path:
    paired = checkpoint.with_name(f"{checkpoint.stem}_vocabulary.json")
    return paired if paired.is_file() else fallback


def _stable_rank(path: str, *, seed: int, domain: int) -> bytes:
    value = f"{seed}\0{domain}\0{_path_key(path)}".encode("utf-8", errors="surrogatepass")
    return hashlib.blake2b(value, digest_size=16).digest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_key(path: str) -> str:
    # Database paths are identities here; resolving every entry touches storage
    # metadata and made million-item split construction prohibitively slow.
    return os.path.normcase(os.path.abspath(path))


def _parse_domain_values(values: Sequence[str], name: str) -> dict[int, float]:
    parsed: dict[int, float] = {}
    for value in values:
        try:
            raw_domain, raw_amount = value.split("=", 1)
            parsed[int(raw_domain)] = float(raw_amount)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} entries must use DATASET_ID=VALUE") from error
    return parsed


def _parse_domain_counts(values: Sequence[str], name: str) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for value in values:
        try:
            raw_domain, raw_amount = value.split("=", 1)
            amount = int(raw_amount)
            if amount < 0:
                raise ValueError
            parsed[int(raw_domain)] = amount
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} entries must use DATASET_ID=NON_NEGATIVE_INTEGER") from error
    return parsed


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _logged_batches(loader, *, epoch: int, phase: str, every: int):
    total = len(loader)
    for index, batch in enumerate(loader, start=1):
        if index == 1 or index % every == 0 or index == total:
            print(
                f"[FoveaTagControl] epoch={epoch} phase={phase} batch={index}/{total}",
                flush=True,
            )
        yield batch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", action="append", required=True)
    parser.add_argument("--validation-count", action="append", required=True)
    parser.add_argument("--domain-weight", action="append", required=True)
    parser.add_argument("--holdout-manifest", type=Path, action="append", default=[])
    parser.add_argument("--photo-dataset-id", type=int)
    parser.add_argument("--photo-loss-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--epoch-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-views", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--max-source-megapixels",
        type=float,
        default=40.0,
        help="Skip sources above this full-decode cap; use 0 to disable",
    )
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_control_training(
        checkpoint=args.checkpoint,
        vocabulary_path=args.vocabulary,
        datasets_db=args.datasets_db,
        output_dir=args.output_dir,
        train_counts=_parse_domain_counts(args.train_count, "train-count"),
        validation_counts=_parse_domain_counts(args.validation_count, "validation-count"),
        domain_weights=_parse_domain_values(args.domain_weight, "domain-weight"),
        holdout_manifests=args.holdout_manifest,
        photo_dataset_id=args.photo_dataset_id,
        photo_loss_weight=args.photo_loss_weight,
        epochs=args.epochs,
        epoch_samples=args.epoch_samples,
        batch_size=args.batch_size,
        local_views=args.local_views,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        seed=args.seed,
        max_source_pixels=(
            round(args.max_source_megapixels * 1_000_000)
            if args.max_source_megapixels > 0
            else None
        ),
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
