"""Measure compute-matched Phase-3 A/B/C optimizer throughput on real images."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ..tag_vocabulary import TagVocabulary
from ..tagger_dataset import RawTaggerSample, SourcePixelLimitError, _load_image
from ..tagger_loss import AsymmetricLossOptimized
from .model import FoveaTagger
from .pretraining_benchmark import _summary, registered_image_paths
from .supervised_data import (
    FoveaInputMode,
    FoveaSupervisedDataset,
    PHASE3_INPUT_MODES,
    collate_supervised_samples,
)
from .views import ViewConfig


class _BenchmarkRawDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        *,
        num_tags: int,
        max_source_pixels: Optional[int],
    ) -> None:
        if not paths or num_tags <= 0:
            raise ValueError("paths and num_tags must be non-empty and positive")
        self.paths = tuple(str(path) for path in paths)
        self.num_tags = int(num_tags)
        self.max_source_pixels = max_source_pixels

    def __len__(self) -> int:
        return len(self.paths)

    def get_raw_item(self, index: int) -> Optional[RawTaggerSample]:
        path = self.paths[index]
        try:
            image = _load_image(path, max_pixels=self.max_source_pixels)
        except (SourcePixelLimitError, OSError, ValueError):
            return None
        label = torch.zeros(self.num_tags, dtype=torch.float32)
        label[index % self.num_tags] = 1.0
        return RawTaggerSample(
            image=image,
            label=label,
            loss_mask=torch.ones(self.num_tags, dtype=torch.float32),
            source_path=path,
        )

    def label_and_mask_for_index(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = torch.zeros(self.num_tags, dtype=torch.float32)
        label[index % self.num_tags] = 1.0
        return label, torch.ones(self.num_tags, dtype=torch.float32)


def benchmark_supervised_modes(
    image_paths: Sequence[str],
    *,
    num_tags: int,
    modes: Sequence[FoveaInputMode] = PHASE3_INPUT_MODES,
    batch_size: int = 1,
    workers: int = 2,
    local_views: int = 4,
    warmup_steps: int = 2,
    measured_steps: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 20260929,
    max_source_pixels: Optional[int] = 40_000_000,
    projected_epoch_samples: Optional[int] = None,
) -> dict[str, Any]:
    _validate_arguments(
        image_paths=image_paths,
        num_tags=num_tags,
        modes=modes,
        batch_size=batch_size,
        workers=workers,
        local_views=local_views,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_source_pixels=max_source_pixels,
        projected_epoch_samples=projected_epoch_samples,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase-3 throughput benchmark")
    view_config = ViewConfig(overview_size=448, local_size=256, local_views=local_views)
    reports = {}
    for mode in modes:
        reports[mode] = _benchmark_mode(
            image_paths,
            num_tags=num_tags,
            input_mode=mode,
            view_config=view_config,
            batch_size=batch_size,
            workers=workers,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            max_source_pixels=max_source_pixels,
            projected_epoch_samples=projected_epoch_samples,
        )
    properties = torch.cuda.get_device_properties(0)
    return {
        "format_version": 1,
        "kind": "foveatag_phase3_supervised_throughput",
        "device": properties.name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "num_tags": int(num_tags),
        "selected_images": len(image_paths),
        "batch_size": batch_size,
        "workers": workers,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "seed": seed,
        "views": {
            "overview_size": view_config.overview_size,
            "local_size": view_config.local_size,
            "local_views": view_config.local_views,
        },
        "max_source_pixels": max_source_pixels,
        "projected_epoch_samples": projected_epoch_samples,
        "classification_loss": {
            "kind": "asymmetric_loss",
            "gamma_negative": 4.0,
            "gamma_positive": 1.0,
            "negative_clip": 0.05,
            "reduction": "global_batch_by_tag_mean",
            "targets": "synthetic_one_positive_per_image",
        },
        "modes": reports,
        "scope": (
            "Real-image full decode, deterministic view rendering, host-to-device "
            "copy, full scratch tagger forward/backward, gradient clipping, and AdamW update."
        ),
        "quality_scope": "Synthetic labels; timing and memory evidence only.",
    }


def _benchmark_mode(
    image_paths: Sequence[str],
    *,
    num_tags: int,
    input_mode: FoveaInputMode,
    view_config: ViewConfig,
    batch_size: int,
    workers: int,
    warmup_steps: int,
    measured_steps: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    max_source_pixels: Optional[int],
    projected_epoch_samples: Optional[int],
) -> dict[str, Any]:
    raw = _BenchmarkRawDataset(
        image_paths,
        num_tags=num_tags,
        max_source_pixels=max_source_pixels,
    )
    dataset = FoveaSupervisedDataset(
        raw,
        view_config=view_config,
        seed=seed,
        training=True,
        input_mode=input_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collate_supervised_samples,
        drop_last=True,
    )
    device = torch.device("cuda")
    random_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = FoveaTagger(num_tags).to(device)
    torch.random.set_rng_state(random_state)
    criterion = AsymmetricLossOptimized()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    iterator = iter(loader)
    data_ms: list[float] = []
    transfer_ms: list[float] = []
    step_ms: list[float] = []
    end_to_end_ms: list[float] = []
    losses: list[float] = []
    measured_samples = successful = attempts = 0
    required = warmup_steps + measured_steps
    torch.cuda.reset_peak_memory_stats(device)
    try:
        while successful < required:
            attempts += 1
            if attempts > required + max(16, len(image_paths) * 2):
                raise RuntimeError("too many unreadable images for the benchmark")
            waiting = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            loaded = time.perf_counter()
            if batch is None:
                continue
            batch = batch.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            transferred = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(
                    batch.views.overview_images,
                    batch.views.overview_geometry,
                    batch.views.local_images,
                    batch.views.local_geometry,
                    local_view_mask=batch.views.local_view_mask,
                )
                loss = criterion(logits, batch.labels, batch.loss_masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            ended = time.perf_counter()
            if successful >= warmup_steps:
                data = (loaded - waiting) * 1000.0
                transfer = (transferred - loaded) * 1000.0
                step = (ended - transferred) * 1000.0
                data_ms.append(data)
                transfer_ms.append(transfer)
                step_ms.append(step)
                end_to_end_ms.append(data + transfer + step)
                losses.append(float(loss.detach()))
                measured_samples += int(batch.labels.shape[0])
            successful += 1
        seconds = sum(end_to_end_ms) / 1000.0
        samples_per_second = measured_samples / seconds
        report = {
            "input_mode": input_mode,
            "model_parameters": model.parameter_count(),
            "local_tensor_views": int(batch.views.local_images.shape[1]),
            "measured_samples": measured_samples,
            "data_wait_ms": _summary(data_ms),
            "host_to_device_ms": _summary(transfer_ms),
            "optimizer_step_ms": _summary(step_ms),
            "end_to_end_step_ms": _summary(end_to_end_ms),
            "samples_per_second": samples_per_second,
            "hours_per_million_samples": 1_000_000 / samples_per_second / 3600.0,
            "synthetic_asl_first": losses[0],
            "synthetic_asl_last": losses[-1],
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
        if projected_epoch_samples is not None:
            report["projected_epoch_hours"] = (
                projected_epoch_samples / samples_per_second / 3600.0
            )
        return report
    finally:
        del iterator, loader, scaler, optimizer, criterion, model
        gc.collect()
        torch.cuda.empty_cache()


def _validate_arguments(**values: Any) -> None:
    if not values["image_paths"]:
        raise ValueError("image_paths must not be empty")
    if int(values["num_tags"]) <= 0:
        raise ValueError("num_tags must be positive")
    if len(values["image_paths"]) < int(values["batch_size"]):
        raise ValueError("image_paths must contain at least one full batch")
    modes = tuple(values["modes"])
    if not modes or len(set(modes)) != len(modes) or any(
        mode not in PHASE3_INPUT_MODES for mode in modes
    ):
        raise ValueError("modes must be unique registered Phase-3 input modes")
    for name in ("batch_size", "measured_steps", "local_views"):
        if int(values[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(values["workers"]) < 0 or int(values["warmup_steps"]) < 0:
        raise ValueError("workers and warmup_steps must be non-negative")
    if float(values["learning_rate"]) <= 0 or float(values["weight_decay"]) < 0:
        raise ValueError("optimizer settings are invalid")
    cap = values["max_source_pixels"]
    if cap is not None and int(cap) <= 0:
        raise ValueError("max_source_pixels must be positive when supplied")
    projection = values["projected_epoch_samples"]
    if projection is not None and int(projection) <= 0:
        raise ValueError("projected_epoch_samples must be positive when supplied")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--local-views", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-source-megapixels", type=float, default=40.0)
    parser.add_argument("--projected-epoch-samples", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    vocabulary = TagVocabulary.from_dict(
        json.loads(args.vocabulary.read_text(encoding="utf-8"))
    )
    cap = (
        None
        if args.max_source_megapixels == 0
        else round(args.max_source_megapixels * 1_000_000)
    )
    paths, eligible = registered_image_paths(
        args.datasets_db,
        args.dataset_id,
        limit=args.sample_count,
        minimum_side=256,
        selection="hashed",
        seed=args.seed,
        max_source_pixels=cap,
    )
    report = benchmark_supervised_modes(
        paths,
        num_tags=vocabulary.num_tags,
        batch_size=args.batch_size,
        workers=args.workers,
        local_views=args.local_views,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_source_pixels=cap,
        projected_epoch_samples=args.projected_epoch_samples,
    )
    report["dataset_id"] = args.dataset_id
    report["eligible_registered_images"] = eligible
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
