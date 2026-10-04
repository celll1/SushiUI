"""Measure data-loaded FoveaTag pretraining steps on registered images."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .pretraining import FoveaImagePathDataset, FoveaPretrainer, collate_pretraining_samples
from .views import ViewConfig


def registered_image_paths(
    database_path: Path | str,
    dataset_id: int,
    *,
    limit: int,
    minimum_side: int,
    selection: str = "first",
    seed: int = 0,
    max_source_pixels: Optional[int] = 40_000_000,
) -> tuple[list[str], int]:
    """Read a bounded path list without importing the application's DB session."""
    if limit <= 0 or minimum_side <= 0:
        raise ValueError("limit and minimum_side must be positive")
    if selection not in ("first", "hashed"):
        raise ValueError("selection must be first or hashed")
    if max_source_pixels is not None and max_source_pixels <= 0:
        raise ValueError("max_source_pixels must be positive when provided")
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        condition = "dataset_id = ? AND width >= ? AND height >= ?"
        parameters: tuple[int, ...] = (dataset_id, minimum_side, minimum_side)
        if max_source_pixels is not None:
            condition += " AND width * height <= ?"
            parameters += (max_source_pixels,)
        total = int(connection.execute(
            f"SELECT COUNT(*) FROM dataset_items WHERE {condition}", parameters
        ).fetchone()[0])
        order = "id" if selection == "first" else "((id * 1103515245 + ?) & 2147483647), id"
        order_parameters = () if selection == "first" else (seed,)
        rows = connection.execute(
            f"SELECT image_path FROM dataset_items WHERE {condition} ORDER BY {order} LIMIT ?",
            (*parameters, *order_parameters, limit),
        ).fetchall()
    return [str(row[0]) for row in rows if row[0]], total


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": ordered[p95_index],
        "max": max(values),
    }


def benchmark_pretraining(
    image_paths: Sequence[str],
    *,
    eligible_image_count: int,
    batch_size: int,
    workers: int,
    warmup_steps: int,
    measured_steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(batch_size, measured_steps) <= 0 or min(workers, warmup_steps) < 0:
        raise ValueError("invalid benchmark dimensions")
    view_config = ViewConfig(local_views=2)
    dataset = FoveaImagePathDataset(image_paths, view_config=view_config, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collate_pretraining_samples,
        drop_last=True,
    )
    device = torch.device("cuda")
    model = FoveaPretrainer().to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    use_bfloat16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    iterator = iter(loader)
    data_ms: list[float] = []
    transfer_ms: list[float] = []
    step_ms: list[float] = []
    losses: list[float] = []
    correspondence_losses: list[float] = []
    reconstruction_losses: list[float] = []
    chroma_losses: list[float] = []
    variance_losses: list[float] = []
    covariance_losses: list[float] = []
    centering_losses: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(warmup_steps + measured_steps):
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
        started = transferred
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            output = model(batch)
        output.losses.loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        ended = time.perf_counter()
        if step >= warmup_steps:
            data_ms.append((loaded - waiting) * 1000.0)
            transfer_ms.append((transferred - loaded) * 1000.0)
            step_ms.append((ended - started) * 1000.0)
            losses.append(float(output.losses.loss.detach()))
            correspondence_losses.append(float(output.losses.correspondence.detach()))
            reconstruction_losses.append(float(output.losses.reconstruction.detach()))
            chroma_losses.append(float(output.losses.chroma.detach()))
            variance_losses.append(float(output.losses.variance.detach()))
            covariance_losses.append(float(output.losses.covariance.detach()))
            centering_losses.append(float(output.losses.centering.detach()))

    if not step_ms:
        raise RuntimeError("no complete benchmark steps were measured")
    combined = [data + transfer + step for data, transfer, step in zip(data_ms, transfer_ms, step_ms)]
    epoch_steps = math.ceil(eligible_image_count / batch_size)
    properties = torch.cuda.get_device_properties(device)
    return {
        "format_version": 1,
        "kind": "foveatag_real_data_pretraining",
        "device": properties.name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": str(amp_dtype).removeprefix("torch."),
        "batch_size": batch_size,
        "workers": workers,
        "selected_images": len(image_paths),
        "eligible_images": eligible_image_count,
        "warmup_steps": warmup_steps,
        "measured_steps": len(step_ms),
        "seed": seed,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "data_wait_ms": _summary(data_ms),
        "host_to_device_ms": _summary(transfer_ms),
        "optimizer_step_ms": _summary(step_ms),
        "end_to_end_step_ms": _summary(combined),
        "estimated_epoch_hours": statistics.fmean(combined) * epoch_steps / 3_600_000.0,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss": _summary(losses),
        "correspondence_loss": _summary(correspondence_losses),
        "reconstruction_loss": _summary(reconstruction_losses),
        "chroma_loss": _summary(chroma_losses),
        "variance_loss": _summary(variance_losses),
        "covariance_loss": _summary(covariance_losses),
        "centering_loss": _summary(centering_losses),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "scope": "Includes image decode, paired-view rendering, host-to-device copy, forward, backward, and AdamW update.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--minimum-side", type=int, default=256)
    parser.add_argument(
        "--max-source-megapixels",
        type=float,
        default=40.0,
        help="Skip sources above this full-decode cap; use 0 to disable",
    )
    parser.add_argument("--selection", choices=("first", "hashed"), default="first")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    paths, total = registered_image_paths(
        args.datasets_db,
        args.dataset_id,
        limit=args.sample_count,
        minimum_side=args.minimum_side,
        selection=args.selection,
        seed=args.seed,
        max_source_pixels=(
            None
            if args.max_source_megapixels == 0
            else round(args.max_source_megapixels * 1_000_000)
        ),
    )
    if len(paths) < args.batch_size:
        raise ValueError("not enough registered images satisfy the benchmark filter")
    report = benchmark_pretraining(
        paths,
        eligible_image_count=total,
        batch_size=args.batch_size,
        workers=args.workers,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    report["sample_selection"] = args.selection
    report["max_source_megapixels"] = args.max_source_megapixels or None
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
