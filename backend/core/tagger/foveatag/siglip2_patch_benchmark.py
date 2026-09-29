"""Measure the current NaFlex tagger at multiple inference patch budgets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from PIL import Image

from ..siglip2_inference_manager import SigLIP2InferenceManager
from .pretraining_benchmark import registered_image_paths


def timing_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("timing samples must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": ordered[index],
        "max": max(values),
    }


def _run_one(manager: SigLIP2InferenceManager, image_path: str, patch_budget: int) -> dict[str, float]:
    device = torch.device(manager.device)
    started = time.perf_counter()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.load()
    decoded = time.perf_counter()
    inputs = manager.processor(
        images=image,
        return_tensors="pt",
        max_num_patches=patch_budget,
    )
    processed = time.perf_counter()
    pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
    attention_mask = inputs["pixel_attention_mask"].to(device, non_blocking=True)
    spatial_shapes = inputs["spatial_shapes"].to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    transferred = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        manager.model(pixel_values, attention_mask, spatial_shapes)
    torch.cuda.synchronize(device)
    finished = time.perf_counter()
    return {
        "decode": (decoded - started) * 1000.0,
        "preprocess": (processed - decoded) * 1000.0,
        "host_to_device": (transferred - processed) * 1000.0,
        "forward": (finished - transferred) * 1000.0,
        "end_to_end": (finished - started) * 1000.0,
        "effective_patches": float(pixel_values.shape[1]),
        "patch_rows": float(spatial_shapes[0, 0]),
        "patch_columns": float(spatial_shapes[0, 1]),
    }


def benchmark_patch_budgets(
    checkpoint: Path | str,
    vocabulary: Path | str,
    image_paths: Sequence[str],
    *,
    patch_budgets: Sequence[int],
    warmups: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not image_paths or not patch_budgets:
        raise ValueError("image_paths and patch_budgets must not be empty")
    if warmups < 0 or any(value <= 0 for value in patch_budgets):
        raise ValueError("invalid warmup count or patch budget")
    manager = SigLIP2InferenceManager()
    load_started = time.perf_counter()
    load_result = manager.load_model(str(checkpoint), str(vocabulary))
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    if not manager.is_naflex:
        manager.unload()
        raise ValueError("patch-budget comparison requires a NaFlex checkpoint")

    results: dict[str, Any] = {}
    try:
        for budget in patch_budgets:
            for _ in range(warmups):
                _run_one(manager, image_paths[0], budget)
            torch.cuda.reset_peak_memory_stats()
            stages: dict[str, list[float]] = {}
            patch_counts: list[float] = []
            patch_rows: list[float] = []
            patch_columns: list[float] = []
            for path in image_paths:
                row = _run_one(manager, path, budget)
                patch_counts.append(row.pop("effective_patches"))
                patch_rows.append(row.pop("patch_rows"))
                patch_columns.append(row.pop("patch_columns"))
                for name, value in row.items():
                    stages.setdefault(name, []).append(value)
            results[str(budget)] = {
                "images": len(image_paths),
                "effective_patches": timing_summary(patch_counts),
                "patch_rows": timing_summary(patch_rows),
                "patch_columns": timing_summary(patch_columns),
                "stage_ms": {name: timing_summary(values) for name, values in stages.items()},
                "images_per_second": 1000.0 / statistics.fmean(stages["end_to_end"]),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            }
        properties = torch.cuda.get_device_properties(0)
        return {
            "format_version": 1,
            "kind": "siglip2_naflex_patch_budget",
            "checkpoint": str(Path(checkpoint).resolve()),
            "vocabulary": str(Path(vocabulary).resolve()),
            "model_type": load_result["model_type"],
            "num_tags": load_result["num_tags"],
            "device": properties.name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "dtype": "bfloat16",
            "load_seconds": load_seconds,
            "budgets": results,
            "scope": "Native image decode, NaFlex preprocessing, transfer, and model forward; excludes probability sorting and tag filtering.",
        }
    finally:
        manager.unload()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--minimum-side", type=int, default=256)
    parser.add_argument("--patch-budgets", type=int, nargs="+", default=(256, 512, 1024))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    paths, _ = registered_image_paths(
        args.datasets_db,
        args.dataset_id,
        limit=args.sample_count,
        minimum_side=args.minimum_side,
    )
    report = benchmark_patch_budgets(
        args.checkpoint,
        args.vocabulary,
        paths,
        patch_budgets=args.patch_budgets,
        warmups=args.warmups,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
