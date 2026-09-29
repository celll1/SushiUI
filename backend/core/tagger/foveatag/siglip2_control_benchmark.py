"""Measure the Phase-1 SigLIP2 encoder-swap control by local-view count."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from PIL import Image

from ..siglip2_inference_manager import SigLIP2InferenceManager
from .pretraining_benchmark import registered_image_paths
from .siglip2_control import SigLIP2ControlTagger
from .siglip2_control_data import SigLIP2ControlBatch, collate_siglip2_control_samples
from .siglip2_patch_benchmark import timing_summary
from .supervised_data import FoveaSupervisedSample
from .views import ViewConfig, build_views


def _sample(path: str, local_views: int, seed: int) -> FoveaSupervisedSample:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.load()
    views = build_views(
        image,
        ViewConfig(local_views=local_views),
        seed=seed,
        training=False,
    )
    return FoveaSupervisedSample(
        views=views,
        label=torch.empty(0),
        loss_mask=torch.empty(0),
    )


def _forward(
    model: SigLIP2ControlTagger,
    batch: SigLIP2ControlBatch,
) -> None:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(
            batch.overview,
            batch.local,
            local_view_mask=batch.local_view_mask,
        )


def benchmark_control_views(
    checkpoint: Path | str,
    vocabulary: Path | str,
    image_paths: Sequence[str],
    *,
    local_view_counts: Sequence[int],
    warmups: int,
    overview_patch_budget: int = 784,
    local_patch_budget: int = 256,
) -> dict[str, Any]:
    if not image_paths or not local_view_counts:
        raise ValueError("image_paths and local_view_counts must not be empty")
    if warmups < 0 or any(count < 0 for count in local_view_counts):
        raise ValueError("warmups and local-view counts must be non-negative")
    manager = SigLIP2InferenceManager()
    load_started = time.perf_counter()
    load_result = manager.load_model(str(checkpoint), str(vocabulary))
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    is_naflex = manager.is_naflex

    original = manager.model
    encoder = original.vision_encoder
    hidden_size = int(encoder.config.hidden_size)
    manager.model = None
    del original
    gc.collect()
    torch.cuda.empty_cache()
    model = SigLIP2ControlTagger(
        encoder,
        hidden_size=hidden_size,
        feature_dim=512,
        num_tags=load_result["num_tags"],
        evidence_slots_per_view=2,
        is_naflex=is_naflex,
        freeze_encoder=True,
    ).to("cuda").eval()

    results: dict[str, Any] = {}
    try:
        for local_views in local_view_counts:
            rows: dict[str, list[float]] = {}
            if warmups:
                warm_sample = _sample(image_paths[0], local_views, 0)
                warm_batch = collate_siglip2_control_samples(
                    [warm_sample],
                    processor=manager.processor,
                    is_naflex=is_naflex,
                    overview_patch_budget=overview_patch_budget,
                    local_patch_budget=local_patch_budget,
                )
                assert warm_batch is not None
                warm_batch = warm_batch.to("cuda", non_blocking=True)
                for _ in range(warmups):
                    _forward(model, warm_batch)
                    torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for index, path in enumerate(image_paths):
                started = time.perf_counter()
                sample = _sample(path, local_views, index)
                rendered = time.perf_counter()
                batch = collate_siglip2_control_samples(
                    [sample],
                    processor=manager.processor,
                    is_naflex=is_naflex,
                    overview_patch_budget=overview_patch_budget,
                    local_patch_budget=local_patch_budget,
                )
                processed = time.perf_counter()
                assert batch is not None
                batch = batch.to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                transferred = time.perf_counter()
                forwarded = time.perf_counter()
                _forward(model, batch)
                torch.cuda.synchronize()
                finished = time.perf_counter()
                values = {
                    "view_render": (rendered - started) * 1000.0,
                    "processor": (processed - rendered) * 1000.0,
                    "host_to_device": (transferred - processed) * 1000.0,
                    "forward": (finished - forwarded) * 1000.0,
                    "end_to_end": (finished - started) * 1000.0,
                }
                for name, value in values.items():
                    rows.setdefault(name, []).append(value)
            results[str(local_views)] = {
                "images": len(image_paths),
                "encoded_patches": (
                    overview_patch_budget + local_views * local_patch_budget
                    if is_naflex else None
                ),
                "stage_ms": {name: timing_summary(values) for name, values in rows.items()},
                "images_per_second": 1000.0 / statistics.fmean(rows["end_to_end"]),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            }
        return {
            "format_version": 1,
            "kind": "foveatag_siglip2_control",
            "checkpoint": str(Path(checkpoint).resolve()),
            "num_tags": load_result["num_tags"],
            "load_seconds": load_seconds,
            "overview_patch_budget": overview_patch_budget,
            "local_patch_budget": local_patch_budget,
            "encoder_input_contract": "naflex" if is_naflex else "fixed",
            "device": torch.cuda.get_device_properties(0).name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "dtype": "bfloat16",
            "modes": results,
            "scope": "Decode, FoveaTag view rendering, SigLIP2 processing, transfer, and encoder-swap control forward.",
        }
    finally:
        del model
        manager.unload()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--datasets-db", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--local-view-counts", type=int, nargs="+", default=(0, 1, 4))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    paths, _ = registered_image_paths(
        args.datasets_db, args.dataset_id, limit=args.sample_count, minimum_side=256
    )
    report = benchmark_control_views(
        args.checkpoint,
        args.vocabulary,
        paths,
        local_view_counts=args.local_view_counts,
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
