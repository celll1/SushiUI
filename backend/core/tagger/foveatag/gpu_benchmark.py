"""Standalone CUDA timing harness for the FoveaTag scratch model."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, Sequence

import torch

from .artifact import FoveaArtifactConfig, InferenceMode
from .model import FoveaTagger


@dataclass(frozen=True)
class BenchmarkModeResult:
    overview_size: int
    local_views: int
    local_size: int
    encoded_megapixels: float
    inference_ms_mean: float
    inference_ms_median: float
    inference_ms_p95: float
    train_step_ms_mean: float | None
    train_step_ms_median: float | None
    peak_allocated_mib: float
    peak_reserved_mib: float


def benchmark_foveatag(
    *,
    config: FoveaArtifactConfig,
    modes: Sequence[str],
    iterations: int,
    warmup: int,
    train_warmup: int,
    train_steps: int,
    num_tags: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FoveaTag GPU benchmark requires an available CUDA device")
    if iterations <= 0 or min(warmup, train_warmup, train_steps) < 0:
        raise ValueError("iterations must be positive; warmup counts must be non-negative")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    model = FoveaTagger(
        num_tags,
        encoder_config=config.encoder,
        feature_dim=config.feature_dim,
        evidence_slots_per_view=config.evidence_slots_per_view,
        pool_heads=config.pool_heads,
        temperature=config.temperature,
    ).to(device=device, dtype=dtype)
    results: Dict[str, object] = {}
    for name in modes:
        if name not in config.inference_modes:
            raise ValueError(f"unknown mode: {name}")
        mode = config.inference_modes[name]
        inputs = synthetic_inputs(config, mode, device=device, dtype=dtype)
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                _forward(model, inputs)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            inference_times = [_cuda_timed(lambda: _forward(model, inputs), device) for _ in range(iterations)]
        inference_peak_allocated = torch.cuda.max_memory_allocated(device)
        inference_peak_reserved = torch.cuda.max_memory_reserved(device)

        training_times = []
        if train_steps:
            model.train()
            targets = torch.zeros((1, num_tags), device=device, dtype=dtype)
            for step_index in range(train_warmup + train_steps):
                model.zero_grad(set_to_none=True)

                def training_step() -> None:
                    logits = _forward(model, inputs)
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).backward()

                elapsed = _cuda_timed(training_step, device)
                if step_index >= train_warmup:
                    training_times.append(elapsed)
            training_peak_allocated = torch.cuda.max_memory_allocated(device)
            training_peak_reserved = torch.cuda.max_memory_reserved(device)
        else:
            training_peak_allocated = inference_peak_allocated
            training_peak_reserved = inference_peak_reserved

        results[name] = asdict(BenchmarkModeResult(
            overview_size=mode.overview_size,
            local_views=mode.local_views,
            local_size=config.views.local_size,
            encoded_megapixels=(
                mode.overview_size**2 + mode.local_views * config.views.local_size**2
            ) / 1_000_000.0,
            inference_ms_mean=mean(inference_times),
            inference_ms_median=median(inference_times),
            inference_ms_p95=_percentile(inference_times, 0.95),
            train_step_ms_mean=mean(training_times) if training_times else None,
            train_step_ms_median=median(training_times) if training_times else None,
            peak_allocated_mib=training_peak_allocated / 2**20,
            peak_reserved_mib=training_peak_reserved / 2**20,
        ))
        del inputs
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": 1,
        "timestamp_unix": time.time(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": properties.name,
        "device_total_mib": properties.total_memory / 2**20,
        "dtype": str(dtype).removeprefix("torch."),
        "model_parameters": model.parameter_count(),
        "encoder_parameters": model.encoder.parameter_count(),
        "num_tags": num_tags,
        "iterations": iterations,
        "warmup": warmup,
        "train_warmup": train_warmup,
        "train_steps": train_steps,
        "modes": results,
    }


def synthetic_inputs(
    config: FoveaArtifactConfig,
    mode: InferenceMode,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    overview = torch.randn(
        (1, 3, mode.overview_size, mode.overview_size),
        device=device,
        dtype=dtype,
    )
    local = torch.randn(
        (1, mode.local_views, 3, config.views.local_size, config.views.local_size),
        device=device,
        dtype=dtype,
    )
    overview_geometry = _full_canvas_geometry(1, device)
    local_geometry = _full_canvas_geometry(mode.local_views, device).unsqueeze(0)
    return {
        "overview_images": overview,
        "overview_geometry": overview_geometry,
        "local_images": local,
        "local_geometry": local_geometry,
    }


def _full_canvas_geometry(count: int, device: torch.device) -> torch.Tensor:
    geometry = torch.zeros((count, 10), device=device, dtype=torch.float32)
    geometry[:, :4] = geometry.new_tensor([0.5, 0.5, 1.0, 1.0])
    geometry[:, 8:10] = 1.0
    return geometry


def _forward(model: FoveaTagger, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        inputs["overview_images"],
        inputs["overview_geometry"],
        inputs["local_images"],
        inputs["local_geometry"],
    )


def _cuda_timed(operation, device: torch.device) -> float:
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    operation()
    finished.record()
    finished.synchronize()
    return float(started.elapsed_time(finished))


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _parse_dtype(value: str) -> torch.dtype:
    choices = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return choices[value]
    except KeyError as error:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {value}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=["fast", "balanced", "detail"])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--train-warmup", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--num-tags", type=int, default=10_000)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark_foveatag(
        config=FoveaArtifactConfig(num_tags=args.num_tags),
        modes=args.modes,
        iterations=args.iterations,
        warmup=args.warmup,
        train_warmup=args.train_warmup,
        train_steps=args.train_steps,
        num_tags=args.num_tags,
        dtype=args.dtype,
        device=torch.device(args.device),
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
