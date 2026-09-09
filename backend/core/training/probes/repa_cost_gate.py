"""Measure REPA's online cost and the latent-stem cost gate on one CUDA device.

This is a component probe, not a training benchmark. It loads the real frozen
teacher, uses the production projector/loss, and reports CUDA-event timings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
from core.training.repa import (
    PROJECTOR_PARAM_DTYPE,
    RepaProjector,
    encode_repa_targets,
    load_repa_encoder,
    repa_loss,
)
from core.training.repa_latent_stem import LatentRepaStem


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p95_ms": p95,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _cuda_samples(
    fn: Callable[[], object], *, warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del result
    return samples


def _schedule_samples(
    encoder: nn.Module,
    projector: nn.Module,
    pixels: torch.Tensor,
    token_template: torch.Tensor,
    *,
    size: int,
    early_item: bool,
    warmup: int,
    iterations: int,
) -> tuple[list[float], list[float]]:
    total_samples: list[float] = []
    item_wait_samples: list[float] = []
    for index in range(warmup + iterations):
        projector.zero_grad(set_to_none=True)
        tokens = token_template.detach().clone().requires_grad_(True)
        torch.cuda.synchronize()
        total_start = time.perf_counter()
        targets = encode_repa_targets(encoder, pixels, 16, 16, size)
        loss = repa_loss(tokens, targets, projector)
        if early_item:
            item_start = time.perf_counter()
            loss.detach().item()
            item_wait = (time.perf_counter() - item_start) * 1000.0
        loss.backward()
        if not early_item:
            item_start = time.perf_counter()
            loss.detach().item()
            item_wait = (time.perf_counter() - item_start) * 1000.0
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - total_start) * 1000.0
        if index >= warmup:
            total_samples.append(total_ms)
            item_wait_samples.append(item_wait)
    return total_samples, item_wait_samples


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("repa_cost_gate requires CUDA")
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    torch.manual_seed(args.seed)
    encoder, enc_dim, native_size = load_repa_encoder(
        args.encoder_source,
        tagger_model_dir=args.tagger_dir,
        siglip2_repo=args.siglip2_repo,
        dtype=dtype,
        device=device,
        attn_implementation=args.attention,
    )
    size = int(native_size or 384)
    projector = RepaProjector(args.tap_hidden, enc_dim).to(
        device=device, dtype=PROJECTOR_PARAM_DTYPE).train()

    cpu_pixels = torch.rand(args.batch, 3, size, size, dtype=torch.float32)
    pixels = cpu_pixels.to(device=device, dtype=dtype)
    tokens = torch.randn(
        args.batch, 16 * 16, args.tap_hidden, device=device, dtype=dtype)

    h2d = _cuda_samples(
        lambda: cpu_pixels.to(device=device, dtype=dtype),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    teacher = _cuda_samples(
        lambda: encode_repa_targets(encoder, pixels, 16, 16, size),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    targets = encode_repa_targets(encoder, pixels, 16, 16, size)
    projector_forward = _cuda_samples(
        lambda: repa_loss(tokens, targets, projector),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    backward_samples = []
    for index in range(args.warmup + args.iterations):
        projector.zero_grad(set_to_none=True)
        leaf = tokens.detach().clone().requires_grad_(True)
        loss = repa_loss(leaf, targets, projector)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        end.synchronize()
        if index >= args.warmup:
            backward_samples.append(float(start.elapsed_time(end)))

    early_total, early_wait = _schedule_samples(
        encoder, projector, pixels, tokens,
        size=size, early_item=True, warmup=args.warmup, iterations=args.iterations)
    deferred_total, deferred_wait = _schedule_samples(
        encoder, projector, pixels, tokens,
        size=size, early_item=False, warmup=args.warmup, iterations=args.iterations)

    latent = torch.randn(
        args.batch, args.latent_channels, args.latent_height, args.latent_width,
        device=device, dtype=dtype)
    stems = {}
    for width in args.stem_width:
        stem = LatentRepaStem(args.latent_channels, enc_dim, width).to(
            device=device, dtype=dtype).eval()
        with torch.no_grad():
            samples = _cuda_samples(
                lambda stem=stem: stem(latent),
                warmup=args.warmup,
                iterations=args.iterations)
        stems[str(width)] = {
            **_summary(samples),
            "parameters": sum(p.numel() for p in stem.parameters()),
            "passes_cost_gate": (
                statistics.median(samples) / args.batch
                < args.replacement_budget_ms_per_item
            ),
            "median_ms_per_item": statistics.median(samples) / args.batch,
        }
        del stem

    return {
        "device": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "batch": args.batch,
        "teacher_size": size,
        "teacher_dim": enc_dim,
        "tap_hidden": args.tap_hidden,
        "latent_shape": [args.batch, args.latent_channels,
                         args.latent_height, args.latent_width],
        "iterations": args.iterations,
        "seed": args.seed,
        "replacement_budget_ms_per_item": args.replacement_budget_ms_per_item,
        "h2d": _summary(h2d),
        "teacher_forward": _summary(teacher),
        "projector_forward_and_loss": _summary(projector_forward),
        "repa_backward_from_tap": _summary(backward_samples),
        "early_item_schedule": {
            "total": _summary(early_total),
            "item_wait": _summary(early_wait),
        },
        "deferred_item_schedule": {
            "total": _summary(deferred_total),
            "item_wait": _summary(deferred_wait),
        },
        "latent_stems": stems,
        "limits": [
            "Component probe: it excludes the diffusion model forward/backward.",
            "The backward number starts at the tapped tensor, not model inputs.",
            "CPU teacher-pixel preparation belongs to the actual-run repa_profile_steps instrument.",
            "The stem gate measures forward cost only; it says nothing about representation quality.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-source", choices=("tagger", "siglip2"), default="tagger")
    parser.add_argument("--tagger-dir", default="")
    parser.add_argument("--siglip2-repo", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tap-hidden", type=int, default=1280)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-height", type=int, default=192)
    parser.add_argument("--latent-width", type=int, default=192)
    parser.add_argument("--stem-width", type=int, nargs="+", default=(128, 256))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--replacement-budget-ms-per-item", type=float, default=6.02)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    _args = _parser().parse_args()
    _result = run(_args)
    _rendered = json.dumps(_result, indent=2)
    print(_rendered)
    if _args.output is not None:
        _args.output.parent.mkdir(parents=True, exist_ok=True)
        _args.output.write_text(_rendered + "\n", encoding="utf-8")
