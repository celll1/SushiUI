"""Bounded CUDA validation for the shared immutable transfer engine."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.memory_management.offload_transfer_engine import FrozenSequentialTransferEngine


def _run_resident(masters, x, repeats):
    weights = [masters[key][torch.float32].cuda() for key in masters]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = x
        for weight in weights:
            result = torch.sin(result @ weight.view(x.shape[1], x.shape[1]))
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    out = {
        "median_s": statistics.median(samples),
        "p95_s": max(samples),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "steady_allocated_gib": torch.cuda.memory_allocated() / 2**30,
    }
    result = result.cpu()
    del weights
    return out, result


def _run_ring(masters, x, repeats, ring_size):
    current = {}

    def point(key, bundle):
        current[key] = bundle[torch.float32]

    engine = FrozenSequentialTransferEngine(
        keys=list(masters), masters=masters, ring_size=ring_size,
        device=torch.device("cuda"), point_bundle=point,
    )
    engine.prime()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = x
        for key in masters:
            engine.acquire(key)
            result = torch.sin(result @ current[key].view(x.shape[1], x.shape[1]))
            engine.release(key)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    stats = engine.stats()
    out = {
        "ring_size": ring_size,
        "median_s": statistics.median(samples),
        "p95_s": max(samples),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "steady_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "h2d_gib": stats.h2d_bytes / 2**30,
        "h2d_submissions": stats.h2d_submissions,
        "acquire_misses": stats.acquire_misses,
        "consumer_waits": stats.consumer_waits,
    }
    result = result.cpu()
    engine.close()
    return out, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=3072)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(1234)
    masters = {}
    for key in range(args.blocks):
        tensor = torch.randn(args.dim * args.dim, dtype=torch.float32).mul_(0.01)
        masters[key] = {torch.float32: tensor.pin_memory()}
    x = torch.randn(args.batch, args.dim, device="cuda", dtype=torch.float32)

    resident, reference = _run_resident(masters, x, args.repeats)
    torch.cuda.empty_cache()
    ring1, result1 = _run_ring(masters, x, args.repeats, 1)
    torch.cuda.empty_cache()
    ring2, result2 = _run_ring(masters, x, args.repeats, 2)
    report = {
        "device": torch.cuda.get_device_name(),
        "shape": vars(args),
        "resident": resident,
        "ring1": ring1,
        "ring2": ring2,
        "bit_exact": {
            "resident_vs_ring1": bool(torch.equal(reference, result1)),
            "resident_vs_ring2": bool(torch.equal(reference, result2)),
            "ring1_vs_ring2": bool(torch.equal(result1, result2)),
        },
        "ring2_speedup_vs_ring1_percent":
            (ring1["median_s"] / ring2["median_s"] - 1.0) * 100.0,
        "ring2_peak_reduction_vs_resident_percent":
            (1.0 - ring2["peak_allocated_gib"] / resident["peak_allocated_gib"]) * 100.0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
