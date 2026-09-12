"""Standalone Sol-Attn latency, memory, and approximation probe."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def _timed(call, warmup: int, repeats: int) -> tuple[torch.Tensor, list[float], int]:
    output = None
    for _ in range(warmup):
        output = call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = call()
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    return output, elapsed, torch.cuda.max_memory_allocated()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--threshold-type", choices=("diag", "exact"), default="diag")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    from sol_attn import get_sol_attn_backend, sol_attn

    generator = torch.Generator(device="cuda").manual_seed(1234)
    q, k, v = (
        torch.randn(
            1, args.tokens, args.heads, 128,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )

    def dense():
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    def sparse():
        return sol_attn(
            q,
            k,
            v,
            tau=args.tau,
            thresh_type=args.threshold_type,
            sink_start=0,
            sink_tokens=min(args.prefix_tokens, args.tokens),
        )

    with torch.inference_mode():
        dense_output, dense_ms, dense_peak = _timed(dense, args.warmup, args.repeats)
        sparse_output, sparse_ms, sparse_peak = _timed(sparse, args.warmup, args.repeats)
        delta = (sparse_output.float() - dense_output.float()).flatten()
        reference = dense_output.float().flatten()
        relative_l2 = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference))
        max_abs = float(delta.abs().max())

    payload = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "sol_backend": get_sol_attn_backend(q.device),
        "shape": list(q.shape),
        "tau": args.tau,
        "threshold_type": args.threshold_type,
        "prefix_tokens": min(args.prefix_tokens, args.tokens),
        "dense_median_ms": statistics.median(dense_ms),
        "sol_median_ms": statistics.median(sparse_ms),
        "speedup": statistics.median(dense_ms) / statistics.median(sparse_ms),
        "dense_peak_allocated_bytes": dense_peak,
        "sol_peak_allocated_bytes": sparse_peak,
        "relative_l2": relative_l2,
        "max_abs": max_abs,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
