"""Measure checkpoint-aware mutable block swap against resident fused SGD."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.memory_management.layer_offload_conductor import LayerOffloadConductor


def build_blocks(count: int, dim: int) -> torch.nn.ModuleList:
    return torch.nn.ModuleList([
        torch.nn.Sequential(
            torch.nn.Linear(dim, dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, dim, bias=False),
        )
        for _ in range(count)
    ])


def install_fused_sgd(module: torch.nn.Module, lr: float):
    handles = []
    for parameter in module.parameters():
        def update(tensor):
            tensor.data.add_(tensor.grad, alpha=-lr)
            tensor.grad = None
        handles.append(parameter.register_post_accumulate_grad_hook(update))
    return handles


def digest(module: torch.nn.Module) -> str:
    result = hashlib.sha256()
    for parameter in module.parameters():
        raw = parameter.detach().cpu().contiguous().view(torch.uint8)
        result.update(raw.numpy().tobytes())
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("resident", "ring1", "ring2", "ring3"), required=True
    )
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--swap", type=int, default=10)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.manual_seed(20260911)
    blocks = build_blocks(args.blocks, args.dim).to(dtype=torch.bfloat16)
    conductor = None
    if args.mode == "resident":
        blocks.cuda()
    else:
        conductor = LayerOffloadConductor(
            blocks, blocks_to_swap=args.swap, device=torch.device("cuda"),
            ring_size=int(args.mode[-1]),
        )
        conductor.register_hooks()
    handles = install_fused_sgd(blocks, 1e-4)
    if conductor is not None:
        conductor.register_optimizer_hooks()

    generator = torch.Generator(device="cuda").manual_seed(991)
    sample = torch.randn(args.batch, args.dim, device="cuda", dtype=torch.bfloat16,
                         generator=generator)
    target = torch.randn(args.batch, args.dim, device="cuda", dtype=torch.bfloat16,
                         generator=generator)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        value = sample.detach().clone().requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for block in blocks:
            value = torch.utils.checkpoint.checkpoint(block, value, use_reentrant=False)
        torch.nn.functional.mse_loss(value.float(), target.float()).backward()
        if conductor is not None:
            conductor.finish_backward()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    if conductor is not None:
        conductor.flush()
    result = {
        "mode": args.mode,
        "median_ms": statistics.median(times),
        "times_ms": times,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "steady_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "parameter_sha256": digest(blocks),
        "transfers": conductor.engine.stats().__dict__ if conductor is not None else None,
    }
    print(json.dumps(result, sort_keys=True))
    if conductor is not None:
        conductor.cleanup()
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
