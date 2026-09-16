"""Isolated CUDA cost probe for the SenseNova latent refiner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.models.sensenova.latent_refiner import LatentRefiner


def measure(resolution: int, *, width: int, depth: int,
            checkpoint_blocks: bool, base_grad: bool) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    cells = resolution // 8
    module = LatentRefiner(4, width, depth).to(device=device, dtype=torch.bfloat16)
    torch.nn.init.normal_(module.out.weight, std=0.01)
    x0 = torch.randn(
        1, 4, cells, cells, device=device, dtype=torch.bfloat16,
        requires_grad=base_grad,
    )
    z = torch.randn_like(x0)
    timestep = torch.tensor([0.5], device=device)

    def step() -> None:
        module.zero_grad(set_to_none=True)
        if x0.grad is not None:
            x0.grad = None
        output = module(
            x0, z, timestep, 1.0, checkpoint_blocks=checkpoint_blocks
        )
        output.float().square().mean().backward()

    step()
    torch.cuda.synchronize()
    module.zero_grad(set_to_none=True)
    if x0.grad is not None:
        x0.grad = None
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "gpu": torch.cuda.get_device_name(device),
        "resolution": resolution,
        "latent_grid": [cells, cells],
        "width": width,
        "depth": depth,
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "checkpoint_blocks": checkpoint_blocks,
        "base_input_grad": base_grad,
        "seconds_forward_backward": elapsed,
        "peak_allocated_delta_gib": (
            torch.cuda.max_memory_allocated(device) - baseline_allocated
        ) / 2**30,
        "peak_reserved_delta_gib": (
            torch.cuda.max_memory_reserved(device) - baseline_reserved
        ) / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--checkpoint-blocks", action="store_true")
    parser.add_argument("--base-grad", action="store_true")
    args = parser.parse_args()
    print(json.dumps(measure(
        args.resolution,
        width=args.width,
        depth=args.depth,
        checkpoint_blocks=args.checkpoint_blocks,
        base_grad=args.base_grad,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
