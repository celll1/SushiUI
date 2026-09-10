"""Compare GPU-resident and pinned-host RB optimizer state on real CUDA kernels."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training.optimizers.adamw8bit_ringbuffer import AdamW8bit_RingBuffer
from core.training.optimizers.lion8bit_ringbuffer import Lion8bit_RingBuffer


def _host_buffer(param, *, dtype):
    return torch.empty_like(param, dtype=dtype, device="cpu").pin_memory()


def _run(cls, initial, gradients, host_state):
    param = torch.nn.Parameter(initial.cuda())
    allocator = _host_buffer if host_state else None
    optimizer = cls(
        [param], lr=1e-4, weight_decay=0.01, use_8bit=True,
        stochastic_rounding=False, get_state_buffer=allocator,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for grad in gradients:
        param.grad = grad.cuda()
        start = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    state = {
        key: value.detach().cpu().clone()
        for key, value in optimizer.state[param].items()
        if isinstance(value, torch.Tensor)
    }
    result = {
        "param": param.detach().cpu().clone(),
        "state": state,
        "median_s": statistics.median(times),
        "p95_s": max(times),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "steady_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "state_devices": {
            key: str(value.device)
            for key, value in optimizer.state[param].items()
            if isinstance(value, torch.Tensor)
        },
    }
    del optimizer, param
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _compare(name, cls, initial, gradients):
    gpu = _run(cls, initial, gradients, False)
    host = _run(cls, initial, gradients, True)
    state_keys = sorted(set(gpu["state"]) | set(host["state"]))
    report = {
        "optimizer": name,
        "parameter_bit_exact": bool(torch.equal(gpu["param"], host["param"])),
        "state_bit_exact": {
            key: bool(torch.equal(gpu["state"][key].reshape(-1), host["state"][key].reshape(-1)))
            for key in state_keys
        },
        "state_shape_equal": {
            key: gpu["state"][key].shape == host["state"][key].shape
            for key in state_keys
        },
        "gpu_state": {k: v for k, v in gpu.items() if k not in ("param", "state")},
        "host_state": {k: v for k, v in host.items() if k not in ("param", "state")},
    }
    report["host_speedup_percent"] = (gpu["median_s"] / host["median_s"] - 1.0) * 100.0
    report["host_steady_vram_reduction_percent"] = (
        1.0 - host["steady_allocated_gib"] / gpu["steady_allocated_gib"]
    ) * 100.0
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(5678)
    initial = torch.randn(args.rows, args.cols, dtype=torch.float32).mul_(0.01)
    gradients = [torch.randn_like(initial).mul_(0.01) for _ in range(args.steps)]
    reports = [
        _compare("adamw8bit_ringbuffer", AdamW8bit_RingBuffer, initial, gradients),
        _compare("lion8bit_ringbuffer", Lion8bit_RingBuffer, initial, gradients),
    ]
    print(json.dumps({"device": torch.cuda.get_device_name(), "results": reports}, indent=2))


if __name__ == "__main__":
    main()
