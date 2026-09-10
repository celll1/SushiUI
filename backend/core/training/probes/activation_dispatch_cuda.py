"""CUDA validation probe for the production saved-activation offload hook.

This is a mechanism gate, not an architecture validation.  It exercises audio,
image, and video-shaped inputs with identical model state and inputs in OFF/ON
arms, then records numerical equivalence, offloaded bytes, memory, host RSS,
and warmed iteration timing.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.memory_management import ActivationDispatcher, offload_activations

try:
    import psutil
except ImportError:  # pragma: no cover - the probe still runs without RSS data
    psutil = None


GIB = 1024 ** 3
MEMORY_FRACTION = 0.25
MIN_FREE_GIB = 4.0
THRESHOLD_BYTES = 64 * 1024
BF16_ATOL = 1.0e-3
BF16_RTOL = 1.0e-2
SCENARIOS: Dict[str, Tuple[int, ...]] = {
    "audio_3d": (2, 1024, 64),
    "image_4d": (2, 4, 128, 128),
    "video_5d": (1, 8, 8, 64, 64),
}


def _host_rss_gib() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / GIB


def _p95(values: Iterable[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _as_rows(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:  # [B, sequence, channels]
        return value.reshape(-1, value.shape[-1])
    if value.ndim == 4:  # [B, channels, H, W]
        return value.permute(0, 2, 3, 1).reshape(-1, value.shape[1])
    if value.ndim == 5:  # [B, channels, T, H, W]
        return value.permute(0, 2, 3, 4, 1).reshape(-1, value.shape[1])
    raise ValueError(f"Unsupported probe rank: {value.ndim}")


class SavedActivationWorkload(nn.Module):
    def __init__(self, input_width: int, hidden_width: int = 128, depth: int = 5):
        super().__init__()
        self.input = nn.Linear(input_width, hidden_width)
        self.blocks = nn.ModuleList(
            nn.Linear(hidden_width, hidden_width) for _ in range(depth)
        )
        self.output = nn.Linear(hidden_width, 16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.input(_as_rows(value))
        for block in self.blocks:
            hidden = hidden + F.silu(block(hidden))
        return self.output(hidden).float().square().mean()


def _clean_cuda() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _run_once(
    model: nn.Module,
    value: torch.Tensor,
    enabled: bool,
    capture: bool,
) -> dict:
    model.zero_grad(set_to_none=True)
    value.grad = None
    stats = {"bytes": 0}

    torch.cuda.reset_peak_memory_stats()
    rss_before = _host_rss_gib()
    started = time.perf_counter()
    with offload_activations(
        enabled,
        threshold_bytes=THRESHOLD_BYTES,
        stats=stats,
    ):
        loss = model(value)
        loss.backward()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    result = {
        "loss": float(loss.detach()),
        "finite": bool(torch.isfinite(loss).item()) and all(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in model.parameters()
        ),
        "offloaded_bytes": int(stats["bytes"]),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "host_rss_before_gib": rss_before,
        "host_rss_after_gib": _host_rss_gib(),
        "elapsed_ms": elapsed_ms,
    }
    if capture:
        result["input_grad"] = value.grad.detach().cpu().clone()
        result["parameter_grads"] = {
            name: param.grad.detach().cpu().clone()
            for name, param in model.named_parameters()
        }

    del loss
    return result


def _run_arm(
    shape: Tuple[int, ...],
    state: Dict[str, torch.Tensor],
    source: torch.Tensor,
    enabled: bool,
    warmup: int,
    steps: int,
) -> tuple[dict, dict]:
    input_width = shape[-1] if len(shape) == 3 else shape[1]
    model = SavedActivationWorkload(input_width).cuda().to(torch.bfloat16)
    model.load_state_dict(state)
    value = source.cuda().to(torch.bfloat16).requires_grad_(True)
    captured = _run_once(model, value, enabled, capture=True)
    samples = []
    for index in range(warmup + steps):
        result = _run_once(model, value, enabled, capture=False)
        if index >= warmup:
            samples.append(result["elapsed_ms"])
    timing = {
        "median_ms": statistics.median(samples),
        "p95_ms": _p95(samples),
    }
    del value, model
    _clean_cuda()
    return captured, timing


def _compare(off: dict, on: dict) -> dict:
    grad_pairs = [("input", off["input_grad"], on["input_grad"])]
    grad_pairs.extend(
        (name, grad, on["parameter_grads"][name])
        for name, grad in off["parameter_grads"].items()
    )
    exact_mismatches = [
        name for name, left, right in grad_pairs if not torch.equal(left, right)
    ]
    tolerance_mismatches = [
        name for name, left, right in grad_pairs
        if not torch.allclose(left, right, atol=BF16_ATOL, rtol=BF16_RTOL)
    ]
    max_abs = max(
        (float((left.float() - right.float()).abs().max())
         for _, left, right in grad_pairs),
        default=0.0,
    )
    return {
        "loss_exact": off["loss"] == on["loss"],
        "gradients_exact": not exact_mismatches,
        "gradient_exact_mismatches": exact_mismatches,
        "gradients_within_bf16_tolerance": not tolerance_mismatches,
        "gradient_tolerance_mismatches": tolerance_mismatches,
        "gradient_max_abs_error": max_abs,
        "bf16_atol": BF16_ATOL,
        "bf16_rtol": BF16_RTOL,
    }


def _without_tensors(result: dict) -> dict:
    return {
        key: value for key, value in result.items()
        if key not in {"input_grad", "parameter_grads"}
    }


def run_scenario(name: str, shape: Tuple[int, ...], warmup: int, steps: int) -> dict:
    torch.manual_seed(20260910)
    input_width = shape[-1] if len(shape) == 3 else shape[1]
    template = SavedActivationWorkload(input_width).to(torch.bfloat16)
    state = {key: value.clone() for key, value in template.state_dict().items()}
    source = torch.randn(shape, generator=torch.Generator().manual_seed(127))
    del template

    captured = {}
    timings = {}
    for enabled in (False, True):
        captured[enabled], timings[enabled] = _run_arm(
            shape, state, source, enabled, warmup, steps
        )

    off = captured[False]
    on = captured[True]
    comparison = _compare(off, on)
    dispatcher = ActivationDispatcher(
        budget_gb=torch.cuda.get_device_properties(0).total_memory / GIB,
        threshold_bytes=THRESHOLD_BYTES,
    )
    rank = len(shape)
    if rank == 3:
        lh, lw, lt, batch = shape[1], 1, 1, shape[0]
    elif rank == 4:
        lh, lw, lt, batch = shape[2], shape[3], 1, shape[0]
    else:
        lh, lw, lt, batch = shape[3], shape[4], shape[2], shape[0]
    resident = torch.cuda.memory_allocated() / GIB
    dispatcher.record(
        lh, lw, batch, "base", off["peak_allocated_gib"], resident, lt=lt
    )
    dispatcher.record(
        lh,
        lw,
        batch,
        "offload",
        on["peak_allocated_gib"],
        resident,
        offloaded_gb=on["offloaded_bytes"] / GIB,
        measured_threshold_bytes=THRESHOLD_BYTES,
        lt=lt,
    )
    base = dispatcher.base_act(lh, lw, batch, lt=lt)
    movable = dispatcher.predicted_offloadable(lh, lw, batch, lt=lt)
    tight_headroom = max(0.0, base - movable / 2.0)

    result = {
        "name": name,
        "shape": shape,
        "off": _without_tensors(off),
        "on": _without_tensors(on),
        "comparison": comparison,
        "timing_off": timings[False],
        "timing_on": timings[True],
        "timing_median_overhead_percent": (
            timings[True]["median_ms"] / timings[False]["median_ms"] - 1.0
        ) * 100.0,
        "calibrated_tight_decision": dispatcher.decide(
            lh, lw, batch, tight_headroom, lt=lt
        ),
    }
    result["passed"] = (
        comparison["loss_exact"]
        and comparison["gradients_within_bf16_tolerance"]
        and off["finite"]
        and on["finite"]
        and on["offloaded_bytes"] > 0
        and result["calibrated_tight_decision"] in {"offload", "escalate"}
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup must be non-negative and --steps must be positive")
    if not torch.cuda.is_available():
        print("CUDA is required for this probe.", file=sys.stderr)
        return 2

    torch.cuda.init()
    torch.use_deterministic_algorithms(True)
    free, total = torch.cuda.mem_get_info()
    if free / GIB < MIN_FREE_GIB:
        print(f"Refusing probe: only {free / GIB:.2f} GiB is free", file=sys.stderr)
        return 3
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    payload = {
        "probe": "activation_dispatch_cuda",
        "scope": "mechanism_only_not_architecture_validation",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "entry_free_gib": free / GIB,
        "total_gib": total / GIB,
        "allocator_fraction": MEMORY_FRACTION,
        "threshold_bytes": THRESHOLD_BYTES,
        "warmup": args.warmup,
        "steps": args.steps,
        "scenarios": [
            run_scenario(name, shape, args.warmup, args.steps)
            for name, shape in SCENARIOS.items()
        ],
    }
    payload["passed"] = all(item["passed"] for item in payload["scenarios"])
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
