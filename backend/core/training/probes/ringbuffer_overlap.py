"""G-RB1: does the Ring Buffer optimizers' HOST optimizer state hide behind backward?

SENSENOVA_TRAINING_DESIGN.md section 6.5 registers G-RB1 as an unmeasured gate. The
16.2B both-branch full FT is not implemented, so its step wall cannot be measured.
This probe therefore measures the *threshold* instead: at what compute intensity does
the HOST-state step wall converge on the GPU-state one.

The axis is **tokens per step**, not parameter count. For a dense Linear chain the
work is 6 FLOP per parameter per token while the optimizer's state traffic is a fixed
2 B/param (AdamW) or 1 B/param (Lion), so compute/transfer depends only on the token
count -- which is what makes a synthetic answer transferable to a model this probe
never loads. Parameter count is held fixed across the sweep for exactly that reason.

Arms (ONE PER PROCESS -- ``--arm``; a HOST arm and a GPU arm in the same process share
an allocator and a warm PCIe path, which is the thing being compared):

* ``compute``       -- forward+backward with no optimizer at all. The t_compute(N) baseline.
* ``adamw_host``    -- AdamW8bit_RingBuffer, state in pinned host memory (probe-supplied
                       ``get_state_buffer``; no production caller supplies one).
* ``adamw_gpu``     -- same optimizer, production wiring (state on GPU).
* ``lion_host`` / ``lion_gpu`` -- Lion8bit_RingBuffer, one state tensor instead of two.
* ``adafactor``     -- zero state traffic, the reference line.

All optimizer arms drive the **fused post-accumulate-grad seam**, which is the seam a
Block-Swap full FT uses and the only one where the update runs inside backward.

**[corrected, U-2-6]** This docstring used to say that seam reads the host buffer
"over UVA per element, not staged with a bulk copy", because the Python hook hands
``state['exp_avg']`` to the extension unchanged. That is true of the Python layer and
false of what runs: ``cuda/adamw8bit_cuda.cpp:145-243`` (and the Lion binding at
``:170-250``) stage the host state with ``.to(device, non_blocking=true)`` on a
**dedicated per-device transfer stream**, order the update kernel behind it with a
CUDA event, and put the D2H writeback back on that same stream so it overlaps the
following parameters' backward. The 26.5 GB/s below is a bulk DMA at PCIe 4.0 x16
line rate, which is what that machinery produces and not what per-element UVA reads
would. The measurements are unaffected; the mechanism attributed to them was wrong.

Read-only with respect to production code.

Measured on an RTX 6000 Ada at d311d1ba (plus the uncommitted ``record_fused_grad_norm``
hook line, a no-op here: no accumulator is attached), 100.66 M BF16 parameters, warmup 5:

* HOST state costs 15.2 ms of GPU time per step for AdamW (402.7 MB both directions ->
  26.5 GB/s, PCIe 4.0 x16 line rate) and 8.2 ms for Lion, exactly half.
* That cost is **fully absorbed** once compute exceeds it: HOST/GPU step-wall ratio is
  4.17 at 64 tokens, 1.32 at 2048, and 1.00 from 4096 tokens up. The excess sits in the
  drain (GPU), not the launch, and disappears -- it does not merely stop growing.
* Threshold, parameter-count-free and matching the measurement to ~2%:
  ``N_tokens >= 2 * bytes_per_param_of_state * achieved_FLOPS / (6 * PCIe_bytes_per_s)``
  = 2038 tokens (AdamW) / 1019 (Lion) at 81 TFLOP/s and 26.5 GB/s.
* HOST and GPU state produce bit-identical parameters after 10 identical steps
  (``--mode verify``), so the hidden traffic is real work, not skipped work.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MEMORY_FRACTION = 0.72  # repo GPU-probe discipline

LR = 1e-5
WEIGHT_DECAY = 0.0

DEPTH = 24
WIDTH = 2048  # 24 x 2048^2 = 100.66 M parameters, held fixed across the sweep
TOKENS = [256, 1024, 4096, 16384, 65536]

try:
    import psutil
except ImportError:
    psutil = None


def gpu_gate() -> Dict[str, Any]:
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    free, total = torch.cuda.mem_get_info()
    info = {
        "device": torch.cuda.get_device_name(0),
        "total_gib": total / 2**30,
        "free_gib": free / 2**30,
        "cap_gib": MEMORY_FRACTION * total / 2**30,
    }
    print(f"[gate] device={info['device']} total={info['total_gib']:.1f} GiB "
          f"free={info['free_gib']:.1f} GiB cap={info['cap_gib']:.2f} GiB")
    return info


def host_rss_gb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / 2**30


def announce(arm: str, gpu_peak_gb: float, host_peak_gb: float) -> None:
    print(f"[announce] arm={arm} estimated peak: GPU {gpu_peak_gb:.2f} GiB, "
          f"host RAM {host_peak_gb:.2f} GiB (current RSS {host_rss_gb():.2f} GiB)")


def sync_free() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class Chain(nn.Module):
    """Depth x (width -> width) with variance-preserving init.

    Default nn.Linear init shrinks the signal over 24 layers far enough that BF16
    gradients underflow to zero; the update kernels would still run (timing is what is
    measured) but a silently all-zero gradient is not worth the ambiguity.
    """

    def __init__(self, depth: int, width: int, seed: int = 1234):
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Linear(width, width, bias=False)
            with torch.no_grad():
                layer.weight.copy_(
                    torch.randn(width, width, generator=gen, dtype=torch.float32)
                    / (width ** 0.5)
                )
            self.layers.append(layer.to(device="cuda", dtype=torch.bfloat16))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class HostStateAllocator:
    """The ``get_state_buffer`` no production caller supplies.

    Signature from the ``_init_param_state`` call sites: ``(p, dtype=torch.uint8)``
    returning a flat ``p.numel()`` buffer. The optimizers pin it themselves.
    """

    def __init__(self) -> None:
        self.buffers: List[torch.Tensor] = []
        self.bytes = 0

    def __call__(self, p: torch.Tensor, dtype=torch.uint8) -> torch.Tensor:
        buf = torch.zeros(p.numel(), dtype=dtype, device="cpu")
        self.buffers.append(buf)
        self.bytes += buf.numel() * buf.element_size()
        return buf


def build_and_hook(
    name: str, model: Chain, get_state_buffer: Optional[Callable]
) -> Any:
    """Build the optimizer as OptimizerFactory would and register the fused seam."""
    from core.training.optimizer_factory import OptimizerFactory

    params = [p for p in model.parameters() if p.requires_grad]
    kwargs: Dict[str, Any] = {"stochastic_rounding": False}
    if get_state_buffer is not None:
        kwargs["get_state_buffer"] = get_state_buffer

    opt = OptimizerFactory.create_optimizer(
        name, params, learning_rate=LR, weight_decay=WEIGHT_DECAY, **kwargs
    )

    if name == "adamw8bit_ringbuffer":
        from core.training.optimizers.adamw8bit_ringbuffer import patch_adamw8bit_ringbuffer
        patch_adamw8bit_ringbuffer(model, opt)
    elif name == "lion8bit_ringbuffer":
        from core.training.optimizers.lion8bit_ringbuffer import register_lion8bit_fused_backward
        register_lion8bit_fused_backward(opt, model)
    elif name == "adafactor":
        from core.training.optimizers.adafactor_fused import patch_adafactor_fused
        patch_adafactor_fused(opt)
        for group in opt.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue

                def hook(tensor: torch.Tensor, pg=group):
                    opt.step_param(tensor, pg)
                    tensor.grad = None

                p.register_post_accumulate_grad_hook(hook)
    else:
        raise ValueError(f"unknown optimizer for this probe: {name}")
    return opt


def state_residency(opt: Any, params: List[nn.Parameter]) -> Dict[str, Any]:
    by_tag: Dict[str, int] = {}
    for p in params:
        for key, value in opt.state.get(p, {}).items():
            if isinstance(value, torch.Tensor):
                tag = (f"{key}:{value.device.type}:"
                       f"{str(value.dtype).replace('torch.', '')}"
                       f"{':pinned' if value.is_cpu and value.is_pinned() else ''}")
                by_tag[tag] = by_tag.get(tag, 0) + value.numel() * value.element_size()
    return by_tag


def state_nonzero_frac(opt: Any, params: List[nn.Parameter]) -> Dict[str, float]:
    """A host buffer the kernel could not dereference would stay at its zero init."""
    out: Dict[str, float] = {}
    for key in ("exp_avg", "exp_avg_sq"):
        tot = nz = 0
        for p in params:
            t = opt.state.get(p, {}).get(key)
            if isinstance(t, torch.Tensor):
                tot += t.numel()
                nz += int((t != 0).sum())
        if tot:
            out[f"{key}_nonzero_frac"] = nz / tot
    return out


def measure(model: Chain, opt: Optional[Any], tokens: int,
            warmup: int, reps: int) -> Dict[str, Any]:
    """Median wall of one forward+backward (+ fused optimizer update, if any)."""
    params = [p for p in model.parameters() if p.requires_grad]
    x = torch.randn(tokens, WIDTH, device="cuda", dtype=torch.bfloat16) / (WIDTH ** 0.5)
    target = torch.randn(tokens, WIDTH, device="cuda", dtype=torch.bfloat16)

    def one() -> None:
        loss = (model(x) * target).sum(dtype=torch.float32) / tokens
        loss.backward()
        if opt is None:
            for p in params:
                p.grad = None

    torch.cuda.synchronize()
    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Split the step into the CPU-side launch (backward returns) and the drain (the
    # queued GPU work finishes). A stall that shows up in ``launch`` is on the CPU
    # thread and can be absorbed by queue depth; one that shows up in ``drain`` is
    # GPU time and cannot.
    times: List[float] = []
    launch: List[float] = []
    drain: List[float] = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        times.append(t2 - t0)
        launch.append(t1 - t0)
        drain.append(t2 - t1)

    peak = torch.cuda.max_memory_allocated()
    grad_nonzero = None
    if opt is None:
        # One extra backward with the gradients left in place, to record that the
        # chain actually produces a nonzero gradient at this token count.
        loss = (model(x) * target).sum(dtype=torch.float32) / tokens
        loss.backward()
        torch.cuda.synchronize()
        tot = sum(p.grad.numel() for p in params if p.grad is not None)
        nz = sum(int((p.grad != 0).sum()) for p in params if p.grad is not None)
        grad_nonzero = nz / tot if tot else None
        for p in params:
            p.grad = None

    del x, target
    res = {
        "tokens": tokens,
        "reps": reps,
        "warmup": warmup,
        "sec_median": statistics.median(times),
        "sec_min": min(times),
        "sec_max": max(times),
        "sec_mean": statistics.fmean(times),
        "launch_sec_median": statistics.median(launch),
        "drain_sec_median": statistics.median(drain),
        "peak_allocated_gib": peak / 2**30,
        "grad_nonzero_frac": grad_nonzero,
    }
    return res


ARM_SPECS = {
    # arm: (optimizer or None, host state?)
    "compute": (None, False),
    "adamw_host": ("adamw8bit_ringbuffer", True),
    "adamw_gpu": ("adamw8bit_ringbuffer", False),
    "lion_host": ("lion8bit_ringbuffer", True),
    "lion_gpu": ("lion8bit_ringbuffer", False),
    "adafactor": ("adafactor", False),
}


def reps_for(tokens: int) -> int:
    if tokens <= 1024:
        return 20
    if tokens <= 16384:
        return 12
    return 8


def run_arm(arm: str, tokens_list: List[int]) -> Dict[str, Any]:
    name, host = ARM_SPECS[arm]

    n_params = DEPTH * WIDTH * WIDTH
    param_gib = n_params * 2 / 2**30
    # Activations dominate at the top token count: depth boundaries x tokens x width.
    act_gib = max(tokens_list) * WIDTH * 2 * (DEPTH + 2) / 2**30
    host_state_gib = 0.0
    if host:
        host_state_gib = n_params * (2 if name.startswith("adamw") else 1) / 2**30
    announce(arm,
             gpu_peak_gb=param_gib * 3 + act_gib + 1.0,
             host_peak_gb=host_rss_gb() + host_state_gib * 2 + 1.0)

    sync_free()
    model = Chain(DEPTH, WIDTH)
    params = [p for p in model.parameters() if p.requires_grad]
    alloc = HostStateAllocator() if host else None
    opt = build_and_hook(name, model, alloc) if name else None

    rows = []
    for tokens in tokens_list:
        print(f"\n===== arm={arm} tokens={tokens} =====")
        row = measure(model, opt, tokens, warmup=5, reps=reps_for(tokens))
        print(json.dumps(row, indent=2))
        rows.append(row)
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "arm": arm,
        "optimizer": name,
        "state_residency": ("host (probe-supplied get_state_buffer)" if host
                            else ("gpu (production wiring)" if name else "none")),
        "seam": "fused post_accumulate_grad hook" if name else "no optimizer",
        "depth": DEPTH,
        "width": WIDTH,
        "n_params": n_params,
        "param_bytes": n_params * 2,
        "get_state_buffer_bytes_requested": alloc.bytes if alloc else 0,
        "state_tensors_by_device": state_residency(opt, params) if opt else {},
        "state_written": state_nonzero_frac(opt, params) if opt else {},
        "rows": rows,
        "host_rss_gib_end": host_rss_gb(),
    }
    del model, opt, params, alloc
    sync_free()
    return result


def run_verify(arm: str, tokens: int, steps: int) -> Dict[str, Any]:
    """Same seed, same inputs, K steps: do HOST and GPU state produce the same run?

    Timing that looks 'hidden' would be worthless if the host path quietly did less
    work, so the arms are made bit-comparable: identical model seed, identical fixed
    input, identical step count, separate processes.
    """
    name, host = ARM_SPECS[arm]
    if name is None:
        raise ValueError("verify needs an optimizer arm")

    sync_free()
    model = Chain(DEPTH, WIDTH)
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone().float() for p in params]
    alloc = HostStateAllocator() if host else None
    opt = build_and_hook(name, model, alloc)

    gen = torch.Generator(device="cuda").manual_seed(777)
    x = torch.randn(tokens, WIDTH, generator=gen, device="cuda",
                    dtype=torch.bfloat16) / (WIDTH ** 0.5)
    target = torch.randn(tokens, WIDTH, generator=gen, device="cuda", dtype=torch.bfloat16)

    for _ in range(steps):
        ((model(x) * target).sum(dtype=torch.float32) / tokens).backward()
    torch.cuda.synchronize()

    after = [p.detach().clone().float() for p in params]
    n = sum(b.numel() for b in before)
    moved = sum(int((a != b).sum()) for a, b in zip(after, before))
    drift = sum(float((a - b).abs().sum()) for a, b in zip(after, before)) / n
    checksum = sum(float(a.double().sum()) for a in after)

    res = {
        "arm": arm,
        "mode": "verify",
        "optimizer": name,
        "state_residency": "host" if host else "gpu",
        "tokens": tokens,
        "steps": steps,
        "n_params": n,
        "moved_frac": moved / n,
        "mean_abs_drift": drift,
        "expected_drift_upper_bound": steps * LR,
        "param_checksum": checksum,
        "state_tensors_by_device": state_residency(opt, params),
        "state_written": state_nonzero_frac(opt, params),
    }
    del model, opt, params, before, after, x, target, alloc
    sync_free()
    return res


def main() -> int:
    global DEPTH, WIDTH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_SPECS))
    parser.add_argument("--tokens", type=int, nargs="*", default=TOKENS)
    parser.add_argument("--out", default=None)
    parser.add_argument("--mode", default="sweep", choices=("sweep", "verify"))
    # Same parameter count in more, smaller tensors: SenseNova's 588 Linears are not
    # 24 big ones, and per-tensor launch cost is not per-parameter cost.
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--verify-steps", type=int, default=10)
    parser.add_argument("--verify-tokens", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this probe.")
        return 2

    DEPTH, WIDTH = args.depth, args.width

    gate = gpu_gate()
    t0 = time.time()
    if args.mode == "verify":
        announce(f"{args.arm}/verify", gpu_peak_gb=2.0, host_peak_gb=host_rss_gb() + 2.0)
        result = run_verify(args.arm, args.verify_tokens, args.verify_steps)
    else:
        result = run_arm(args.arm, list(args.tokens))
    result["gate"] = gate
    print(f"\n[done] arm={args.arm} in {time.time() - t0:.1f}s, "
          f"host RSS {host_rss_gb():.2f} GiB")
    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
