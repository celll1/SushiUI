"""U-2-6 exit gates for the ring-buffer optimizers: G-RB2 and G-RB3.

G-RB1 (bandwidth hiding) was closed by ``ringbuffer_overlap.py`` (8c13c493).
This probe discharges the other two, through the PRODUCTION wiring rather than a
probe-local allocator: the state buffers come from
``BaseTrainer._ringbuffer_optimizer_kwargs`` -> ``HostOptimizerStateAllocator``,
and the census from ``optimizers/update_census.py``.

Arms (one per process -- ``--arm``):

* ``pinalias`` -- CPU only. Whether the optimizers' own ``pin_memory()`` call on
  an allocator-returned buffer keeps ONE buffer or creates a second one. An
  allocator that returns unpinned buffers and holds a reference doubles the host
  RAM G-RB2 is about, without changing any device census.
* ``hoststate`` -- G-RB2. Host-resident state through the production kwargs:
  where the state actually lands (device census, not a flag), bytes per
  parameter on the GPU and on the host, pinned fraction, and process peak
  working set against the pinned total. **Runs exactly ONE case**, selected with
  ``--optimizer`` / ``--host-resident`` / ``--path``.
* ``census`` -- G-RB3. Updated-parameter census over a real fused backward, with
  a negative control (one parameter denied its hook) that must fail.
* ``censuscost`` -- what the census costs per step. CPU only.

ONE CASE PER PROCESS, and the ``hoststate`` arm enforces it rather than merely
saying so. An earlier version swept eight cases in one process and its host
numbers were unusable: the working set never shrinks, so the second and later
cases reported an RSS delta near zero while genuinely allocating 0.375 GiB of
pinned state. Only the first allocation of a process shows up in a delta.

Peak VRAM is read BEFORE the trailing cleanup, because ``sync_free()`` calls
``reset_peak_memory_stats()`` and reading afterwards records retained current
allocation instead of the peak.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MEMORY_FRACTION = 0.72
LR = 1e-5

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is present in this venv
    psutil = None


def host_rss_gb() -> Optional[float]:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / 2 ** 30


def host_peak_wset_gb() -> Optional[float]:
    """Process peak working set. The high-water mark an RSS delta cannot see.

    A working set never shrinks on the timescale of these cases, so the delta
    between two RSS samples only records an allocation the process has not made
    before -- which is why this arm runs one case per process.
    """
    if psutil is None:
        return None
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 2 ** 30


def announce(arm: str, gpu_peak_gb: float, host_peak_gb: float) -> None:
    print(f"[announce] arm={arm} estimated peak: GPU {gpu_peak_gb:.2f} GiB, "
          f"host RAM {host_peak_gb:.2f} GiB (current RSS "
          f"{host_rss_gb() or float('nan'):.2f} GiB)")


def gpu_gate() -> None:
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    free, total = torch.cuda.mem_get_info()
    print(f"[gate] device={torch.cuda.get_device_name(0)} "
          f"total={total / 2 ** 30:.1f} GiB free={free / 2 ** 30:.1f} GiB "
          f"cap={MEMORY_FRACTION * total / 2 ** 30:.2f} GiB")


def sync_free() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class ParamBag(nn.Module):
    """Bare parameters whose gradient is exactly a tensor we choose.

    ``loss = sum((p * g).sum())`` makes ``p.grad == g``, and the backward still
    fires the post-accumulate-grad hooks the fused path is built on.
    """

    def __init__(self, shapes, dtype=torch.bfloat16, seed=1234):
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.ps = nn.ParameterList([
            nn.Parameter((torch.randn(s, generator=gen) * 0.02).to("cuda", dtype))
            for s in shapes
        ])
        self.grads = [
            torch.randn(s, generator=gen).to("cuda", dtype) for s in shapes
        ]

    def loss(self):
        return sum((p * g).sum() for p, g in zip(self.ps, self.grads))


def build_trainer_stub(host_resident: bool, census: bool):
    """A stub carrying exactly the attributes _ringbuffer_optimizer_kwargs reads.

    The method itself is the production one, taken off BaseTrainer, so the probe
    cannot drift from what a run does.
    """
    from core.training.base_trainer import BaseTrainer

    class Stub:
        _ringbuffer_optimizer_kwargs = BaseTrainer._ringbuffer_optimizer_kwargs
        optimizer_cautious = False
        optimizer_schedule_free = False
        optimizer_warmup_steps = 0
        optimizer_schedule_free_r = 0.0
        optimizer_schedule_free_weight_lr_power = 2.0
        optimizer_use_radam = False
        optimizer_stochastic_rounding = True
        optimizer_state_host_resident = host_resident
        optimizer_update_census = census
        _host_state_allocator = None
        log_prefix = "[probe]"

    return Stub()


def build_optimizer(name: str, params: List[nn.Parameter], stub) -> Any:
    from core.training.optimizer_factory import OptimizerFactory

    kwargs = stub._ringbuffer_optimizer_kwargs()
    return OptimizerFactory.create_optimizer(
        name, params, learning_rate=LR, weight_decay=0.0, **kwargs
    )


def register_hooks(name: str, optimizer, module) -> None:
    if name == "adamw8bit_ringbuffer":
        from core.training.optimizers.adamw8bit_ringbuffer import patch_adamw8bit_ringbuffer
        patch_adamw8bit_ringbuffer(module, optimizer)
    else:
        from core.training.optimizers.lion8bit_ringbuffer import register_lion8bit_fused_backward
        register_lion8bit_fused_backward(optimizer, module)


# ---------------------------------------------------------------------------
# Arm: pinalias -- does pin_memory() on an allocator buffer duplicate it?
# ---------------------------------------------------------------------------


def arm_pinalias() -> List[Dict[str, Any]]:
    announce("pinalias", gpu_peak_gb=0.0, host_peak_gb=0.5)
    from core.training.optimizers.host_state_allocator import HostOptimizerStateAllocator

    out = []
    for pin in (True, False):
        alloc = HostOptimizerStateAllocator(pin=pin)
        template = torch.empty(1024 * 1024, dtype=torch.uint8)
        buf = alloc(template, dtype=torch.uint8)
        pinned = buf.pin_memory()
        out.append({
            "allocator_pin": pin,
            "returned_is_pinned": bool(buf.is_pinned()),
            # The optimizers do state[k] = state[k].pin_memory(). If that returns
            # a different storage, the pre-pin buffer is a second allocation --
            # only harmless because this allocator keeps no reference to it.
            "pin_memory_returned_same_storage":
                pinned.data_ptr() == buf.data_ptr(),
            "allocator_accounted_bytes": alloc.bytes,
            "allocator_pinned_bytes": alloc.pinned_bytes,
        })
    return out


# ---------------------------------------------------------------------------
# Arm: hoststate -- G-RB2
# ---------------------------------------------------------------------------


def run_hoststate_case(name: str, tiles: int, host_resident: bool, fused: bool) -> Dict[str, Any]:
    sync_free()
    from core.training.optimizers.host_state_allocator import state_device_census

    shapes = [(4096, 4096)] * tiles
    bag = ParamBag(shapes)
    params = list(bag.parameters())
    n = sum(p.numel() for p in params)
    before = [p.detach().float().clone() for p in params]

    rss0 = host_rss_gb()
    torch.cuda.synchronize()
    gpu_before = torch.cuda.memory_allocated()

    stub = build_trainer_stub(host_resident, census=False)
    opt = build_optimizer(name, params, stub)

    if fused:
        register_hooks(name, opt, bag)
        bag.loss().backward()
    else:
        for p, g in zip(params, bag.grads):
            p.grad = g.clone()
        opt.step()
    torch.cuda.synchronize()

    gpu_after = torch.cuda.memory_allocated()
    rss1 = host_rss_gb()
    census = state_device_census(opt)

    host_bytes = sum(b["cpu"] for b in census.values())
    pinned_bytes = sum(b["cpu_pinned"] for b in census.values())
    cuda_state_bytes = sum(b["cuda"] for b in census.values())

    moved = sum(
        int((p.detach().float() != b).sum()) for p, b in zip(params, before)
    )
    alloc = stub._host_state_allocator
    result = {
        "optimizer": name,
        "params": n,
        "host_resident": host_resident,
        "path": "fused" if fused else "step()",
        "state_device_census": census,
        # The authoritative per-parameter state figures: summed from the state
        # TENSORS, by device.
        "cuda_state_tensor_bytes_per_param": cuda_state_bytes / n,
        "host_state_bytes_per_param": host_bytes / n,
        "host_state_pinned_fraction": (pinned_bytes / host_bytes) if host_bytes else None,
        # NOT a state figure. A memory_allocated() delta also contains the
        # stochastic-rounding scratch, the extension's staging buffers and
        # whatever the gradients did -- e.g. Lion gpu-control step() reports
        # 3.3489 here against a census of 1.015625. Kept for the record, named
        # so it cannot be quoted as state size.
        "allocated_delta_bytes_per_param_NOT_STATE": (gpu_after - gpu_before) / n,
        "allocator": alloc.summary() if alloc is not None else None,
        "rss_gib_before": rss0,
        "rss_gib_after": rss1,
        "rss_delta_gib": (rss1 - rss0) if (rss0 and rss1) else None,
        "host_peak_wset_gib": host_peak_wset_gb(),
        # Read BEFORE the cleanup below: sync_free() resets the peak counter.
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        # A count of CHANGED ELEMENTS only -- no magnitude, no checksum. Enough
        # to show the update ran; NOT evidence that host and GPU state do
        # bit-identical work. 5dce52ee established that, with moved fraction AND
        # mean drift AND parameter checksum AND state occupancy.
        "params_changed_fraction_not_a_parity_check": moved / n,
    }
    del opt, bag, params, before
    sync_free()
    return result


def arm_hoststate(optimizer: str, host_resident: bool, path: str) -> List[Dict[str, Any]]:
    """ONE case. Sweeping in-process destroys the host-RAM measurement (see module docstring)."""
    # 24 tiles of 4096x4096 bf16 = 402 M params: 0.81 GiB weights + 0.81 GiB
    # grads, up to 0.81 GiB of GPU state, plus stochastic-rounding scratch and
    # the extension's staging buffers. Announced 3.0 GiB on the first run and
    # measured up to 4.78 (gpu-control, step()), so the figure is the measured
    # worst case, not the estimate that was wrong. Host: 2 x 402 MB pinned
    # (AdamW) or 1 x 402 MB (Lion) on a ~0.7 GiB interpreter -> 2.2 GiB peak
    # working set measured.
    announce("hoststate", gpu_peak_gb=5.0, host_peak_gb=2.5)
    gpu_gate()
    label = (f"{optimizer} state={'HOST' if host_resident else 'gpu-control'} "
             f"path={path}")
    print(f"\n===== {label} =====")
    res = run_hoststate_case(optimizer, 24, host_resident, path == "fused")
    print(json.dumps(res, indent=2, default=str))
    return [res]


# ---------------------------------------------------------------------------
# Arm: census -- G-RB3
# ---------------------------------------------------------------------------


def run_census_case(name: str, tiles: int, deny_hook_for: Optional[int]) -> Dict[str, Any]:
    """One fused backward with the census armed.

    ``deny_hook_for`` is the negative control: that parameter's gradient is
    withheld, so its hook never fires and nothing updates it -- exactly the
    silent-skip failure, which the census must catch.
    """
    sync_free()
    from core.training.optimizers.update_census import enable_update_census

    shapes = [(1024, 1024)] * tiles
    bag = ParamBag(shapes)
    params = list(bag.parameters())

    stub = build_trainer_stub(host_resident=False, census=True)
    opt = build_optimizer(name, params, stub)
    register_hooks(name, opt, bag)
    census = enable_update_census(opt, bag)

    if deny_hook_for is None:
        loss = bag.loss()
    else:
        loss = sum(
            (p * g).sum()
            for i, (p, g) in enumerate(zip(bag.ps, bag.grads))
            if i != deny_hook_for
        )
    loss.backward()
    torch.cuda.synchronize()

    error = None
    try:
        census.assert_complete("probe")
    except RuntimeError as exc:
        error = str(exc)

    result = {
        "optimizer": name,
        "expected": census.expected_count,
        "updated": census.updated_count,
        "missing": census.missing(),
        "unexpected": census.unexpected_count(),
        "negative_control_param": deny_hook_for,
        "assert_complete_raised": error is not None,
        "error": (error[:220] + "...") if error else None,
    }
    del opt, bag, params
    sync_free()
    return result


def arm_census() -> List[Dict[str, Any]]:
    # 32 tiles of 1024x1024 bf16 = 33.5 M params: tiny.
    announce("census", gpu_peak_gb=0.5, host_peak_gb=1.0)
    gpu_gate()
    out = []
    for name in ("adamw8bit_ringbuffer", "lion8bit_ringbuffer"):
        for deny in (None, 7):
            label = f"{name} {'negative-control' if deny is not None else 'complete'}"
            print(f"\n===== {label} =====")
            res = run_census_case(name, 32, deny)
            print(json.dumps(res, indent=2, default=str))
            out.append(res)
    return out


# ---------------------------------------------------------------------------

def arm_censuscost() -> List[Dict[str, Any]]:
    """What the census costs per step at SenseNova's parameter count.

    Pure Python set work, no device work: measured on the census object itself
    at 588 parameters (both MoT halves' Linears), so the number is not diluted
    by a synthetic model's step time.
    """
    announce("censuscost", gpu_peak_gb=0.0, host_peak_gb=0.2)
    import time

    from core.training.optimizers.update_census import UpdateCensus

    out = []
    for count in (294, 588):
        params = [nn.Parameter(torch.empty(1)) for _ in range(count)]
        census = UpdateCensus()
        census.expect(params)
        reps = 2000
        start = time.perf_counter()
        for _ in range(reps):
            census.begin_step(True)
            for p in params:
                census.record(p)
            census.assert_complete("cost")
        elapsed = time.perf_counter() - start
        out.append({
            "params": count,
            "reps": reps,
            "us_per_step": elapsed / reps * 1e6,
            "ns_per_param": elapsed / reps / count * 1e9,
        })
    return out


ARMS = {
    "pinalias": arm_pinalias,
    "hoststate": arm_hoststate,
    "census": arm_census,
    "censuscost": arm_censuscost,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--optimizer", default="adamw8bit_ringbuffer",
                        choices=("adamw8bit_ringbuffer", "lion8bit_ringbuffer"),
                        help="hoststate only")
    parser.add_argument("--host-resident", default="1", choices=("0", "1"),
                        help="hoststate only")
    parser.add_argument("--path", default="fused", choices=("fused", "step"),
                        help="hoststate only")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.arm == "hoststate":
        results = arm_hoststate(args.optimizer, args.host_resident == "1", args.path)
    else:
        results = ARMS[args.arm]()
    print("\n===== SUMMARY =====")
    print(json.dumps(results, indent=2, default=str))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
