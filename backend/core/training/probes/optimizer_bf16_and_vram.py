"""Measure the in-repo optimizer stack: does a BF16 parameter actually move, and
what does its optimizer state cost per parameter.

Every number SENSENOVA_TRAINING_DESIGN.md section 6.5 quotes for optimizer state is a
structural estimate. This probe replaces the ones that can be measured without
loading a model: synthetic BF16 parameters, exact known gradients, one optimizer
at a time.

Arms (one per process -- ``--arm``; the VRAM arms are meaningless if a previous
arm's allocator state is still around):

* ``correctness`` -- fraction of BF16 elements that changed after one step, for
  every (optimizer x stochastic-rounding x step()/fused-seam) combination that
  the trainer can actually construct. A near-zero fraction with rounding off is
  the known BF16 full-FT rounding defect reproducing.
* ``vram`` -- optimizer state bytes per parameter, at three parameter counts, so
  the per-parameter slope is separated from the constant term (quantization
  maps, stochastic-rounding scratch).
* ``cpuring`` -- the ring-buffer CPU-state path, reached by passing
  ``get_state_buffer`` from here. No production caller supplies one
  (``optimizer_factory.py:130``), so this is the unwired path, driven through
  both ``step()`` and the fused hook.
* ``fusedgrad`` -- whether the post-accumulate-grad hooks' ``tensor.grad = None``
  really keeps gradient residency to one parameter.

Read-only with respect to production code: optimizers are built the way
``OptimizerFactory`` builds them and patched the way ``BaseTrainer`` patches
them, via the production functions.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Fraction of the card this probe may touch, per the repo's GPU-probe discipline.
MEMORY_FRACTION = 0.72

LR = 1e-5
WEIGHT_DECAY = 0.0
PARAM_STD = 0.02  # DiT-like weight scale; sets the BF16 ULP the update competes with

try:
    import psutil
except ImportError:
    psutil = None


def gpu_gate() -> None:
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    free, total = torch.cuda.mem_get_info()
    print(f"[gate] device={torch.cuda.get_device_name(0)} "
          f"total={total / 2**30:.1f} GiB free={free / 2**30:.1f} GiB "
          f"cap={MEMORY_FRACTION * total / 2**30:.2f} GiB")


def host_rss_gb() -> Optional[float]:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / 2**30


def announce(arm: str, gpu_peak_gb: float, host_peak_gb: float) -> None:
    print(f"[announce] arm={arm} estimated peak: GPU {gpu_peak_gb:.2f} GiB, "
          f"host RAM {host_peak_gb:.2f} GiB (current RSS "
          f"{host_rss_gb() or float('nan'):.2f} GiB)")


def sync_free() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Building optimizers the way the trainer does
# ---------------------------------------------------------------------------

class ParamBag(nn.Module):
    """Bare parameters with a loss whose gradient is exactly a tensor we choose.

    ``loss = sum((p * g).sum())`` gives ``p.grad == g``, so correctness does not
    depend on a forward pass's numerics, and the backward still fires the
    post-accumulate-grad hooks the fused paths rely on.
    """

    def __init__(self, shapes: List[Tuple[int, ...]], dtype: torch.dtype, seed: int = 1234):
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.ps = nn.ParameterList([
            nn.Parameter(
                (torch.randn(s, generator=gen, dtype=torch.float32) * PARAM_STD).to(
                    device="cuda", dtype=dtype
                )
            )
            for s in shapes
        ])
        self.grads = [
            torch.randn(s, generator=gen, dtype=torch.float32).to(device="cuda", dtype=dtype)
            for s in shapes
        ]

    def loss(self) -> torch.Tensor:
        return sum((p * g).sum() for p, g in zip(self.ps, self.grads))

    def set_grads_directly(self) -> None:
        for p, g in zip(self.ps, self.grads):
            p.grad = g.clone()


def build_optimizer(
    name: str,
    params: List[nn.Parameter],
    sr: bool,
    fused: bool,
    get_state_buffer: Optional[Callable] = None,
) -> Tuple[Any, List[str]]:
    """Construct + patch exactly as OptimizerFactory / BaseTrainer would.

    ``get_state_buffer`` is supplied only by this probe: no production caller
    passes one. Returns (optimizer, notes-about-what-was-attached).
    """
    from core.training.optimizer_factory import OptimizerFactory
    from core.training.optimizers.stochastic_rounding import attach_stochastic_rounding

    notes: List[str] = []
    kwargs: Dict[str, Any] = {"stochastic_rounding": sr}
    if get_state_buffer is not None:
        kwargs["get_state_buffer"] = get_state_buffer

    opt = OptimizerFactory.create_optimizer(
        name, params, learning_rate=LR, weight_decay=WEIGHT_DECAY, **kwargs
    )

    native = name in ("adamw8bit_ringbuffer", "lion8bit_ringbuffer")
    if native:
        notes.append("sr=native(constructor)" if sr else "sr=off")

    # BaseTrainer._setup_fused_backward_pass: patch first...
    if fused and not native:
        if name == "adafactor":
            from core.training.optimizers.adafactor_fused import patch_adafactor_fused
            patch_adafactor_fused(opt)
            notes.append("patched:adafactor_fused")
        elif name == "adamw8bit":
            from core.training.optimizers.adamw8bit_fused import patch_adamw8bit_fused
            patch_adamw8bit_fused(opt, sr)
            notes.append("patched:adamw8bit_fused")

    # ..._attach_stochastic_rounding second, and never for the ring buffers.
    if sr and not native:
        if name == "adafactor" and not hasattr(opt, "step_param"):
            from core.training.optimizers.adafactor_fused import patch_adafactor_fused
            patch_adafactor_fused(opt)
            notes.append("patched:adafactor_fused(for sr)")
        covered = attach_stochastic_rounding(opt)
        notes.append(f"sr_seam={covered or 'NONE'}")

    return opt, notes


def register_fused_hooks(name: str, opt: Any, bag: ParamBag) -> str:
    """Register the fused-backward hooks BaseTrainer would register."""
    if name == "adamw8bit_ringbuffer":
        from core.training.optimizers.adamw8bit_ringbuffer import patch_adamw8bit_ringbuffer
        patch_adamw8bit_ringbuffer(bag, opt)
        return "patch_adamw8bit_ringbuffer"
    if name == "lion8bit_ringbuffer":
        from core.training.optimizers.lion8bit_ringbuffer import register_lion8bit_fused_backward
        register_lion8bit_fused_backward(opt, bag)
        return "register_lion8bit_fused_backward"

    # base_trainer.py:3818-3838 generic loop
    for group in opt.param_groups:
        for p in group["params"]:
            if not p.requires_grad:
                continue

            def hook(tensor: torch.Tensor, pg=group):
                opt.step_param(tensor, pg)
                tensor.grad = None

            p.register_post_accumulate_grad_hook(hook)
    return "step_param hooks"


# ---------------------------------------------------------------------------
# Arm: correctness
# ---------------------------------------------------------------------------

# Small enough that the whole matrix fits in one process; the ULP question is
# per-element and does not depend on the tensor count.
CORRECTNESS_SHAPES = [(1024, 1024), (512, 2048), (2048,)]
CORRECTNESS_STEPS = 20


def run_correctness_case(
    name: str, sr: bool, fused: bool, dtype: torch.dtype
) -> Dict[str, Any]:
    sync_free()
    bag = ParamBag(CORRECTNESS_SHAPES, dtype)
    params = list(bag.parameters())
    before = [p.detach().clone().float() for p in params]

    opt, notes = build_optimizer(name, params, sr, fused)
    seam = register_fused_hooks(name, opt, bag) if fused else "step()"

    # Step 1 -- the headline measurement.
    if fused:
        bag.loss().backward()
        torch.cuda.synchronize()
    else:
        bag.set_grads_directly()
        opt.step()
        torch.cuda.synchronize()

    after1 = [p.detach().clone().float() for p in params]

    # Steps 2..N with the same gradient, to distinguish "frozen" from "slow".
    for _ in range(CORRECTNESS_STEPS - 1):
        if fused:
            bag.loss().backward()
        else:
            bag.set_grads_directly()
            opt.step()
    torch.cuda.synchronize()
    afterN = [p.detach().clone().float() for p in params]

    n = sum(b.numel() for b in before)
    moved1 = sum(int((a != b).sum()) for a, b in zip(after1, before))
    movedN = sum(int((a != b).sum()) for a, b in zip(afterN, before))

    # Descent direction: for AdamW/Lion/Adafactor with weight_decay=0 the step is
    # -lr * (something with the sign of the gradient).
    agree = 0
    delta_abs_sum = 0.0
    for a, b, g in zip(after1, before, bag.grads):
        d = a - b
        mask = d != 0
        if int(mask.sum()) == 0:
            continue
        agree += int((torch.sign(d[mask]) == -torch.sign(g.float()[mask])).sum())
        delta_abs_sum += float(d[mask].abs().sum())

    driftN = sum(float((a - b).abs().sum()) for a, b in zip(afterN, before)) / n
    mean_abs_p = sum(float(b.abs().sum()) for b in before) / n

    res = {
        "optimizer": name,
        "dtype": str(dtype).replace("torch.", ""),
        "stochastic_rounding": sr,
        "path": seam,
        "notes": notes,
        "n_params": n,
        "moved_frac_step1": moved1 / n,
        "moved_frac_step20": movedN / n,
        "descent_agree_frac": (agree / moved1) if moved1 else None,
        "mean_abs_delta_moved_step1": (delta_abs_sum / moved1) if moved1 else None,
        "mean_abs_drift_step20": driftN,
        "expected_drift_step20": CORRECTNESS_STEPS * LR,
        "mean_abs_param": mean_abs_p,
        "lr": LR,
    }
    del bag, opt, params, before, after1, afterN
    sync_free()
    return res


def arm_correctness() -> List[Dict[str, Any]]:
    n = sum(int(torch.tensor(s).prod()) for s in CORRECTNESS_SHAPES)
    announce("correctness", gpu_peak_gb=0.5, host_peak_gb=3.0)
    print(f"[correctness] {n} elements per case, lr={LR}, weight_decay={WEIGHT_DECAY}, "
          f"param std={PARAM_STD}")

    cases: List[Tuple[str, bool, bool, torch.dtype]] = [
        # fp32 reference: rounding cannot bite.
        ("adamw", False, False, torch.float32),
        # bf16 references and the full matrix.
        ("adamw", False, False, torch.bfloat16),
        ("adamw", True, False, torch.bfloat16),
        ("adamw8bit", False, False, torch.bfloat16),
        ("adamw8bit", True, False, torch.bfloat16),
        ("adamw8bit", False, True, torch.bfloat16),
        ("adamw8bit", True, True, torch.bfloat16),
        ("adafactor", False, False, torch.bfloat16),
        ("adafactor", True, False, torch.bfloat16),
        ("adafactor", False, True, torch.bfloat16),
        ("adafactor", True, True, torch.bfloat16),
        ("adamw8bit_ringbuffer", False, False, torch.bfloat16),
        ("adamw8bit_ringbuffer", True, False, torch.bfloat16),
        ("adamw8bit_ringbuffer", False, True, torch.bfloat16),
        ("adamw8bit_ringbuffer", True, True, torch.bfloat16),
        ("lion8bit_ringbuffer", False, False, torch.bfloat16),
        ("lion8bit_ringbuffer", True, False, torch.bfloat16),
        ("lion8bit_ringbuffer", False, True, torch.bfloat16),
        ("lion8bit_ringbuffer", True, True, torch.bfloat16),
    ]

    out = []
    for name, sr, fused, dtype in cases:
        label = f"{name} dtype={str(dtype).replace('torch.', '')} sr={sr} " \
                f"path={'fused' if fused else 'step'}"
        print(f"\n===== {label} =====")
        try:
            res = run_correctness_case(name, sr, fused, dtype)
        except Exception as exc:  # a combination that cannot run is a result
            print(f"[correctness] FAILED: {type(exc).__name__}: {exc}")
            res = {
                "optimizer": name, "dtype": str(dtype).replace("torch.", ""),
                "stochastic_rounding": sr, "path": "fused" if fused else "step()",
                "error": f"{type(exc).__name__}: {exc}",
            }
            sync_free()
        print(json.dumps(res, indent=2, default=str))
        out.append(res)
    return out


# ---------------------------------------------------------------------------
# Arm: vram
# ---------------------------------------------------------------------------

TILE = (2048, 4096)  # 8.389 M elements per tensor
TILE_COUNTS = [6, 24, 96]  # ~50 M / 201 M / 805 M parameters


def run_vram_case(name: str, sr: bool, fused: bool, tiles: int) -> Dict[str, Any]:
    sync_free()
    shapes = [TILE] * tiles
    bag = ParamBag(shapes, torch.bfloat16)
    params = list(bag.parameters())
    n = sum(p.numel() for p in params)

    torch.cuda.synchronize()
    mem_params = torch.cuda.memory_allocated()

    bag.set_grads_directly()
    torch.cuda.synchronize()
    mem_with_grads = torch.cuda.memory_allocated()

    opt, notes = build_optimizer(name, params, sr, fused)
    torch.cuda.synchronize()
    mem_after_ctor = torch.cuda.memory_allocated()

    if fused:
        # step_param on each parameter, the seam the hooks drive, without
        # building a backward graph the size of the model.
        seam = register_fused_hooks(name, opt, bag)
        if name in ("adamw8bit_ringbuffer", "lion8bit_ringbuffer"):
            bag.loss().backward()
        else:
            for group in opt.param_groups:
                for p in group["params"]:
                    opt.step_param(p, group)
    else:
        seam = "step()"
        opt.step()
    torch.cuda.synchronize()
    mem_after_step = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    # Gradients may have been consumed and freed by the fused hooks, so the
    # state delta is measured against the post-construction floor plus whatever
    # gradient memory survived.
    grads_live = sum(p.grad.numel() * p.grad.element_size() for p in params if p.grad is not None)
    state_bytes = mem_after_step - mem_params - grads_live

    res = {
        "optimizer": name,
        "stochastic_rounding": sr,
        "path": seam,
        "notes": notes,
        "tiles": tiles,
        "n_params": n,
        "param_bytes": mem_params,
        "grad_bytes_at_step": mem_with_grads - mem_params,
        "grad_bytes_live_after": grads_live,
        "ctor_bytes": mem_after_ctor - mem_with_grads,
        "state_bytes_total": state_bytes,
        "state_bytes_per_param": state_bytes / n,
        "peak_allocated_gb": peak / 2**30,
    }
    del bag, opt, params
    sync_free()
    return res


def arm_vram() -> List[Dict[str, Any]]:
    # Worst case: 805 M bf16 params (1.6 GiB) + grads (1.6 GiB) + fp32 AdamW-shaped
    # state (up to 6.4 GiB) + transients.
    announce("vram", gpu_peak_gb=12.0, host_peak_gb=4.0)

    cases: List[Tuple[str, bool, bool]] = [
        ("adamw", False, False),
        ("adamw8bit", False, False),
        ("adamw8bit", False, True),
        ("adafactor", False, False),
        ("adafactor", False, True),
        ("adamw8bit_ringbuffer", False, False),
        ("lion8bit_ringbuffer", False, False),
    ]

    out = []
    for name, sr, fused in cases:
        for tiles in TILE_COUNTS:
            label = f"{name} sr={sr} path={'fused' if fused else 'step'} tiles={tiles}"
            print(f"\n===== {label} =====")
            try:
                res = run_vram_case(name, sr, fused, tiles)
            except Exception as exc:
                print(f"[vram] FAILED: {type(exc).__name__}: {exc}")
                res = {"optimizer": name, "stochastic_rounding": sr, "tiles": tiles,
                       "path": "fused" if fused else "step()",
                       "error": f"{type(exc).__name__}: {exc}"}
                sync_free()
            print(json.dumps(res, indent=2, default=str))
            out.append(res)

    # Stochastic-rounding scratch is a constant term (one buffer the size of the
    # largest parameter, times slots), so it is measured at two sizes only.
    for name in ("adamw8bit", "adamw8bit_ringbuffer"):
        for tiles in (6, 24):
            print(f"\n===== {name} sr=True path=step tiles={tiles} =====")
            try:
                res = run_vram_case(name, True, False, tiles)
            except Exception as exc:
                print(f"[vram] FAILED: {type(exc).__name__}: {exc}")
                res = {"optimizer": name, "stochastic_rounding": True, "tiles": tiles,
                       "error": f"{type(exc).__name__}: {exc}"}
                sync_free()
            print(json.dumps(res, indent=2, default=str))
            out.append(res)

    return out


# ---------------------------------------------------------------------------
# Arm: cpuring -- the unwired CPU-state path
# ---------------------------------------------------------------------------

class HostStateAllocator:
    """The ``get_state_buffer`` no production caller supplies.

    Signature taken from the two call sites in ``_init_param_state``:
    ``get_state_buffer(p, dtype=torch.uint8)``, returning a flat buffer of
    ``p.numel()`` elements. The optimizers pin CPU buffers themselves.
    """

    def __init__(self) -> None:
        self.buffers: List[torch.Tensor] = []
        self.bytes = 0

    def __call__(self, p: torch.Tensor, dtype=torch.uint8) -> torch.Tensor:
        buf = torch.zeros(p.numel(), dtype=dtype, device="cpu")
        self.buffers.append(buf)
        self.bytes += buf.numel() * buf.element_size()
        return buf


def run_cpuring_case(name: str, tiles: int, fused: bool, host_state: bool = True) -> Dict[str, Any]:
    sync_free()
    shapes = [TILE] * tiles
    bag = ParamBag(shapes, torch.bfloat16)
    params = list(bag.parameters())
    n = sum(p.numel() for p in params)
    before = [p.detach().clone().float() for p in params]

    rss0 = host_rss_gb()
    torch.cuda.synchronize()
    mem_params = torch.cuda.memory_allocated()
    bag.set_grads_directly()
    torch.cuda.synchronize()
    mem_with_grads = torch.cuda.memory_allocated()

    alloc = HostStateAllocator() if host_state else None
    opt, notes = build_optimizer(name, params, sr=False, fused=fused, get_state_buffer=alloc)

    error = None
    if fused:
        seam = register_fused_hooks(name, opt, bag)
        try:
            bag.loss().backward()
            torch.cuda.synchronize()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    else:
        seam = "step()"
        try:
            opt.step()
            torch.cuda.synchronize()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    rss1 = host_rss_gb()

    # Where did the state actually land?
    state_devices: Dict[str, int] = {}
    for p in params:
        for key, value in opt.state.get(p, {}).items():
            if isinstance(value, torch.Tensor):
                tag = f"{key}:{value.device.type}:{str(value.dtype).replace('torch.', '')}" \
                      f"{':pinned' if value.is_cpu and value.is_pinned() else ''}"
                state_devices[tag] = state_devices.get(tag, 0) + value.numel() * value.element_size()

    # Did the 8-bit state itself actually get written? If the kernel had been
    # handed a host pointer it could not dereference, the momentum would stay at
    # its zero initialisation (and the parameters would not move either).
    state_written = {}
    for key in ("exp_avg", "exp_avg_sq"):
        tot = nz = 0
        for p in params:
            t = opt.state.get(p, {}).get(key)
            if isinstance(t, torch.Tensor):
                tot += t.numel()
                nz += int((t != 0).sum())
        if tot:
            state_written[f"{key}_nonzero_frac"] = nz / tot
            state_written[f"{key}_device"] = str(
                opt.state[params[0]][key].device
            )

    # Steady-state wall per step: the CPU-state path's cost is PCIe traffic, and
    # step() shuttles the state explicitly while the fused hook does not.
    per_step = None
    if error is None:
        torch.cuda.synchronize()
        t0 = time.time()
        reps = 3
        for _ in range(reps):
            if fused:
                bag.loss().backward()
            else:
                bag.set_grads_directly()
                opt.step()
        torch.cuda.synchronize()
        per_step = (time.time() - t0) / reps

    moved = 0
    if error is None:
        after = [p.detach().clone().float() for p in params]
        moved = sum(int((a != b).sum()) for a, b in zip(after, before))

    grads_live = sum(p.grad.numel() * p.grad.element_size() for p in params if p.grad is not None)
    res = {
        "optimizer": name,
        "state_residency": "host (probe-supplied get_state_buffer)" if host_state
                           else "gpu (production wiring)",
        "path": "fused hook" if fused else "step()",
        "seam": seam,
        "notes": notes,
        "n_params": n,
        "sec_per_step": per_step,
        "state_written": state_written,
        "get_state_buffer_bytes_requested": alloc.bytes if alloc else 0,
        "gpu_state_bytes": mem_after - mem_params - grads_live,
        "gpu_param_bytes": mem_params,
        "gpu_grad_bytes": mem_with_grads - mem_params,
        "state_tensors_by_device": state_devices,
        "host_rss_gb_before": rss0,
        "host_rss_gb_after": rss1,
        "moved_frac": moved / n if error is None else None,
        "error": error,
    }
    del bag, opt, params, before, alloc
    sync_free()
    return res


def arm_cpuring() -> List[Dict[str, Any]]:
    # 201 M params: 0.4 GiB bf16 weights + 0.4 GiB grads on GPU; on the host,
    # two uint8 state buffers of 201 MB each, pinned by the optimizer (which
    # copies, so transiently double) -> under 1.5 GiB host.
    announce("cpuring", gpu_peak_gb=2.0, host_peak_gb=5.0)

    out = []
    for name in ("adamw8bit_ringbuffer", "lion8bit_ringbuffer"):
        for tiles in (6, 24):
            for fused in (False, True):
                for host_state in (True, False):
                    label = (f"{name} tiles={tiles} path={'fused' if fused else 'step'} "
                             f"state={'HOST' if host_state else 'gpu-control'}")
                    print(f"\n===== {label} =====")
                    try:
                        res = run_cpuring_case(name, tiles, fused, host_state)
                    except Exception as exc:
                        print(f"[cpuring] FAILED OUTRIGHT: {type(exc).__name__}: {exc}")
                        res = {"optimizer": name, "tiles": tiles,
                               "host_state": host_state,
                               "path": "fused" if fused else "step()",
                               "error_outer": f"{type(exc).__name__}: {exc}"}
                        sync_free()
                    print(json.dumps(res, indent=2, default=str))
                    out.append(res)
    return out


# ---------------------------------------------------------------------------
# Arm: fusedgrad -- does tensor.grad = None actually cap gradient residency
# ---------------------------------------------------------------------------

class Chain(nn.Module):
    """A sequential chain, so gradients become ready one tensor at a time."""

    def __init__(self, depth: int, width: int, dtype: torch.dtype):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(width, width, bias=False, device="cuda", dtype=dtype)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def run_fusedgrad_case(name: str, fused: bool, depth: int, width: int) -> Dict[str, Any]:
    sync_free()
    model = Chain(depth, width, torch.bfloat16)
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    param_bytes = sum(p.numel() * p.element_size() for p in params)

    opt, notes = build_optimizer(name, params, sr=False, fused=fused)
    seam = register_fused_hooks(name, opt, model) if fused else "step()"

    x = torch.randn(8, width, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    out = model(x).float().pow(2).mean()
    out.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    after = torch.cuda.memory_allocated()

    grads_live = sum(p.grad.numel() * p.grad.element_size() for p in params if p.grad is not None)
    grads_present = sum(1 for p in params if p.grad is not None)

    # Two more iterations, so a step counter that never advances is visible.
    for _ in range(2):
        if fused:
            model(x).float().pow(2).mean().backward()
        else:
            model(x).float().pow(2).mean().backward()
            opt.step()
            opt.zero_grad()
    if not fused:
        opt.step()
    torch.cuda.synchronize()

    res = {
        "optimizer": name,
        "path": seam,
        "notes": notes,
        "depth": depth,
        "width": width,
        "n_params": n,
        "param_bytes": param_bytes,
        "all_grads_would_be_bytes": param_bytes,
        "one_grad_bytes": params[0].numel() * params[0].element_size(),
        "grads_live_after_backward_bytes": grads_live,
        "grad_tensors_present_after_backward": grads_present,
        "grad_tensors_total": len(params),
        "backward_peak_over_base_bytes": peak - base,
        # The trainer does not call optimizer.step() under fused backward
        # (base_trainer.py:11671), so whatever advances the bias-correction step
        # counter has to be the hook itself.
        "optimizer_step_count_attr": getattr(opt, "step_count", None),
        "optimizer_state_step": next(
            (int(s["step"]) for s in opt.state.values() if isinstance(s, dict) and "step" in s),
            None,
        ),
    }
    del model, opt, params, x, out
    sync_free()
    return res


def arm_fusedgrad() -> List[Dict[str, Any]]:
    depth, width = 24, 2048
    # 24 x 2048^2 bf16 = 201 MB of weights; the same again if every gradient is
    # resident at once. Optimizer state on top.
    announce("fusedgrad", gpu_peak_gb=2.0, host_peak_gb=3.0)

    out = []
    for name, fused in [
        ("adamw8bit", False), ("adamw8bit", True),
        ("adafactor", False), ("adafactor", True),
        ("adamw8bit_ringbuffer", True),
        ("lion8bit_ringbuffer", True),
    ]:
        print(f"\n===== fusedgrad {name} path={'fused' if fused else 'step'} =====")
        try:
            res = run_fusedgrad_case(name, fused, depth, width)
        except Exception as exc:
            print(f"[fusedgrad] FAILED: {type(exc).__name__}: {exc}")
            res = {"optimizer": name, "path": "fused" if fused else "step()",
                   "error": f"{type(exc).__name__}: {exc}"}
            sync_free()
        print(json.dumps(res, indent=2, default=str))
        out.append(res)
    return out


ARMS = {
    "correctness": arm_correctness,
    "vram": arm_vram,
    "cpuring": arm_cpuring,
    "fusedgrad": arm_fusedgrad,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--out", default=None, help="write results as JSON here")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this probe.")
        return 2

    gpu_gate()
    t0 = time.time()
    results = ARMS[args.arm]()
    print(f"\n[done] arm={args.arm} in {time.time() - t0:.1f}s, "
          f"host RSS peak-ish {host_rss_gb() or float('nan'):.2f} GiB")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
