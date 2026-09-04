"""Executed capability probe: a backend is usable for a region only after its
output has been compared against the fp32 oracle FOR THAT REGION.

A declaration of support is not evidence, so this module is the executed half:
it builds a probe COPY of a live branch, runs the backend and
``core.adapters.reference`` on the same input, and records a per-region verdict.
Why a copy (the zero-init trap, and keeping the live run untouched), why the
oracle runs on the host in fp32, and why an over-budget region is refused rather
than certified are in ``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` phase 4.

A REGION is ``(algorithm, weight_decompose, device kind and index, activation
dtype, branch dtype, out_features, in_features)`` -- the acceptance matrix's
device/dtype/shape axes plus the two that decide WHICH function is computed. A
verdict never generalises across regions.
"""

from __future__ import annotations

import copy
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import AdapterBackend, declared_support

#: Worst relative error MEASURED for the shipped algebras against this oracle,
#: over 40 seeds x 3 (rank, alpha) pairs x 5 strengths x all six algebras, taken
#: over the forward, forward_delta, the input gradient and every parameter
#: gradient: 8.2e-7 fp32, 2.7e-3 fp16, 2.9e-2 bf16 -- 3 to 7 ulps of each dtype.
#: Headroom 2.4x / 2.3x / 1.7x. A candidate backend is admitted against the SAME
#: bar the shipped path is held to; inventing a looser one for a fused kernel
#: would make the probe unable to fail. ``adapter_oracle_gate_cheap_test``
#: imports this table and pins its values, so loosening it here fails that gate
#: rather than silently widening it.
ORACLE_TOLERANCE = {
    torch.float32: 2e-6,
    torch.float16: 6e-3,
    torch.bfloat16: 5e-2,
}

#: Fill for a factor that starts as a no-op. Sized as the oracle gate's is, by
#: the smallest delta of the eleven rows (LoHa's Hadamard product).
PROBE_FACTOR_STD = 2.0

#: The randomised branch must move the base output by at least this fraction
#: before any verdict is trusted. Below it the comparison proves nothing.
PROBE_MIN_MOVE = 0.10

#: Host bytes the fp32 oracle may allocate for one region; ``_oracle_bytes``
#: estimates against it before anything is allocated.
PROBE_ORACLE_BUDGET_BYTES = 2 * 1024 ** 3

#: Rows of the probe input: enough to exercise a batch-tiling defect, few
#: enough that its own memory is not a term in the budget above.
PROBE_BATCH = 4

_SEED = 20260904


class AdapterRegion(NamedTuple):
    """The unit a probe verdict applies to. Never generalised across."""

    algorithm: str
    weight_decompose: bool
    device_kind: str
    device_index: Optional[int]
    dtype: torch.dtype
    param_dtype: torch.dtype
    out_features: int
    in_features: int

    def describe(self) -> str:
        family = self.algorithm + ("+wd" if self.weight_decompose else "")
        index = "" if self.device_index is None else f":{self.device_index}"
        return (f"{family} on {self.device_kind}{index} "
                f"[{self.out_features}x{self.in_features}] "
                f"act={_dtype_name(self.dtype)} branch={_dtype_name(self.param_dtype)}")


class ProbeResult(NamedTuple):
    """``usable`` is the only field dispatch reads; the rest is for reporting.

    ``error`` is set only when the backend RAISED. That is a launch or compile
    failure, which latches the backend off for the process; a numerical
    disagreement is ``usable=False`` and leaves the backend selectable for other
    regions.
    """

    usable: bool
    reason: Optional[str]
    forward_error: Optional[float] = None
    grad_error: Optional[float] = None
    error: Optional[BaseException] = None


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def region_of(branch: nn.Module, x: torch.Tensor) -> AdapterRegion:
    """The region of a live branch and the activation it was handed."""
    return region_for(branch, x.dtype, x.device)


def region_for(branch: nn.Module, dtype: torch.dtype,
               device: Optional[torch.device] = None) -> AdapterRegion:
    """The region a branch would occupy for an activation of ``dtype``.

    Warm-up needs this before any activation exists; ``device`` defaults to the
    base weight's own.
    """
    base = branch.original_module
    weight = getattr(base, "weight", None)
    if device is None:
        device = weight.device if weight is not None else torch.device("cpu")
    return AdapterRegion(
        algorithm=getattr(branch, "ADAPTER_ALGORITHM", ""),
        weight_decompose=bool(getattr(branch, "WEIGHT_DECOMPOSE", False)),
        device_kind=device.type,
        device_index=device.index,
        dtype=dtype,
        param_dtype=getattr(branch, "lora_dtype", None)
        or (weight.dtype if weight is not None else dtype),
        out_features=int(getattr(base, "out_features", 0)),
        in_features=int(getattr(base, "in_features", 0)),
    )


# -- verdict cache ---------------------------------------------------------

_verdicts: Dict[Tuple[str, AdapterRegion], ProbeResult] = {}


def cached_verdict(backend_name: str,
                   region: AdapterRegion) -> Optional[ProbeResult]:
    return _verdicts.get((backend_name, region))


def record_verdict(backend_name: str, region: AdapterRegion,
                   result: ProbeResult) -> ProbeResult:
    _verdicts[(backend_name, region)] = result
    return result


def probed_regions() -> Dict[Tuple[str, AdapterRegion], ProbeResult]:
    """Every verdict taken in this process, for reporting and for tests."""
    return dict(_verdicts)


def clear_probe_cache() -> None:
    """Drop every verdict. For tests; a running process never re-probes."""
    _verdicts.clear()


# -- the probe itself ------------------------------------------------------

def _oracle_bytes(region: AdapterRegion, rank: Optional[int],
                  with_grad: bool) -> int:
    """Host bytes the fp32 oracle needs for this region, before allocating.

    ``2 * rank`` because ``_low_rank_product`` holds the list of rank-1 terms
    AND the ``torch.stack`` of them live at once, plus 3 for the Kronecker
    assembly, the delta, and the linear's saved tensors. Doubled again for a
    backward graph. MEASURED at 512x512 rank 48, ``with_grad=False``: a
    ``rank + 3`` estimate gave 51.0 MiB against a 97.2 MiB peak, 1.91x
    optimistic -- and an inference-only backend is exactly the case that takes
    this arm.
    """
    terms = 2 * max(int(rank or 1), 1) + 3
    if with_grad:
        terms *= 2
    return 4 * region.out_features * region.in_features * terms


def _randomize_no_op_factors(branch: nn.Module) -> None:
    """Fill every all-zero factor, so the delta cannot be vacuously right."""
    generator = torch.Generator().manual_seed(_SEED)
    with torch.no_grad():
        for name, tensor in branch.branch_tensors().items():
            if not isinstance(tensor, nn.Parameter) or bool(tensor.any()):
                continue
            fill = torch.randn(tensor.shape, generator=generator) * PROBE_FACTOR_STD
            tensor.copy_(fill.to(device=tensor.device, dtype=tensor.dtype))


def _oracle_leaves(branch: nn.Module,
                   with_grad: bool) -> Dict[str, torch.Tensor]:
    """The branch's tensors as independent host fp32 leaves."""
    leaves = {}
    for name, tensor in branch.branch_tensors().items():
        leaf = tensor.detach().to(device="cpu", dtype=torch.float32)
        if with_grad and isinstance(tensor, nn.Parameter):
            leaf.requires_grad_(True)
        leaves[name] = leaf
    return leaves


def _oracle_delta_weight(branch: nn.Module,
                         leaves: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The delta weight the fp32 oracle says this branch has.

    The oracle is imported HERE, not at module scope, so importing
    ``core.adapters`` does not load it: ``adapter_layering_test`` allows this
    one production importer and gates both halves of that.
    """
    from ..reference import adapter_delta_weight, dora_effective_delta_weight

    tensors = dict(leaves)
    algorithm = branch.ADAPTER_ALGORITHM
    rank, alpha = branch.rank, branch.alpha
    strength = branch.adapter_strength()
    if not branch.WEIGHT_DECOMPOSE:
        return adapter_delta_weight(algorithm, tensors, rank=rank, alpha=alpha,
                                    strength=strength)
    dora_scale = tensors.pop("dora_scale")
    unit = adapter_delta_weight(algorithm, tensors, rank=rank, alpha=alpha)
    base = branch.original_module.weight.detach().to(device="cpu",
                                                     dtype=torch.float32)
    return dora_effective_delta_weight(base, unit, dora_scale, strength=strength)


def _relative_error(got: torch.Tensor, expected: torch.Tensor) -> float:
    got32 = got.detach().to(device="cpu", dtype=torch.float32)
    expected32 = expected.detach().to(device="cpu", dtype=torch.float32)
    scale = max(expected32.abs().max().item(), 1e-12)
    return (got32 - expected32).abs().max().item() / scale


def probe_region(backend: AdapterBackend, branch: nn.Module,
                 region: AdapterRegion) -> ProbeResult:
    """Run ``backend`` against the oracle for ``region`` and return the verdict.

    Cached: a region is probed once per process. Never raises -- a backend that
    raises is reported through ``ProbeResult.error`` so the caller can latch it
    off, rather than having the exception escape into a training run.
    """
    cached = cached_verdict(backend.name, region)
    if cached is not None:
        return cached
    if not backend.needs_probe:
        return record_verdict(backend.name, region, ProbeResult(True, None))
    declared = declared_support(backend, region)
    if declared is not None:
        return record_verdict(backend.name, region, ProbeResult(False, declared))
    try:
        result = _run_probe(backend, branch, region)
    except Exception as error:  # the oracle or the copy failed, not the backend
        result = ProbeResult(
            False,
            f"the probe for {region.describe()} could not be run "
            f"({type(error).__name__}: {error})")
    return record_verdict(backend.name, region, result)


def _run_probe(backend: AdapterBackend, branch: nn.Module,
               region: AdapterRegion) -> ProbeResult:
    with_grad = backend.trainable
    needed = _oracle_bytes(region, getattr(branch, "rank", None), with_grad)
    if needed > PROBE_ORACLE_BUDGET_BYTES:
        return ProbeResult(
            False,
            f"the fp32 oracle for {region.describe()} would need about "
            f"{needed / 1024 ** 3:.1f} GiB of host memory, over the "
            f"{PROBE_ORACLE_BUDGET_BYTES / 1024 ** 3:.1f} GiB probe budget, so "
            f"this region cannot be checked and is not admitted")

    try:
        probe_branch = copy.deepcopy(branch)
    except Exception as error:
        return ProbeResult(
            False,
            f"this branch cannot be copied for probing "
            f"({type(error).__name__}); {region.describe()} is not admitted")
    _randomize_no_op_factors(probe_branch)

    generator = torch.Generator().manual_seed(_SEED + 1)
    x_host = torch.randn((PROBE_BATCH, region.in_features), generator=generator)
    x = x_host.to(device=probe_branch.original_module.weight.device,
                  dtype=region.dtype)
    # The oracle sees exactly the values the backend saw, cast losses included.
    x32 = x.detach().to(device="cpu", dtype=torch.float32)
    if with_grad:
        x = x.detach().requires_grad_(True)
        x32 = x32.requires_grad_(True)

    leaves = _oracle_leaves(probe_branch, with_grad)
    with torch.enable_grad() if with_grad else torch.no_grad():
        delta_w = _oracle_delta_weight(probe_branch, leaves)
        expected = F.linear(x32, delta_w)

    with torch.no_grad():
        base_out = probe_branch.original_module(x.detach())
        moved = _relative_error(
            base_out + expected.detach().to(base_out.device, base_out.dtype),
            base_out)
    if moved < PROBE_MIN_MOVE:
        return ProbeResult(
            False,
            f"the probe branch moves the output by {moved:.3g}, below "
            f"{PROBE_MIN_MOVE} -- this comparison could not have failed, so "
            f"{region.describe()} is not admitted")

    try:
        with torch.enable_grad() if with_grad else torch.no_grad():
            got = backend.fn(probe_branch, x)
    except Exception as error:
        return ProbeResult(
            False,
            f"{backend.name} raised on {region.describe()} "
            f"({type(error).__name__}: {error})",
            error=error)
    if got is None:
        return ProbeResult(False, f"{backend.name} declined {region.describe()}")
    if tuple(got.shape) != tuple(expected.shape):
        return ProbeResult(
            False,
            f"{backend.name} returned shape {tuple(got.shape)} for "
            f"{region.describe()}, expected {tuple(expected.shape)}")

    tolerance = ORACLE_TOLERANCE.get(region.dtype)
    if tolerance is None:
        return ProbeResult(
            False,
            f"no measured tolerance for activation dtype "
            f"{_dtype_name(region.dtype)}; only "
            f"{sorted(_dtype_name(d) for d in ORACLE_TOLERANCE)} are admitted")

    forward_error = _relative_error(got, expected)
    if forward_error > tolerance:
        return ProbeResult(
            False,
            f"{backend.name} disagrees with the fp32 oracle by "
            f"{forward_error:.3e} on {region.describe()} (tolerance "
            f"{tolerance:.1e})",
            forward_error=forward_error)
    if not with_grad:
        return ProbeResult(True, None, forward_error=forward_error)

    try:
        grad_error = _compare_gradients(probe_branch, x, x32, got, expected,
                                        leaves)
    except Exception as error:
        return ProbeResult(
            False,
            f"{backend.name} failed its backward on {region.describe()} "
            f"({type(error).__name__}: {error})",
            forward_error=forward_error,
            error=error)
    if grad_error > tolerance:
        return ProbeResult(
            False,
            f"{backend.name}'s gradients disagree with the fp32 oracle by "
            f"{grad_error:.3e} on {region.describe()} (tolerance "
            f"{tolerance:.1e})",
            forward_error=forward_error, grad_error=grad_error)
    return ProbeResult(True, None, forward_error=forward_error,
                       grad_error=grad_error)


def _compare_gradients(branch: nn.Module, x: torch.Tensor, x32: torch.Tensor,
                       got: torch.Tensor, expected: torch.Tensor,
                       leaves: Dict[str, torch.Tensor]) -> float:
    """Worst relative error over the input gradient and every parameter gradient.

    ``torch.autograd.grad`` rather than ``backward()``: this branch is a copy,
    but a probe that writes ``.grad`` buffers would still be a probe with side
    effects.
    """
    names: List[str] = [name for name, leaf in leaves.items()
                        if leaf.requires_grad]
    live = branch.branch_tensors()
    grads_ref = torch.autograd.grad(expected.sum(),
                                    [x32] + [leaves[n] for n in names],
                                    allow_unused=True)
    grads_got = torch.autograd.grad(got.sum(),
                                    [x] + [live[n] for n in names],
                                    allow_unused=True)
    worst = 0.0
    for reference, candidate in zip(grads_ref, grads_got):
        if reference is None and candidate is None:
            continue
        if reference is None or candidate is None:
            return float("inf")
        worst = max(worst, _relative_error(candidate, reference))
    return worst


__all__ = [
    "ORACLE_TOLERANCE",
    "PROBE_BATCH",
    "PROBE_FACTOR_STD",
    "PROBE_MIN_MOVE",
    "PROBE_ORACLE_BUDGET_BYTES",
    "AdapterRegion",
    "ProbeResult",
    "cached_verdict",
    "clear_probe_cache",
    "probe_region",
    "probed_regions",
    "record_verdict",
    "region_for",
    "region_of",
]
