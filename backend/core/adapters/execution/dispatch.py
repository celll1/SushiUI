"""The adapter-algebra conduit, the process latch, and warm-up.

``adapter_forward_delta(branch, x)`` is the ONE point where the branch delta is
computed: every branch class in ``core.adapters.layers`` routes its
``forward_delta`` through it and keeps its unfused body as ``reference_delta``.
With nothing selected it calls ``branch.reference_delta`` directly -- what the
``reference`` backend's ``fn`` does -- so the unselected path costs a
module-global read, an identity test, and this function's own frame.

The latch is per PROCESS and warm-up is what keeps a run's mathematical
function fixed; both decisions, and why, are in
``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` phase 4.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .probe import (AdapterRegion, ProbeResult, cached_verdict, probe_region,
                    region_for, region_of)
from .registry import REFERENCE, AdapterBackend

#: Warning code for a backend that latched off after warm-up, i.e. mid-run.
LATCH_CODE = "lora_backend_latched"

#: Warning/refusal code for a backend that was asked for and cannot serve.
BACKEND_UNAVAILABLE_CODE = "lora_backend_unavailable"

WarnCallback = Callable[[str, str], None]

# ``None`` means the reference path. Not the ``reference`` descriptor: the
# unselected forward must not pay an extra call.
_active: Optional[AdapterBackend] = None

#: backend name -> why it is latched off for the rest of this process.
_latched: Dict[str, str] = {}

#: Set by ``warm_up_adapter_backend``.
_warmed = False

#: The backend that has returned a LIVE result from ``adapter_forward_delta``,
#: i.e. whose output some real work was computed with. NOT set by the probe: a
#: probe runs on a copy, so "warm-up ran" is not evidence that anything was
#: computed with the backend, and a latch message keyed on it can tell the
#: operator that steps used a function they did not.
_served: Optional[str] = None

_warn_callback: Optional[WarnCallback] = None
_log: Callable[[str], None] = print


class WarmUpReport(NamedTuple):
    backend: str
    regions: int
    usable: int
    refused: Tuple[str, ...]
    latched: bool

    def summary(self) -> str:
        if self.backend == REFERENCE:
            return "adapter execution: reference (nothing selected)"
        state = "LATCHED OFF" if self.latched else f"{self.usable}/{self.regions} regions admitted"
        return f"adapter execution: {self.backend} -- {state}"


# -- process state ---------------------------------------------------------

def active_backend() -> Optional[AdapterBackend]:
    """The selected backend, or ``None`` for the reference path."""
    return _active


def active_backend_name() -> str:
    return REFERENCE if _active is None else _active.name


def set_active_backend(backend: Optional[AdapterBackend], *,
                       warn: Optional[WarnCallback] = None,
                       log: Optional[Callable[[str], None]] = None) -> None:
    """Install the selected backend. ``selection.py`` is the only caller."""
    global _active, _warn_callback, _log
    if backend is not None and backend.name in _latched:
        raise ValueError(
            f"{backend.name} is latched off for this process: {_latched[backend.name]}")
    _active = None if backend is None or backend.name == REFERENCE else backend
    if warn is not None:
        _warn_callback = warn
    if log is not None:
        _log = log


def latched_reason(name: str) -> Optional[str]:
    return _latched.get(name)


def is_latched(name: str) -> bool:
    return name in _latched


def latch_off(name: str, reason: str) -> None:
    """Turn ``name`` off for the rest of the process. Idempotent.

    Called on a launch, compile or execution failure. The current call still
    returns the reference result; so does every later one.
    """
    global _active
    if name in _latched:
        return
    _latched[name] = reason
    served = _served == name
    if _active is not None and _active.name == name:
        _active = None
    where = ("after it had already produced results: work computed before this "
             "point used it, everything after uses the reference path"
             if served else
             "before it produced any result, so nothing in this process was "
             "computed with it")
    message = (f"adapter backend '{name}' latched off for this process "
               f"({where}) -- {reason}")
    _log(f"[Adapter] {message}")
    if _warn_callback is not None:
        try:
            _warn_callback(message, LATCH_CODE)
        except Exception as error:
            _log(f"[Adapter] warning channel failed ({type(error).__name__})")


def reset_execution_state() -> None:
    """Clear selection, latch and warm-up state. For tests only.

    A running process never calls this: un-latching a backend that failed to
    launch is exactly what the latch exists to prevent.
    """
    global _active, _warmed, _served, _warn_callback, _log
    _active = None
    _warmed = False
    _served = None
    _warn_callback = None
    _log = print
    _latched.clear()


# -- the conduit -----------------------------------------------------------

def adapter_forward_delta(branch: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The branch's contribution alone -- through the selected backend if one is
    selected, admitted for this region, and has not latched off.

    Never raises on a backend's behalf: a backend that fails latches off and
    this call returns the reference result, because a training step must not die
    because an experimental kernel did.
    """
    backend = _active
    if backend is None:
        return branch.reference_delta(x)

    region = region_of(branch, x)
    verdict = cached_verdict(backend.name, region)
    if verdict is None:
        _log(f"[Adapter] probing {backend.name} for an unwarmed region "
             f"({region.describe()}); this runs once")
        verdict = probe_region(backend, branch, region)
        _report_verdict(backend, region, verdict)
    if not verdict.usable:
        return branch.reference_delta(x)

    global _served
    try:
        out = backend.fn(branch, x)
    except Exception as error:
        latch_off(backend.name,
                  f"{type(error).__name__} on {region.describe()}: {error}")
        return branch.reference_delta(x)
    if out is None:
        return branch.reference_delta(x)
    _served = backend.name
    return out


def _report_verdict(backend: AdapterBackend, region: AdapterRegion,
                    verdict: ProbeResult) -> None:
    """Log a verdict, and latch on the one kind that is not a region fact."""
    if verdict.error is not None:
        latch_off(backend.name, verdict.reason or str(verdict.error))
    elif not verdict.usable:
        _log(f"[Adapter] {backend.name} not admitted for {region.describe()}: "
             f"{verdict.reason}")


# -- warm-up ---------------------------------------------------------------

def warm_up_adapter_backend(
    branches: Iterable[nn.Module],
    *,
    activation_dtypes: Sequence[torch.dtype] = (),
    warn: Optional[WarnCallback] = None,
    log: Optional[Callable[[str], None]] = None,
    strict: bool = False,
) -> WarmUpReport:
    """Probe the regions these branches are expected to use, before the step.

    Called from the trainer's startup path (see ``base_trainer``): a fused
    backend's first-use search inside step 1 is indistinguishable from a hung
    run.

    ``activation_dtypes`` is a HINT, not the whole truth, and is UNIONED with
    each branch's own base dtype rather than replacing it. A run's training
    dtype is not what every branch sees: MiniMax-H3 runs bf16 blocks with fp32
    I/O heads and AdaLN projections and no ``autocast``, so the run dtype alone
    both fabricates a bf16 region for an fp32 head (whose probe fails on a
    genuine dtype mismatch) and leaves the fp32 region that head really uses
    unwarmed -- which is the stall warm-up exists to prevent. A branch's device
    can move after this too (block swap), so the set is expected, not certain;
    ``adapter_forward_delta`` probes an unwarmed region on first sight.

    ``strict`` raises when some branch has NO admitted region, not when one
    member of the union fails: a branch served in fp32 and refused in bf16 is
    served. KNOWN GAP: a backend admitted here and latched on a later region
    leaves ``usable > 0``, so strict does not raise and the run continues on the
    reference path with a ``lora_backend_latched`` warning only.
    """
    global _warmed, _warn_callback, _log
    if warn is not None:
        _warn_callback = warn
    if log is not None:
        _log = log

    backend = _active
    if backend is None:
        _warmed = True
        return WarmUpReport(REFERENCE, 0, 0, (), False)

    seen: Dict[AdapterRegion, nn.Module] = {}
    per_branch: List[Tuple[nn.Module, Tuple[AdapterRegion, ...]]] = []
    for branch in branches:
        dtypes = dict.fromkeys((*activation_dtypes,
                                _branch_activation_dtype(branch)))
        regions = tuple(region_for(branch, dtype) for dtype in dtypes)
        for region in regions:
            seen.setdefault(region, branch)
        per_branch.append((branch, regions))

    admitted: Dict[AdapterRegion, bool] = {}
    refused: List[str] = []
    for region, branch in seen.items():
        verdict = probe_region(backend, branch, region)
        admitted[region] = verdict.usable
        if not verdict.usable:
            refused.append(f"{region.describe()}: {verdict.reason}")
        _report_verdict(backend, region, verdict)
        if is_latched(backend.name):
            break

    _warmed = True
    usable = sum(1 for ok in admitted.values() if ok)
    report = WarmUpReport(backend.name, len(seen), usable, tuple(refused),
                          is_latched(backend.name))
    _log(f"[Adapter] {report.summary()}")
    for line in report.refused:
        _log(f"[Adapter]   not admitted -- {line}")

    unserved = [regions for _, regions in per_branch
                if not any(admitted.get(region) for region in regions)]
    if unserved and strict:
        from .selection import backend_refusal

        raise backend_refusal(
            f"adapter backend '{report.backend}' was selected but can serve no "
            f"region of {len(unserved)} of this run's {len(per_branch)} adapter "
            f"branch(es), e.g. {unserved[0][0].describe()}; "
            + (report.refused[0] if report.refused else "it has no usable region"))
    return report


def _branch_activation_dtype(branch: nn.Module) -> torch.dtype:
    weight = getattr(branch.original_module, "weight", None)
    if weight is not None and weight.dtype.is_floating_point:
        return weight.dtype
    return getattr(branch, "lora_dtype", torch.float32)


__all__ = [
    "BACKEND_UNAVAILABLE_CODE",
    "LATCH_CODE",
    "WarmUpReport",
    "active_backend",
    "active_backend_name",
    "adapter_forward_delta",
    "is_latched",
    "latch_off",
    "latched_reason",
    "reset_execution_state",
    "set_active_backend",
    "warm_up_adapter_backend",
]
