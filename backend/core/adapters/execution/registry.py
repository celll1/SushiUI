"""Adapter-algebra execution backend registry.

One frozen :class:`AdapterBackend` descriptor per backend captures what it can
serve -- which ``(algorithm, weight_decompose)`` pairs, which activation
dtypes, which device kinds, whether it has a backward -- alongside the callable
that computes the branch delta. ``selection.py`` reads these descriptors to
decide whether a request may name a backend; ``dispatch.py`` reads ``fn`` to
run it and ``probe.py`` checks it against the fp32 oracle before it is allowed
to run anything.

Modelled on ``core/attention/registry.py``, which is this repo's existing
answer to the same problem: adding a backend is ONE entry here plus its
callable, and no architecture, loader or trainer changes.

``reference`` is the built-in entry: it is the shipped path, it is what runs
when nothing is selected, and it is the only implementation this build has. A
fused backend (LyCORIS 4.0.0's Triton/TileLang operations, or any other) is not
registered here -- see ``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` phase 4 for
what registering one requires.
"""

from dataclasses import dataclass
from typing import Callable, FrozenSet, Optional

import torch

from ..capability import ADAPTER_PAIRS, AdapterPair

#: The built-in backend's name. Never absent, never latched, never probed.
REFERENCE = "reference"


@dataclass(frozen=True)
class AdapterBackend:
    """Immutable capability descriptor for an adapter execution backend.

    Attributes:
        name: Canonical backend string ("reference" | ...).
        fn: Delta callable. Signature ``fn(branch, x) -> Tensor | None``,
            returning the branch's contribution ALONE (the ``forward_delta``
            contract), or ``None`` to decline this call without failing. A
            raised exception latches the backend off; see ``dispatch.py``.
        pairs: ``(algorithm, weight_decompose)`` pairs it implements.
        dtypes: Activation dtypes it accepts; ``None`` means unrestricted. This
            is the ALLOWLIST axis -- upstream's device guard accepts floating
            dtypes beyond the three it documents, and a region whose dtype has
            no measured tolerance is refused by ``probe.ORACLE_TOLERANCE``
            regardless of what a backend declares here.
        device_kinds: ``torch.device.type`` values it runs on; ``None`` means
            unrestricted.
        trainable: False when the backend has no backward, which makes it
            unusable for training and for the probe's gradient arm.
        requires_matching_dtypes: True when the activation and the branch
            parameter dtype must be equal -- the MIXED-dtype axis, upstream
            refusing fp16 mixed with bf16 in one operation. Declared here rather
            than left to the kernel to discover.
        needs_probe: True when a region must pass an executed comparison
            against the fp32 oracle before the backend may serve it. Only
            ``reference`` sets this False, because it IS the implementation the
            oracle checks.
        availability: Called once when the backend is selected. Returns None if
            the backend can run in this process, or a sentence saying why not
            (missing dependency, unsupported toolchain). Never raises.
    """

    name: str
    fn: Callable
    pairs: FrozenSet[AdapterPair]
    dtypes: Optional[FrozenSet[torch.dtype]]
    device_kinds: Optional[FrozenSet[str]]
    trainable: bool
    requires_matching_dtypes: bool
    needs_probe: bool
    availability: Callable[[], Optional[str]]


def _reference_delta(branch, x: torch.Tensor) -> torch.Tensor:
    """The shipped path: the branch's own unfused PyTorch delta.

    ``dispatch.adapter_forward_delta`` calls ``branch.reference_delta`` directly
    when nothing is selected rather than routing through this descriptor, so the
    unselected path adds no indirection at all. The two are the same function
    and ``adapter_execution_backend_cheap_test`` asserts they are.
    """
    return branch.reference_delta(x)


def _always_available() -> Optional[str]:
    return None


#: Backend table. Adding a real backend is one entry here and one callable.
BACKENDS = {
    REFERENCE: AdapterBackend(
        name=REFERENCE,
        fn=_reference_delta,
        pairs=frozenset(ADAPTER_PAIRS),
        dtypes=None,
        device_kinds=None,
        trainable=True,
        requires_matching_dtypes=False,
        needs_probe=False,
        availability=_always_available,
    ),
}


def reference_backend() -> AdapterBackend:
    """The built-in backend. Raises if it is missing, which is a build defect."""
    try:
        return BACKENDS[REFERENCE]
    except KeyError:  # pragma: no cover - a build without it cannot run adapters
        raise RuntimeError(
            "the reference adapter backend is not registered; it is the shipped "
            "execution path and every other backend falls back to it") from None


def declared_support(backend: AdapterBackend, region) -> Optional[str]:
    """Why ``backend`` cannot serve ``region`` by its own DECLARATION, or None.

    A declaration is not permission: a region also has to pass
    ``probe.probe_region`` before ``dispatch`` will use it. This is the cheap
    half, and it mirrors ``attention.config.resolve_backend``'s guard chain --
    except that it returns the reason instead of silently downgrading, because
    an adapter request that quietly changes the base mathematical function is
    the thing phase 4 exists to prevent.
    """
    pair = (region.algorithm, region.weight_decompose)
    if pair not in backend.pairs:
        return (f"{backend.name} does not implement "
                f"{region.algorithm}{' + weight_decompose' if region.weight_decompose else ''}")
    if backend.dtypes is not None and region.dtype not in backend.dtypes:
        return f"{backend.name} does not accept activation dtype {region.dtype}"
    if backend.device_kinds is not None and region.device_kind not in backend.device_kinds:
        return f"{backend.name} does not run on device kind {region.device_kind!r}"
    if backend.requires_matching_dtypes and region.dtype != region.param_dtype:
        return (f"{backend.name} refuses a mixed-dtype operation "
                f"(activation {region.dtype}, branch {region.param_dtype})")
    return None


__all__ = [
    "AdapterBackend",
    "BACKENDS",
    "REFERENCE",
    "declared_support",
    "reference_backend",
]
