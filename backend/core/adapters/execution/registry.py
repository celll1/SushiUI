"""Adapter-algebra execution backend registry.

One frozen :class:`AdapterBackend` per backend: what it can serve, plus the
callable computing the branch delta. ``selection`` reads it to admit a name,
``probe`` to check it against the fp32 oracle, ``dispatch`` to run it.

Modelled on ``core/attention/registry.py``: adding a backend is one entry plus
a callable, with no architecture or trainer change. ``reference`` is the only
entry and is what runs when nothing is selected; registering a fused one is
``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` phase 4.
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

    ``fn(branch, x)`` returns the branch's contribution alone (the
    ``forward_delta`` contract), or ``None`` to decline one call; raising
    latches the backend off for the process.

    ``dtypes`` is the allowlist axis and ``requires_matching_dtypes`` the
    mixed-dtype one -- two different questions that read alike. A region whose
    dtype has no measured ``probe.ORACLE_TOLERANCE`` is refused whatever
    ``dtypes`` says.

    ``trainable`` False means no backward: no training, no gradient arm in the
    probe. ``needs_probe`` is False only for ``reference``, which is the
    implementation the oracle checks. ``availability()`` returns None or a
    sentence saying why the backend cannot run here, and never raises.
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
