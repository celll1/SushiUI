"""Which adapter algebras each architecture has ENABLED -- one table, one edit.

The enablement decision lives HERE rather than in
``core.training.arch.base_arch`` because generation has to reach it, and
importing ``core.training`` from a generation path costs 8.9 s, 5801 modules and
a CUDA context in a fresh process (``backend/tests/adapter_layering_test.py``).
The dependency therefore runs the other way: ``declare_adapter_capability``
READS this table, so there is no mirrored set to drift.

Nothing beyond ordinary LoRA is enabled today. The layer classes and the
tensor-group engine exist; the codec and training integration do not. See
``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` Phases 2 and 3.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import FrozenSet, Mapping, Optional, Tuple

from .spec import ALGORITHM_LORA, ALGORITHMS, FAMILY_NAMES

AdapterPair = Tuple[str, bool]

ORDINARY_LORA: AdapterPair = (ALGORITHM_LORA, False)

#: Every ``(algorithm, weight_decompose)`` pair the two-axis form can name.
ADAPTER_PAIRS: Tuple[AdapterPair, ...] = tuple(
    (algorithm, decompose)
    for algorithm in ALGORITHMS
    for decompose in (False, True)
)

_ORDINARY_ONLY: FrozenSet[AdapterPair] = frozenset({ORDINARY_LORA})

#: THE capability flip point, keyed by the ``ARCH_REGISTRY`` spelling. Enabling
#: LoHa on one architecture is an edit to its row and nothing else: training
#: reads it through ``declare_adapter_capability``, generation through
#: ``AdapterSession``.
ENABLED_ADAPTER_PAIRS: Mapping[str, FrozenSet[AdapterPair]] = MappingProxyType({
    "sd15": _ORDINARY_ONLY,
    "sdxl": _ORDINARY_ONLY,
    "zimage": _ORDINARY_ONLY,
    "anima": _ORDINARY_ONLY,
    "lens": _ORDINARY_ONLY,
    "ideogram4": _ORDINARY_ONLY,
    "minit2i": _ORDINARY_ONLY,
    "krea2": _ORDINARY_ONLY,
    "flux2": _ORDINARY_ONLY,
    "ltx2": _ORDINARY_ONLY,
    "minimax_h3": _ORDINARY_ONLY,
    "acestep": _ORDINARY_ONLY,
    "sensenova": _ORDINARY_ONLY,
})

#: Reasons shared by the architectures the design-doc table treats alike.
PHASE2_PENDING = ("LoHa and LoKr reference paths are designed but not "
                  "implemented (Phase 2), so no checkpoint of either can be "
                  "written, resumed or applied")
PHASE3_PENDING = ("DoRA is planned for dense Linear targets but the dense-DoRA "
                  "phase (Phase 3) has not landed")
PHASE3_PENDING_DENSE_ONLY = (
    "DoRA is planned for dense Linear targets ONLY because this architecture's "
    "base may be weight-only quantized, and Phase 3 has not landed")
QUANTIZED_ADDITIVE_PENDING = (
    "additive branches over an INT8/FP8/W4A8 base are the second half of "
    "Phase 2 and are not enabled")

__all__ = [
    "ADAPTER_PAIRS",
    "AdapterPair",
    "ENABLED_ADAPTER_PAIRS",
    "ORDINARY_LORA",
    "PHASE2_PENDING",
    "PHASE3_PENDING",
    "PHASE3_PENDING_DENSE_ONLY",
    "QUANTIZED_ADDITIVE_PENDING",
    "adapter_refusal_reason",
    "declared_pairs",
    "is_adapter_supported",
    "supported_pairs",
]


def supported_pairs(architecture: Optional[str]) -> FrozenSet[AdapterPair]:
    """What ``architecture`` has enabled, EMPTY for a name this build does not
    know -- an unrecognized architecture must not inherit an enablement."""
    if not architecture:
        return frozenset()
    return ENABLED_ADAPTER_PAIRS.get(architecture, frozenset())


def declared_pairs(architecture: str) -> FrozenSet[AdapterPair]:
    """Strict lookup, for a DECLARATION rather than a runtime check.

    Raises instead of defaulting: an architecture missing from the table would
    otherwise declare that ordinary LoRA is refused for it.
    """
    try:
        return ENABLED_ADAPTER_PAIRS[architecture]
    except KeyError:
        raise KeyError(
            f"{architecture!r} has no row in ENABLED_ADAPTER_PAIRS "
            f"(core/adapters/capability.py); add one before declaring its "
            f"adapter capability") from None


def is_adapter_supported(architecture: Optional[str], algorithm: str,
                         weight_decompose: bool = False) -> bool:
    return (algorithm, bool(weight_decompose)) in supported_pairs(architecture)


def adapter_refusal_reason(architecture: Optional[str], algorithm: str,
                           weight_decompose: bool = False) -> Optional[str]:
    """Why this architecture will not apply the pair, or ``None`` if it will."""
    pair = (algorithm, bool(weight_decompose))
    if pair in supported_pairs(architecture):
        return None
    where = architecture or "this build"
    if pair not in ADAPTER_PAIRS:
        return f"{where}: adapter algorithm {algorithm!r} is not recognized"
    if pair[1] and algorithm != ALGORITHM_LORA:
        # Blocked twice over: by the decomposition AND by the algebra under it.
        why = f"{PHASE3_PENDING}; and {PHASE2_PENDING}"
    else:
        why = PHASE3_PENDING if pair[1] else PHASE2_PENDING
    return f"{where}: {FAMILY_NAMES[pair]} adapters are not enabled -- {why}"
