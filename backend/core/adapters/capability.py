"""Which adapter algebras each architecture has ENABLED -- one table, one edit.

The enablement decision lives HERE rather than in
``core.training.arch.base_arch`` because generation has to reach it, and
importing ``core.training`` from a generation path costs 8.9 s, 5801 modules and
a CUDA context in a fresh process (``backend/tests/adapter_layering_test.py``).
The dependency therefore runs the other way: ``declare_adapter_capability``
READS this table, so there is no mirrored set to drift.

GENERATION ONLY. A row here says a checkpoint of that family LOADS AND
GENERATES on that architecture; it says nothing about training, which
constructs ``LoRALinearLayer`` and only that, and nothing about the API, which
still describes ordinary LoRA. See ``docs/guides/LYCORIS_ADAPTER_DESIGN.md``
Phases 2 and 3.
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

#: Ordinary LoRA plus the two additive LyCORIS algebras, no weight
#: decomposition -- DoRA needs the base weight's direction and norm and is
#: Phase 3.
_ADDITIVE_LYCORIS: FrozenSet[AdapterPair] = frozenset(
    {ORDINARY_LORA, ("loha", False), ("lokr", False)})

#: THE capability flip point, keyed by the ``ARCH_REGISTRY`` spelling. Enabling
#: LoHa on one architecture is an edit to its row and nothing else: training
#: reads it through ``declare_adapter_capability``, generation through
#: ``AdapterSession``.
#:
#: The four LyCORIS rows are the ones whose generation branch builder goes
#: through ``core.adapters.groups.build_adapter_branch`` AND that have no fused
#: target and no quantization policy in the way; each is gated by
#: ``backend/tests/adapter_lycoris_roundtrip_cheap_test.py``. SD1.5/SDXL cannot
#: be flipped from here at all -- they load through diffusers and never reach
#: ``AdapterSession``.
ENABLED_ADAPTER_PAIRS: Mapping[str, FrozenSet[AdapterPair]] = MappingProxyType({
    "sd15": _ORDINARY_ONLY,
    "sdxl": _ORDINARY_ONLY,
    "zimage": _ADDITIVE_LYCORIS,
    "anima": _ORDINARY_ONLY,
    "lens": _ORDINARY_ONLY,
    "ideogram4": _ORDINARY_ONLY,
    "minit2i": _ADDITIVE_LYCORIS,
    "krea2": _ADDITIVE_LYCORIS,
    "flux2": _ORDINARY_ONLY,
    "ltx2": _ADDITIVE_LYCORIS,
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

#: Does this backend's generate function install adapters BEFORE or AFTER its
#: offloader splits the blocks? No session can observe that, and it decides
#: whether a branch the offloader cannot carry (``branch_survives_block_swap``)
#: is a refusal or an advisory. Rows exist for the LyCORIS-enabled
#: architectures; add one before flipping another. Design doc, phase 2.
BEFORE_SPLIT = "before_split"
AFTER_SPLIT = "after_split"
NO_BLOCK_SWAP = "no_block_swap"

BLOCK_SWAP_ADAPTER_ORDER: Mapping[str, str] = MappingProxyType({
    # prepare_block_devices does blocks[i].to(device) for EVERY tensor and only
    # then returns the swapped blocks' Linear weights to the host, so factors
    # installed first are swept to the device and never offloaded again.
    "zimage": BEFORE_SPLIT,
    "krea2": NO_BLOCK_SWAP,
    # _minit2i_stage_transformer / _ensure_ltx2_block_swap_wrapper have already
    # run: a branch over a swapped-out block is built on the HOST and nothing
    # ever moves it.
    "minit2i": AFTER_SPLIT,
    "ltx2": AFTER_SPLIT,
})

#: The refusal, and the advisory. Both name the mechanism, because "not
#: supported" without it reads as a policy choice rather than a missing move.
BLOCK_SWAP_REFUSAL_CODE = "lora_blockswap_unsupported"
BLOCK_SWAP_WARNING_CODE = "lora_blockswap_not_offloaded"
_BLOCK_SWAP_MECHANISM = (
    "a block offloader moves modules whose class name ends in 'Linear', which "
    "covers an ordinary LoRA branch's lora_down/lora_up and covers nothing "
    "inside a {family} branch, whose factors are bare parameters")

__all__ = [
    "ADAPTER_PAIRS",
    "AFTER_SPLIT",
    "BEFORE_SPLIT",
    "BLOCK_SWAP_ADAPTER_ORDER",
    "BLOCK_SWAP_REFUSAL_CODE",
    "BLOCK_SWAP_WARNING_CODE",
    "NO_BLOCK_SWAP",
    "AdapterPair",
    "ENABLED_ADAPTER_PAIRS",
    "ORDINARY_LORA",
    "PHASE2_PENDING",
    "PHASE3_PENDING",
    "PHASE3_PENDING_DENSE_ONLY",
    "QUANTIZED_ADDITIVE_PENDING",
    "adapter_refusal_reason",
    "block_swap_strands_branches",
    "block_swap_refusal_reason",
    "block_swap_warning_text",
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


def block_swap_strands_branches(architecture: Optional[str]) -> bool:
    """Whether this architecture installs adapters AFTER its offloader has split
    the blocks, which is what leaves a LyCORIS branch stranded on the host."""
    return BLOCK_SWAP_ADAPTER_ORDER.get(architecture or "") == AFTER_SPLIT


def block_swap_refusal_reason(architecture: Optional[str], family: str) -> str:
    """Why a branch cannot be installed under a live block offloader.

    ``family`` names the branch CLASS that was built, not the file's label."""
    return (
        f"{architecture or 'this build'}: {family} adapters cannot be applied "
        f"while block swap is active -- "
        f"{_BLOCK_SWAP_MECHANISM.format(family=family)}, so they would stay "
        f"on the host and fail mid-generation. Set blocks_to_swap to 0, or use an "
        f"ordinary LoRA.")


def block_swap_warning_text(architecture: Optional[str], count: int) -> str:
    """The advisory for the other ordering: correct, but never offloaded."""
    return (
        f"{architecture or 'this build'}: {count} LyCORIS adapter branch(es) are "
        f"resident on the compute device for the whole denoise loop and are not "
        f"offloaded with their block -- "
        f"{_BLOCK_SWAP_MECHANISM.format(family='LoHa/LoKr')}. The images are "
        f"unaffected; only the block-swap saving is smaller.")


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
    if (pair[1] and algorithm != ALGORITHM_LORA
            and (algorithm, False) not in supported_pairs(architecture)):
        # Blocked twice over: by the decomposition AND by the algebra under it.
        why = f"{PHASE3_PENDING}; and {PHASE2_PENDING}"
    else:
        why = PHASE3_PENDING if pair[1] else PHASE2_PENDING
    return f"{where}: {FAMILY_NAMES[pair]} adapters are not enabled -- {why}"
