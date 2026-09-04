"""Which adapter algebras each architecture has enabled -- one table, one edit.

The decision lives here rather than in ``core.training.arch.base_arch`` because
generation must reach it, and importing ``core.training`` from a generation path
costs 8.9 s, 5801 modules and a CUDA context (``adapter_layering_test``). So the
dependency runs the other way: ``declare_adapter_capability`` reads this table
and there is no mirror to drift.

``ENABLED_ADAPTER_PAIRS`` says a checkpoint of that family loads and generates;
``TRAINABLE_ADAPTER_PAIRS`` says a trainer can construct, save and resume it.
Separate so a generation flip does not silently enable training, which has its
own optimizer census, resume slice and block-swap contract.
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
#: decomposition.
_ADDITIVE_LYCORIS: FrozenSet[AdapterPair] = frozenset(
    {ORDINARY_LORA, ("loha", False), ("lokr", False)})

#: ...plus DENSE DoRA. The decomposed pairs over LoHa/LoKr (DoHa/DoKr) are NOT
#: here: the engine builds all three the same way, but each pair needs its own
#: per-architecture round trip and Phase 3's declared scope is dense DoRA.
_ADDITIVE_LYCORIS_WITH_DORA: FrozenSet[AdapterPair] = (
    _ADDITIVE_LYCORIS | frozenset({("lora", True)}))

#: THE capability flip point, keyed by the ``ARCH_REGISTRY`` spelling. Enabling
#: LoHa on one architecture is an edit to its row and nothing else: training
#: reads it through ``declare_adapter_capability``, generation through
#: ``AdapterSession``.
#:
#: The LyCORIS rows are the ones whose generation branch builder goes through
#: ``core.adapters.groups.build_adapter_branch``; each is gated by
#: ``backend/tests/adapter_lycoris_roundtrip_cheap_test.py``. ACE-Step carries
#: the pair for its sd-scripts codec only -- its diffusers/PEFT branch bakes
#: ``(lora_A|lora_B)`` into its key regexes, so a LyCORIS file cannot reach a
#: grouper there and falls out as zero targets. MiniMax-H3 splits a fused-QKV
#: group with ``split_group_on_out_rows``, which refuses a LoKr whose ``w1``
#: rows are not divisible by three. SD1.5/SDXL cannot be flipped from here at
#: all -- they load through diffusers and never reach ``AdapterSession``.
ENABLED_ADAPTER_PAIRS: Mapping[str, FrozenSet[AdapterPair]] = MappingProxyType({
    "sd15": _ORDINARY_ONLY,
    "sdxl": _ORDINARY_ONLY,
    "zimage": _ADDITIVE_LYCORIS_WITH_DORA,
    "anima": _ADDITIVE_LYCORIS,
    "lens": _ADDITIVE_LYCORIS_WITH_DORA,
    "ideogram4": _ADDITIVE_LYCORIS,
    "minit2i": _ADDITIVE_LYCORIS_WITH_DORA,
    "krea2": _ADDITIVE_LYCORIS,
    "flux2": _ADDITIVE_LYCORIS,
    "ltx2": _ADDITIVE_LYCORIS,
    "minimax_h3": _ADDITIVE_LYCORIS,
    "acestep": _ADDITIVE_LYCORIS,
    "sensenova": _ADDITIVE_LYCORIS,
})

#: THE TRAINING axis: which families a trainer may CONSTRUCT, save and resume
#: on that architecture. Necessarily a subset of the generation row -- a
#: checkpoint no loader accepts is not a trained adapter -- and asserted as one
#: at import (below).
#:
#: Each open row is gated by
#: ``backend/tests/adapter_lycoris_training_roundtrip_cheap_test.py`` (trainer
#: save -> real generation loader -> same delta, plus resume).
#: SD1.5/SDXL train through the same adapters but load through diffusers, which
#: does not understand ``hada_*``. MiniMax-H3 and SenseNova generate a LoHa/LoKr
#: but do not train one: their Tier-3 training gate (fused-QKV row splitting,
#: the two MoT halves and phase eviction) is separate work.
TRAINABLE_ADAPTER_PAIRS: Mapping[str, FrozenSet[AdapterPair]] = MappingProxyType({
    "sd15": _ORDINARY_ONLY,
    "sdxl": _ORDINARY_ONLY,
    "zimage": _ADDITIVE_LYCORIS_WITH_DORA,
    "anima": _ADDITIVE_LYCORIS,
    "lens": _ADDITIVE_LYCORIS_WITH_DORA,
    "ideogram4": _ADDITIVE_LYCORIS,
    "minit2i": _ADDITIVE_LYCORIS_WITH_DORA,
    "krea2": _ADDITIVE_LYCORIS,
    "flux2": _ADDITIVE_LYCORIS,
    "ltx2": _ADDITIVE_LYCORIS,
    "minimax_h3": _ORDINARY_ONLY,
    "acestep": _ADDITIVE_LYCORIS,
    "sensenova": _ORDINARY_ONLY,
})

#: The two axes a caller may ask about. ``require()`` takes one explicitly:
#: reading the wrong table is the failure this split exists to prevent.
AXIS_GENERATION = "generation"
AXIS_TRAINING = "training"
ADAPTER_AXES = (AXIS_GENERATION, AXIS_TRAINING)

_AXIS_TABLES: Mapping[str, Mapping[str, FrozenSet[AdapterPair]]] = MappingProxyType({
    AXIS_GENERATION: ENABLED_ADAPTER_PAIRS,
    AXIS_TRAINING: TRAINABLE_ADAPTER_PAIRS,
})

for _arch, _pairs in TRAINABLE_ADAPTER_PAIRS.items():
    _extra = sorted(_pairs - ENABLED_ADAPTER_PAIRS.get(_arch, frozenset()))
    if _extra:
        raise RuntimeError(
            f"{_arch}: {_extra} is trainable but not loadable -- a trained "
            f"checkpoint no generation path accepts is not a feature")
del _arch, _pairs, _extra

#: Reasons shared by the architectures the design-doc table treats alike.
PHASE2_PENDING = ("LoHa and LoKr generate on the architectures whose row "
                  "enables them, but this one is not among them yet, so no "
                  "checkpoint of either can be applied to it")
PHASE3_PENDING = ("DoRA is planned for dense Linear targets but this "
                  "architecture's row is not open")
PHASE3_PENDING_DENSE_ONLY = (
    "DoRA is planned for dense Linear targets ONLY because this architecture's "
    "base may be weight-only quantized, and its row is not open")
PHASE3_DECOMPOSED_PENDING = (
    "dense DoRA ships here, but DoHa and DoKr do not: the engine builds all "
    "three the same way, and each decomposed pair still needs its own "
    "per-architecture round trip")
DORA_DIFFUSERS_STRIPS_MAGNITUDES = (
    "DoRA cannot be applied here: this architecture loads through diffusers, "
    "whose lora_state_dict DROPS every dora_scale key -- with a log line and "
    "nothing else -- before its Kohya converter can see them (measured on "
    "diffusers 0.38.0), so the file would apply as an ordinary LoRA at the "
    "wrong numbers instead of failing")
DORA_QUANTIZED_BASE_REFUSAL = (
    "a weight-decomposed adapter needs the base weight's direction and norm, "
    "so a weight-only quantized base would have to be dequantized every "
    "forward and the fused base GEMM abandoned")
QUANTIZED_ADDITIVE_PENDING = (
    "additive branches over an INT8/FP8/W4A8 base are the second half of "
    "Phase 2 and are not enabled")
QUANTIZED_ADDITIVE_SHIPPED = (
    "additive branches over a weight-only quantized base ship here because "
    "this architecture has no dense configuration to ship first: every LoRA "
    "target it has is a quantized Linear, so enabling LoHa/LoKr at all IS the "
    "quantized-base case. Claimed: they build and forward correctly over the "
    "real quantized layer. NOT claimed: any quality or speed measurement, and "
    "nothing about DoRA, which still needs the base weight's direction and norm")
PHASE2_TRAINING_PENDING = (
    "the training round trip -- build the algebra, save it, resume it, and load "
    "the saved file back through this architecture's own generation path -- is "
    "gated per architecture, and this row is not open")

#: What blocks an architecture whose GENERATION row is open but whose training
#: row is not. It lives here, not on the ArchHandler, because
#: ``api/arch_capabilities.py`` may not import the trainer stack (see its
#: ``TRAINING_DECLARED_ARCHS`` comment) and a sentence no client can read is not
#: a reason. Absent = ``PHASE2_TRAINING_PENDING``.
TRAINING_REFUSAL_REASONS: Mapping[str, str] = MappingProxyType({
    "minimax_h3": (
        "LoHa/LoKr need a training gate of their own: the ConvRot forward and "
        "the activation dtype policy (this forward runs without "
        "torch.autocast) dominate more of the step, and the fused-QKV row "
        "split has no save side yet"),
    "sensenova": (
        "LoHa/LoKr need a training gate of their own: the two MoT halves, "
        "phase eviction and the INT8/ConvRot policy"),
})


def training_refusal_reason(architecture: Optional[str]) -> str:
    """The prose for a closed training row: this architecture's, or the generic."""
    return TRAINING_REFUSAL_REASONS.get(architecture or "", PHASE2_TRAINING_PENDING)


#: What blocks the DECOMPOSITION axis on an architecture whose row is closed.
#: Here rather than on the ArchHandler for the reason ``TRAINING_REFUSAL_REASONS``
#: is: ``api/arch_capabilities.py`` may not import the trainer stack, so a
#: sentence declared on the handler cannot reach a client, and
#: ``declare_adapter_capability`` READS this table so the handler and the
#: payload cannot word the same refusal differently.
DECOMPOSE_REFUSAL_REASONS: Mapping[str, str] = MappingProxyType({
    "sd15": DORA_DIFFUSERS_STRIPS_MAGNITUDES,
    "sdxl": DORA_DIFFUSERS_STRIPS_MAGNITUDES,
    "flux2": PHASE3_PENDING_DENSE_ONLY,
    "anima": PHASE3_PENDING_DENSE_ONLY,
    "ltx2": PHASE3_PENDING_DENSE_ONLY,
    "acestep": PHASE3_PENDING_DENSE_ONLY,
    "krea2": (f"DoRA is deferred here: {DORA_QUANTIZED_BASE_REFUSAL}, and this "
              f"loader can produce INT8/FP8 bases"),
    "ideogram4": (f"DoRA is deferred here: {DORA_QUANTIZED_BASE_REFUSAL}, and "
                  f"either transformer can be loaded FP8"),
    "minimax_h3": (f"DoRA is deferred here: {DORA_QUANTIZED_BASE_REFUSAL}, and "
                   f"this architecture has no dense configuration -- its whole "
                   f"DiT block stack is Fp8Linear. The custom QKV row mapping "
                   f"has no decomposed split either: dora_scale's (1, in) form "
                   f"has no row axis to slice"),
    "sensenova": (f"DoRA is deferred here: {DORA_QUANTIZED_BASE_REFUSAL}, and "
                  f"this architecture has no dense configuration -- all 294 "
                  f"targets per MoT half are Int8Linear"),
})


def decompose_refusal_reason(architecture: Optional[str],
                             algorithm: str = ALGORITHM_LORA) -> str:
    """The prose for a closed decomposition row.

    Two different facts share the axis. Where dense DoRA is OPEN, a DoHa/DoKr
    request is refused because that PAIR has no round trip, not because
    decomposition is unimplemented -- telling a Z-Image user that DoRA has not
    landed is false there, the same way "LoHa is unimplemented" became false at
    the Phase 2 flips.
    """
    if algorithm != ALGORITHM_LORA and (ALGORITHM_LORA, True) in supported_pairs(architecture):
        return PHASE3_DECOMPOSED_PENDING
    return DECOMPOSE_REFUSAL_REASONS.get(architecture or "", PHASE3_PENDING)

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
    # FLUX.2 loads its LoRAs in stage 1 and builds create_flux_block_offloader
    # hundreds of lines later, in stage 3 of the same generate function.
    # MiniMax-H3 loads them in the stage_transformer phase and builds its
    # per-generation TransformerBlockOffloader three lines later, in
    # _ensure_minimax_h3_swap_and_offload.
    "zimage": BEFORE_SPLIT,
    "flux2": BEFORE_SPLIT,
    "minimax_h3": BEFORE_SPLIT,
    "krea2": NO_BLOCK_SWAP,
    "acestep": NO_BLOCK_SWAP,
    # SenseNova's blocks_to_swap is inert (its backend never reads it); its MoT
    # phase evictor is not a TransformerBlockOffloader and moves a module's own
    # parameters, so it carries a LyCORIS branch with the half it sits under.
    "sensenova": NO_BLOCK_SWAP,
    # The offloader is already built when the adapters install, so a branch over
    # a swapped-out block is built on the HOST and nothing ever moves it:
    # _minit2i_stage_transformer / _ensure_ltx2_block_swap_wrapper, and
    # _anima_stage_transformer / _lens_stage_transformer /
    # _ideogram4_stage_transformers, each one call before the LoRA load.
    "minit2i": AFTER_SPLIT,
    "ltx2": AFTER_SPLIT,
    "anima": AFTER_SPLIT,
    "lens": AFTER_SPLIT,
    "ideogram4": AFTER_SPLIT,
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
    "ADAPTER_AXES",
    "ADAPTER_PAIRS",
    "AXIS_GENERATION",
    "AXIS_TRAINING",
    "PHASE2_TRAINING_PENDING",
    "TRAINING_REFUSAL_REASONS",
    "training_refusal_reason",
    "TRAINABLE_ADAPTER_PAIRS",
    "trainable_pairs",
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
    "DECOMPOSE_REFUSAL_REASONS",
    "DORA_DIFFUSERS_STRIPS_MAGNITUDES",
    "DORA_QUANTIZED_BASE_REFUSAL",
    "PHASE3_DECOMPOSED_PENDING",
    "PHASE3_PENDING",
    "PHASE3_PENDING_DENSE_ONLY",
    "decompose_refusal_reason",
    "QUANTIZED_ADDITIVE_PENDING",
    "QUANTIZED_ADDITIVE_SHIPPED",
    "adapter_refusal_reason",
    "adapter_training_refusal_reason",
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


def _axis_table(axis: str) -> Mapping[str, FrozenSet[AdapterPair]]:
    try:
        return _AXIS_TABLES[axis]
    except KeyError:
        raise ValueError(
            f"axis {axis!r} is not one of {ADAPTER_AXES}") from None


def declared_pairs(architecture: str,
                   axis: str = AXIS_GENERATION) -> FrozenSet[AdapterPair]:
    """Strict lookup, for a DECLARATION rather than a runtime check.

    Raises instead of defaulting: an architecture missing from the table would
    otherwise declare that ordinary LoRA is refused for it.
    """
    table = _axis_table(axis)
    try:
        return table[architecture]
    except KeyError:
        raise KeyError(
            f"{architecture!r} has no row in the {axis} adapter table "
            f"(core/adapters/capability.py); add one before declaring its "
            f"adapter capability") from None


def trainable_pairs(architecture: Optional[str]) -> FrozenSet[AdapterPair]:
    """What ``architecture`` may TRAIN, empty for an unknown name."""
    if not architecture:
        return frozenset()
    return TRAINABLE_ADAPTER_PAIRS.get(architecture, frozenset())


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


def adapter_training_refusal_reason(architecture: Optional[str], algorithm: str,
                                    weight_decompose: bool = False
                                    ) -> Optional[str]:
    """Why this architecture will not TRAIN the pair, or ``None`` if it will.

    A pair that does not generate here cannot be trained here either, and that
    is the reason worth reporting, so the generation refusal wins when it
    applies.
    """
    pair = (algorithm, bool(weight_decompose))
    if pair in trainable_pairs(architecture):
        return None
    generation = adapter_refusal_reason(architecture, algorithm, weight_decompose)
    if generation is not None:
        return generation
    where = architecture or "this build"
    why = PHASE3_PENDING if pair[1] else training_refusal_reason(architecture)
    return f"{where}: {FAMILY_NAMES[pair]} adapters cannot be trained -- {why}"


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
    if not pair[1]:
        why = PHASE2_PENDING
    else:
        why = decompose_refusal_reason(architecture, algorithm)
        if (algorithm != ALGORITHM_LORA
                and (algorithm, False) not in supported_pairs(architecture)):
            # Blocked twice over: by the decomposition AND by the algebra under it.
            why = f"{why}; and {PHASE2_PENDING}"
    return f"{where}: {FAMILY_NAMES[pair]} adapters are not enabled -- {why}"
