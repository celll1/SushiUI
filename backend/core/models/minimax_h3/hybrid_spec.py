"""MiniMax-H3 hybrid DiT: the spec, the header-only preflight and the selector.

C2 of ``docs/guides/MINIMAX_H3_HYBRID_LOADER_DESIGN.md`` (rev2), sections 4.1,
4.2, 4.4. Reads headers only; builds no model and reads no tensor bytes. The
reader (C3) and the component lifecycle (C4) are separate commits; nothing here
is wired into the base-only load path.

Attribution (doc section 1.1): the recipe -- overlay a ``ref2va`` checkpoint's
per-block AdaLN projection onto an ``fl2va`` base over a block range, default
25..49 -- comes from ``ComfyUI_MinimaxH3HybridLoader`` by scottmudge, MIT
licence, https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader . The
validation, digest, sidecar-atomicity rule and refusal set are this repo's own;
upstream merges on a key-set equality check alone.

THE HEADER-SOURCE CONTRACT (doc section 4.2, closing paragraph)
--------------------------------------------------------------
Checks 4/5/7/8 prove the two files agree on key set, shape, dtype and
quantization contract, so every downstream consumer of "the header" or "the
metadata" reads the BASE file's, and no implementer decides that per call site.
Encoded structurally rather than by comment: ``MiniMaxH3HybridPreflight``
carries the base file's ``header`` / ``metadata`` and no overlay HEADER DICT at
all, so a consumer holding it cannot reach for the wrong one. The overlay's
PATH does remain reachable (``spec.overlay_dit_path``, ``overlay_layout``) --
C3 needs it to open the second handle. The guarantee is about header dicts, not
about hiding the file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .loader import (
    _w4a8_layers_from_metadata,
    detect_minimax_h3_layout,
    is_minimax_h3_safetensors,
    read_safetensors_header,
)
from .reload import _SHARED_LAYOUT_KEYS, same_path


# Bumped independently: the spec's FIELD SET and the digest's INPUTS change for
# different reasons.
SCHEMA_VERSION = 1
DIGEST_VERSION = 1

#: Which of the two files every header/metadata consumer downstream reads.
HEADER_SOURCE = "base"

PRESET_BLOCK_RANGE_ADALN = "block_range_adaln"

#: Non-MVP recipes, enumerated so asking for one gets a refusal that says why
#: rather than an "unknown preset" that reads like a typo.
_REFUSED_PRESETS: Dict[str, str] = {
    "custom_glob": "rule precedence between overlapping globs is undesigned, so a glob "
                   "would silently pick a source for keys nobody validated",
    "full_overlay": "that is a plain single-file load of the overlay, not a hybrid",
    "all": "that is a plain single-file load of the overlay, not a hybrid",
    "multi_overlay": "with more than one overlay, per-key precedence is undesigned",
}

# The doc's MVP defaults, and the ONLY declaration of them in the repo. C5's
# ``H3_HYBRID_LOAD_DEFAULTS`` (backend/api/param_defaults.py) must import these
# rather than re-type 25/49, so the API default and the loader default cannot
# drift. api -> core is the direction this repo already uses; core must not
# import api.
DEFAULT_BLOCK_RANGE_START = 25
DEFAULT_BLOCK_RANGE_END = 49
DEFAULT_FINAL_ADALN_FROM_OVERLAY = False

BASE = "base"
OVERLAY = "overlay"

# Real sidecar key names, audited against loader.py (doc section 4.4).
# ``.comfy_quant`` IS the INT8-ConvRot marker; there is no separate marker key.
QUANT_SIDECAR_SUFFIXES: Tuple[str, ...] = (
    ".weight_scale", ".comfy_quant", ".weight_s_rel", ".weight_s_channel",
    ".weight_codebook", ".weight_correction",
)

# Section 4.4's atomicity rule: an overlaid weight brings these with it. Shipped
# checkpoints never exercise this (adaln_proj is not quantized -- the 200
# quantized Linears are qkv / out_proj / fc1 / fc2), so it is a guard against a
# future export, not machinery in use today.
_WEIGHT_FAMILY_SUFFIXES: Tuple[str, ...] = (".weight",) + QUANT_SIDECAR_SUFFIXES
_BIAS_FAMILY_SUFFIXES: Tuple[str, ...] = (".bias",)

# Dropped by loader policy (``_DIT_DROPPED_KEYS`` neighbourhood), so they have
# no provenance to get wrong.
_DROPPED_SUFFIXES: Tuple[str, ...] = (".input_scale",)

_ADALN_KEY_RE = re.compile(
    r"^(?:blocks\.(?P<block>\d+)|(?P<final>final_layer))\.adaln_proj\.linear"
    r"(?P<suffix>\.[A-Za-z0-9_]+)$"
)

_BLOCK_INDEX_RE = re.compile(r"^blocks\.(\d+)\.")

GEOMETRY_PRUNED = "pruned_adaln_curve"
GEOMETRY_FULL = "full_modulation"
GEOMETRY_CONTRADICTORY = "contradictory"


class MiniMaxH3HybridRefusal(ValueError):
    """A hybrid pair (or recipe) this loader will not combine.

    Carries a stable ``code`` so callers and tests can assert WHICH check
    refused without matching prose.
    """

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def _refuse(code: str, message: str) -> "MiniMaxH3HybridRefusal":
    return MiniMaxH3HybridRefusal(code, message)


# ---------------------------------------------------------------------------
# 4.1 -- the spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MiniMaxH3HybridSpec:
    """The logical input of a hybrid load (doc section 4.1).

    Field order differs from the doc's listing only because Python forbids a
    defaulted field before an undefaulted one; ``as_dict()`` emits the doc's
    order. ``compatibility_digest`` is ``None`` until the preflight fills it in
    -- an unvalidated spec is a request, not a contract, and ``validated`` says
    which one you hold.
    """

    base_dit_path: str
    overlay_dit_path: str
    preset: str = PRESET_BLOCK_RANGE_ADALN
    block_range_start: int = DEFAULT_BLOCK_RANGE_START
    block_range_end: int = DEFAULT_BLOCK_RANGE_END
    final_adaln_from_overlay: bool = DEFAULT_FINAL_ADALN_FROM_OVERLAY
    base_variant: Optional[str] = None
    overlay_variant: Optional[str] = None
    compatibility_digest: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    @property
    def validated(self) -> bool:
        return self.compatibility_digest is not None

    def as_dict(self) -> Dict[str, Any]:
        """The doc's section 4.1 field order, for provenance and persistence."""
        return {
            "schema_version": self.schema_version,
            "base_dit_path": self.base_dit_path,
            "overlay_dit_path": self.overlay_dit_path,
            "preset": self.preset,
            "block_range_start": self.block_range_start,
            "block_range_end": self.block_range_end,
            "final_adaln_from_overlay": self.final_adaln_from_overlay,
            "base_variant": self.base_variant,
            "overlay_variant": self.overlay_variant,
            "compatibility_digest": self.compatibility_digest,
        }

    def recipe(self) -> Dict[str, Any]:
        """The part that changes what the model IS, minus the file identities.

        C4 folds this into model identity so "same pair, different range" does
        not hit the same-model early return.
        """
        return {
            "preset": self.preset,
            "block_range_start": self.block_range_start,
            "block_range_end": self.block_range_end,
            "final_adaln_from_overlay": self.final_adaln_from_overlay,
        }


def validate_preset(preset: str, overlay_dit_path: Any) -> None:
    """Refuse every non-MVP recipe shape by name (doc section 8).

    ``overlay_dit_path`` is inspected here too: a list/tuple is the "multiple
    overlays" request, and it must refuse rather than be indexed into.
    """
    if isinstance(overlay_dit_path, (list, tuple, set)):
        raise _refuse(
            "multiple_overlays",
            f"{len(overlay_dit_path)} overlay checkpoints were requested; exactly one is "
            "supported. With more than one, per-key precedence is undesigned.")
    if preset in _REFUSED_PRESETS:
        raise _refuse("preset_unsupported",
                      f"overlay preset {preset!r} is refused: {_REFUSED_PRESETS[preset]}")
    if preset != PRESET_BLOCK_RANGE_ADALN:
        raise _refuse(
            "preset_unknown",
            f"unknown overlay preset {preset!r}; the only implemented preset is "
            f"{PRESET_BLOCK_RANGE_ADALN!r}")


# ---------------------------------------------------------------------------
# 4.3 -- the structured selector (pure, over raw key names)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockRangeAdalnSelector:
    """Which file a RAW checkpoint key is read from. Pure; no I/O; no header.

    The whole rule is ``start <= N <= end`` (inclusive both ends) over
    ``blocks.<N>.adaln_proj.linear.*``, plus one explicit, default-off toggle
    for ``final_layer.adaln_proj.linear.*``. Everything else -- including
    ``adaln_t_table`` -- stays base. No glob, no fallthrough.

    TOTAL for every key the selection does not touch: selection is decided
    first, so only a key that WOULD be overlaid can raise. An unclassified
    sidecar on an out-of-range block is unambiguously base, and refusing it
    would kill the feature for ranges that never touch it.

    ``overlay_bias`` is decided by the preflight, not assumed (check 9).
    """

    block_range_start: int
    block_range_end: int
    final_adaln_from_overlay: bool = False
    overlay_bias: bool = False

    def in_range(self, block: int) -> bool:
        return self.block_range_start <= block <= self.block_range_end

    def source_for(self, key: str) -> str:
        """``"base"`` or ``"overlay"`` for a raw (Comfy-spelling) key."""
        match = _ADALN_KEY_RE.match(key)
        if match is None:
            return BASE

        if match.group("final") is not None:
            selected = self.final_adaln_from_overlay
        else:
            selected = self.in_range(int(match.group("block")))
        if not selected:
            return BASE

        suffix = match.group("suffix")
        if suffix in _DROPPED_SUFFIXES:
            return BASE
        if suffix in _WEIGHT_FAMILY_SUFFIXES:
            return OVERLAY
        if suffix in _BIAS_FAMILY_SUFFIXES:
            return OVERLAY if self.overlay_bias else BASE
        # A SELECTED Linear carrying a sidecar nobody has classified. Guessing
        # is how you get a load that succeeds and infers garbage (section 4.4).
        # The preflight refuses these too; this is the belt to that braces.
        raise _refuse(
            "adaln_sidecar_unknown",
            f"the selected AdaLN key {key!r} carries an unrecognised suffix {suffix!r}; its "
            "provenance cannot be decided, and reading a sidecar from a different file "
            "than its weight loads cleanly and infers garbage.")

    def overlay_keys(self, keys: Sequence[str]) -> List[str]:
        """The subset of ``keys`` this selector reads from the overlay, sorted."""
        return sorted(k for k in keys if self.source_for(k) == OVERLAY)


# ---------------------------------------------------------------------------
# header helpers (no tensor bytes)
# ---------------------------------------------------------------------------

def _read_header_and_metadata(path: str, *, side: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        header = read_safetensors_header(path)
    except Exception as exc:
        raise _refuse("header_unreadable",
                      f"the {side} checkpoint {os.path.basename(path)!r} has no readable "
                      f"safetensors header: {exc}") from exc
    metadata = header.pop("__metadata__", None) or {}
    return header, metadata


def _entry(header: Mapping[str, Any], key: str) -> Dict[str, Any]:
    entry = header.get(key)
    return entry if isinstance(entry, dict) else {}


def _dtype_of(header: Mapping[str, Any], key: str) -> Optional[str]:
    return _entry(header, key).get("dtype")


def _shape_of(header: Mapping[str, Any], key: str) -> Optional[List[int]]:
    shape = _entry(header, key).get("shape")
    return [int(x) for x in shape] if isinstance(shape, list) else None


def _num_blocks(header: Mapping[str, Any]) -> int:
    highest = -1
    for key in header:
        match = _BLOCK_INDEX_RE.match(key)
        if match is not None:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def geometry_of(header: Mapping[str, Any]) -> str:
    """Pruned AdaLN-curve, full modulation, or a file that is neither.

    Read the way loader.py reads it: an ``adaln_t_table`` and NO
    ``time_embedder.*`` is the pruned variant; both present is the file the
    loader calls "matches neither", and it gets its own answer here so the
    refusal does not describe it as full-modulation.
    """
    has_table = "adaln_t_table" in header
    has_embedder = any(k.startswith("time_embedder.") for k in header)
    if has_table and has_embedder:
        return GEOMETRY_CONTRADICTORY
    return GEOMETRY_PRUNED if has_table else GEOMETRY_FULL


def quantization_format(header: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    """A coarse format label from the header alone. No tensor bytes.

    Classified INDEPENDENTLY, not first-match-wins, and joined with ``+``: this
    repo's own ``--format int8`` exporter emits a MIXED file on purpose
    (loader.py's DiT quantization policy detects INT8 and e4m3 separately and
    runs both swaps), and section 5.4 puts this label into generation metadata,
    where calling such a file "fp8_scaled" would be a false record.

    ``.comfy_quant`` CONTENTS cannot be consulted -- that is a tensor read -- so
    the evidence is what the header states: ``.weight_s_rel`` sidecars for W4A8,
    an F8 weight dtype for fp8, I8 weights plus a marker for INT8 ConvRot.
    """
    keys = list(header)
    weight_dtypes = {_dtype_of(header, k) for k in keys if k.endswith(".weight")}
    has_markers = any(k.endswith(".comfy_quant") for k in keys)

    labels = []
    if any(k.endswith(".weight_s_rel") for k in keys):
        labels.append("w4a8_mixed")
    if weight_dtypes & {"F8_E4M3", "F8_E5M2"}:
        labels.append("fp8_scaled")
    if "I8" in weight_dtypes:
        labels.append("int8_convrot" if has_markers else "int8_unmarked")
    if not labels:
        if has_markers:
            labels.append("comfy_quant_unclassified")
        elif "_quantization_metadata" in metadata:
            labels.append("declared_quant_no_sidecars")
        else:
            labels.append("unquantized")
    return "+".join(sorted(labels))


def _declared_quant_layers(metadata: Mapping[str, Any], *, side: str) -> Optional[Dict[str, Any]]:
    """``_quantization_metadata``'s ``layers`` object, or ``None`` when absent."""
    payload = _quant_metadata_payload(metadata, side=side)
    if payload is None:
        return None
    layers = payload.get("layers") if isinstance(payload, dict) else None
    if not isinstance(layers, dict):
        raise _refuse("quant_metadata_malformed",
                      f"the {side} checkpoint's _quantization_metadata has no 'layers' object")
    return layers


def _quant_metadata_payload(metadata: Mapping[str, Any], *, side: str) -> Optional[Any]:
    """The parsed ``_quantization_metadata``, or ``None``. Refuses, never raises raw."""
    raw = metadata.get("_quantization_metadata")
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise _refuse("quant_metadata_malformed",
                      f"the {side} checkpoint's _quantization_metadata is not valid JSON: "
                      f"{exc}") from exc


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _key_census(header: Mapping[str, Any]) -> str:
    """A hash over (key, dtype, shape) for every key. Order-independent."""
    lines = sorted(
        f"{key}\t{_dtype_of(header, key)}\t{_shape_of(header, key)}" for key in header)
    return _sha256("\n".join(lines))


def _file_size(path: str) -> int:
    try:
        return int(os.path.getsize(path))
    except OSError:
        return -1


# ---------------------------------------------------------------------------
# 4.2 -- the preflight result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MiniMaxH3HybridPreflight:
    """The validated contract: what C3/C4 need, and not the overlay's header.

    ``header`` / ``metadata`` are the BASE file's -- see the module docstring's
    header-source contract.
    """

    spec: MiniMaxH3HybridSpec
    header: Dict[str, Any]
    metadata: Dict[str, Any]
    base_layout: Dict[str, Optional[str]]
    overlay_layout: Dict[str, Optional[str]]
    selector: BlockRangeAdalnSelector
    overlay_keys: Tuple[str, ...]
    quant_format: str
    num_blocks: int
    overlay_bias_eligible: bool
    header_source: str = HEADER_SOURCE

    def provenance(self) -> Dict[str, Any]:
        """Sanitised provenance for ``current_model_info`` and image metadata.

        Basenames only: doc section 5.4 forbids absolute paths in gallery
        metadata.
        """
        return {
            "variant": "hybrid",
            "base_variant": self.spec.base_variant,
            "overlay_variant": self.spec.overlay_variant,
            "base_file": os.path.basename(self.spec.base_dit_path),
            "overlay_file": os.path.basename(self.spec.overlay_dit_path),
            "hybrid_recipe": self.spec.recipe(),
            "compatibility_digest": self.spec.compatibility_digest,
            "quantization_format": self.quant_format,
            "overlay_key_count": len(self.overlay_keys),
        }


# ---------------------------------------------------------------------------
# 4.2 -- the preflight itself
# ---------------------------------------------------------------------------

def preflight_minimax_h3_hybrid(
    base_dit_path: str,
    overlay_dit_path: str,
    *,
    preset: str = PRESET_BLOCK_RANGE_ADALN,
    block_range_start: int = DEFAULT_BLOCK_RANGE_START,
    block_range_end: int = DEFAULT_BLOCK_RANGE_END,
    final_adaln_from_overlay: bool = DEFAULT_FINAL_ADALN_FROM_OVERLAY,
) -> MiniMaxH3HybridPreflight:
    """Validate a (base, overlay) DiT pair from HEADERS ONLY. Zero tensor reads.

    Runs the doc's section 4.2 checks in order, each with its own refusal code,
    and returns the validated contract with a ``compatibility_digest``. Raises
    ``MiniMaxH3HybridRefusal`` on the first failure; there is no warn-and-
    continue path, because each of these produces a load that either crashes
    late or succeeds and infers garbage.
    """
    validate_preset(preset, overlay_dit_path)

    # --- 0. both files exist ----------------------------------------------
    # Precedes the doc's ten: without it a typo'd overlay surfaces as "not the
    # same tree", which is a true statement about the wrong thing.
    for side, path in (("base", base_dit_path), ("overlay", overlay_dit_path)):
        if not path:
            raise _refuse(f"{side}_missing", f"no {side} DiT path was given")
        if not os.path.isfile(path):
            raise _refuse(f"{side}_missing",
                          f"the {side} DiT {path!r} does not exist (or is not a file)")

    # --- 1. same H3 tree, same shared components --------------------------
    base_layout = detect_minimax_h3_layout(base_dit_path)
    overlay_layout = detect_minimax_h3_layout(overlay_dit_path)
    if base_layout is None or overlay_layout is None:
        missing = "base" if base_layout is None else "overlay"
        raise _refuse("not_an_h3_tree",
                      f"the {missing} DiT does not sit in a MiniMax-H3 tree "
                      "(no diffusion_models/ parent was found)")
    differing = [key for key in _SHARED_LAYOUT_KEYS
                 if not same_path(base_layout.get(key), overlay_layout.get(key))]
    if differing:
        raise _refuse(
            "different_tree",
            f"base and overlay resolve to different shared components ({', '.join(differing)}). "
            "A hybrid shares one official/, one video VAE, one audio VAE and one text encoder; "
            "combining DiTs across trees is out of scope.")

    # --- 2. direction: base fl2va, overlay ref2va -------------------------
    base_variant = base_layout.get("variant")
    overlay_variant = overlay_layout.get("variant")
    if base_variant != "fl2va" or overlay_variant != "ref2va":
        raise _refuse(
            "variant_direction",
            f"the supported hybrid is base=fl2va + overlay=ref2va; got base={base_variant!r}, "
            f"overlay={overlay_variant!r}. The recipe is not symmetric -- it puts the "
            "reference-trained AdaLN onto the keyframe-trained body -- and the reverse has "
            "never been measured.")

    # --- 3. both are MiniMax-H3 single files ------------------------------
    for side, path in (("base", base_dit_path), ("overlay", overlay_dit_path)):
        if not is_minimax_h3_safetensors(path):
            raise _refuse("not_h3_checkpoint",
                          f"the {side} file {os.path.basename(path)!r} does not carry the "
                          "MiniMax-H3 single-file key signature")

    base_header, base_metadata = _read_header_and_metadata(base_dit_path, side="base")
    overlay_header, overlay_metadata = _read_header_and_metadata(overlay_dit_path,
                                                                side="overlay")

    # --- 4. identical raw key sets ----------------------------------------
    base_keys = set(base_header)
    overlay_keys_set = set(overlay_header)
    if base_keys != overlay_keys_set:
        only_base = sorted(base_keys - overlay_keys_set)
        only_overlay = sorted(overlay_keys_set - base_keys)
        raise _refuse(
            "key_set_mismatch",
            f"base and overlay declare different tensors: {len(only_base)} only in base "
            f"(first 5: {only_base[:5]}), {len(only_overlay)} only in overlay "
            f"(first 5: {only_overlay[:5]}).")

    # --- 5. identical shapes, then identical dtypes -----------------------
    # Shape first: a shape mismatch is a different model, a dtype mismatch is
    # usually a different export of the same one, so the deeper difference is
    # the more actionable message.
    for key in sorted(base_keys):
        base_shape, overlay_shape = _shape_of(base_header, key), _shape_of(overlay_header, key)
        if base_shape != overlay_shape:
            raise _refuse(
                "shape_mismatch",
                f"tensor {key!r} is {base_shape} in the base and {overlay_shape} in the "
                "overlay; the two checkpoints are not the same geometry.")
    for key in sorted(base_keys):
        base_dtype, overlay_dtype = _dtype_of(base_header, key), _dtype_of(overlay_header, key)
        if base_dtype != overlay_dtype:
            raise _refuse(
                "dtype_mismatch",
                f"tensor {key!r} is {base_dtype} in the base and {overlay_dtype} in the "
                "overlay; mixing storage formats across the pair is out of scope.")

    # --- 6. pruned AdaLN-curve geometry on both sides ---------------------
    base_geometry = geometry_of(base_header)
    overlay_geometry = geometry_of(overlay_header)
    if base_geometry != overlay_geometry:
        raise _refuse(
            "geometry_mismatch",
            f"base is {base_geometry} and overlay is {overlay_geometry}; the AdaLN "
            "projections these variants carry are not the same operator.")
    if base_geometry == GEOMETRY_CONTRADICTORY:
        raise _refuse(
            "geometry_contradictory",
            "both checkpoints carry BOTH an 'adaln_t_table' and 'time_embedder.*' keys. The "
            "AdaLN-curve ('pruned') variant and the full-modulation variant are mutually "
            "exclusive; these files match neither.")
    if base_geometry != GEOMETRY_PRUNED:
        raise _refuse(
            "geometry_unsupported",
            "both checkpoints are the full-modulation variant. The MVP hybrid covers the "
            "pruned AdaLN-curve geometry only; the full variant's AdaLN is driven by a "
            "time_embedder MLP whose overlay semantics are unmeasured.")

    # --- 7. declared quantization metadata agrees -------------------------
    # Key-level agreement (marker layers, sidecar presence) is already proven by
    # check 4, which is why this looks at ``__metadata__`` instead: it is popped
    # before the key census, and it is where format, group sizes and ConvRot
    # semantics are declared.
    base_layers = _declared_quant_layers(base_metadata, side="base")
    overlay_layers = _declared_quant_layers(overlay_metadata, side="overlay")
    if (base_layers is None) != (overlay_layers is None):
        present, absent = ("base", "overlay") if base_layers is not None else ("overlay", "base")
        raise _refuse(
            "quant_metadata_mismatch",
            f"the {present} checkpoint declares _quantization_metadata and the {absent} one "
            "does not; the pair does not share a quantization contract.")
    if base_layers is not None and _canonical(base_layers) != _canonical(overlay_layers):
        only_base = sorted(set(base_layers) - set(overlay_layers or {}))
        only_overlay = sorted(set(overlay_layers or {}) - set(base_layers))
        differing_fields = sorted(
            layer for layer in set(base_layers) & set(overlay_layers or {})
            if _canonical(base_layers[layer]) != _canonical((overlay_layers or {})[layer]))
        raise _refuse(
            "quant_metadata_mismatch",
            "base and overlay declare different quantization contracts "
            f"(layers only in base: {only_base[:5]}, only in overlay: {only_overlay[:5]}, "
            f"declared differently: {differing_fields[:5]}).")

    # --- 8. W4A8: the per-layer contract validates identically ------------
    # Reuses the loader's own validator, header-only, on both files, so a hybrid
    # can never accept a W4A8 contract the single-file path would refuse.
    base_w4a8 = _validated_w4a8(base_metadata, base_header, path=base_dit_path, side="base")
    overlay_w4a8 = _validated_w4a8(overlay_metadata, overlay_header,
                                   path=overlay_dit_path, side="overlay")
    if _canonical(base_w4a8) != _canonical(overlay_w4a8):
        raise _refuse(
            "w4a8_contract_mismatch",
            "the two checkpoints' validated W4A8 layer contracts differ (group sizes or "
            "ConvRot group sizes are not the same on every layer).")

    # --- 9. the selected AdaLN keys exist on both sides -------------------
    num_blocks = _num_blocks(base_header)
    overlay_bias_eligible = _validate_block_range(
        base_header, overlay_header,
        start=block_range_start, end=block_range_end,
        num_blocks=num_blocks, final_adaln_from_overlay=final_adaln_from_overlay)

    # --- 10. one and the same quantization format -------------------------
    # UNREACHABLE after checks 5/7/8, which leave a format split no route to
    # take. Kept because computing the label on BOTH files is what makes "the
    # label describes the pair" true rather than assumed -- and that label is
    # what the digest and the generation metadata carry.
    base_format = quantization_format(base_header, base_metadata)
    overlay_format = quantization_format(overlay_header, overlay_metadata)
    if base_format != overlay_format:
        raise _refuse(
            "format_mismatch",
            f"the base is {base_format} and the overlay is {overlay_format}. Mixing "
            "quantization formats is out of scope: the quantized module contract and the "
            "sidecar conversion differ per format.")

    selector = BlockRangeAdalnSelector(
        block_range_start=block_range_start,
        block_range_end=block_range_end,
        final_adaln_from_overlay=final_adaln_from_overlay,
        overlay_bias=overlay_bias_eligible,
    )
    overlay_keys = tuple(selector.overlay_keys(list(base_header)))

    digest = compatibility_digest(
        base_path=base_dit_path,
        overlay_path=overlay_dit_path,
        base_variant=base_variant,
        overlay_variant=overlay_variant,
        header=base_header,
        metadata=base_metadata,
        overlay_header=overlay_header,
        quant_format=base_format,
        num_blocks=num_blocks,
    )
    spec = MiniMaxH3HybridSpec(
        base_dit_path=base_dit_path,
        overlay_dit_path=overlay_dit_path,
        preset=preset,
        block_range_start=block_range_start,
        block_range_end=block_range_end,
        final_adaln_from_overlay=final_adaln_from_overlay,
        base_variant=base_variant,
        overlay_variant=overlay_variant,
        compatibility_digest=digest,
    )
    return MiniMaxH3HybridPreflight(
        spec=spec,
        header=base_header,
        metadata=base_metadata,
        base_layout=base_layout,
        overlay_layout=overlay_layout,
        selector=selector,
        overlay_keys=overlay_keys,
        quant_format=base_format,
        num_blocks=num_blocks,
        overlay_bias_eligible=overlay_bias_eligible,
    )


def _validated_w4a8(metadata: Mapping[str, Any], header: Mapping[str, Any], *,
                    path: str, side: str) -> Dict[str, Dict[str, Any]]:
    try:
        return _w4a8_layers_from_metadata(dict(metadata), dict(header), path=path)
    except MiniMaxH3HybridRefusal:
        raise
    except Exception as exc:
        raise _refuse("w4a8_contract_invalid",
                      f"the {side} checkpoint's W4A8 contract does not validate: {exc}") from exc


def _validate_block_range(
    base_header: Mapping[str, Any],
    overlay_header: Mapping[str, Any],
    *,
    start: int,
    end: int,
    num_blocks: int,
    final_adaln_from_overlay: bool,
) -> bool:
    """Check 9. Returns whether the AdaLN BIAS is overlay-eligible.

    The bias is eligible only when BOTH files carry it for every selected block,
    and it is NOT assumed to exist: the loader's geometry synthesis reads only
    the weight, and whether the shipped exports carry an AdaLN bias was never
    confirmed.
    """
    if not isinstance(start, int) or not isinstance(end, int) \
            or isinstance(start, bool) or isinstance(end, bool):
        raise _refuse("block_range_invalid",
                      f"the block range must be two integers, got ({start!r}, {end!r})")
    if start > end:
        raise _refuse(
            "block_range_empty",
            f"the block range {start}..{end} selects no blocks. An empty overlay range is "
            "refused rather than quietly degrading to a base-only load; ask for a base-only "
            "load if that is what you want.")
    if start < 0:
        raise _refuse("block_range_out_of_range",
                      f"the block range starts at {start}; block indices start at 0.")
    if num_blocks <= 0:
        raise _refuse("no_transformer_blocks",
                      "the base checkpoint declares no 'blocks.N.*' tensors at all")
    if end >= num_blocks:
        raise _refuse(
            "block_range_out_of_range",
            f"the block range {start}..{end} runs past the checkpoint's last block "
            f"({num_blocks - 1}); it declares {num_blocks} blocks.")

    weight_keys = [f"blocks.{n}.adaln_proj.linear.weight" for n in range(start, end + 1)]
    if final_adaln_from_overlay:
        weight_keys.append("final_layer.adaln_proj.linear.weight")
    missing = [k for k in weight_keys if k not in base_header or k not in overlay_header]
    if missing:
        raise _refuse(
            "adaln_weight_missing",
            f"{len(missing)} selected AdaLN weight(s) are absent from the checkpoints "
            f"(first 5: {missing[:5]}); the range names tensors this pair does not have.")

    bias_keys = [k[: -len(".weight")] + ".bias" for k in weight_keys]
    in_base = [k for k in bias_keys if k in base_header]
    in_overlay = [k for k in bias_keys if k in overlay_header]
    if len(in_base) not in (0, len(bias_keys)) or len(in_overlay) not in (0, len(bias_keys)):
        raise _refuse(
            "adaln_bias_partial",
            "the AdaLN bias exists for some selected blocks and not others; the selection "
            "cannot be applied consistently.")
    if bool(in_base) != bool(in_overlay):
        # UNREACHABLE after check 4: a one-sided bias key makes the key sets
        # differ, so check 4 speaks first. Kept as defence in depth for a future
        # caller that reaches this function without the full preflight.
        side = "base" if in_base else "overlay"
        raise _refuse(
            "adaln_bias_one_sided",
            f"only the {side} checkpoint carries an AdaLN bias for the selected blocks; the "
            "bias is overlay-eligible only when both sides have it.")

    # Section 4.4 atomicity: any sidecar hanging off a SELECTED AdaLN Linear
    # must be one whose provenance we know how to decide. Shipped checkpoints
    # have none (adaln_proj is not quantized).
    selected_prefixes = tuple(k[: -len(".weight")] + "." for k in weight_keys)
    known = set(_WEIGHT_FAMILY_SUFFIXES) | set(_BIAS_FAMILY_SUFFIXES) | set(_DROPPED_SUFFIXES)
    unknown = sorted(
        key for key in base_header
        if key.startswith(selected_prefixes) and "." + key.rsplit(".", 1)[-1] not in known)
    if unknown:
        raise _refuse(
            "adaln_sidecar_unknown",
            f"selected AdaLN Linears carry unclassified sidecar tensors (first 5: "
            f"{unknown[:5]}). Reading a sidecar from a different file than its weight loads "
            "cleanly and infers garbage, so an unknown one is refused.")

    return bool(in_base and in_overlay)


# ---------------------------------------------------------------------------
# 4.2 -- the compatibility digest
# ---------------------------------------------------------------------------

def compatibility_digest(
    *,
    base_path: str,
    overlay_path: str,
    base_variant: Optional[str],
    overlay_variant: Optional[str],
    header: Mapping[str, Any],
    metadata: Mapping[str, Any],
    overlay_header: Mapping[str, Any],
    quant_format: str,
    num_blocks: int,
) -> str:
    """A stable digest of the VALIDATED CONTRACT (not of the recipe).

    Reproducible: the same two files digest the same on any machine and run.
    Paths are reduced to basenames because doc section 5.4 keeps absolute paths
    out of provenance.

    BOTH files' censuses and sizes are inputs, because the digest has to
    identify the actual files: every tree holds the same release filenames, so
    a base-only digest collides across trees, and doc section 7 needs an overlay
    replaced between preflight and the real read to be detectable. A size costs
    one stat and no header.

    The block range and the final-AdaLN toggle are deliberately NOT inputs: they
    are the recipe, not the contract. C4's identity is digest + recipe, which is
    what lets "same pair, different range" read as the same validated pair
    rather than an unrelated model.
    """
    payload = {
        "digest_version": DIGEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "base": {
            "file": os.path.basename(base_path),
            "variant": base_variant,
            "size": _file_size(base_path),
            "key_census_sha256": _key_census(header),
        },
        "overlay": {
            "file": os.path.basename(overlay_path),
            "variant": overlay_variant,
            "size": _file_size(overlay_path),
            "key_census_sha256": _key_census(overlay_header),
        },
        "geometry": geometry_of(header),
        "num_blocks": int(num_blocks),
        "quant_format": quant_format,
        "quant_metadata_sha256": _sha256(
            _canonical(_quant_metadata_payload(metadata, side="base"))),
        "header_source": HEADER_SOURCE,
    }
    return f"h3hybrid{DIGEST_VERSION}:{_sha256(_canonical(payload))}"
