"""Refuse a weight-only quantized checkpoint at a loader that cannot read one.

WHY THIS EXISTS
---------------
Several DiT loaders call ``load_state_dict(..., strict=False)`` because their
checkpoints legitimately carry extra or missing sections (embedded VAE/TE
blocks, tied weights, a base module that is then overridden). That tolerance is
correct for those cases and catastrophic for one specific input: hand such a
loader a checkpoint produced by ``subapps/fp8_quantize/quantize_transformer_fp8.py``
or by ``POST /models/export-quantized`` and

  * every ``.weight_scale`` lands in ``unexpected_keys`` (the module has no such
    buffer), and
  * every quantized ``.weight`` is an int8 / float8 tensor being copied into a
    bf16 parameter -- which ``load_state_dict`` performs as a DTYPE CAST, so the
    int8 CODES (-127..127) or the e4m3 values are written as if they were the
    real weights, without their per-row scale.

The load then "succeeds", one warning line scrolls past, and a silently wrong
model generates noise. The archs that DO support these files
(``int8_runtime_quantize.QUANTIZED_LINEAR_ARCHS``) detect the layout first and
swap in ``Int8Linear`` / ``Fp8Linear`` before loading; every other arch must
fail loudly instead.

This is deliberately a check on the STATE DICT, not on the file name or its
metadata: the offline tool's output is named by the user, and a shard index
carries no format flag the reader must trust.

THE SECOND HALF: DECLARED SEMANTICS
-----------------------------------
Everything above answers "are these tensors codes rather than weights". A
LAYOUT-compatible file can still be numerically incompatible, and the guards
above cannot see it, because the incompatibility is declared in a SIDECAR
tensor rather than in a dtype. Comfy-Org's ``int8_tensorwise`` distribution
carries, per quantized Linear:

    <layer>.weight        int8    [out, in]
    <layer>.weight_scale  float32 [out, 1]
    <layer>.comfy_quant   uint8   [N]   -> UTF-8 JSON

and the JSON may read
``{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}``.
``convrot`` means the stored codes quantize ``W @ H^T``, a Hadamard-rotated
weight, not ``W``. ``int8 * weight_scale`` therefore reconstructs the ROTATED
weight, and ``F.linear`` computes ``x H W^T`` -- the input silently passes
through an orthogonal mixing. Not degraded: wrong. Correct inference needs the
activation rotated too (or the weight un-rotated at load).

Nothing in the layout guards notices: ``.comfy_quant`` ends in neither
``.weight`` nor ``.weight_scale``, so the census ignores it; the swap helpers
gate on "scale sibling present AND weight is int8", which such a file
satisfies; and ``verify_quantized_swap``'s three counts then agree. The ONLY
thing that stops the load today is incidental -- ``Int8Linear`` registers
``weight_scale`` as ``(out,)`` and the file stores ``[out, 1]``, so
``load_state_dict`` raises a size mismatch. Squeezing that scale, the obvious
"fix", turns every guard green on a rotated model.

So the semantic refusal below runs FIRST, ahead of the census and ahead of any
shape adaptation, and a marker whose meaning this build does not implement is
refused rather than ignored. The same trap exists for NVFP4/AWQ files, whose
``.pre_quant_scale`` ``[in_features]`` vectors ComfyUI applies to the INPUT at
runtime; ignoring those is equally wrong, so they are refused here too.

SUPPORT IS DELIBERATELY NOT IMPLEMENTED HERE. The un-rotation is understood --
the normalized regular Hadamard Comfy builds from ``convrot_groupsize`` is
symmetric AND involutory (``H @ H == I``), so applying the same block rotation
a second time recovers the original basis, which is exactly what
comfy-kitchen's own ``dequantize_int8_convrot_weight`` does. An implementation
would live in ``Int8Linear``'s dequant forward (carry ``convrot_groupsize``,
rotate ``codes * scale`` by a cached ``[gs, gs]`` constant) and cost
``out*in*gs`` FLOPs per forward. It is NOT written, and until it is, this guard
is the whole of this repo's convrot handling. Evidence and the derivation:
``scratchpad/minimax_h3_weight_formats.md``.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import torch

__all__ = [
    "COMFY_QUANT_MARKER_SUFFIX",
    "FLOAT8_WEIGHT_DTYPES",
    "INT_WEIGHT_DTYPES",
    "KNOWN_COMFY_QUANT_FIELDS",
    "KNOWN_COMFY_QUANT_FORMATS",
    "PRE_QUANT_SCALE_SUFFIX",
    "QUANT_SCALE_SUFFIX",
    "QUANT_WEIGHT_DTYPES",
    "UnsupportedQuantSemanticsError",
    "cast_float8_tensors",
    "decode_comfy_quant_marker",
    "comfy_quant_markers",
    "unsupported_quant_semantics_report",
    "refuse_unsupported_quant_semantics",
    "quantized_state_dict_report",
    "refuse_quantized_state_dict",
    "scaled_quantization_report",
    "verify_quantized_swap",
]

# Both weight-only formats in this repo use the same scale suffix; only the
# weight dtype tells them apart (see ``int8_linear.INT8_SCALE_SUFFIX`` and
# ``fp8_linear.FP8_SCALE_SUFFIX``, which are both ".weight_scale").
QUANT_SCALE_SUFFIX = ".weight_scale"

_FLOAT8_WEIGHT_DTYPES = tuple(
    d for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    ) if d is not None
)

# int8 / uint8 weights are CODES: without their scale they are not an
# approximation of the weight, they are a different number entirely. Tracked
# apart from float8 because a scale-less float8 file IS readable (see
# ``scaled_quantization_report``) and a scale-less integer one is not.
_INT_WEIGHT_DTYPES = tuple(
    d for d in (torch.int8, getattr(torch, "uint8", None)) if d is not None
)

_QUANT_WEIGHT_DTYPES = _INT_WEIGHT_DTYPES + _FLOAT8_WEIGHT_DTYPES

# Public spellings. ``FLOAT8_WEIGHT_DTYPES`` is for the one loader that must
# cast a pure-cast checkpoint itself (see ``cast_float8_tensors``);
# ``QUANT_WEIGHT_DTYPES`` is what the OFFLINE TOOL's already-quantized refusal
# tests its source's header dtypes against, so the tool and this guard cannot
# disagree about what "already quantized" means.
FLOAT8_WEIGHT_DTYPES = _FLOAT8_WEIGHT_DTYPES
INT_WEIGHT_DTYPES = _INT_WEIGHT_DTYPES
QUANT_WEIGHT_DTYPES = _QUANT_WEIGHT_DTYPES


# ---------------------------------------------------------------------------
# Declared quantization semantics (Comfy-Org markers)
# ---------------------------------------------------------------------------

# The sidecar tensor Comfy-Org writes next to a quantized Linear. uint8, holding
# UTF-8 JSON. Ends in neither ".weight" nor QUANT_SCALE_SUFFIX, which is exactly
# why the census below cannot see it.
COMFY_QUANT_MARKER_SUFFIX = ".comfy_quant"

# The AWQ smoothing vector of an NVFP4/AWQ file: bf16 [in_features], applied to
# the layer's INPUT at runtime (ComfyUI ``comfy/ops.py``:
# ``input = input * pre_quant_scale``). A reader that installs the weights and
# ignores this is as wrong as one that ignores a rotation, so its mere presence
# is refused -- there is nothing to decode and nothing to be tolerant about.
PRE_QUANT_SCALE_SUFFIX = ".pre_quant_scale"

# Marker ``format`` strings whose weight LAYOUT this repo's Int8Linear/Fp8Linear
# can express (per-output-row codes + scale). Membership here is not a promise
# that the file loads -- Comfy stores the scale as [out, 1] and per-tensor
# scalars, which our modules still reject on shape -- only that its declared
# format is not, by itself, a numerically different contract.
KNOWN_COMFY_QUANT_FORMATS = frozenset({
    "int8_tensorwise",
    "float8_e4m3fn",
    "float8_e5m2",
})

# Marker fields whose meaning has been READ, from comfy-kitchen's quantizer and
# Comfy-Org's format spec. Anything else is a semantic this build has never
# looked at, and silently ignoring an unread field is the whole failure mode
# this module exists to prevent -- so an unknown field is refused, not skipped.
# ``convrot`` is listed as known and refused when TRUE: knowing what it means is
# precisely why it cannot be ignored.
KNOWN_COMFY_QUANT_FIELDS = frozenset({
    "format",
    "convrot",
    "convrot_groupsize",
    "full_precision_matrix_mult",
})


class UnsupportedQuantSemanticsError(RuntimeError):
    """The checkpoint declares a quantization contract this build does not implement.

    A ``RuntimeError`` subclass so every existing ``except RuntimeError`` around a
    load still catches it, and a distinct type so a caller that wants to tell
    "unreadable layout" from "unimplemented semantics" can.
    """


def decode_comfy_quant_marker(tensor: "torch.Tensor") -> Optional[Dict[str, object]]:
    """Parse one ``.comfy_quant`` marker tensor; ``None`` if it cannot be read.

    The marker is a 1-D uint8 tensor of UTF-8 JSON bytes. ``None`` means
    UNKNOWN, never "absent": a marker that is present but garbled (truncated
    shard, zero-element header proxy, a future non-JSON encoding) is positive
    evidence that the file declares something, and the caller treats it as
    unsupported for that reason.
    """
    try:
        if tensor.numel() == 0:
            return None
        raw = bytes(tensor.detach().to(torch.uint8).reshape(-1).cpu().tolist())
        parsed = json.loads(raw.decode("utf-8").rstrip("\x00"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def comfy_quant_markers(
    state_dict: Dict[str, "torch.Tensor"],
) -> Dict[str, Optional[Dict[str, object]]]:
    """``{layer path: parsed marker or None}`` for every ``.comfy_quant`` key.

    The key is the layer stem (the marker suffix stripped), so it lines up with
    the ``.weight`` / ``.weight_scale`` names the rest of this module speaks.
    A ``None`` value is an undecodable marker -- see
    ``decode_comfy_quant_marker``.
    """
    out: Dict[str, Optional[Dict[str, object]]] = {}
    for key, value in (state_dict or {}).items():
        if key.endswith(COMFY_QUANT_MARKER_SUFFIX):
            out[key[: -len(COMFY_QUANT_MARKER_SUFFIX)]] = decode_comfy_quant_marker(value)
    return out


def unsupported_quant_semantics_report(
    state_dict: Dict[str, "torch.Tensor"],
) -> Optional[Dict[str, object]]:
    """``None`` when the declared semantics are implementable here; else a census.

    Pure inspection, no raising -- ``refuse_unsupported_quant_semantics`` turns
    the census into the message. Four independent findings, any one sufficient:

    * ``convrot_layers``      -- a marker declaring ``convrot: true``. The codes
      quantize ``W @ H^T``; reconstructing with the scales alone gives a rotated
      weight (see the module docstring).
    * ``unknown_format_layers`` -- a marker whose ``format`` is not in
      ``KNOWN_COMFY_QUANT_FORMATS``.
    * ``unknown_field_layers`` -- a marker carrying a field this build has never
      read. Ignoring an unread field is the failure mode itself.
    * ``undecodable_markers`` -- a marker present but not parseable.
    * ``pre_quant_scale_keys`` -- AWQ input smoothing vectors, applied to the
      INPUT at runtime by the writer's own runtime.
    """
    if not state_dict:
        return None
    markers = comfy_quant_markers(state_dict)
    pre_quant = [k for k in state_dict if k.endswith(PRE_QUANT_SCALE_SUFFIX)]
    if not markers and not pre_quant:
        return None

    convrot: List[str] = []
    unknown_format: List[str] = []
    unknown_field: List[str] = []
    undecodable: List[str] = []
    formats: set = set()
    groupsizes: set = set()
    fields: set = set()
    for layer, marker in sorted(markers.items()):
        if marker is None:
            undecodable.append(layer)
            continue
        declared = marker.get("format")
        if isinstance(declared, str):
            formats.add(declared)
        if declared not in KNOWN_COMFY_QUANT_FORMATS:
            unknown_format.append(layer)
        extra = set(marker) - KNOWN_COMFY_QUANT_FIELDS
        if extra:
            unknown_field.append(layer)
            fields |= extra
        if marker.get("convrot"):
            convrot.append(layer)
            gs = marker.get("convrot_groupsize")
            if isinstance(gs, int):
                groupsizes.add(gs)

    if not (convrot or unknown_format or unknown_field or undecodable or pre_quant):
        # Markers exist, but every one declares a format and a field set this
        # build implements and none asks for a rotation: an ordinary per-row
        # scaled file that merely carries provenance. Let the layout guards
        # decide it, exactly as they would with no marker at all.
        return None

    return {
        "marker_keys": len(markers),
        "convrot_layers": convrot,
        "unknown_format_layers": unknown_format,
        "unknown_field_layers": unknown_field,
        "undecodable_markers": undecodable,
        "pre_quant_scale_keys": pre_quant,
        "declared_formats": sorted(formats),
        "convrot_groupsizes": sorted(groupsizes),
        "unknown_fields": sorted(str(f) for f in fields),
    }


def refuse_unsupported_quant_semantics(
    state_dict: Dict[str, "torch.Tensor"],
    *,
    arch: Optional[str] = None,
    path: Optional[str] = None,
    label: str = "transformer",
) -> None:
    """Raise ``UnsupportedQuantSemanticsError`` on an unimplementable declaration.

    Silent on every checkpoint that carries no ``.comfy_quant`` marker and no
    ``.pre_quant_scale``, i.e. on everything this repo writes and on every file
    it reads today, so it is safe to call unconditionally and EARLY. It is
    called from ``quantized_state_dict_report`` (the census every supporting
    loader runs before it swaps) and from the int8/fp8 detectors and swap
    entry points themselves, so no ordering change and no scale-shape
    adaptation can get in front of it.
    """
    report = unsupported_quant_semantics_report(state_dict)
    if report is None:
        return

    where = f" ({path})" if path else ""
    what = f"the {arch} {label}" if arch else f"the {label}"
    reasons: List[str] = []
    if report["convrot_layers"]:
        layers = report["convrot_layers"]
        gs = report["convrot_groupsizes"]
        reasons.append(
            f"{len(layers)} layer(s) declare \"convrot\": true"
            + (f" (groupsize {', '.join(str(g) for g in gs)})" if gs else "")
            + f" -- e.g. {', '.join(layers[:3])}. Their int8 codes quantize the "
            f"HADAMARD-ROTATED weight W @ H^T, not W, so dequantizing with "
            f"'{QUANT_SCALE_SUFFIX}' alone reconstructs a rotated weight and the "
            f"forward silently mixes the input through an orthogonal matrix. That is "
            f"a wrong model, not a degraded one, and it would load without a warning"
        )
    if report["unknown_format_layers"]:
        layers = report["unknown_format_layers"]
        fmts = report["declared_formats"] or ["<none>"]
        reasons.append(
            f"{len(layers)} layer(s) declare a quantization format this build does not "
            f"implement ({', '.join(fmts)}) -- e.g. {', '.join(layers[:3])}"
        )
    if report["unknown_field_layers"]:
        layers = report["unknown_field_layers"]
        reasons.append(
            f"{len(layers)} marker(s) carry field(s) whose meaning this build has never "
            f"read ({', '.join(report['unknown_fields'])}) -- e.g. "
            f"{', '.join(layers[:3])}; an unread field may declare another weight "
            f"transform, and ignoring it is precisely the failure being guarded"
        )
    if report["undecodable_markers"]:
        layers = report["undecodable_markers"]
        reasons.append(
            f"{len(layers)} '{COMFY_QUANT_MARKER_SUFFIX}' marker(s) are present but could "
            f"not be decoded as UTF-8 JSON -- e.g. {', '.join(layers[:3])}; a marker that "
            f"exists declares something, so an unreadable one is treated as unsupported "
            f"rather than as absent"
        )
    if report["pre_quant_scale_keys"]:
        keys = report["pre_quant_scale_keys"]
        reasons.append(
            f"{len(keys)} '{PRE_QUANT_SCALE_SUFFIX}' tensor(s) are present -- e.g. "
            f"{', '.join(keys[:3])}. Those are AWQ input-smoothing vectors that the "
            f"writer's runtime multiplies into the layer's INPUT; installing the weights "
            f"without them is as wrong as ignoring a rotation"
        )

    raise UnsupportedQuantSemanticsError(
        f"{what} checkpoint{where} declares weight-only quantization SEMANTICS this "
        f"build does not implement: " + "; ".join(reasons) + ". "
        f"What IS supported: per-output-row scaled int8/e4m3 weights with a "
        f"'{QUANT_SCALE_SUFFIX}' sibling and NO '{COMFY_QUANT_MARKER_SUFFIX}' marker and "
        f"NO '{PRE_QUANT_SCALE_SUFFIX}' tensor -- the layout that "
        f"subapps/fp8_quantize/quantize_transformer_fp8.py and "
        f"POST /models/export-quantized produce. Refusing rather than continuing: every "
        f"other guard in this module would pass this file, because its LAYOUT is the "
        f"supported one and only its meaning differs. Load an unquantized checkpoint (or "
        f"one quantized by this repo) instead."
    )


def quantized_state_dict_report(
    state_dict: Dict[str, "torch.Tensor"],
    *,
    arch: Optional[str] = None,
    path: Optional[str] = None,
    label: str = "transformer",
) -> Optional[Dict[str, object]]:
    """``None`` for an ordinary checkpoint; a report dict for a quantized one.

    Two independent pieces of evidence, either of which is sufficient:

    * a ``.weight_scale`` key -- the sibling that gates both swap helpers;
    * a ``.weight`` whose dtype is int8 / float8 / uint8. Kept as a separate
      test so a file whose scales were stripped, or a future layout that names
      them differently, is still caught. Restricted to ``.weight`` keys so an
      ordinary integer BUFFER (a mask, an index table) cannot trip it.

    The report carries ``scale_keys`` / ``quantized_weight_keys`` counts, the
    float8-versus-integer split of the latter (``float8_weight_keys`` /
    ``int_weight_keys``, which ``scaled_quantization_report`` needs to tell a
    plain dtype cast from a file whose scales are missing), and a few example
    key names for the message.

    RAISES before it counts anything if the state dict declares quantization
    SEMANTICS this build does not implement (``.comfy_quant`` markers,
    ``.pre_quant_scale`` vectors). That check has to be first: such a file is
    LAYOUT-compatible, so every count below, and every guard keyed off them,
    agrees with it. ``arch``/``path``/``label`` are optional and feed only that
    message.
    """
    refuse_unsupported_quant_semantics(state_dict, arch=arch, path=path, label=label)
    if not state_dict:
        return None
    scale_keys: List[str] = []
    quant_weights: List[str] = []
    n_float8 = 0
    n_int = 0
    for key, value in state_dict.items():
        if key.endswith(QUANT_SCALE_SUFFIX):
            scale_keys.append(key)
            continue
        if not key.endswith(".weight"):
            continue
        dtype = getattr(value, "dtype", None)
        if dtype in _FLOAT8_WEIGHT_DTYPES:
            quant_weights.append(key)
            n_float8 += 1
        elif dtype in _INT_WEIGHT_DTYPES:
            quant_weights.append(key)
            n_int += 1
    if not scale_keys and not quant_weights:
        return None
    return {
        "scale_keys": len(scale_keys),
        "quantized_weight_keys": len(quant_weights),
        "float8_weight_keys": n_float8,
        "int_weight_keys": n_int,
        "examples": (scale_keys[:3] + quant_weights[:3])[:5],
    }


def scaled_quantization_report(
    report: Optional[Dict[str, object]],
    *,
    arch: str,
    path: Optional[str] = None,
    label: str = "transformer",
) -> Optional[Dict[str, object]]:
    """``report``, unless it describes a plain FLOAT8 DTYPE CAST -- then ``None``.

    ``quantized_state_dict_report`` answers "does this file contain anything
    quantized-looking"; a LOADER needs the narrower question "is this file
    weight-only quantized in the SCALED sense, i.e. does reading it require the
    Int8Linear / Fp8Linear swap". The two differ on exactly one input, and it is
    a common one:

        a pure cast -- every ``.weight`` stored as e4m3 (or e5m2) with NO
        ``.weight_scale`` anywhere. This is the dominant ComfyUI "fp8" community
        distribution shape, and it is not a quantization format at all: it is the
        bf16 model with its weights rounded to 8-bit floats, meant to be read by
        casting them back. Every loader here does exactly that already
        (``model.to(bf16)`` + ``load_state_dict``, which performs the cast), and
        e4m3's range and 3-bit mantissa sit wholly inside bf16, so the cast back
        is exact and the forward is the one the file's author intended.

    Refusing it would misdiagnose a legitimate file as one whose scales were
    stripped. So the refusal is kept for the cases where there IS positive
    evidence of scaled quantization, or where the cast interpretation is
    impossible:

    * ``scale_keys > 0`` -- something wrote per-row scales, so the weights are
      codes; a swap must cover them (``verify_quantized_swap``);
    * ``int_weight_keys > 0`` -- int8/uint8 weights are codes in -127..127 (or a
      bitsandbytes 4-bit pack) with no meaning as numbers. Nothing legitimately
      distributes those without their scales, and casting them into a bf16
      parameter is the 103020%-error failure this module exists to prevent.

    Call it immediately after ``quantized_state_dict_report`` and use the result
    everywhere the loader branches on "is this a quantized checkpoint": the pure
    cast then takes the ordinary path, byte-for-byte as it did before any of
    these guards existed.
    """
    if report is None:
        return None
    if int(report.get("scale_keys", 0) or 0):
        return report
    if int(report.get("int_weight_keys", 0) or 0):
        return report
    n_float8 = int(report.get("float8_weight_keys", 0) or 0)
    if not n_float8:
        return report
    where = f" ({path})" if path else ""
    print(f"[QuantGuard] the {arch} {label} checkpoint{where} stores {n_float8} "
          f"'.weight' tensor(s) as float8 with no '{QUANT_SCALE_SUFFIX}' sibling: a plain "
          f"dtype cast, not a scaled weight-only quantization. Loading it normally; the "
          f"cast back to the compute dtype is exact.")
    return None


def verify_quantized_swap(
    report: Optional[Dict[str, object]],
    swapped: int,
    *,
    arch: str,
    path: Optional[str] = None,
    label: str = "transformer",
) -> None:
    """Raise unless the quantized-Linear swap covered the WHOLE quantized file.

    Call it on a loader that DOES support these checkpoints, immediately after
    the swap and before the ``strict=False`` load. ``report`` is what
    ``quantized_state_dict_report`` returned for the same state dict and
    ``swapped`` is the count the swap helper(s) returned. ``report is None``
    (an ordinary checkpoint) is a no-op.

    WHY AN EQUALITY AND NOT "swapped > 0"
    ------------------------------------
    ``quantized_state_dict_report`` fires on EITHER piece of evidence -- a
    ``.weight_scale`` key OR an int8/float8 ``.weight`` -- while both swap
    helpers require BOTH (the scale sibling AND the weight dtype, so a mixed
    int8/e4m3 file cannot have one format claim the other's layers). Anything
    the report saw and the swap did not take is a layer that reaches
    ``load_state_dict`` as a plain ``nn.Linear``: its scale lands in
    ``unexpected_keys`` (a print) and its quantized codes are CAST into a bf16
    parameter (silently, because a dtype cast is what ``load_state_dict``
    does). That is the exact silently-wrong model this module exists to
    prevent, so the three counts must agree exactly:

    * a file whose scales were stripped by a foreign tool, or a shard set
      missing its scale-bearing shard, reports ``quantized_weight_keys > 0``
      with ``scale_keys == 0`` and swaps NOTHING;
    * a file whose module paths do not match the model the loader built (a
      config/artifact mismatch) reports both counts > 0 and swaps FEWER.

    Both loaded silently wrong before this check existed.

    WHAT THIS MUST NOT REFUSE. A file with NO scales at all whose quantized
    weights are all float8 is a plain dtype cast, which every one of these
    loaders reads correctly by casting back (the ComfyUI "fp8" distribution
    shape). ``scaled_quantization_report`` above filters that case out, and
    every caller runs the report through it, so a ``report`` reaching here with
    ``scale_keys == 0`` means integer codes with no scales -- unreadable.
    """
    if report is None:
        return
    scale_keys = int(report.get("scale_keys", 0) or 0)
    weight_keys = int(report.get("quantized_weight_keys", 0) or 0)
    if swapped == scale_keys == weight_keys:
        return
    where = f" ({path})" if path else ""
    examples = ", ".join(str(e) for e in (report.get("examples") or [])) or "none"
    if scale_keys != weight_keys:
        int_keys = int(report.get("int_weight_keys", 0) or 0)
        kind = "int8/uint8" if int_keys == weight_keys else "int8/float8"
        diagnosis = (
            f"the file carries {weight_keys} {kind} '.weight' tensor(s) but "
            f"{scale_keys} '{QUANT_SCALE_SUFFIX}' sibling(s) -- every quantized weight "
            f"needs its per-row scale, so a scale-less (or partially scale-less) file "
            f"cannot be read back. Producing it again with "
            f"subapps/fp8_quantize/quantize_transformer_fp8.py, or supplying the shard "
            f"that holds the scales, is the fix"
        )
    else:
        diagnosis = (
            f"the file attests {scale_keys} quantized Linear(s) but only {swapped} "
            f"matching module(s) exist in the {arch} {label} this loader built -- the "
            f"checkpoint's module paths and the model's geometry/config disagree"
        )
    raise RuntimeError(
        f"the {arch} {label} checkpoint{where} is weight-only QUANTIZED, and "
        f"{swapped} of its quantized Linear(s) could be swapped in "
        f"(scales={scale_keys}, quantized weights={weight_keys}, swapped={swapped}; "
        f"e.g. {examples}). {diagnosis}. Refusing rather than continuing: the "
        f"unswapped layers would reach load_state_dict as plain nn.Linear, dropping "
        f"any scales they do have as unexpected keys and casting their quantized "
        f"codes into bf16 parameters -- a model that loads without a warning and "
        f"generates noise."
    )


def cast_float8_tensors(
    state_dict: Dict[str, "torch.Tensor"], dtype: "torch.dtype",
) -> Dict[str, "torch.Tensor"]:
    """A copy of ``state_dict`` with every float8 tensor cast to ``dtype``.

    ONLY for a loader that installs the checkpoint's tensors with
    ``load_state_dict(..., assign=True)`` (Anima, whose module is built on the
    meta device, so assignment is the only option). A plain
    ``load_state_dict`` CASTS into the existing parameter and needs nothing from
    this; ``assign=True`` would instead leave float8 parameters behind, which no
    ``nn.Linear`` forward can multiply.

    The input is not mutated -- it may be the caller's own dict -- and only the
    float8 entries are copied, so the transient cost is the float8 half of the
    checkpoint expanded once, not the whole file twice.
    """
    return {
        key: (value.to(dtype)
              if getattr(value, "dtype", None) in _FLOAT8_WEIGHT_DTYPES else value)
        for key, value in state_dict.items()
    }


def refuse_quantized_state_dict(
    state_dict: Dict[str, "torch.Tensor"],
    *,
    arch: str,
    path: Optional[str] = None,
    label: str = "transformer",
) -> None:
    """Raise ``RuntimeError`` when ``state_dict`` is weight-only quantized.

    Call it immediately before a ``strict=False`` load on an architecture that
    has no quantized-Linear swap at all (Lens, MiniT2I -- neither is in
    ``int8_runtime_quantize.QUANTIZED_LINEAR_ARCHS`` and neither loader has a
    swap to verify). Silent on every ordinary checkpoint, so it is safe to
    call unconditionally.

    Runs the census through ``scaled_quantization_report`` before deciding
    whether to raise, same as every swap-capable loader does: a checkpoint
    whose float8 ``.weight`` tensors carry no ``.weight_scale`` sibling at all
    is a plain dtype cast (the ComfyUI "fp8" distribution shape), and the
    ``strict=False`` load these callers already do reads it correctly by
    casting back -- refusing it would misdiagnose a legitimate file as one
    whose scales were stripped. What remains refused here is exactly what
    these loaders genuinely cannot read: a file with positive evidence of
    SCALED quantization (a ``.weight_scale`` key), or scale-less int8/uint8
    weights (codes with no meaning as numbers, unlike a float8 cast).
    """
    report = scaled_quantization_report(
        quantized_state_dict_report(state_dict, arch=arch, path=path, label=label),
        arch=arch, path=path, label=label)
    if report is None:
        return
    try:
        from core.models.common.int8_runtime_quantize import (
            QUANTIZED_LINEAR_ARCHS, arch_names,
        )
        supported = arch_names(QUANTIZED_LINEAR_ARCHS)
    except Exception:  # pragma: no cover - defensive
        supported = "the architectures whose loaders swap in quantized Linear layers"
    where = f" ({path})" if path else ""
    raise RuntimeError(
        f"the {arch} {label} checkpoint{where} is weight-only QUANTIZED "
        f"({report['scale_keys']} '{QUANT_SCALE_SUFFIX}' key(s), "
        f"{report['quantized_weight_keys']} int8/float8 weight(s); e.g. "
        f"{', '.join(report['examples'])}), and the {arch} loader does not support "
        f"quantized checkpoints. Loading it would drop every scale as an unexpected "
        f"key and cast the quantized codes into bf16 parameters, producing a "
        f"silently wrong model. Weight-only quantized checkpoints are readable only "
        f"on {supported}; load an unquantized {arch} checkpoint instead."
    )
