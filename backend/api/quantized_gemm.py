"""Per-generation selection of the quantized-GEMM path (``quantized_gemm_mode``).

Two process-level flags govern how ALREADY-quantized Linear weights are
multiplied on the three architectures whose loaders swap in the quantized
Linear classes (Ideogram 4, Krea 2, Anima):

* ``core.models.ideogram4.vendor.fp8_linear`` -- W8A8 ``torch._scaled_mm`` vs a
  dequantized matmul, for e4m3 weights.
* ``core.models.ideogram4.vendor.int8_linear`` -- W8A8 ``torch._int_mm`` vs a
  dequantized matmul, for int8 weights.

``quantized_gemm_mode`` exposes them as ONE per-generation parameter with three
values:

``None``
    Do not touch process state. This is the default and it must stay the
    default: writing ``False``/``"dequant"`` here would override an env-var
    opt-in (``SUSHI_FP8_SCALED_MM=1`` / ``SUSHI_INT8_MM=1``, reported as
    ``origin: "env"``) for every existing raw-API caller that never sends the
    field.
``"w8a8"``
    Force BOTH flags on for this generation.
``"dequant"``
    Force BOTH flags off for this generation.

One axis rather than two booleans: whether a checkpoint's quantized layers are
FP8 or INT8 is decided by the checkpoint format, not by the caller (Ideogram 4
is FP8/nf4, Krea 2 is FP8 or INT8, Anima is INT8, and the offline int8 tool
emits MIXED files). Two booleans would make "int8 on an fp8 checkpoint"
representable and meaningless; one axis makes it unrepresentable.

WHY NOT CALL THE ``/system/*`` ENDPOINTS. ``apply_quantized_gemm_mode`` calls
``set_scaled_mm_enabled`` / ``set_int8_mm_enabled`` DIRECTLY. Routing through
``POST /system/fp8-scaled-mm`` from inside a generation handler cannot work:
by that point ``generation_status`` is already ``running`` and
``_fp8_scaled_mm_busy_reason`` -- which exists to stop a MANUAL flip landing
mid-run -- would return 409 against the very generation whose value it is. The
``/system/*`` endpoints, their 409 and that fail-closed busy check are
deliberately left unchanged; they still guard manual flips.

Requesting ``"w8a8"`` on a model with no quantized layers (SDXL, a bf16 Krea 2
checkpoint, ...) is a NO-OP, not an error: the flags are process-wide and
harmless when nothing consumes them. It is reported through the generation's
``warnings[]`` channel instead (``report_quantized_gemm_outcome``), because
silent degradation is exactly how this feature became hard to find.
"""
from typing import Any, Dict, Optional

from core.models.common.int8_runtime_quantize import QUANTIZED_LINEAR_ARCHS

# The architectures whose loaders swap in Fp8Linear / Int8Linear, i.e. the only
# ones where this parameter can change what runs. IMPORTED, not restated: the
# tuple lives with the quantization selection rule in
# ``core.models.common.int8_runtime_quantize`` so that adding an architecture
# there cannot leave this module (or the ``quantized_gemm`` entries in
# ``arch_capabilities.py``, or ``generation_utils.extract_fp8_gemm_info``'s
# component map) naming a stale set.
QUANTIZED_GEMM_ARCHS = QUANTIZED_LINEAR_ARCHS

VALID_MODES = ("w8a8", "dequant")

# ``origin`` recorded on the process flags when a generation forces them. A
# distinct value from "api" so `GET /system/*` can say a generation moved it.
GENERATION_ORIGIN = "generation"


def normalize_mode(mode: Any) -> Optional[str]:
    """Coerce a request value to ``None`` / ``"w8a8"`` / ``"dequant"``.

    Multipart ``Form()`` fields arrive as strings, so the empty string and the
    literal ``"none"``/``"null"`` (what a frontend select can send for its
    "Default" option) mean "not set" and map to ``None`` -- the tier that must
    not touch process state. Anything else that is not a valid mode raises
    ``ValueError``; the caller turns that into a 400.
    """
    if mode is None:
        return None
    if isinstance(mode, str):
        stripped = mode.strip().lower()
        if stripped in ("", "none", "null", "default"):
            return None
        if stripped in VALID_MODES:
            return stripped
    raise ValueError(
        f"quantized_gemm_mode must be null, 'w8a8' or 'dequant', got {mode!r}"
    )


def apply_quantized_gemm_mode(mode: Optional[str]) -> Optional[Dict[str, Any]]:
    """Force the process quantized-GEMM flags for this generation, or no-op.

    Returns ``None`` when ``mode`` is ``None`` (NOTHING is read, imported or
    written in that case, so an env-var opt-in survives untouched), otherwise a
    dict with the resulting ``fp8``/``int8`` state dicts for logging.

    Best-effort on the vendor imports: a build where those modules cannot be
    imported has no quantized Linear layers to govern either, so a failure here
    must not fail the generation.

    Call this BEFORE the generation's first forward. Both setters clear their
    probe caches, so the next quantized Linear forward re-probes under the new
    setting.
    """
    if mode is None:
        return None
    if mode not in VALID_MODES:
        raise ValueError(
            f"quantized_gemm_mode must be null, 'w8a8' or 'dequant', got {mode!r}"
        )
    enabled = mode == "w8a8"
    result: Dict[str, Any] = {}
    try:
        from core.models.ideogram4.vendor.fp8_linear import set_scaled_mm_enabled

        result["fp8"] = set_scaled_mm_enabled(enabled, origin=GENERATION_ORIGIN)
    except Exception as exc:  # pragma: no cover - import/environment failure
        print(f"[QuantizedGemm] Could not set the FP8 GEMM path: {exc}")
    try:
        from core.models.ideogram4.vendor.int8_linear import set_int8_mm_enabled

        result["int8"] = set_int8_mm_enabled(enabled, origin=GENERATION_ORIGIN)
    except Exception as exc:  # pragma: no cover - import/environment failure
        print(f"[QuantizedGemm] Could not set the INT8 GEMM path: {exc}")
    return result


def _gemm_flags_enabled() -> Dict[str, Optional[bool]]:
    """Current process state of the two W8A8 flags (``None`` when unreadable).

    Read back rather than assumed: ``apply_quantized_gemm_mode`` sets both flags
    best-effort, so a build where a vendor module cannot be imported leaves the
    flag where it was, and the resulting dequant has a THIRD cause that is
    neither the hardware nor a policy pin.
    """
    state: Dict[str, Optional[bool]] = {"fp8": None, "int8": None}
    try:
        from core.models.ideogram4.vendor.fp8_linear import get_scaled_mm_state

        state["fp8"] = bool(get_scaled_mm_state().get("enabled"))
    except Exception:
        pass
    try:
        from core.models.ideogram4.vendor.int8_linear import get_int8_mm_state

        state["int8"] = bool(get_int8_mm_state().get("enabled"))
    except Exception:
        pass
    return state


def _dequant_cause(label: str, arch: Optional[str]) -> str:
    """Explain WHY a ``w8a8`` request resolved to the dequantized matmul.

    "The W8A8 path is unavailable on this device/build" -- what this used to say
    for every case -- is false for an architecture whose LOADER pins its layers
    to the dequant path, and it points the reader at a GPU upgrade that would
    change nothing. MiniMax-H3 is exactly that case: `disable_scaled_mm` is
    called over the whole DiT at load time because 50 of the checkpoint's 200
    quantized tensors are marked `full_precision_matrix_mult` and the other 150
    carry an `input_scale` this repo's `Fp8Linear` does not read, and that pin
    outranks both the env flag and this request.

    The cause is DERIVED from the resolved label plus the process flag, not from
    an arch list, because `describe_gemm_path` already separates them:

    * a BARE ``dequant`` / ``int8_dequant`` stem means "the flag is off, or
      every owned layer opted out" -- so with the flag confirmed ON it can only
      be the per-module opt-out, i.e. the loader's policy pin;
    * ``(scaled_mm unavailable)`` / ``(int_mm unavailable)`` means the
      per-device probe rejected every scaling mode -- the genuine
      device/build limitation;
    * ``(... unprobed)`` means no quantized forward reached the probe at all.
    """
    flags = _gemm_flags_enabled()
    reasons = []
    for stem, fmt, flag_key, kernel in (
        ("int8_dequant", "INT8", "int8", "torch._int_mm"),
        ("dequant", "FP8", "fp8", "torch._scaled_mm"),
    ):
        parts = [p for p in label.split("+") if p.startswith(stem)]
        if not parts:
            continue
        part = parts[0]
        enabled = flags.get(flag_key)
        if "unavailable" in part:
            reasons.append(
                f"The {fmt} W8A8 path is unavailable on this device/build: the "
                f"per-device probe rejected every {kernel} scaling mode for these layers."
            )
        elif "unprobed" in part:
            reasons.append(
                f"The {fmt} W8A8 path was enabled but no quantized Linear forward "
                "reached the probe in this process, so no layer ran it."
            )
        elif enabled is False:
            reasons.append(
                f"The {fmt} W8A8 process flag is off: this request could not set it "
                "(see the console for the setter failure)."
            )
        else:
            reasons.append(
                f"The {fmt} W8A8 flag is on and this is NOT a device or build "
                f"limitation: every quantized Linear layer of the loaded "
                f"'{arch}' model is pinned to the dequantized path by its loader "
                "(disable_scaled_mm), which outranks this request. That pin is a "
                "property of the checkpoint's declared quantization semantics, so "
                "a different GPU would resolve the same way."
            )
        # Only the first matching stem per format is described; `label.split("+")`
        # holds at most one stem per format by construction.
    return " ".join(reasons) if reasons else (
        "The W8A8 path did not run for these layers."
    )


def report_quantized_gemm_outcome(
    mode: Optional[str], fp8_gemm_label: str, arch: Optional[str]
) -> Optional[str]:
    """Warn when an explicit ``"w8a8"`` request did not actually run W8A8.

    Called AFTER the generation, with the label
    ``generation_utils.extract_fp8_gemm_info`` already computed for the image
    metadata -- that label records the RESOLVED path, which is the only honest
    witness (the flag can be on while the per-device probe rejected every
    scaling mode, in which case every layer ran the dequantized matmul).

    Two degradations are reported:

    * the loaded checkpoint carries no quantized Linear layers at all (empty
      label), so the request was a no-op;
    * the label resolved to a ``dequant`` stem (``dequant...`` for FP8,
      ``int8_dequant...`` for INT8) on every owned module.

    A MIXED checkpoint reports both stems joined with "+"; it counts as degraded
    only when NO stem is a ``w8a8`` one, since a file whose fp8 half ran W8A8
    did run the requested path where it applies.

    Returns the warning message, or None. Never raises.
    """
    if mode != "w8a8":
        return None
    try:
        label = (fp8_gemm_label or "").strip()
        if not label:
            if arch not in QUANTIZED_GEMM_ARCHS:
                # Handled by arch_capabilities' unsupported-parameter warning;
                # do not file a second one for the same fact.
                return None
            message = (
                "quantized_gemm_mode='w8a8' had no effect: the loaded "
                f"'{arch}' checkpoint carries no weight-only quantized Linear "
                "layers, so there is no quantized GEMM to select."
            )
        elif any(part.startswith("w8a8") for part in label.split("+")):
            return None
        else:
            message = (
                "quantized_gemm_mode='w8a8' was requested but the dequantized "
                f"matmul ran (resolved path: {label}). "
                + _dequant_cause(label, arch)
            )
        try:
            from api.generation_status import add_warning

            add_warning(message, code="quantization_fallback")
        except Exception:
            pass
        return message
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[QuantizedGemm] Could not report the resolved GEMM path: {exc}")
        return None
