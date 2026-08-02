"""Per-layer INT8 / e4m3 selection, shared by the offline tool and the runtime
converter, plus the in-place runtime converter itself.

WHY THIS MODULE EXISTS
----------------------
The decision rule for "which Linear becomes int8, which falls back to e4m3, and
which is not quantized at all" used to live only in
``subapps/fp8_quantize/quantize_transformer_fp8.py``. That was fine while the
only consumer was the offline tool, but SushiUI now also converts an ordinary
bf16 checkpoint IN PLACE at generation time (``quantize_linears_in_place``
below), and two copies of a selection rule drift. Both callers import from here;
the shared import IS the pin. A pinning test on synthetic weights lives at
``tmp/int8_runtime_selection_pin.py``.

THE RULE, in the order it is applied
------------------------------------
1. **Shape filters** (``select_targets``)
   * no weight in the checkpoint (offline only; a live module always has one)
   * ``in_features`` or ``out_features`` not a multiple of the format's GEMM
     alignment (``FORMAT_MIN_ALIGN``: 8 for int8, 16 for fp8). Such a layer can
     never reach the fast path, so quantizing it buys error for no speed.
   * (optional, int8 only) ``in_features < _MIN_WORK_K`` or
     ``out_features < _MIN_WORK_N``: the runtime min-work gate can never admit
     the layer at any ``m``, so it would always run
     ``Int8Linear._dequant_forward`` -- slower than the ``F.linear`` an
     unquantized checkpoint runs. Whether this pays is per-architecture, which
     is what ``ARCH_QUANT_POLICY`` records.
   * user-supplied exclude regexes.
2. **Per-layer format choice** (``audit_and_quantize_int8``)
   * crest pre-filter: mean per-row crest above ``crest_threshold`` -> fallback.
   * MEASURED backstop: both candidate quantizations are always performed and
     both relative RMS weight errors always measured; unless int8 is STRICTLY
     better than e4m3 the layer falls back. This, not the crest, is the actual
     decision -- the crest is the predictive explanation for it.

Both callers emit the same audit document shape (``audit_document``), so a
runtime conversion can be diffed against the committed offline artifact.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from core.models.ideogram4.vendor.fp8_linear import (
    Fp8Linear,
    quantize_weight_to_fp8,
)
from core.models.ideogram4.vendor.int8_linear import (
    Int8Linear,
    quantize_weight_to_int8,
    weight_crest_factors,
)
# The runtime min-work gate's SHAPE conditions, imported so the offline
# --skip-below-work-gate filter and the runtime converter cannot drift from what
# Int8Linear._int_mm_forward actually enforces. The third condition
# (_MIN_WORK_MKN) depends on m and therefore on the call, not the layer, so it
# has no offline equivalent.
from core.models.ideogram4.vendor.int8_linear import (
    _MIN_WORK_K as INT8_MIN_WORK_K,
    _MIN_WORK_N as INT8_MIN_WORK_N,
)

__all__ = [
    "FORMAT_MIN_ALIGN",
    "DEFAULT_CREST_THRESHOLD",
    "INT8_MIN_WORK_K",
    "INT8_MIN_WORK_N",
    "ARCH_QUANT_POLICY",
    "RUNTIME_INT8_ARCHS",
    "LoraWrappedError",
    "arch_policy",
    "linear_paths",
    "select_targets",
    "audit_and_quantize_int8",
    "audit_document",
    "already_weight_only_quantized",
    "float8_weight_linear_count",
    "quantize_linears_in_place",
]


class LoraWrappedError(RuntimeError):
    """Raised when the module carries LoRA wrappers; NOTHING was converted.

    Distinct from every other failure of ``quantize_linears_in_place`` because
    its consequence is different: the refusal happens before the first layer is
    touched, so the model is byte-identical afterwards. A caller must not report
    it with the partial-conversion message.
    """


# Per-format GEMM alignment. A layer that cannot satisfy its format's fast-path
# alignment can never reach that path, so quantizing it buys error for no speed.
FORMAT_MIN_ALIGN = {"fp8": 16, "int8": 8}

# Default crest-factor threshold above which int8 loses to e4m3. Derived, not
# tuned: int8's relative error is crest/(127*sqrt(12)) = crest/440 and e4m3's is
# flat at ~2.63e-02, so they cross at crest ~= 11.6.
#
# It is NOT true that the real checkpoint leaves a wide empty gap around 12.0 --
# the full 263-layer Krea 2 run has layers at crest 9.22, 9.43, 12.14, 12.44 and
# 32.56, i.e. two of them sit just above the threshold. What makes the placement
# safe is stronger than a gap: on that run the MEASURED backstop alone
# (``err_int8 < err_e4m3``) reproduces exactly the same 4-layer selection, with
# every chosen int8 layer at an int8-over-e4m3 error advantage >= 1.199 and every
# selected-out layer <= 0.928. The two rules agree, and the measurement -- not the
# crest -- is what actually decides.
DEFAULT_CREST_THRESHOLD = 12.0


# ---------------------------------------------------------------------------
# Per-architecture policy
# ---------------------------------------------------------------------------
#
# The knobs that differ per architecture, in ONE place, so the offline
# invocation and the runtime converter cannot disagree about an arch. Anything
# absent falls back to the format defaults.
#
#   skip_below_work_gate  int8 only. True where the arch has enough Linears
#                         under the runtime min-work gate that quantizing them
#                         is a measured loss.
#   excludes              extra module-path regexes (none today).
#   note                  why the entry reads the way it does.
ARCH_QUANT_POLICY: Dict[str, Dict[str, object]] = {
    "krea2": {
        "skip_below_work_gate": False,
        "excludes": (),
        "note": (
            "Krea 2 has few Linears below the runtime min-work gate, so they are "
            "quantized for the VRAM (time_mod_proj alone is 36864x6144). The "
            "shipped krea2_int8 artifact was produced with the flag off; turning "
            "it on here would stop reproducing it."
        ),
    },
    "anima": {
        "skip_below_work_gate": True,
        "excludes": (),
        "note": (
            "283 of Anima's 515 Linears sit below the runtime min-work gate (168 "
            "AdaLN modulation Linears alone). A Linear-only roll-up put the naive "
            "all-int8 artifact below break-even at 384x384 and behind the filtered "
            "one at every resolution measured, so Anima ships filtered."
        ),
    },
}

# Architectures the RUNTIME converter is wired for. A superset entry in
# ARCH_QUANT_POLICY (offline-only arch) would simply not be listed here.
RUNTIME_INT8_ARCHS = ("anima", "krea2")


def arch_policy(arch: Optional[str], fmt: str = "int8") -> Dict[str, object]:
    """Resolved selection knobs for ``arch``/``fmt``.

    Returns ``{"min_align", "skip_below_work_gate", "excludes", "note"}``.
    ``skip_below_work_gate`` is forced False for any format other than int8:
    its two constants are ``int8_linear``'s runtime gate and ``fp8_linear`` has
    no ``_MIN_WORK_*`` at all, so applying them to an e4m3 conversion would
    filter it against a rule that governs nothing it will ever run.
    """
    entry = ARCH_QUANT_POLICY.get(arch or "", {})
    return {
        "min_align": FORMAT_MIN_ALIGN.get(fmt, 0),
        "skip_below_work_gate": bool(entry.get("skip_below_work_gate", False)) if fmt == "int8" else False,
        "excludes": tuple(entry.get("excludes", ()) or ()),
        "note": entry.get("note", ""),
    }


# ---------------------------------------------------------------------------
# Linear enumeration + shape selection
# ---------------------------------------------------------------------------

def linear_paths(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """{module path: (in_features, out_features)} for every ``nn.Linear``."""
    return {
        name: (m.in_features, m.out_features)
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    }


def select_targets(
    linears: Dict[str, Tuple[int, int]],
    present_keys: set,
    min_align: int,
    excludes: Sequence[re.Pattern],
    skip_below_work_gate: bool = False,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Split the Linears into (quantize, [(skipped, reason)]).

    ``present_keys`` holds module paths ALREADY stripped of an arch's
    ``source_prefix``, so it is directly comparable with the model's paths. The
    runtime converter passes the live module's own paths, for which the check is
    trivially true; it is kept in the shared body so both callers run the same
    function rather than two similar ones.

    ``skip_below_work_gate`` is applied verbatim if set; the INT8-only scoping
    lives in the callers (``arch_policy`` for the runtime path, ``main`` for the
    CLI), the same place the other int8-only selectors are scoped.
    """
    targets: List[str] = []
    skipped: List[Tuple[str, str]] = []
    for name, (in_f, out_f) in sorted(linears.items()):
        if f"{name}.weight" not in present_keys:
            skipped.append((name, "no weight in checkpoint"))
            continue
        if min_align and (in_f % min_align or out_f % min_align):
            skipped.append((name, f"unaligned {in_f}x{out_f} (cannot reach the fast GEMM path)"))
            continue
        if skip_below_work_gate and (in_f < INT8_MIN_WORK_K or out_f < INT8_MIN_WORK_N):
            skipped.append((
                name,
                f"{in_f}x{out_f} below the runtime min-work gate "
                f"(k>={INT8_MIN_WORK_K}, n>={INT8_MIN_WORK_N}) at any m: it would always "
                f"run the dequant path, which is slower than the unquantized F.linear"))
            continue
        pattern = next((p for p in excludes if p.search(name)), None)
        if pattern is not None:
            skipped.append((name, f"excluded by /{pattern.pattern}/"))
            continue
        targets.append(name)
    return targets, skipped


# ---------------------------------------------------------------------------
# Per-layer format selection + audit (int8 only)
# ---------------------------------------------------------------------------

def _rel_rms(reference: torch.Tensor, approx: torch.Tensor) -> float:
    """Relative RMS error of ``approx`` against ``reference``, in float32."""
    ref = reference.to(torch.float32)
    err = approx.to(torch.float32) - ref
    denom = ref.pow(2).mean().sqrt()
    if not torch.isfinite(denom) or denom == 0:
        return float("nan")
    return float(err.pow(2).mean().sqrt() / denom)


def audit_and_quantize_int8(
    name: str,
    tensor: torch.Tensor,
    crest_threshold: float,
    fallback: str,
) -> Tuple[str, torch.Tensor, Optional[torch.Tensor], Dict]:
    """Choose int8 / e4m3 / bf16 for one Linear weight and return the audit row.

    BOTH candidate quantizations are always performed and both errors always
    measured, whatever the crest says. That costs one extra pass over a weight
    that is already resident and makes the audit table a record of what was
    actually true rather than of what the heuristic predicted.

    Returns ``(chosen_format, weight, scale_or_None, audit_row)``.
    """
    crest = weight_crest_factors(tensor)
    crest_mean = float(crest.mean())
    crest_p99 = float(crest.quantile(0.99)) if crest.numel() > 1 else crest_mean
    crest_max = float(crest.amax())

    q_i8, s_i8 = quantize_weight_to_int8(tensor)
    q_f8, s_f8 = quantize_weight_to_fp8(tensor)
    err_i8 = _rel_rms(tensor, q_i8.to(torch.float32) * s_i8.unsqueeze(1))
    err_f8 = _rel_rms(tensor, q_f8.to(torch.float32) * s_f8.unsqueeze(1))

    # Two independent reasons to select a layer out. The crest rule is the
    # documented, predictive one; the measured comparison is the backstop for a
    # weight whose distribution the crest model does not describe (it cannot,
    # for instance, see a bimodal row). Either one is sufficient.
    if crest_mean > crest_threshold:
        reason = f"crest {crest_mean:.2f} > {crest_threshold:.2f}"
        chosen = fallback
    elif not (err_i8 < err_f8):
        # Also catches NaN errors (a degenerate all-zero or non-finite weight):
        # `not (a < b)` is False only when int8 is strictly better.
        reason = f"measured int8 {err_i8:.5f} not better than e4m3 {err_f8:.5f}"
        chosen = fallback
    else:
        reason = f"int8 {err_i8:.5f} < e4m3 {err_f8:.5f} at crest {crest_mean:.2f}"
        chosen = "int8"

    row = {
        "name": name,
        "shape": list(tensor.shape),
        "int8_rel_rms": err_i8,
        "e4m3_rel_rms": err_f8,
        "advantage_int8_over_e4m3": (err_f8 / err_i8) if err_i8 else float("inf"),
        "crest_mean": crest_mean,
        "crest_p99": crest_p99,
        "crest_max": crest_max,
        "chosen": chosen,
        "reason": reason,
    }
    if chosen == "int8":
        return chosen, q_i8, s_i8, row
    if chosen == "e4m3":
        return chosen, q_f8, s_f8, row
    return "bf16", tensor, None, row


def audit_document(rows: List[Dict], settings: Dict) -> Dict:
    """The audit JSON body, identical in shape for the offline and runtime paths.

    ``{"settings", "format_counts", "geomean_advantage", "layers"}``. The offline
    tool writes this to ``<stem>.int8_audit.json``; the runtime converter returns
    it so a conversion can be diffed against a committed artifact.
    """
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["chosen"]] = counts.get(r["chosen"], 0) + 1
    finite = [r["advantage_int8_over_e4m3"] for r in rows
              if r["advantage_int8_over_e4m3"] not in (float("inf"),)
              and r["advantage_int8_over_e4m3"] == r["advantage_int8_over_e4m3"]]
    geomean = None
    if finite:
        geomean = float(torch.tensor(finite, dtype=torch.float64).log().mean().exp())
    return {
        "settings": settings,
        "format_counts": counts,
        "geomean_advantage": geomean,
        "layers": rows,
    }


# ---------------------------------------------------------------------------
# In-place runtime conversion
# ---------------------------------------------------------------------------

def already_weight_only_quantized(model: nn.Module) -> int:
    """Count ``Int8Linear`` / ``Fp8Linear`` modules under ``model``.

    Non-zero means the module already owns weight-only quantized Linears --
    either from an offline-quantized checkpoint or from a previous in-place
    runtime conversion. Detection is by module type, not by weight dtype, so it
    cannot be confused by a module that merely happens to store float8.
    """
    return sum(1 for m in model.modules() if isinstance(m, (Int8Linear, Fp8Linear)))


_FLOAT8_DTYPES = tuple(
    getattr(torch, _n) for _n in
    ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, _n)
)


def float8_weight_linear_count(model: nn.Module) -> int:
    """Count plain ``nn.Linear`` modules whose weight ALREADY holds float8.

    The type-based ``already_weight_only_quantized`` above cannot see these: the
    legacy runtime FP8 path (``vram_optimization._anima_patch_linear_fp8``) casts
    ``linear.weight.data`` to e4m3/e5m2 and monkeypatches ``forward``, leaving the
    module an ``nn.Linear``. Quantizing such a weight to int8 would quantize an
    ALREADY-rounded weight -- measured at 0.04400 relative RMS on Anima against
    0.00394 for a direct int8 conversion of the same checkpoint (11.2x), i.e.
    worse than either format alone. The dtype check is the missing half of the
    type check, and the converter's callers refuse on it.
    """
    if not _FLOAT8_DTYPES:
        return 0
    return sum(
        1 for m in model.modules()
        if isinstance(m, nn.Linear) and m.weight is not None
        and m.weight.dtype in _FLOAT8_DTYPES
    )


def _lora_wrapped_count(model: nn.Module) -> int:
    """Count LoRA wrappers under ``model`` (by class name, no import needed).

    A wrapped Linear is no longer an ``nn.Linear``, so converting a LoRA'd module
    would silently skip every wrapped layer and select a DIFFERENT set than the
    offline audit. The converter refuses instead.
    """
    return sum(1 for m in model.modules() if type(m).__name__ == "LoRALinearLayer")


def _resolve_parent(root: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    """``(parent module, attribute name)`` for a dotted module path under root."""
    if "." not in dotted:
        return root, dotted
    parent_path, attr = dotted.rsplit(".", 1)
    return root.get_submodule(parent_path), attr


def _filled_quantized_linear(
    src: nn.Linear,
    chosen: str,
    q: torch.Tensor,
    scale: torch.Tensor,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Build an ``Int8Linear``/``Fp8Linear`` already holding ``q``/``scale``.

    The sibling ``swap_linears_to_*`` helpers are LOADER-shaped: they gate on a
    state dict and construct an EMPTY module for ``load_state_dict`` to fill.
    This is the live-module constructor -- it takes the source ``nn.Linear`` and
    the quantized tensors and assigns the buffers directly, so no second copy of
    the model ever exists.
    """
    cls = Int8Linear if chosen == "int8" else Fp8Linear
    mod = cls(
        src.in_features,
        src.out_features,
        bias=src.bias is not None,
        compute_dtype=compute_dtype,
    )
    mod.weight = q.contiguous().to(device)
    mod.weight_scale = scale.to(torch.float32).contiguous().to(device)
    if src.bias is not None:
        mod.bias = src.bias.detach().to(compute_dtype).to(device)
    return mod


def quantize_linears_in_place(
    model: nn.Module,
    *,
    arch: Optional[str] = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    work_device: Optional[torch.device] = None,
    crest_threshold: float = DEFAULT_CREST_THRESHOLD,
    fallback: str = "e4m3",
    min_align: Optional[int] = None,
    excludes: Iterable[str] = (),
    skip_below_work_gate: Optional[bool] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    label: str = "transformer",
) -> Dict:
    """Convert every selected ``nn.Linear`` under ``model`` to int8/e4m3 IN PLACE.

    MIXED by construction, exactly like the offline tool: a layer the measured
    backstop rejects becomes an ``Fp8Linear`` rather than being left alone (1
    layer on Krea 2, 4 on Anima).

    MEMORY, measured, not asserted. NO SECOND COPY OF THE MODULE IS BUILT: each
    source weight is dropped as its replacement is installed, and the module's own
    parameter bytes fall (Anima 3.895 -> 2.327 GB, Krea 2 23.879 -> 11.948 GB).
    PROCESS RSS is a different quantity and does NOT fall: on real Anima, RSS goes
    0.958 GB after load -> 6.159 GB peak -> 6.159 GB steady after gc, against
    2.327 GB of resulting module bytes. The safetensors mapping of the SOURCE
    checkpoint stays resident because the layers this selection skips (283 of
    Anima's 515 Linears) and every non-Linear parameter still reference it, so
    steady-state host memory is roughly source + quantized module ~= 1.6x the
    source, held until the model is reloaded. Budget host RAM accordingly: ~6 GB
    for Anima, ~36 GB for a 24 GB bf16 Krea 2 transformer. It is still far below
    the SD1.5/SDXL-style ``copy.deepcopy`` + retained second CPU copy, which is
    what makes it viable at Krea 2 scale at all.

    DEVICE. The quantization math runs on ``work_device`` (CUDA when available,
    else wherever the weight lives) and each result is placed back on the
    weight's ORIGINAL device, so this is safe to call with the module on CPU
    (before staging, which is what both callers do) or on GPU. The math itself
    materialises the weight in float32 (``quantize_weight_to_int8``), which is
    why it is done per layer and freed immediately. A layer whose float32 working
    set does not fit on ``work_device`` is retried on the weight's own device
    rather than aborting the conversion; the fallbacks are listed in the returned
    document under ``oom_fallback_layers``.

    ONE-WAY. There is no inverse: the source bf16 weights are dropped. The model
    stays quantized until it is reloaded.

    RESUMABLE. Selection walks ``nn.Linear`` only and a converted layer is no
    longer one, so re-running after a failure converts exactly the layers that
    are still unconverted. On failure the exception carries the partial audit
    document as ``_int8_partial_document`` so the caller can report and later
    merge it.

    Returns the audit document (``audit_document``) with the extra keys
    ``elapsed_s``, ``converted`` and ``oom_fallback_layers``.
    """
    if _lora_wrapped_count(model):
        raise LoraWrappedError(
            f"refusing to quantize a LoRA-wrapped {label}: the wrappers hide the "
            f"underlying Linears, so the selection would silently differ from the "
            f"offline audit. Convert before applying LoRAs."
        )

    policy = arch_policy(arch, "int8")
    if min_align is None:
        min_align = int(policy["min_align"])
    if skip_below_work_gate is None:
        skip_below_work_gate = bool(policy["skip_below_work_gate"])
    patterns = [re.compile(p) for p in (tuple(excludes) + tuple(policy["excludes"]))]

    linears = linear_paths(model)
    present = {f"{name}.weight" for name in linears}
    targets, skipped = select_targets(
        linears, present, min_align, patterns, skip_below_work_gate=skip_below_work_gate)

    print(f"[RuntimeInt8] {label}: {len(linears)} nn.Linear module(s); "
          f"converting {len(targets)}, skipping {len(skipped)} "
          f"(arch={arch}, min_align={min_align}, skip_below_work_gate={skip_below_work_gate})")

    rows: List[Dict] = []
    counts = {"int8": 0, "e4m3": 0, "bf16": 0}
    oom_fallbacks: List[str] = []
    total = len(targets)
    t0 = time.perf_counter()

    def _document() -> Dict:
        doc = audit_document(rows, {
            "arch": arch,
            "format": "int8",
            "mode": "runtime_in_place",
            "min_align": min_align,
            "skip_below_work_gate": skip_below_work_gate,
            "min_work_k": INT8_MIN_WORK_K,
            "min_work_n": INT8_MIN_WORK_N,
            "crest_threshold": crest_threshold,
            "fallback": fallback,
            "compute_dtype": str(compute_dtype),
            "skipped": [{"name": n, "reason": r} for n, r in skipped],
        })
        doc["elapsed_s"] = time.perf_counter() - t0
        doc["converted"] = dict(counts)
        doc["oom_fallback_layers"] = list(oom_fallbacks)
        return doc

    for i, name in enumerate(targets):
        try:
            parent, attr = _resolve_parent(model, name)
            src = getattr(parent, attr)
            if not isinstance(src, nn.Linear):
                # Only reachable if the module tree changed under us.
                continue
            weight = src.weight.detach()
            orig_device = weight.device
            compute_on = work_device if work_device is not None else orig_device
            staged = weight.to(compute_on) if compute_on != orig_device else weight
            try:
                chosen, q, scale, row = audit_and_quantize_int8(
                    name, staged, crest_threshold, fallback)
            except torch.cuda.OutOfMemoryError:
                # This layer's float32 working set did not fit on the work
                # device. Retrying it on the weight's OWN device costs time but
                # keeps the conversion whole -- an abort here is precisely what
                # leaves a half-quantized module behind.
                if compute_on == orig_device:
                    raise
                del staged
                torch.cuda.empty_cache()
                oom_fallbacks.append(name)
                print(f"[RuntimeInt8] {label}: CUDA OOM quantizing {name} "
                      f"({tuple(weight.shape)}); retrying on {orig_device}")
                staged = weight
                chosen, q, scale, row = audit_and_quantize_int8(
                    name, staged, crest_threshold, fallback)
            if chosen == "bf16":
                # fallback="bf16": leave the source Linear untouched.
                rows.append(row)
                counts[chosen] = counts.get(chosen, 0) + 1
                del staged, q
                continue
            setattr(parent, attr, _filled_quantized_linear(
                src, chosen, q, scale, compute_dtype, orig_device))
            rows.append(row)
            counts[chosen] = counts.get(chosen, 0) + 1
            del src, weight, staged, q, scale
        except Exception as err:
            # The module is now PARTIALLY converted. Hand the caller everything
            # measured so far, plus where it stopped, so the failure can be
            # reported accurately and the remaining layers resumed later (the
            # selection walks nn.Linear, and the layers already replaced are no
            # longer nn.Linear, so a re-run picks up exactly the remainder).
            doc = _document()
            doc["partial"] = True
            doc["failed_layer"] = name
            doc["converted_before_failure"] = counts["int8"] + counts["e4m3"]
            doc["remaining"] = total - (counts["int8"] + counts["e4m3"] + counts["bf16"])
            try:
                setattr(err, "_int8_partial_document", doc)
            except Exception:
                pass
            print(f"[RuntimeInt8] {label}: FAILED at {name} after converting "
                  f"{doc['converted_before_failure']} of {total} layer(s): {err}")
            raise
        if progress_cb is not None:
            try:
                progress_cb(i + 1, total, name)
            except Exception:
                pass

    doc = _document()
    elapsed = doc["elapsed_s"]
    print(f"[RuntimeInt8] {label}: {counts.get('int8', 0)} int8 + "
          f"{counts.get('e4m3', 0)} e4m3 Linear(s) in {elapsed:.1f}s "
          f"(skipped {len(skipped)}"
          + (f", {len(oom_fallbacks)} OOM fallback(s)" if oom_fallbacks else "") + ")")
    return doc
