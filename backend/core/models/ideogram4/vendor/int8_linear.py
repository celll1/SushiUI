"""Weight-only INT8 Linear support, with an opt-in W8A8 ``torch._int_mm`` path.

A sibling of ``fp8_linear.Fp8Linear`` with the SAME on-disk shape -- a per-output
row scale stored under ``<name>.weight_scale`` -- but an ``int8`` weight instead
of ``float8_e4m3fn``. The weight dtype is what disambiguates the two formats; the
scale suffix alone does not (see ``is_int8_state_dict``).

WHY IT EXISTS. The shipped FP8 W8A8 path is fast but its ACTIVATION
quantization is the accuracy floor: e4m3 keeps 3 mantissa bits, so its relative
error is ~2.6e-02 no matter what the data looks like, and on a flat gradient that
floor shows up as coarse mottle. int8 with per-row/per-token scales spends its
254 levels linearly across each row's range, which on real Krea 2 weights measured
2.06x lower weight error and 2.19x lower W8A8 error (geomean over 112 layers)
than e4m3 -- EXCEPT on rows with a huge crest factor, where e4m3's floating
exponent wins and the offline quantizer must select the layer out (see
``subapps/fp8_quantize/quantize_transformer_fp8.py --format int8``, which emits
both formats into one mixed checkpoint).

Checkpoint layout (per quantized Linear ``<name>``):
    <name>.weight        int8     (out, in)
    <name>.weight_scale  float32  (out,)
    <name>.bias          compute dtype (out,)  [optional]

Dequantization: ``weight.to(dtype) * weight_scale[:, None]``.

TWO FORWARD PATHS, exactly as ``Fp8Linear`` has:

* **W8A8 integer GEMM** (``torch._int_mm``): the activation is dynamically
  quantized to int8 with a per-token float32 scale, the matmul accumulates in
  int32 on the integer tensor cores, and the two scale vectors are applied in
  float32 afterwards. Opt-in behind ``SUSHI_INT8_MM=1`` at import or
  ``set_int8_mm_enabled`` at runtime, and INFERENCE-ONLY: a module's owner
  declares that by calling ``disable_int8_mm`` (every trainer-side loader does).
* **Dequantized matmul** (fallback and default): the weight is dequantized to
  the compute dtype and a normal matmul runs. Works anywhere, including CPU.

``torch._int_mm``'s constraints, measured on this hardware and corroborated
against ``torch/_meta_registrations.py::meta__int_mm`` (torch 2.10):

    * both operands int8, 2-D, int32 accumulation
    * ``m > 16`` STRICTLY (m=16 raises, m=17 works)
    * ``k % 8 == 0`` and ``n % 8 == 0``
    * ALL FOUR operand layout combinations are accepted -- unlike
      ``_scaled_mm``, which demands row-major x column-major. That is why this
      module has no layout branch and no contiguity juggling: the weight stays
      ``(out, in)`` row-major and ``.t()`` gives the ``(in, out)`` operand with
      no copy.

A FOURTH gate is ours, not torch's: a minimum amount of GEMM work, expressed as
THREE conditions (``_MIN_WORK_MKN`` / ``_MIN_WORK_K`` / ``_MIN_WORK_N``).
Measured on sm_89, the thin real Krea 2 shapes REGRESS on this path -- the fixed
cost of quantizing the activation and running the epilogue is not repaid -- and
a single ``m*k*n`` threshold provably cannot separate them. See those constants
for the sweep. Anything failing any gate is served by the dequant path, which is
the definition of the layer's output.

Both elementwise stages around the GEMM are served by single fused Triton kernels
from ``int8_fused`` where they are available, gated on a probe that verifies them
BITWISE against the eager chains (``_eager_quantize_activation``,
``_eager_epilogue``) which remain the definition of the result. The fused
kernels are NOT an optimization here but a REQUIREMENT: eager int8 measured
1.515x over bf16 -- statistically tied with the shipped fp8 fused path (1.550x)
-- while fused int8 measured 2.561x. Without them this path is not worth having.
"""

from __future__ import annotations

import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F


# Symmetric int8 range. 127, not 128: a symmetric grid has no asymmetric
# most-negative value to special-case, and -128 is excluded by the clamp so the
# negation of any representable value is representable.
INT8_MAX = 127.0
INT8_WEIGHT_DTYPE = torch.int8

# 1/127 pre-rounded to float32, and used as a MULTIPLIER everywhere a value is
# mapped onto the int8 grid -- never ``x / INT8_MAX``.
#
# This is not micro-optimization, it is determinism. ``tensor / python_float`` is
# NOT a true divide on CUDA: torch lowers a scalar divisor to a multiply by its
# reciprocal, while the CPU kernel does a correctly-rounded divide, so the same
# expression gives two different float32 results on the two devices (measured: a
# 1-ulp split on 2.0994208/127, 0x3c876bc0 vs 0x3c876bc1). With the eager chain
# as the bitwise contract for the Triton kernels AND for the offline quantizer,
# a definition that depends on which device evaluated it -- and on a lowering
# torch is free to change -- is not a definition. Spelling the multiply out
# makes CPU, CUDA and Triton agree by construction.
_RECIP_INT8_MAX = float(torch.tensor(1.0, dtype=torch.float32).div(
    torch.tensor(INT8_MAX, dtype=torch.float32)).item())
# Deliberately the SAME suffix fp8_linear uses: the two formats share the
# per-output-row-scale layout and only the weight dtype tells them apart, so a
# reader that keys on the suffix alone must also check the dtype.
INT8_SCALE_SUFFIX = ".weight_scale"


def _refuse_unsupported_quant_semantics(state_dict: dict[str, torch.Tensor]) -> None:
    """Refuse a checkpoint whose DECLARED quantization contract we do not implement.

    Imported lazily: ``core.models.common.int8_runtime_quantize`` imports this
    module at module level, so a top-level import of anything under
    ``core.models.common`` from here risks a cycle.

    Why it is repeated at the int8/fp8 entry points rather than left to
    ``quantized_state_dict_report``: a Comfy ``int8_tensorwise`` file with
    ``convrot: true`` satisfies THIS function's gate (a ``.weight_scale``
    sibling next to an int8 ``.weight``) exactly as a supported file does, and
    at least one caller (``ideogram4/vendor/text_encoder.py``) reaches the swap
    without running the census at all. See
    ``core.models.common.quantized_checkpoint_guard`` for the mechanism.
    """
    try:
        from core.models.common.quantized_checkpoint_guard import (
            refuse_unsupported_quant_semantics,
        )
    except Exception:  # pragma: no cover - defensive; never mask the guard itself
        return
    refuse_unsupported_quant_semantics(state_dict)


def is_int8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """True if the checkpoint carries weight-only INT8 Linear weights.

    Raises ``UnsupportedQuantSemanticsError`` first if the file declares a
    quantization contract this build does not implement (a ``.comfy_quant``
    marker asking for a ConvRot rotation, an unknown format, or AWQ
    ``.pre_quant_scale`` vectors): such a file answers True below and swaps
    cleanly while being numerically wrong.

    Keyed on BOTH the ``.weight_scale`` sibling and an ``int8`` weight. The
    suffix alone is ambiguous -- ``fp8_linear``'s format uses the identical
    suffix -- so a checkpoint is int8 only when a scale's own weight is int8.
    A MIXED checkpoint (some layers int8, some e4m3, produced by the offline
    tool's per-layer selection) answers True here and True to
    ``is_fp8_state_dict`` as well; both swaps must then run, which is what the
    Krea 2 loader does.
    """
    _refuse_unsupported_quant_semantics(state_dict)
    for key in state_dict:
        if not key.endswith(INT8_SCALE_SUFFIX):
            continue
        weight = state_dict.get(key[: -len(INT8_SCALE_SUFFIX)] + ".weight")
        if weight is not None and weight.dtype is INT8_WEIGHT_DTYPE:
            return True
    return False


def quantize_weight_to_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D Linear weight to int8 with per-output-row float32 scales.

    Returns ``(weight_int8, scale)`` where ``weight_int8`` is ``(out, in)``
    ``torch.int8`` and ``scale`` is ``(out,)`` float32 such that
    ``weight ~= weight_int8.to(dtype) * scale[:, None]``.

    Round-half-to-EVEN (``torch.round``), matching the activation quantizer, so
    the weight and activation grids are produced by the same rule. The scale uses
    the ``_RECIP_INT8_MAX`` multiply for the same reason the activation scale does
    -- this runs on CPU in the offline tool and must agree with a GPU evaluation.
    """
    w = weight.detach().to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax * _RECIP_INT8_MAX
    q = (w / scale).round_().clamp_(-INT8_MAX, INT8_MAX).to(INT8_WEIGHT_DTYPE)
    return q, scale.squeeze(1).to(torch.float32)


def weight_crest_factors(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-row crest factor (row amax / row RMS) of a 2-D weight.

    THE predictor of when int8 loses to e4m3. int8's quantization step is
    ``amax/127`` for the whole row, so a uniformly-distributed rounding error has
    RMS ``amax/(127*sqrt(12))``; relative to the row's own RMS that is
    ``crest/440``. e4m3's relative error is flat at ~2.63e-02 regardless of the
    distribution (it spends a floating exponent per element). Setting the two
    equal gives a break-even crest of ~11.6 -- which is why the offline tool
    selects a layer out above a threshold in that neighbourhood rather than at
    some arbitrary percentile.

    Returns a float32 tensor of shape ``(out,)``. Rows that are entirely zero
    get 0.0 (no outlier structure to protect against), not NaN.
    """
    w = weight.detach().to(torch.float32)
    amax = w.abs().amax(dim=1)
    rms = w.pow(2).mean(dim=1).sqrt()
    return torch.where(rms > 0, amax / rms.clamp(min=torch.finfo(torch.float32).tiny),
                       torch.zeros_like(amax))


# ---------------------------------------------------------------------------
# W8A8 integer-GEMM fast path (torch._int_mm)
# ---------------------------------------------------------------------------

# Smallest activation scale we will emit. Guards all-zero rows: without it the
# scale would be 0 and its reciprocal inf/NaN.
_MIN_ACT_SCALE = 1e-12

# torch._int_mm's own constraints (measured; see the module docstring).
_INT_MM_M_FLOOR = 16        # m must be STRICTLY greater than this
_INT_MM_KN_ALIGN = 8

# Minimum-work gate. NOT a torch constraint -- ours, and THREE conditions, not
# one. All must hold for the fast path to be taken.
#
# WHAT IS CLAIMED, precisely: on the 32-shape sweep below, run on sm_89, the rule
# ADMITS NO SHAPE THAT MEASURED SLOWER THAN THE DEQUANT PATH. It is NOT claimed
# to admit every winner -- it refuses several -- and it is NOT claimed that a
# formula can separate the two classes in general (see the non-monotonicity
# below). Every refusal costs speed only: both sides of the gate are numerically
# valid paths, so a mis-placed threshold never changes a pixel.
#
# WHY THREE CONDITIONS. The activation quantizer reads (m, k) and the epilogue
# reads and writes (m, n) whatever the GEMM costs, while the dequant path it is
# competing against pays a full (n, k) weight dequantization on every call. A
# single threshold on m*k*n cannot express that, and MEASUREMENT SAYS SO -- the
# sweep (tmp/int8_gate_sweep.py, output kept at tmp/int8_gate_sweep_round2.txt)
# found m*k*n badly non-monotonic in the speedup:
#
#     m=4608 k=512  n=6144  m*k*n=1.45e10  0.538x   (regresses, high m*k*n)
#     m=512  k=2560 n=2560  m*k*n=3.36e09  2.518x   (wins, low m*k*n)
#     m=17   k=6144 n=6144  m*k*n=6.42e08  3.879x   (wins hugely, tiny m*k*n --
#                                                    the dequant arm re-expands
#                                                    a 6144x6144 weight for a
#                                                    17-row GEMM)
#
# so a threshold placed to exclude the first would exclude both wins, and one
# placed to admit the wins would admit a 2x regression. What separates them is
# the THIN dimension, and BOTH thin dimensions matter:
#
#     k = 512 or less        0.155x-0.782x   every shape regressed
#     k = 1024               0.774x-1.018x   at or below break-even, every shape
#     k = 2048 or more       1.23x-2.52x     every shape won (n >= 1024)
#     k = 6144, n <= 512     0.629x-0.960x   regressed despite k and m*k*n being
#                                            large -- the epilogue's (m, n) pass
#                                            and the dequant arm's tiny weight
#                                            make the integer GEMM not worth it
#
# Hence k >= 2048 and n >= 1024 rather than the earlier 1024/128, which admitted
# measured regressors at exactly k=1024 with large m (e.g. m=2500 k=1024 n=1024,
# 0.914x) and at k=6144 with n in {128, 256, 512}. The cost of the tightening is
# forgone speed on shapes near the boundary: m=4096 k=2560 n=640 measured 1.159x
# and is now refused, and shapes at n=640 measured on both sides of 1.0 across
# runs (0.890x/1.341x at k=2048), which is itself a reason not to admit them.
#
# The m*k*n floor then excludes small-work shapes that clear k/n: the worst
# regressor clearing them is 4.19e8 (m=64 k=n=2560, 0.931x) and 8.39e8
# (m=128 k=n=2560, 0.956x), the best win 3.36e9 (m=512 k=n=2560, 2.518x). 2.5e9
# sits essentially at the geometric centre of that gap. It also refuses
# m=17 k=n=6144 (6.42e8, a forgone 3.88x): a real win, given up because no
# full-width Krea 2 layer ever sees 17 tokens and admitting its m*k*n band would
# also admit the 0.93x-0.96x shapes above.
#
# On the seven REAL Krea 2 shapes the rule is exactly right: it admits all five
# winners (1.750x-2.937x) and refuses both regressors, img_in (4608x64x6144,
# 0.176x, k=64) and final_layer (4608x6144x64, 0.630x, n=64).
#
# The constants are measured on sm_89 and are not claimed to transfer to another
# GPU; what does transfer is the shape of the rule.
_MIN_WORK_MKN = 2_500_000_000
_MIN_WORK_K = 2048
_MIN_WORK_N = 1024

# Largest reduction length whose int32 accumulator provably cannot overflow.
#
# ``torch._int_mm`` accumulates in int32 and neither torch nor the kernel checks
# for wraparound: the worst-case magnitude of one output element is
# ``k * 127 * 127`` (both operands are clamped to +-127 by construction here), so
# any ``k`` above ``(2**31 - 1) // 16129 = 133_144`` CAN wrap, silently and with a
# plausible-looking sign flip. Demonstrated: k=140000, n=128, m=144 returned
# -2036907296 for a true 2258060000.
#
# Real Krea 2 never comes close (max in_features 16384, ~8x under the bound), but
# the gate is a FORMULA applied to whatever module a caller builds, not a
# whitelist of Krea 2 shapes, so the bound is enforced rather than assumed. The
# dequant path has no such limit and serves anything above it.
_MAX_ACC_K = (2 ** 31 - 1) // (127 * 127)

# The W8A8 integer-GEMM path is OPT-IN: set SUSHI_INT8_MM=1 to enable it.
# Default off for the same reason the fp8 one is: it is a different function from
# the dequant path everyone runs, and default-changing a numeric path behind an
# unrelated upgrade is not something an operator can consent to.
#
# Initialized from the environment at import, then MUTABLE for the life of the
# process via ``set_int8_mm_enabled`` (exposed as
# ``GET/POST /api/v1/system/int8-mm``). Not persisted.
_INT8_MM_ENABLED = os.environ.get("SUSHI_INT8_MM", "0") == "1"

# Where the CURRENT value came from: "default"/"env" at import, "api" for a
# POST /system/int8-mm, or "generation" for a per-generation
# `quantized_gemm_mode` (api/quantized_gemm.py). "generation" is distinct from
# "api" on purpose so the diagnostics can tell a manual flip from one a queued
# generation forced. Must stay in lockstep with the ``Int8MmState.origin`` enum
# in openapi.yaml ([default, env, api, generation]).
_INT8_MM_VALID_ORIGINS = frozenset({"default", "env", "api", "generation"})
_INT8_MM_ORIGIN = "env" if "SUSHI_INT8_MM" in os.environ else "default"

# device index -> "int_mm" (probed OK) or None (unusable, latched).
#
# Keyed on the DEVICE ONLY, unlike fp8's (device, activation dtype): both GEMM
# operands are int8 whatever the activation dtype is, so the probe would be
# byte-identical for every dtype on a device. The activation dtype still keys
# the FUSED-kernel latch in ``int8_fused``, where it genuinely matters.
_INT_MM_MODE: dict[int, str | None] = {}
_INT_MM_LOCK = threading.Lock()
_INT_MM_REPORTED: set[str] = set()


def _device_label(index: int) -> str:
    return f"cuda:{index}" if index >= 0 else "cuda:default"


def get_int8_mm_state() -> dict:
    """Current W8A8 integer-GEMM state.

    ``resolved_modes`` exposes the per-device probe results so the case "flag on,
    but the probe latched None and every layer runs the dequant path" is visible
    rather than inferred. An empty dict means no INT8 Linear forward has reached
    the probe yet in this process.
    """
    with _INT_MM_LOCK:
        return {
            "enabled": bool(_INT8_MM_ENABLED),
            "origin": _INT8_MM_ORIGIN,
            "resolved_modes": {_device_label(k): v for k, v in _INT_MM_MODE.items()},
        }


def set_int8_mm_enabled(enabled: bool, *, origin: str = "api") -> dict:
    """Turn the W8A8 integer-GEMM path on or off for THIS PROCESS.

    Both directions clear the probe cache (``_INT_MM_MODE``) and the one-shot
    report set, so the next INT8 forward re-probes -- which also un-latches a key
    that a transient failure had condemned.

    Same scope and limits as ``fp8_linear.set_scaled_mm_enabled``: per-process,
    not persisted, does NOT clear ``int8_fused``'s own toolchain latch (a
    property of the install, not of the selected GEMM path, and its fallback
    changes no result), and does NOT override ``disable_int8_mm`` -- that
    per-module opt-out is the authoritative gate and enabling here cannot
    resurrect a module that declared itself dequant-only.

    ``Int8Linear.forward`` branches on this module global, so a ``torch.compile``
    graph containing an ``Int8Linear`` would bake the value in at trace time.
    Safe today only because nothing compiles these layers (``use_torch_compile``
    is wired solely into the SD1.5/SDXL U-Net staging path and is unsupported for
    every DiT arch); a future change that compiles an arch owning ``Int8Linear``
    must grow a refusal here.

    Returns the same dict as ``get_int8_mm_state()``.
    """
    global _INT8_MM_ENABLED, _INT8_MM_ORIGIN
    enabled = bool(enabled)
    if origin not in _INT8_MM_VALID_ORIGINS:
        raise ValueError(
            f"invalid origin {origin!r}: must be one of {sorted(_INT8_MM_VALID_ORIGINS)} "
            f"(see the Int8MmState.origin enum in openapi.yaml)"
        )
    with _INT_MM_LOCK:
        changed = enabled != _INT8_MM_ENABLED
        _INT8_MM_ENABLED = enabled
        _INT8_MM_ORIGIN = origin
        _INT_MM_MODE.clear()
        _INT_MM_REPORTED.clear()
        state = {"enabled": _INT8_MM_ENABLED, "origin": _INT8_MM_ORIGIN, "resolved_modes": {}}
    print(
        f"[Int8Linear] W8A8 integer-GEMM path "
        f"{'ENABLED' if enabled else 'DISABLED'} (origin={origin}"
        f"{'' if changed else ', unchanged'}); probe cache cleared. "
        f"{'INT8 Linear layers will run torch._int_mm where the shape gates allow.' if enabled else 'INT8 Linear layers run the dequantized matmul.'}"
    )
    return state


def describe_gemm_path(module: nn.Module | None = None) -> str | None:
    """Describe which INT8 GEMM path is in force, for metadata/provenance.

    Returns None when ``module`` is given and owns no ``Int8Linear`` (nothing to
    record). Otherwise one of:

    * ``"w8a8_int_mm(int_mm)"`` -- the flag is on, the module allows it and the
      probe resolved. A ``+fused`` suffix records that the fused Triton kernels
      served at least one key in this process; that does NOT change the pixels
      (they are gated on a bitwise-equality probe) but it does change what ran.
    * ``"int8_dequant"`` -- the flag is off, or every owned layer opted out.
    * ``"int8_dequant(int_mm unavailable)"`` -- the flag is on but the probe
      latched None on every device.
    * ``"int8_dequant(int_mm unprobed)"`` -- the flag is on but no INT8 forward
      has reached the probe in this process.

    Distinct label stems from ``fp8_linear.describe_gemm_path`` on purpose: a
    mixed-format checkpoint reports both, joined, and "dequant" would be
    ambiguous about which format it referred to.

    Same LIMITATIONS as the fp8 reporter: ``_INT_MM_MODE`` is a process-wide
    cache, not scoped to ``module`` or to one generation, and it cannot see the
    per-layer runtime fallbacks (a shape below ``_MIN_WORK_MKN``, or a transient
    allocation failure) that happen WITHIN a generation.
    """
    if module is not None:
        layers = [m for m in module.modules() if isinstance(m, Int8Linear)]
        if not layers:
            return None
        if not any(m._allow_int8_mm for m in layers):
            return "int8_dequant"
    with _INT_MM_LOCK:
        if not _INT8_MM_ENABLED:
            return "int8_dequant"
        modes = sorted({m for m in _INT_MM_MODE.values() if m})
        probed = bool(_INT_MM_MODE)
    if modes:
        return f"w8a8_int_mm({'+'.join(modes)}{_fused_suffix()})"
    return "int8_dequant(int_mm unavailable)" if probed else "int8_dequant(int_mm unprobed)"


def _fused_suffix() -> str:
    """``"+fused"`` when the fused Triton kernels served a key in this process."""
    try:
        from .int8_fused import fused_enabled

        return "+fused" if fused_enabled() else ""
    except Exception:
        return ""


def _report_int_mm_fallback(key: str, reason: str, *, degraded: bool) -> None:
    """Log the fallback, and for unexpected failures surface it to the user.

    ``degraded=True`` means the fast path failed unexpectedly rather than simply
    being unsupported here; only that case reaches the generation warning
    channel. Fires exactly once per ``key``, and every degraded call site latches
    ``_INT_MM_MODE[...] = None`` at or before reporting, so the next forward
    short-circuits and never reaches this function again for that device.
    """
    message = f"INT8 W8A8 via torch._int_mm unavailable, falling back to dequant path: {reason}"
    with _INT_MM_LOCK:
        first_time = key not in _INT_MM_REPORTED
        _INT_MM_REPORTED.add(key)
    if first_time:
        print(f"[Int8Linear] {message}")
    if not degraded:
        return
    try:
        from api.generation_status import add_warning

        add_warning(message, code="quantization_fallback")
    except Exception:
        pass


def _probe_int_mm(device: torch.device) -> str | None:
    """Run one tiny integer GEMM and CHECK ITS VALUE. Returns "int_mm" or None.

    Executed, not inferred: ``hasattr(torch, "_int_mm")`` proves nothing about
    whether this driver/arch actually produces the right int32 accumulation, and
    a silently wrong GEMM here would be far worse than no fast path at all. The
    operands are 17x8 and 8x8 (17 because m must exceed 16, 8 because that is the
    k/n divisibility floor), so the probe sits exactly on every constraint
    boundary and costs no meaningful VRAM or time.

    The reference is computed ON THE CPU: CUDA has no int32 matmul at all
    (``a.int() @ b.int()`` raises ``"addmm_cuda" not implemented for 'Int'``),
    so a GPU-side reference would look like a failure of ``_int_mm`` and latch
    a perfectly good fast path off on every machine.
    """
    if not _INT8_MM_ENABLED:
        # The default state, not a degradation: stay silent.
        return None
    if not hasattr(torch, "_int_mm"):
        _report_int_mm_fallback("missing", "torch._int_mm is not available", degraded=False)
        return None
    if device.type != "cuda":
        return None
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        a = torch.randint(-127, 128, (_INT_MM_M_FLOOR + 1, _INT_MM_KN_ALIGN),
                          device=device, generator=gen, dtype=torch.int32).to(torch.int8)
        b = torch.randint(-127, 128, (_INT_MM_KN_ALIGN, _INT_MM_KN_ALIGN),
                          device=device, generator=gen, dtype=torch.int32).to(torch.int8)
        got = torch._int_mm(a, b)
        want = a.cpu().to(torch.int32) @ b.cpu().to(torch.int32)
        if got.dtype is not torch.int32 or not torch.equal(got.cpu(), want):
            _report_int_mm_fallback(
                f"probe{device.index}",
                "torch._int_mm did not reproduce the int32 reference product",
                degraded=True,
            )
            return None
    except Exception as exc:
        _report_int_mm_fallback(
            f"probe{device.index}",
            f"torch._int_mm unusable ({type(exc).__name__}: {exc})",
            degraded=True,
        )
        return None
    return "int_mm"


def _int_mm_mode(device: torch.device) -> str | None:
    """Cached ``_probe_int_mm`` keyed on the device index."""
    key = device.index if device.index is not None else -1
    try:
        return _INT_MM_MODE[key]
    except KeyError:
        pass
    mode = _probe_int_mm(device)
    with _INT_MM_LOCK:
        return _INT_MM_MODE.setdefault(key, mode)


def _quantize_activation(x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic int8 quantization of a 2-D activation.

    Returns ``(x_int8, scale)`` with ``scale`` of shape ``(m, 1)`` float32 such
    that ``x2 ~= x_int8.float() * scale``. Rows that are entirely zero get the
    floor scale rather than zero, so the reciprocal stays finite.

    Prefers the single fused Triton kernel in ``int8_fused``, gated on an
    EXECUTED probe that checks its output BITWISE against
    ``_eager_quantize_activation``, which stays the definition of the result.
    """
    fused = _try_fused_quantize(x2)
    if fused is not None:
        return fused
    return _eager_quantize_activation(x2)


def _try_fused_quantize(x2: torch.Tensor):
    """``int8_fused.fused_quantize`` if importable/usable, else None."""
    try:
        from .int8_fused import fused_quantize
    except Exception:
        return None
    return fused_quantize(x2)


def _eager_quantize_activation(x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference (and fallback) implementation of ``_quantize_activation``.

    Also the bitwise target the fused kernel's probe compares against, so this
    body -- including WHERE each rounding happens -- is the contract, not an
    implementation detail. Every property is load-bearing:

    * the scaled product is computed in FLOAT32 for every input dtype. The fp8
      quantizer rounds the multiplier to bfloat16 for bf16 inputs because e4m3
      keeps only 3 mantissa bits and the extra precision is thrown away anyway;
      int8's grid is ~5x finer (step 1/127 of the row range), so a bfloat16
      multiplier would perturb values by up to ~0.4% -- about one whole LSB --
      and cost real accuracy for nothing. It costs an ``(m, k)`` float32
      temporary in eager, which is exactly one of the things the fused kernel
      removes.
    * rounding is ``torch.round`` = round-half-to-EVEN, the same rule
      ``quantize_weight_to_int8`` uses.
    * the clamp is symmetric at +-127, so -128 is never produced.
    * the scale is ``amax * _RECIP_INT8_MAX``, a multiply, NOT ``amax / 127``
      -- see that constant for why a scalar divide is device-dependent here.

    NaN AND INF. ``amax`` propagates a NaN into the SCALE, and the epilogue then
    multiplies the whole row by NaN -- so a blown-up activation still reaches the
    output as NaN, loudly, exactly as the dequant path would. The int8 PAYLOAD
    cannot carry NaN (no integer can): ``.to(torch.int8)`` of NaN is 0 on CUDA,
    and the fused kernel reproduces that 0 explicitly rather than relying on the
    undefined float->int conversion. An ``inf`` amax gives an ``inf`` scale, a
    zero reciprocal, and a payload of 0/NaN->0; the output is then +-inf or NaN.
    """
    amax = x2.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (amax * _RECIP_INT8_MAX).clamp_(min=_MIN_ACT_SCALE)
    recip = scale.reciprocal()
    scaled = x2.to(torch.float32) * recip
    return scaled.round_().clamp_(-INT8_MAX, INT8_MAX).to(INT8_WEIGHT_DTYPE), scale


def _eager_epilogue(
    acc: torch.Tensor,
    a_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``((acc.float() * a_scale) * w_scale [+ bias]).to(out_dtype)``.

    THE DEFINITION of the epilogue and the bitwise target the fused kernel's
    probe compares against. Same discipline as ``fp8_linear._eager_epilogue``:

    * both scales are applied in float32 and the ASSOCIATION is
      ``(acc * a_scale) * w_scale`` -- not ``acc * (a_scale * w_scale)``, which
      is what a textbook int8 epilogue does;
    * the bias is widened to float32 and added as a separate ROUNDED step (an
      FMA would round once, which is more accurate and therefore a DIFFERENT
      result -- the fused kernel blocks the contraction with inline asm);
    * the output dtype is reached by exactly ONE rounding, at the end.

    The int32 -> float32 widening is exact for every accumulator this path
    produces: measured max |acc| at the real Krea 2 shapes is 2.94e6, well inside
    float32's exactly-representable integer range (2^24 = 1.68e7), and inside
    int32's range with 730x headroom. That headroom is a fact about Krea 2's
    shapes, NOT a property of the gate -- the accumulator itself is bounded only
    by ``k * 127**2``, which is why ``_int_mm_forward`` enforces ``_MAX_ACC_K``
    separately. Beyond 2^24 the conversion would round
    (round-to-nearest, deterministically) and both paths would round identically,
    so bitwise equality survives even there -- only the accuracy claim would not.

    ``acc`` is NOT mutated (it is int32 and the result is float), but the float32
    temporary it produces is scaled IN PLACE, keeping the transient at one
    ``(m, n)`` float32 buffer plus the narrower output.
    """
    out = acc.to(torch.float32)
    out.mul_(a_scale).mul_(w_scale)
    if bias is not None:
        out.add_(bias.to(torch.float32))
    return out.to(out_dtype)


def _try_fused_epilogue(acc, a_scale, w_scale, bias, out_dtype):
    """``int8_fused.fused_epilogue`` if importable/usable, else None."""
    try:
        from .int8_fused import fused_epilogue
    except Exception:
        return None
    return fused_epilogue(acc, a_scale, w_scale, bias, out_dtype)


def _is_allocation_failure(exc: BaseException) -> bool:
    """True for an allocation shortage however it is spelled.

    Reuses ``fp8_linear``'s marker list rather than restating it: cuBLAS reports
    a workspace shortage as a plain RuntimeError, which is transient in the same
    way an OOM is and must NOT latch the mode off for the process.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    try:
        from .fp8_linear import _is_allocation_failure as _fp8_is_alloc
    except Exception:
        return False
    return _fp8_is_alloc(exc)


class Int8Linear(nn.Module):
    """Linear layer holding an int8 weight + per-output-row float32 scale.

    The weight and scale are registered as buffers (not parameters) so they load
    via ``load_state_dict`` and are excluded from optimizer/grad machinery.

    ``forward`` tries the W8A8 integer GEMM first (``_int_mm_forward``) and falls
    back to the dequantized matmul (``_dequant_forward``, in the activation's
    dtype) whenever the former is not usable.

    ``_allow_int8_mm`` is the explicit per-module opt-out (see
    ``disable_int8_mm``). A CLASS attribute so it costs nothing per instance, is
    not a buffer/parameter (never touches ``state_dict``), and is set per
    instance only where a caller has opted out.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    # Owner-level kill switch for the W8A8 fast path. See disable_int8_mm().
    _allow_int8_mm: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=INT8_WEIGHT_DTYPE),
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._int_mm_forward(x)
        if out is not None:
            return out
        return self._dequant_forward(x)

    def _dequant_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize the weight to the compute dtype and run a normal matmul.

        ``self.weight * s`` is the PROMOTED spelling of
        ``self.weight.to(x.dtype) * s``: an integer tensor times a float tensor
        promotes to the float dtype, so torch does the widening inside the
        multiply and writes one ``(out, in)`` buffer instead of two.

        It is bitwise identical, not merely close, and the reason is exact
        representability rather than luck: every int8 code in [-128, 127] is
        exactly representable in bf16 (8 mantissa bits), fp16 (11) and fp32
        (24), so the widening rounds nothing in either spelling and the single
        remaining rounding is the multiply's. Pinned by
        ``backend/tests/quantized_dequant_bitwise_test.py``, which compares the
        two spellings on INTEGER BIT VIEWS over both devices, all three compute
        dtypes and hostile code/scale/activation values; any future rewrite of
        this line that is not bitwise equal fails that test.

        ``Fp8Linear._dequant_forward`` deliberately keeps the two-step form:
        ``float8_e4m3fn`` has no promoting multiply at all (torch raises
        "Promotion for Float8 Types is not supported" on CPU and CUDA alike).

        The dtype test is not decoration. Promotion only reproduces the explicit
        cast for an INTEGRAL weight; a float weight of a different dtype would
        promote UPWARD (fp32 codes against a bf16 scale give a fp32 ``w``, which
        ``F.linear`` would then reject) where the explicit cast narrows to
        ``x.dtype``. ``_int_mm_forward`` already declines on the same condition,
        so this keeps the two paths agreeing about what the buffer holds.
        """
        codes = self.weight
        if codes.dtype is not INT8_WEIGHT_DTYPE:
            codes = codes.to(x.dtype)
        w = codes * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def _int_mm_forward(self, x: torch.Tensor) -> torch.Tensor | None:
        """INT8 W8A8 matmul on the integer tensor cores, or None if not usable.

        Returning None (rather than raising) lets ``forward`` fall through to the
        dequant path for every case the integer GEMM cannot serve.
        """
        # GATE 0 (cheapest first): the feature is off for this process (the
        # default). A pure short-circuit -- every gate below still behaves
        # correctly without it -- so users who never set SUSHI_INT8_MM=1 do not
        # pay the remaining gates on every Linear forward.
        if not _INT8_MM_ENABLED:
            return None
        w = self.weight
        # GATE 1 (authoritative): the module's owner may forbid the fast path.
        # Grad mode is NOT a usable proxy for "this is inference" -- several
        # @torch.no_grad() helpers are shared by the inference and TRAINING call
        # graphs (ideogram4_pipeline_ops.encode_text_layers is reached from both)
        # and a training subprocess inherits SUSHI_INT8_MM from the backend via
        # training_process.py's os.environ.copy(). The trainer-side loaders call
        # disable_int8_mm on every module they own.
        if not self._allow_int8_mm:
            return None
        # GATE 2 (defence in depth): never run W8A8 where a gradient could flow.
        # x.requires_grad alone is not enough: every Int8Linear before the first
        # LoRA contribution in the graph receives an input with
        # requires_grad=False.
        if x.requires_grad or torch.is_grad_enabled():
            return None
        if not x.is_cuda or w.device != x.device or w.dtype is not INT8_WEIGHT_DTYPE:
            return None
        # GATE 3: torch._int_mm's k/n divisibility. No layout or contiguity gate
        # is needed -- _int_mm accepts all four operand layout combinations, so
        # the (out, in) row-major weight's .t() is a valid operand with no copy.
        if self.in_features % _INT_MM_KN_ALIGN or self.out_features % _INT_MM_KN_ALIGN:
            return None
        if x.shape[-1] != self.in_features or x.numel() == 0:
            return None
        m = x.numel() // self.in_features
        # GATE 4: torch._int_mm's strict m floor (m=16 raises, m=17 works).
        if m <= _INT_MM_M_FLOOR:
            return None
        # GATE 5 (ours, correctness): the int32 accumulator must not be able to
        # wrap. Unlike every other gate here this one is not about speed -- above
        # _MAX_ACC_K the fast path can return a silently wrong number.
        if self.in_features > _MAX_ACC_K:
            return None
        # GATE 6 (ours, performance): enough GEMM work, and no dimension thin
        # enough that the quantizer + epilogue outweigh it. All three conditions;
        # see the constants for the sweep they come from.
        if (self.in_features < _MIN_WORK_K
                or self.out_features < _MIN_WORK_N
                or m * self.in_features * self.out_features < _MIN_WORK_MKN):
            return None

        # Everything only the fast path does lives inside the try, including the
        # probe: ``_int_mm_mode`` allocates real tensors on its first call for a
        # device, so it can OOM exactly like the GEMM below -- and with the
        # toggle clearing the cache on every flip, that first call can land
        # inside somebody's forward pass. Because the exception propagates out of
        # ``_int_mm_mode`` before ``setdefault``, the mode is never cached: the
        # next forward re-probes instead of being latched off by memory pressure.
        # The activation quantizer has the same property (it allocates an (m, k)
        # int8 buffer, and an (m, k) float32 temporary on the eager chain).
        try:
            if _int_mm_mode(x.device) is None:
                return None

            x2 = x.reshape(-1, self.in_features)
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            x_int8, a_scale = _quantize_activation(x2)

            w_scale = self.weight_scale
            if w_scale.dtype is not torch.float32:
                w_scale = w_scale.to(torch.float32)
            w_scale = w_scale.reshape(1, self.out_features)
            if not w_scale.is_contiguous():
                w_scale = w_scale.contiguous()

            # (out, in) row-major -> (in, out) via .t(), no copy. _int_mm takes
            # any layout, so unlike the fp8 path there is nothing to branch on.
            acc = torch._int_mm(x_int8, w.t())

            # ONE fused Triton kernel where available (one read of the int32
            # accumulator, one write of the output; the eager chain is 5 kernels
            # over the same (m, n) extent with a float32 temporary), else the
            # eager chain. The two are bitwise-identical by construction and the
            # fused path is gated on a probe that verifies exactly that.
            out2 = _try_fused_epilogue(acc, a_scale, w_scale, self.bias, x.dtype)
            if out2 is None:
                out2 = _eager_epilogue(acc, a_scale, w_scale, self.bias, x.dtype)
        except torch.cuda.OutOfMemoryError as exc:
            self._report_transient_oom(exc)
            return None
        except Exception as exc:
            if _is_allocation_failure(exc):
                # e.g. RuntimeError: CUBLAS_STATUS_ALLOC_FAILED. Same nature as an
                # OOM (transient, pressure-dependent) but it does not arrive as
                # OutOfMemoryError, so it would otherwise latch the mode off for
                # the whole process.
                self._report_transient_oom(exc)
                return None
            # Anything else is a property of this configuration, not of one call:
            # latch it so we do not pay a failing call on every forward.
            key = x.device.index if x.device.index is not None else -1
            with _INT_MM_LOCK:
                _INT_MM_MODE[key] = None
            _report_int_mm_fallback(
                f"runtime{key}",
                f"torch._int_mm call failed ({type(exc).__name__}: {exc})",
                degraded=True,
            )
            return None

        return out2.reshape(*x.shape[:-1], self.out_features)

    def _report_transient_oom(self, exc: BaseException) -> None:
        """Report an allocation failure that must NOT latch the mode off.

        Transient and shape-specific: one oversized layer (a large batch, or a
        block-swap spike) must not condemn every other layer on this device for
        the process lifetime, and it is not evidence that the hardware or build
        is degraded -- so no ``quantization_fallback`` warning either. Printed
        once per layer shape so a layer that fails on every step cannot flood.
        """
        key = f"oom{self.in_features}x{self.out_features}"
        with _INT_MM_LOCK:
            first_time = key not in _INT_MM_REPORTED
            _INT_MM_REPORTED.add(key)
        if first_time:
            print(
                f"[Int8Linear] integer GEMM out of memory on a "
                f"{self.in_features}x{self.out_features} layer "
                f"({type(exc).__name__}); using the dequant path for these calls. "
                f"The mode stays enabled for every other layer."
            )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, int8=per-row"
        )


def swap_linears_to_int8(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> int:
    """Replace each ``nn.Linear`` with an INT8 saved weight by an ``Int8Linear``.

    Gated on BOTH ``<name>.weight_scale`` being present AND ``<name>.weight``
    being int8 -- the scale suffix alone is shared with the e4m3 format, so a
    mixed checkpoint must be able to run this and ``swap_linears_to_fp8`` in
    either order and have each take only its own layers. That symmetry needs the
    SAME dtype test on the fp8 side, which ``swap_linears_to_fp8`` /
    ``is_fp8_state_dict`` now carry; without it the order does matter, because a
    suffix-only fp8 swap claims int8 layers and copies integer codes into an
    e4m3 buffer silently. Returns the count.

    The DECLARED-semantics refusal runs on the top-level call only (``prefix``
    empty; the recursion below always passes a non-empty one), so a caller that
    reaches the swap without the census -- or a future one that adapts the scale
    SHAPE to make a Comfy file load -- still cannot install a rotated weight.
    """
    if not prefix:
        _refuse_unsupported_quant_semantics(state_dict)
    swapped = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}{name}"
        weight = state_dict.get(f"{child_prefix}.weight")
        if (
            isinstance(child, nn.Linear)
            and f"{child_prefix}{INT8_SCALE_SUFFIX}" in state_dict
            and weight is not None
            and weight.dtype is INT8_WEIGHT_DTYPE
        ):
            setattr(
                module,
                name,
                Int8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                ),
            )
            swapped += 1
        else:
            swapped += swap_linears_to_int8(
                child, state_dict, compute_dtype, prefix=f"{child_prefix}."
            )
    return swapped


def disable_int8_mm(module: nn.Module, *, label: str = "") -> int:
    """Forbid the W8A8 integer-GEMM path on every ``Int8Linear`` under ``module``.

    The dequant path is what a default-config user runs, so anything that must
    match it -- above all TRAINING, where the base function is what a LoRA is
    fitted against -- calls this on the modules it owns.

    Authoritative: it does not depend on grad mode (several ``@torch.no_grad()``
    helpers are shared between the inference and training call graphs) nor on
    ``SUSHI_INT8_MM`` (a training subprocess inherits the backend's environment).
    Idempotent; a no-op on a module with no ``Int8Linear``. Returns the count.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, Int8Linear):
            m._allow_int8_mm = False
            n += 1
    if n and label:
        print(f"[Int8Linear] {label}: W8A8 integer GEMM disabled on {n} layer(s) (dequant only)")
    return n
