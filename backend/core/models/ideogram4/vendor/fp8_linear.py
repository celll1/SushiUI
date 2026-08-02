"""Weight-only FP8 (e4m3) Linear support for Ideogram 4.

Independently implemented for SushiUI (no runtime dependency on the upstream
``ideogram4`` package). Ported from the Apache-2.0 reference loader
(ideogram-oss/ideogram4 ``quantized_loading.py``).

Linear weights are stored as float8 with a per-output-channel (per-row) float32
scale, halving the size of every quantized Linear weight.

Two forward paths exist. The dequantized matmul is the DEFAULT; the scaled GEMM
is opt-in behind ``SUSHI_FP8_SCALED_MM=1`` at import or ``set_scaled_mm_enabled``
at runtime (see ``_SCALED_MM_ENABLED``) and is
inference-only -- a module's owner declares that explicitly by calling
``disable_scaled_mm`` (every trainer-side loader does), because grad mode alone
cannot distinguish inference from training here.

* **W8A8 scaled GEMM** (``torch._scaled_mm``): the activation is dynamically
  quantized to e4m3 with a per-token (per-row) float32 scale and the matmul runs
  directly on the FP8 tensor cores, with the per-token and per-output-channel
  scales applied by the GEMM epilogue. Requires CUDA with FP8 tensor cores
  (compute capability >= 8.9) and shapes that satisfy ``_scaled_mm``'s
  constraints.
* **Dequantized matmul** (fallback): the weight is dequantized back to the
  compute dtype and a normal matmul runs. Works on any device that can store
  float8 (CPU included) and is used whenever the scaled GEMM is disabled
  (globally by env/API, or per-module via ``disable_scaled_mm``), unavailable,
  unsupported for the given shape/dtype, or grad mode is enabled (any training,
  whether or not this particular layer needs a gradient).

Path selection is probed once per (device, activation dtype) and cached; see
``_scaled_mm_mode``.

On a build that only accepts TENSORWISE scaling (this is what sm_89 offers), the
scaled GEMM's two elementwise stages -- the per-token activation quantization and
the epilogue that re-applies both scale vectors -- are served by single fused
Triton kernels from ``fp8_fused`` where they are available. They are gated on a
probe that verifies their output BITWISE against the eager chains
(``_eager_quantize_activation``, ``_eager_epilogue``), which remain the
definition of the result and the fallback for every case the kernels cannot
serve. See ``fp8_fused`` for why bitwise, and for the three places its kernels
deviate from the obvious formulation to stay that way.

Checkpoint layout (per quantized Linear ``<name>``):
    <name>.weight        float8_e4m3fn  (out, in)
    <name>.weight_scale  float32        (out,)
    <name>.bias          compute dtype  (out,)   [optional]

Dequantization: ``weight.to(dtype) * weight_scale[:, None]``.
"""

from __future__ import annotations

import os
import threading
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# Largest magnitude representable by the e4m3 float8 format. Per-row weight
# scales map each row's max abs value onto this so we use the full range.
FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"
# Marker written into the text encoder's config.json so the loader knows to take
# the custom weight-only FP8 path instead of transformers' from_pretrained.
FP8_TEXT_ENCODER_CONFIG_FLAG = "ideogram_fp8_weight_only"


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """True if the checkpoint carries weight-only FP8 Linear weights.

    Keyed on BOTH the ``.weight_scale`` sibling and an ``e4m3`` weight, because
    ``int8_linear`` deliberately uses the IDENTICAL scale suffix and only the
    weight dtype tells the two formats apart. The suffix alone would answer True
    for a pure INT8 checkpoint, and ``swap_linears_to_fp8`` would then claim its
    layers and ``copy_`` int8 codes into an e4m3 buffer -- silently, since the
    load itself succeeds.

    The second clause keeps a checkpoint whose e4m3 weights carry no scales
    (nothing this repo emits, but a valid weight-only file) detected as FP8;
    it cannot fire on an int8 checkpoint, which contains no e4m3 tensor.

    A MIXED int8/e4m3 checkpoint -- what the offline tool's per-layer selection
    produces -- answers True here and True to ``is_int8_state_dict``, and both
    swaps run in either order, each taking only its own layers.
    """
    for key in state_dict:
        if not key.endswith(FP8_SCALE_SUFFIX):
            continue
        weight = state_dict.get(key[: -len(FP8_SCALE_SUFFIX)] + ".weight")
        if weight is not None and weight.dtype == FP8_WEIGHT_DTYPE:
            return True
    return any(v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values())


def quantize_weight_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D Linear weight to e4m3 float8 with per-row scales.

    Returns ``(weight_fp8, scale)`` where ``weight_fp8`` has shape ``(out, in)``
    in ``float8_e4m3fn`` and ``scale`` has shape ``(out,)`` in float32 such that
    ``weight ~= weight_fp8.to(dtype) * scale[:, None]``.
    """
    w = weight.detach().to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    q = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
    return q, scale.squeeze(1).to(torch.float32)


# ---------------------------------------------------------------------------
# W8A8 scaled-GEMM fast path (torch._scaled_mm)
# ---------------------------------------------------------------------------
#
# Constraints taken from the installed torch's own checker,
# ``torch/_meta_registrations.py::_check_scaled_mm_sizes`` (torch 2.10):
#   * both operands 2-D and of an fp8/fp4 dtype
#   * ``self`` row-major, ``mat2`` column-major
#   * ``self.size(1) % 16 == 0`` and both dims of ``mat2`` divisible by 16
#   * tensorwise scaling: both scales are float32 with ``numel() == 1``
#   * rowwise scaling: both scales are float32 and 2-D, ``scale_a`` is ``(m, 1)``
#     and ``scale_b`` is ``(1, n)``, both contiguous
# A weight stored ``(out, in)`` row-major contiguous gives a column-major
# ``(in, out)`` operand for free via ``.t()``.
#
# Rowwise scaling has historically been gated by torch version / GPU
# architecture, so the usable mode is probed once per (device, out dtype) and
# cached instead of being retried on every forward.

# Smallest activation scale we will emit. Guards all-zero rows: without it the
# scale would be 0 and the reciprocal used for quantization would be inf/NaN.
_MIN_ACT_SCALE = 1e-12

# Alignment required by _scaled_mm on both GEMM operand dimensions.
_SCALED_MM_ALIGN = 16

# Minimum compute capability with FP8 tensor cores (Ada / sm_89).
_FP8_MIN_CAPABILITY = (8, 9)

# Fast accumulation keeps the FP8 MMA accumulator in reduced precision inside a
# tile instead of promoting every k-block to fp32. Inference-only here (the
# fast path is skipped whenever gradients must flow). Set the env var to "0" to
# force the fully-promoted accumulation.
_USE_FAST_ACCUM = os.environ.get("SUSHI_FP8_FAST_ACCUM", "1") != "0"

# The W8A8 scaled-GEMM path is OPT-IN: set SUSHI_FP8_SCALED_MM=1 to enable it.
#
# It is off by default because its cost is measured and its benefit is not. The
# extra activation quantization raises the error against an fp32 reference from
# ~2.6e-02 (dequant path) to ~3.7e-02 rel RMS, i.e. ~44% more error, on both
# Ideogram 4 (transformer and text encoder) and Krea 2. No throughput number has
# been produced on this hardware yet; until the measurement gate in
# ``examples/api/bench_fp8_scaled_mm.py`` passes, the dequant path -- which is
# what users already run -- stays the default.
#
# Initialized from the environment at import, then MUTABLE for the life of the
# process via ``set_scaled_mm_enabled`` (exposed as
# ``GET/POST /api/v1/system/fp8-scaled-mm``). Not persisted: every process starts
# from the environment again, and default-off is the correct state for a path
# whose benefit is unmeasured.
_SCALED_MM_ENABLED = os.environ.get("SUSHI_FP8_SCALED_MM", "0") == "1"

# Where the CURRENT value came from: "env" (the variable was present at import),
# "default" (it was not), or "api" (a later set_scaled_mm_enabled call). Must
# stay in lockstep with the `Fp8ScaledMmState.origin` enum in openapi.yaml
# ([default, env, api]) -- see the validation in set_scaled_mm_enabled().
_SCALED_MM_VALID_ORIGINS = frozenset({"default", "env", "api"})
_SCALED_MM_ORIGIN = "env" if "SUSHI_FP8_SCALED_MM" in os.environ else "default"

# (device index, activation dtype) -> mode string, or None for "unusable".
#   "rowwise_bias" : rowwise scales, bias fused into the GEMM epilogue
#   "rowwise"      : rowwise scales, bias added afterwards
#   "tensorwise"   : unit scalar scales, both scales applied after the GEMM
_SCALED_MM_MODE: dict[tuple[int, torch.dtype], str | None] = {}
_SCALED_MM_LOCK = threading.Lock()
_SCALED_MM_REPORTED: set[str] = set()


def _mode_label(key: tuple[int, torch.dtype]) -> str:
    """JSON-friendly label for a ``_SCALED_MM_MODE`` key."""
    index, dtype = key
    device = f"cuda:{index}" if index >= 0 else "cuda:default"
    return f"{device}/{str(dtype).rsplit('.', 1)[-1]}"


def get_scaled_mm_state() -> dict:
    """Current W8A8 scaled-GEMM state.

    ``resolved_modes`` exposes the per-(device, activation dtype) probe results
    so the case "flag on, but the probe latched None and every layer runs the
    dequant path" is visible rather than inferred. An empty dict means no probe
    has run yet in this process (no FP8 Linear forward has reached the probe).
    """
    with _SCALED_MM_LOCK:
        return {
            "enabled": bool(_SCALED_MM_ENABLED),
            "origin": _SCALED_MM_ORIGIN,
            "resolved_modes": {_mode_label(k): v for k, v in _SCALED_MM_MODE.items()},
        }


def set_scaled_mm_enabled(enabled: bool, *, origin: str = "api") -> dict:
    """Turn the W8A8 scaled-GEMM path on or off for THIS PROCESS.

    Both directions clear the probe cache (``_SCALED_MM_MODE``) and the one-shot
    report set (``_SCALED_MM_REPORTED``), so the next FP8 forward re-probes. That
    also un-latches a key that a transient failure had condemned for the process.

    Scope and limits:

    * Per-process, not per-generation, and not persisted across restarts.
    * Does NOT clear ``fp8_fused``'s own latch. That one records whether this
      machine's triton toolchain can compile and run the fused kernels
      bitwise-identically -- a property of the install, not of which GEMM path is
      selected -- and its fallback changes no result, so re-probing it on every
      flip would only re-pay the compile.
    * Does NOT override ``disable_scaled_mm``: that per-module opt-out is the
      authoritative gate (every trainer-side loader calls it), and enabling the
      path here cannot resurrect a module that declared itself dequant-only.
    * ``Fp8Linear.forward`` branches on this module global, so a ``torch.compile``
      graph containing an ``Fp8Linear`` would bake the value in as a constant and
      keep running the path that was active at trace time. This is safe today
      only because nothing compiles these layers: ``use_torch_compile`` is wired
      solely into the SD1.5/SDXL U-Net staging path
      (``core/vram_optimization.move_unet_to_gpu``) and is listed as unsupported
      for every DiT arch in ``api/arch_capabilities.py``, while ``Fp8Linear``
      exists only on Ideogram 4 and Krea 2. The trainer's block-level
      ``torch.compile`` runs in a separate process that is hard-off. If a future
      change compiles an arch that owns ``Fp8Linear`` modules, this setter must
      grow a refusal for that case.

    ``origin`` must be one of ``_SCALED_MM_VALID_ORIGINS`` (kept identical to
    the ``Fp8ScaledMmState.origin`` enum in ``openapi.yaml``): raises
    ``ValueError`` otherwise, so a caller cannot silently drift the two apart
    by passing an ad-hoc string that only ``get_scaled_mm_state()`` would ever
    have surfaced as a mismatch against the documented contract.

    Returns the same dict as ``get_scaled_mm_state()``.
    """
    global _SCALED_MM_ENABLED, _SCALED_MM_ORIGIN
    enabled = bool(enabled)
    if origin not in _SCALED_MM_VALID_ORIGINS:
        raise ValueError(
            f"invalid origin {origin!r}: must be one of {sorted(_SCALED_MM_VALID_ORIGINS)} "
            f"(see the Fp8ScaledMmState.origin enum in openapi.yaml)"
        )
    with _SCALED_MM_LOCK:
        changed = enabled != _SCALED_MM_ENABLED
        _SCALED_MM_ENABLED = enabled
        _SCALED_MM_ORIGIN = origin
        # Cleared on EVERY flip, in both directions: the cached mode (and the
        # "already reported" marker) describe a probe taken under the previous
        # setting, and a latched None must not survive a re-enable.
        _SCALED_MM_MODE.clear()
        _SCALED_MM_REPORTED.clear()
        state = {
            "enabled": _SCALED_MM_ENABLED,
            "origin": _SCALED_MM_ORIGIN,
            "resolved_modes": {},
        }
    print(
        f"[Fp8Linear] W8A8 scaled-GEMM path "
        f"{'ENABLED' if enabled else 'DISABLED'} (origin={origin}"
        f"{'' if changed else ', unchanged'}); probe cache cleared. "
        f"{'FP8 Linear layers will run torch._scaled_mm where the probe accepts a scaling mode.' if enabled else 'FP8 Linear layers run the dequantized matmul.'}"
    )
    return state


def describe_gemm_path(module: nn.Module | None = None) -> str | None:
    """Describe which FP8 GEMM path is in force, for metadata/provenance.

    Returns None when ``module`` is given and owns no ``Fp8Linear`` (nothing to
    record). Otherwise one of:

    * ``"w8a8_scaled_mm(<mode>)"`` -- the flag is on, the module allows it and the
      probe resolved a usable scaling mode. A ``+fused`` suffix on the mode
      (e.g. ``"w8a8_scaled_mm(tensorwise+fused)"``) records that the fused Triton
      epilogue/quantizer served at least one key in this process. That does NOT
      change the pixels -- the fused kernels are gated on a bitwise-equality
      probe against the eager chain -- but it does change what ran, and a
      toolchain that latched the kernels off is exactly the kind of thing a
      provenance label should not hide.
    * ``"dequant"`` -- the flag is off, or every owned layer opted out.
    * ``"dequant(scaled_mm unavailable)"`` -- the flag is on but every probed
      key latched None.
    * ``"dequant(scaled_mm unprobed)"`` -- the flag is on but no FP8 forward has
      reached the probe in this process, so no layer has run the scaled GEMM.

    Derived from state only; it adds nothing to any per-forward path.

    LIMITATIONS: ``_SCALED_MM_MODE`` is a process-wide cache keyed on (device,
    activation dtype), not scoped to ``module`` or to one generation, so this
    aggregates over every key ever probed in this process -- a mode latched by
    an earlier, unrelated model's forward can still be reported here even if
    ``module``'s own layers latched ``None`` on their own key (mixed devices or
    activation dtypes). It also cannot see a per-layer runtime fallback within
    THIS generation: ``_scaled_mm_forward``'s OOM/allocation-failure path
    (``_report_transient_oom``) intentionally does not touch
    ``_SCALED_MM_MODE``, so a generation where every layer fell back to dequant
    for a transient reason is still labelled with whatever mode is cached, not
    ``"dequant"``.
    """
    if module is not None:
        layers = [m for m in module.modules() if isinstance(m, Fp8Linear)]
        if not layers:
            return None
        if not any(m._allow_scaled_mm for m in layers):
            return "dequant"
    with _SCALED_MM_LOCK:
        if not _SCALED_MM_ENABLED:
            return "dequant"
        modes = sorted({m for m in _SCALED_MM_MODE.values() if m})
        probed = bool(_SCALED_MM_MODE)
    if modes:
        return f"w8a8_scaled_mm({'+'.join(modes)}{_fused_suffix()})"
    return "dequant(scaled_mm unavailable)" if probed else "dequant(scaled_mm unprobed)"


def _fused_suffix() -> str:
    """``"+fused"`` when the fused Triton kernels served a key in this process.

    Derived from state only (no import side effects that matter, no probe is
    forced): if ``fp8_fused`` has not been imported, or every key latched off, the
    label stays exactly what it was before the fused kernels existed.
    """
    try:
        from .fp8_fused import fused_enabled

        return "+fused" if fused_enabled() else ""
    except Exception:
        return ""


def _report_scaled_mm_fallback(key: str, reason: str, *, degraded: bool) -> None:
    """Log the fallback, and for unexpected failures surface it to the user.

    ``degraded=True`` means the fast path failed unexpectedly rather than simply
    being unsupported on this hardware; only that case is worth putting on the
    generation's warning channel.

    Both the console print and the user-facing warning fire exactly ONCE PER
    (device, activation dtype) -- not once per process: the print is one-shot
    per ``key``, and every degraded call site latches
    ``_SCALED_MM_MODE[key] = None`` at or before reporting (probe failure,
    runtime failure), so the next forward short-circuits on ``mode is None`` and
    never reaches this function again for that key. A second GPU, or a second
    activation dtype on the same GPU, gets its own key and therefore its own
    single report.

    That is deliberate. Re-filing the warning on the ``mode is None`` early
    return would put an ``add_warning`` call (plus an import) on the hot path of
    every Linear forward of every generation, to restate a fact that does not
    change for the life of the process. The one warning plus the console line
    are the signal; the fallback itself is safe (it is the default path).
    """
    message = f"FP8 W8A8 via torch._scaled_mm unavailable, falling back to dequant path: {reason}"
    with _SCALED_MM_LOCK:
        first_time = key not in _SCALED_MM_REPORTED
        _SCALED_MM_REPORTED.add(key)
    if first_time:
        print(f"[Fp8Linear] {message}")
    if not degraded:
        # Plain capability miss (no FP8 tensor cores, no torch._scaled_mm, ...).
        # Expected on most hardware: print-only, never a user-facing warning.
        return
    try:
        from api.generation_status import add_warning

        add_warning(message, code="quantization_fallback")
    except Exception:
        pass


def _probe_scaled_mm(device: torch.device, out_dtype: torch.dtype) -> str | None:
    """Run one tiny scaled GEMM to find which scaling mode this build accepts.

    Uses 16x16 operands, so the probe costs no meaningful VRAM or time. Returns
    the mode string, or None if no variant works.
    """
    if not _SCALED_MM_ENABLED:
        # The default state, not a degradation: stay silent. Printing here would
        # put a "falling back" line on every model load for every user.
        return None
    if not hasattr(torch, "_scaled_mm"):
        _report_scaled_mm_fallback("missing", "torch._scaled_mm is not available", degraded=False)
        return None
    if device.type != "cuda":
        return None
    try:
        capability = torch.cuda.get_device_capability(device)
    except Exception as exc:  # pragma: no cover - driver-level failure
        _report_scaled_mm_fallback("capability", f"could not query compute capability ({exc})", degraded=False)
        return None
    if capability < _FP8_MIN_CAPABILITY:
        _report_scaled_mm_fallback(
            f"capability{capability}",
            f"compute capability {capability[0]}.{capability[1]} has no FP8 tensor cores",
            degraded=False,
        )
        return None

    n = _SCALED_MM_ALIGN
    a = torch.zeros(n, n, dtype=FP8_WEIGHT_DTYPE, device=device)
    # (out, in) row-major -> .t() is the column-major operand _scaled_mm wants.
    b = torch.zeros(n, n, dtype=FP8_WEIGHT_DTYPE, device=device).t()
    row_a = torch.ones(n, 1, dtype=torch.float32, device=device)
    row_b = torch.ones(1, n, dtype=torch.float32, device=device)
    one = torch.ones((), dtype=torch.float32, device=device)
    bias = torch.zeros(n, dtype=out_dtype, device=device)

    # The tensorwise probe must use the same out_dtype the forward will ask for
    # (see Fp8Linear._scaled_mm_forward): always float32, because the per-token
    # and per-output-channel scales are applied in float32 afterwards.
    tensorwise_out = torch.float32
    attempts: list[tuple[str, dict]] = [
        ("rowwise_bias", {"scale_a": row_a, "scale_b": row_b, "bias": bias, "out_dtype": out_dtype}),
        ("rowwise", {"scale_a": row_a, "scale_b": row_b, "out_dtype": out_dtype}),
        ("tensorwise", {"scale_a": one, "scale_b": one, "out_dtype": tensorwise_out}),
    ]
    errors: list[str] = []
    for mode, kwargs in attempts:
        try:
            torch._scaled_mm(a, b, use_fast_accum=_USE_FAST_ACCUM, **kwargs)
            return mode
        except Exception as exc:
            errors.append(f"{mode}: {type(exc).__name__}: {exc}")
    device_key = device.index if device.index is not None else -1
    _report_scaled_mm_fallback(
        f"probe{device_key}{out_dtype}",
        f"no supported scaling mode for {out_dtype} ({' | '.join(errors)})",
        degraded=True,
    )
    return None


def _scaled_mm_mode(device: torch.device, out_dtype: torch.dtype) -> str | None:
    """Cached ``_probe_scaled_mm`` keyed on (device index, activation dtype)."""
    key = (device.index if device.index is not None else -1, out_dtype)
    try:
        return _SCALED_MM_MODE[key]
    except KeyError:
        pass
    mode = _probe_scaled_mm(device, out_dtype)
    with _SCALED_MM_LOCK:
        return _SCALED_MM_MODE.setdefault(key, mode)


def _quantize_activation(x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic e4m3 quantization of a 2-D activation.

    Returns ``(x_fp8, scale)`` with ``scale`` of shape ``(m, 1)`` float32 such
    that ``x2 ~= x_fp8.float() * scale``. Rows that are entirely zero get the
    floor scale rather than zero, so the reciprocal stays finite.

    Prefers the single fused Triton kernel in ``fp8_fused`` (~8 kernels and
    ~13 B/element of traffic collapse to 1 and 3), which is gated on an EXECUTED
    probe that checks its output BITWISE against ``_eager_quantize_activation``
    below. Anything that is not usable falls through to the eager chain, which
    stays the definition of the result.
    """
    fused = _try_fused_quantize(x2)
    if fused is not None:
        return fused
    return _eager_quantize_activation(x2)


def _try_fused_quantize(x2: torch.Tensor):
    """``fp8_fused.fused_quantize`` if importable/usable, else None."""
    try:
        from .fp8_fused import fused_quantize
    except Exception:
        return None
    return fused_quantize(x2)


def _eager_quantize_activation(x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference (and fallback) implementation of ``_quantize_activation``.

    Also the bitwise target the fused kernel's probe compares against, so this
    body -- including WHERE each rounding happens -- is the contract, not an
    implementation detail.
    """
    amax = x2.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (amax / FP8_E4M3_MAX).clamp_(min=_MIN_ACT_SCALE)
    recip = scale.reciprocal()
    if x2.dtype is torch.bfloat16:
        # bfloat16 covers the full float32 exponent range, so the reciprocal
        # never overflows and the product carries more mantissa than e4m3 keeps.
        scaled = x2 * recip.to(torch.bfloat16)
    elif x2.dtype is torch.float32:
        scaled = x2 * recip
    else:
        # float16 would overflow on the reciprocal of a very small amax.
        scaled = x2.to(torch.float32) * recip
    return scaled.clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE), scale


# Allocation failures that do not arrive as ``torch.cuda.OutOfMemoryError``.
# cuBLAS reports a workspace shortage as a plain RuntimeError; it is transient in
# the same way an OOM is, so it must not latch the mode off for the process.
_ALLOC_FAILURE_MARKERS = ("CUBLAS_STATUS_ALLOC_FAILED", "out of memory", "CUDA_ERROR_OUT_OF_MEMORY")


def _eager_epilogue(
    acc: torch.Tensor,
    a_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``((acc * a_scale) * w_scale [+ bias]).to(out_dtype)``, in place on ``acc``.

    THE DEFINITION of the tensorwise decomposition's epilogue, and the bitwise
    target the fused kernel's probe compares against. Every property here is
    load-bearing:

    * both scales are applied in float32 and the ASSOCIATION is
      ``(acc * a_scale) * w_scale`` -- not ``acc * (a_scale * w_scale)``;
    * the bias is widened to float32 and added as a separate rounded step (a
      fused multiply-add would round once, which is more accurate and therefore
      a different result);
    * the output dtype is reached by exactly ONE rounding, at the end. Scaling in
      the narrower activation dtype instead would save the float32 ``(m, n)``
      temporary but measured 4.31e-03 rel RMS worse on bf16.

    ``acc`` is MUTATED: callers pass a fresh ``_scaled_mm`` output with no other
    reference, which keeps the transient at 1.5x the accumulator instead of 3x
    (measured at m=n=2048: 24 MiB vs 48 MiB peak delta over a 16 MiB
    accumulator; tmp/fp8_scaled_mm_memory.py).
    """
    acc.mul_(a_scale).mul_(w_scale)
    if bias is not None:
        acc.add_(bias.to(torch.float32))
    return acc.to(out_dtype)


def _try_fused_epilogue(acc, a_scale, w_scale, bias, out_dtype):
    """``fp8_fused.fused_epilogue`` if importable/usable, else None."""
    try:
        from .fp8_fused import fused_epilogue
    except Exception:
        return None
    return fused_epilogue(acc, a_scale, w_scale, bias, out_dtype)


# Cached float32 scalar 1.0 per device: the tensorwise GEMM needs unit scales on
# every call, and allocating two 4-byte tensors per Linear forward is pure
# allocator churn on a path whose whole point is to remove per-call overhead.
_ONE_CACHE: dict[torch.device, torch.Tensor] = {}


def _scalar_one(device: torch.device) -> torch.Tensor:
    one = _ONE_CACHE.get(device)
    if one is None:
        one = torch.ones((), dtype=torch.float32, device=device)
        # Benign race: two threads may each build one; both are the value 1.0 and
        # neither is ever mutated, so whichever wins the dict is equivalent.
        _ONE_CACHE[device] = one
    return one


def _is_allocation_failure(exc: BaseException) -> bool:
    try:
        message = str(exc)
    except Exception:
        # An exception whose own __str__ raises must not propagate out of the
        # except block in _scaled_mm_forward that calls this helper.
        return False
    return any(marker in message for marker in _ALLOC_FAILURE_MARKERS)


class Fp8Linear(nn.Module):
    """Linear layer holding an e4m3 float8 weight + per-row float32 scale.

    The weight and scale are registered as buffers (not parameters) so they load
    via ``load_state_dict`` and are excluded from optimizer/grad machinery.

    ``forward`` tries the W8A8 scaled GEMM first (``_scaled_mm_forward``) and
    falls back to the dequantized matmul (``_dequant_forward``, which runs in the
    activation's dtype) whenever the former is not usable.

    ``_allow_scaled_mm`` is the explicit per-module opt-out (see
    ``disable_scaled_mm``). It is a CLASS attribute so it costs nothing per
    instance, is not a buffer/parameter (so it never touches ``state_dict``), and
    is set per instance only where a caller has opted out.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    # Owner-level kill switch for the W8A8 fast path. See disable_scaled_mm().
    _allow_scaled_mm: bool = True

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
            torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._scaled_mm_forward(x)
        if out is not None:
            return out
        return self._dequant_forward(x)

    def _dequant_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize the weight to the compute dtype and run a normal matmul."""
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def _scaled_mm_forward(self, x: torch.Tensor) -> torch.Tensor | None:
        """FP8 W8A8 matmul on the tensor cores, or None if it is not usable here.

        Returning None (rather than raising) lets ``forward`` fall through to the
        dequant path for every case the scaled GEMM cannot serve.
        """
        # GATE 0 (cheapest first): the feature is off entirely for this process
        # (the default). Every other gate below still runs correctly when this
        # check is skipped -- ``_scaled_mm_mode`` calls ``_probe_scaled_mm``,
        # which itself checks ``_SCALED_MM_ENABLED`` and returns None -- so this
        # is a pure short-circuit, not a behavior change: it just avoids paying
        # gates 1-8 (module ownership check, grad-mode check, device/dtype/shape
        # checks, and the probe/cache lookup) on every Linear forward of every
        # generation for users who never set ``SUSHI_FP8_SCALED_MM=1``.
        if not _SCALED_MM_ENABLED:
            return None
        w = self.weight
        # GATE 1 (authoritative): the module's owner may forbid the fast path
        # outright. Grad mode is NOT a usable proxy for "this is inference":
        # several no_grad-decorated helpers are shared by the inference and the
        # TRAINING call graphs -- ``encode_text_layers``
        # (ideogram4_pipeline_ops.py) is @torch.no_grad() and is reached both from
        # the pipeline and from ``training/ops/ideogram4_ops.encode_prompt``, and
        # a training subprocess inherits ``SUSHI_FP8_SCALED_MM`` from the backend
        # via ``training_process.py``'s ``os.environ.copy()``. Without this gate an
        # operator who enabled the fast path for inference speed would silently
        # fit every LoRA against W8A8 conditioning (~2.7e-02 rel RMS noisier than
        # the dequant path everyone else runs). The trainer-side loaders call
        # ``disable_scaled_mm`` on every module they own, so a training process is
        # dequant-only regardless of env or grad mode.
        if not self._allow_scaled_mm:
            return None
        # GATE 2 (defence in depth): never run W8A8 where a gradient could flow.
        # Checking only ``x.requires_grad`` would not be enough on its own, since
        # every Fp8Linear sitting BEFORE the first LoRA contribution in the graph
        # (and every layer in a branch with no LoRA at all) receives an input with
        # requires_grad=False.
        if x.requires_grad or torch.is_grad_enabled():
            return None
        if not x.is_cuda or w.device != x.device or w.dtype is not FP8_WEIGHT_DTYPE:
            return None
        if self.in_features % _SCALED_MM_ALIGN or self.out_features % _SCALED_MM_ALIGN:
            return None
        if not w.is_contiguous():
            return None
        if x.shape[-1] != self.in_features or x.numel() == 0:
            return None

        # Everything that only the fast path does lives inside the try, including
        # the mode probe itself. ``_scaled_mm_mode`` allocates real tensors on its
        # first call for a given (device, dtype) key (see ``_probe_scaled_mm``),
        # so it can OOM exactly like the GEMM calls below -- and with the toggle
        # (``set_scaled_mm_enabled``) clearing ``_SCALED_MM_MODE`` on every flip,
        # that first call can now land inside somebody's forward pass instead of
        # only once at process start. Catching it here means a transient OOM
        # during the probe falls through the same ``except`` branches as every
        # other allocation on this path, and -- because the exception propagates
        # out of ``_scaled_mm_mode`` before it reaches
        # ``_SCALED_MM_MODE.setdefault`` -- the mode is never cached: the next
        # forward call re-probes instead of being latched off by a false
        # negative that was really just memory pressure. The activation
        # quantization below has the same property: it allocates a float32
        # (m, k) temporary for fp16/fp32 inputs, so it can OOM where the dequant
        # path would not, and an exception there must fall back rather than
        # escape forward().
        try:
            mode = _scaled_mm_mode(x.device, x.dtype)
            if mode is None:
                return None

            x2 = x.reshape(-1, self.in_features)
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            x_fp8, a_scale = _quantize_activation(x2)

            w_scale = self.weight_scale
            if w_scale.dtype is not torch.float32:
                w_scale = w_scale.to(torch.float32)
            w_scale = w_scale.reshape(1, self.out_features)
            if not w_scale.is_contiguous():
                w_scale = w_scale.contiguous()
            # (out, in) row-major -> (in, out) column-major, which is the layout
            # _scaled_mm requires for the second operand. No copy.
            w_t = w.t()

            # UNEXERCISED: the two rowwise branches below have never run on the
            # only hardware this was tested on (sm_89 / Ada), where the probe only
            # ever accepts "tensorwise". They were checked by inspection against
            # torch's own _check_scaled_mm_sizes and are internally consistent
            # (and "rowwise_bias" with bias=None degrades harmlessly to
            # "rowwise"), but they carry NO measured evidence. Anyone enabling
            # them -- a Blackwell GPU, or a torch upgrade that widens rowwise
            # support -- must re-run examples/api/bench_fp8_scaled_mm.py and the
            # numerics check rather than trusting this code.
            if mode == "rowwise_bias":
                out2 = torch._scaled_mm(
                    x_fp8,
                    w_t,
                    scale_a=a_scale,
                    scale_b=w_scale,
                    bias=self.bias.to(x.dtype) if self.bias is not None else None,
                    out_dtype=x.dtype,
                    use_fast_accum=_USE_FAST_ACCUM,
                )
            elif mode == "rowwise":
                out2 = torch._scaled_mm(
                    x_fp8,
                    w_t,
                    scale_a=a_scale,
                    scale_b=w_scale,
                    out_dtype=x.dtype,
                    use_fast_accum=_USE_FAST_ACCUM,
                )
                if self.bias is not None:
                    out2 = out2 + self.bias.to(x.dtype)
            else:
                # Tensorwise-only build (this is what sm_89 offers): the GEMM
                # takes scalar scales, so the per-token and per-output-channel
                # scales are applied afterwards.
                #
                # This is NOT numerically identical to a fused rowwise epilogue,
                # which scales the fp32 accumulator once and rounds once. The
                # closest we can get with scalar scales is to take the GEMM out in
                # float32, apply both scale vectors in float32, and round to the
                # activation dtype exactly once at the end. That costs one float32
                # (m, n) temporary; scaling in the (narrower) activation dtype
                # instead would save it but add three roundings, which measured
                # 4.31e-03 rel RMS worse than this decomposition on bf16.
                one = _scalar_one(x.device)
                acc = torch._scaled_mm(
                    x_fp8,
                    w_t,
                    scale_a=one,
                    scale_b=one,
                    out_dtype=torch.float32,
                    use_fast_accum=_USE_FAST_ACCUM,
                )
                # ONE fused Triton kernel where it is available (one read of the
                # accumulator, one write of the output -- the eager chain is 4
                # kernels and ~5x the traffic over the same (m, n) buffer), else
                # the eager chain. The two are bitwise-identical by construction
                # and the fused path is gated on a probe that verifies exactly
                # that; see fp8_fused.py. Both leave the transient at 1.5x the
                # accumulator: the eager form scales in place and allocates only
                # the narrower output, the fused form allocates the same output
                # while the accumulator is still live.
                bias = self.bias
                out2 = _try_fused_epilogue(acc, a_scale, w_scale, bias, x.dtype)
                if out2 is None:
                    out2 = _eager_epilogue(acc, a_scale, w_scale, bias, x.dtype)
        except torch.cuda.OutOfMemoryError as exc:
            self._report_transient_oom(exc)
            return None
        except Exception as exc:
            if _is_allocation_failure(exc):
                # e.g. RuntimeError: CUBLAS_STATUS_ALLOC_FAILED -- cuBLAS could not
                # get its workspace. Same nature as an OOM (transient, pressure-
                # dependent) but it does not arrive as OutOfMemoryError, so it
                # would otherwise latch the mode off for the whole process.
                self._report_transient_oom(exc)
                return None
            # Anything else (unsupported shape/dtype combination, a kernel or
            # driver error) is a property of this configuration, not of one call:
            # latch it so we do not pay a failing call on every forward.
            key = (x.device.index if x.device.index is not None else -1, x.dtype)
            with _SCALED_MM_LOCK:
                _SCALED_MM_MODE[key] = None
            _report_scaled_mm_fallback(
                f"runtime{key}",
                f"{mode} call failed ({type(exc).__name__}: {exc})",
                degraded=True,
            )
            return None

        return out2.reshape(*x.shape[:-1], self.out_features)

    def _report_transient_oom(self, exc: BaseException) -> None:
        """Report an allocation failure that must NOT latch the mode off.

        Transient and shape-specific: one oversized layer (a large batch, or a
        block-swap spike) must not condemn every other layer on this device for
        the process lifetime, and it is not evidence that the hardware or the
        build is degraded -- so no ``quantization_fallback`` warning either.
        Printed once per layer shape so a layer that fails on every step cannot
        flood the log.
        """
        key = f"oom{self.in_features}x{self.out_features}"
        with _SCALED_MM_LOCK:
            first_time = key not in _SCALED_MM_REPORTED
            _SCALED_MM_REPORTED.add(key)
        if first_time:
            print(
                f"[Fp8Linear] scaled-GEMM out of memory on a "
                f"{self.in_features}x{self.out_features} layer "
                f"({type(exc).__name__}); using the dequant path for these "
                f"calls. The mode stays enabled for every other layer."
            )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, fp8=e4m3"
        )


def swap_linears_to_fp8(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> int:
    """Replace each ``nn.Linear`` with an e4m3 saved weight by an ``Fp8Linear``.

    Gated on BOTH ``<name>.weight_scale`` being present AND ``<name>.weight``
    being ``float8_e4m3fn``. The scale suffix alone is NOT sufficient: it is
    shared with ``int8_linear``'s format, and without the dtype test this would
    claim int8 layers and ``copy_`` their integer codes into an e4m3 buffer
    without raising. With the dtype test, a mixed checkpoint can run this and
    ``swap_linears_to_int8`` in either order and each takes only its own layers.
    Everything else loads normally in the compute dtype. Returns the count.
    """
    swapped = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}{name}"
        weight = state_dict.get(f"{child_prefix}.weight")
        if (
            isinstance(child, nn.Linear)
            and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict
            and weight is not None
            and weight.dtype == FP8_WEIGHT_DTYPE
        ):
            setattr(
                module,
                name,
                Fp8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                ),
            )
            swapped += 1
        else:
            swapped += swap_linears_to_fp8(
                child, state_dict, compute_dtype, prefix=f"{child_prefix}."
            )
    return swapped


def disable_scaled_mm(module: nn.Module, *, label: str = "") -> int:
    """Forbid the W8A8 scaled-GEMM fast path on every ``Fp8Linear`` under ``module``.

    The dequant path is what every published checkpoint and every default-config
    user runs, so anything that must match it bit-for-bit-ish -- above all
    TRAINING, where the base function is what a LoRA is fitted against -- calls
    this on the modules it owns.

    This is the authoritative gate: it does not depend on grad mode (several
    ``@torch.no_grad()`` helpers are shared between the inference and training
    call graphs) nor on ``SUSHI_FP8_SCALED_MM`` (a training subprocess inherits
    the backend's environment). Idempotent; a no-op on a module with no
    ``Fp8Linear``. Returns the number of layers switched off.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, Fp8Linear):
            m._allow_scaled_mm = False
            n += 1
    if n and label:
        print(f"[Fp8Linear] {label}: W8A8 scaled-GEMM disabled on {n} layer(s) (dequant only)")
    return n


_BNB_SIBLING_SUFFIXES = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
)


def is_bnb4bit_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """True if the checkpoint carries bitsandbytes 4-bit (nf4) quantized weights."""
    return any(".quant_state.bitsandbytes__" in k for k in state_dict)


def swap_linears_to_bnb4bit(
    module: nn.Module,
    compute_dtype: torch.dtype,
    *,
    quant_type: str = "nf4",
    compress_statistics: bool = False,
) -> int:
    """Replace every ``nn.Linear`` with a bitsandbytes ``Linear4bit``. Returns the count."""
    import bitsandbytes as bnb

    swapped = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    compress_statistics=compress_statistics,
                    quant_type=quant_type,
                ),
            )
            swapped += 1
        else:
            swapped += swap_linears_to_bnb4bit(
                child, compute_dtype, quant_type=quant_type, compress_statistics=compress_statistics
            )
    return swapped


def load_bnb4bit_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Load a bitsandbytes 4-bit (nf4) checkpoint into ``model`` (requires CUDA)."""
    import bitsandbytes as bnb

    consumed: set = set()
    for full_name, tensor in state_dict.items():
        if ".quant_state." in full_name or full_name.endswith(_BNB_SIBLING_SUFFIXES):
            continue
        parent_path, _, param_name = full_name.rpartition(".")
        try:
            parent = model.get_submodule(parent_path) if parent_path else model
        except AttributeError:
            continue
        current = parent._parameters.get(param_name)
        if not isinstance(current, bnb.nn.Params4bit):
            continue
        prefix = full_name + "."
        quantized_stats = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
        consumed.add(full_name)
        consumed.update(quantized_stats.keys())
        parent._parameters[param_name] = bnb.nn.Params4bit.from_prequantized(
            data=tensor,
            quantized_stats=quantized_stats,
            requires_grad=False,
            device=device,
        )

    remaining = {k: v for k, v in state_dict.items() if k not in consumed}
    for k in list(remaining):
        if remaining[k].is_floating_point():
            remaining[k] = remaining[k].to(device=device, dtype=dtype)
        else:
            remaining[k] = remaining[k].to(device=device)

    missing, unexpected = model.load_state_dict(remaining, strict=False)
    real_missing = [m for m in missing if m not in consumed]
    if real_missing:
        raise RuntimeError(f"missing keys after bnb4bit load: {real_missing[:10]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys after bnb4bit load: {unexpected[:10]}")

    for p in model.parameters():
        if isinstance(p, bnb.nn.Params4bit):
            continue
        if p.is_floating_point() and p.dtype != dtype:
            p.data = p.data.to(dtype=dtype)
        if p.device != device:
            p.data = p.data.to(device=device)
    model.to(device)


def load_fp8_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    *,
    assign: bool = False,
    strict: bool = True,
) -> None:
    """Load a weight-only FP8 checkpoint into ``model``.

    ``model`` must already have its FP8 Linear layers swapped in (see
    ``swap_linears_to_fp8``). FP8 weights are kept as float8, scales stay float32,
    and every other floating tensor is cast to ``dtype``.

    ``assign=True`` replaces the module's tensors with the prepared ones rather
    than copying into them. Use it when the model was built with ``from_config`` so
    the non-quantized params take the loaded dtype directly and computed
    non-persistent buffers (e.g. rotary caches) are left untouched.

    ``strict=False`` downgrades missing keys to a warning (e.g. tied weights that a
    ``transformers`` model resolves itself); unexpected keys always raise.
    """
    prepared: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if v.dtype == FP8_WEIGHT_DTYPE:
            prepared[k] = v.to(device=device)
        elif k.endswith(FP8_SCALE_SUFFIX):
            prepared[k] = v.to(device=device, dtype=torch.float32)
        elif v.is_floating_point():
            prepared[k] = v.to(device=device, dtype=dtype)
        else:
            prepared[k] = v.to(device=device)

    missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
    if unexpected:
        raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
    if missing:
        if strict:
            raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
        warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)

    model.to(device)
