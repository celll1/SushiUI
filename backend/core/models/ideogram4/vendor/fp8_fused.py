"""Fused Triton kernels for the FP8 W8A8 fast path's non-GEMM stages.

On sm_89 ``torch._scaled_mm`` rejects rowwise scaling, so ``Fp8Linear`` calls it
TENSORWISE and re-applies the per-token and per-output-channel scales by hand.
That decomposition puts two elementwise passes around the GEMM, and both were
measured (``tmp/fp8_epilogue_bench.py``) to cost a large fraction of the layer:

* the epilogue -- ``acc.mul_(a_scale).mul_(w_scale) [.add_(bias)] .to(dtype)`` --
  is 4 kernels over an ``(m, n)`` float32 buffer, ~30 B of traffic per output
  element where one fused pass needs 6. Measured at 7-40% of the whole fast path.
* the activation quantization -- ``abs / amax / div / clamp / reciprocal / mul /
  clamp / to`` -- is ~8 kernels over an ``(m, k)`` buffer, ~13 B per input
  element where one fused pass needs 3. Measured at up to 36% (k-heavy layers)
  and, on the small text-encoder shapes, ~85 us of pure launch overhead.

Both are replaced here by ONE Triton kernel each.

BITWISE EQUALITY IS THE CONTRACT, not "close enough". The eager ordering is
deliberate (see the comment in ``Fp8Linear._scaled_mm_forward``): both scales are
applied in float32 and the result is rounded to the output dtype exactly once,
which is worth 4.31e-03 rel RMS over the cheaper alternatives, and the committed
numerics figure (bf16 ``fast_vs_ideal`` = 1.2656e-03) depends on it. Three
places therefore deviate from the obvious kernel:

1. The epilogue keeps the ``(acc * a_scale) * w_scale`` ASSOCIATION (not
   ``acc * (a_scale * w_scale)``, which is what a textbook int8 epilogue does).
2. With a bias, the final multiply is emitted as ``mul.rn.f32`` inline asm so the
   NVPTX backend cannot contract multiply-then-add into an FMA. An FMA would be
   *more* accurate and therefore *different*.
3. The quantizer takes the reciprocal with ``div_rn`` (torch's
   ``Tensor.reciprocal()`` is correctly rounded; triton's default ``/`` need not
   be) and, for bfloat16 inputs, rounds both the reciprocal and the product to
   bfloat16 exactly where the eager path does.
4. Every min/max in the quantizer carries ``propagate_nan=tl.PropagateNan.ALL``,
   and the row amax carries a separate "a NaN was seen" flag because the
   ``tl.max`` REDUCTION has no such option. Triton defaults to
   ``PropagateNan.NONE`` while ``torch.amax``/``torch.clamp_`` propagate, so the
   obvious kernel turns a row containing a NaN into a finite scale and a full row
   of ``+-448`` -- it would LAUNDER a blown-up activation (fp16 overflow, a
   corrupted or mis-merged FP8 checkpoint) into plausible finite noise that flows
   on through the GEMM, where the eager chain fails loudly and unmistakably.
   Measured cost: 0.8x-1.4x on the quantizer stage, a few percent of the layer.

Availability is decided by an EXECUTED probe (``_ensure``): the kernels are
compiled, launched on tiny tensors, and their output compared BITWISE against the
eager reference for that device and dtype. A successful ``import triton`` proves
nothing -- a toolchain that cannot compile, a driver that rejects the inline asm,
or a triton release that changes a rounding default all pass the import and fail
here. Any failure, at probe time or at call time, latches the key off for the
process and the caller silently runs the eager chain (which is what produced
every image before this module existed).

WHY TRITON AND NOT ``torch.compile``. The same two helpers were prototyped under
inductor (``tmp/fp8_fused_proto.py``) and REJECTED on numerics before speed was
even weighed: inductor contracts the epilogue's multiply-and-add into an FMA
(1-2 differing elements per 9.4e6 on bf16, ~1.8e4 on fp32) and drops the
intermediate bfloat16 rounding in the quantizer (6.0e5 differing e4m3 values out
of 1.9e7 -- 3% of the tensor), and neither is steerable from the caller the way
an explicit ``mul.rn.f32`` and an explicit ``.to(bfloat16)`` are. It was also
slower on every shape measured (epilogue 1.0-1.8x, quantizer 1.4-2.5x the triton
time), and its per-call guard/dispatch overhead lands hardest on exactly the
small text-encoder shapes where this path was already losing to the dequant
matmul on launch overhead alone.

Set ``SUSHI_FP8_FUSED=0`` to disable the fused path entirely.
"""

from __future__ import annotations

import os
import threading

import torch


# Off switch, read once at import (matches SUSHI_FP8_FAST_ACCUM's shape).
_FUSED_ENABLED = os.environ.get("SUSHI_FP8_FUSED", "1") != "0"

# (device index, dtype) -> True (probed and bitwise-exact) / False (latched off).
_FUSED_OK: dict[tuple[int, torch.dtype], bool] = {}
_FUSED_LOCK = threading.Lock()
_FUSED_REPORTED: set[str] = set()

# Populated by _import_triton() on first use; None means "not imported yet",
# False means "unavailable".
_triton = None
_tl = None
_KERNELS: dict = {}


def fused_enabled() -> bool:
    """True if the fused kernels are permitted AND at least one key probed OK."""
    if not _FUSED_ENABLED:
        return False
    with _FUSED_LOCK:
        return any(_FUSED_OK.values())


def fused_state() -> dict:
    """Introspection for provenance/metadata (no cost on any hot path)."""
    with _FUSED_LOCK:
        return {
            "enabled": bool(_FUSED_ENABLED),
            "resolved": {
                f"cuda:{i if i >= 0 else 'default'}/{str(d).rsplit('.', 1)[-1]}": v
                for (i, d), v in _FUSED_OK.items()
            },
        }


def _report(key: str, message: str) -> None:
    with _FUSED_LOCK:
        first = key not in _FUSED_REPORTED
        _FUSED_REPORTED.add(key)
    if first:
        print(f"[Fp8Fused] {message}")


def _import_triton() -> bool:
    """Import triton and define the kernels. Idempotent; never raises."""
    global _triton, _tl
    if _triton is not None:
        return _triton is not False
    try:
        import triton
        import triton.language as tl
    except Exception as exc:
        _triton = False
        _report("import", f"triton unavailable ({type(exc).__name__}: {exc}); "
                          "FP8 fast path uses the eager epilogue/quantizer")
        return False

    @triton.jit
    def _epilogue_kernel(acc_ptr, as_ptr, ws_ptr, b_ptr, out_ptr, N,
                         HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr):
        """out = ((acc * a_scale[row]) * w_scale[col] [+ bias[col]]).to(out_dtype)

        One read of the float32 accumulator, one write of the output.
        """
        row = tl.program_id(0)
        col = tl.program_id(1)
        offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        base = row.to(tl.int64) * N + offs
        acc = tl.load(acc_ptr + base, mask=mask, other=0.0)
        a_s = tl.load(as_ptr + row)
        w_s = tl.load(ws_ptr + offs, mask=mask, other=0.0)
        out = acc * a_s
        if HAS_BIAS:
            # Opaque mul.rn.f32: blocks multiply+add contraction into an FMA,
            # which would round once where the eager chain rounds twice.
            out = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;", "=r,r,r", [out, w_s],
                dtype=tl.float32, is_pure=True, pack=1,
            )
            out = out + tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            out = out * w_s
        tl.store(out_ptr + base, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _quant_kernel(x_ptr, out_ptr, scale_ptr, K,
                      FP8_MAX: tl.constexpr, MIN_SCALE: tl.constexpr,
                      MODE: tl.constexpr, BLOCK_K: tl.constexpr):
        """Per-token e4m3 quantization: row amax, then scale + convert.

        MODE 0 = bfloat16 input (reciprocal and product rounded to bfloat16,
        exactly where the eager path rounds them), 1 = float32, 2 = float16
        (float32 math, as the eager path does to avoid overflowing the
        reciprocal).
        """
        row = tl.program_id(0)
        base = row.to(tl.int64) * K
        amax = 0.0
        # NaN BOOKKEEPING. torch.amax/clamp_ PROPAGATE NaN; triton's tl.max /
        # tl.maximum default to PropagateNan.NONE, so without this a row holding
        # a single NaN would produce a finite scale and a full row of +-448 --
        # laundering a blown-up activation into plausible finite garbage where
        # the eager chain fails loudly. tl.maximum/tl.minimum take an explicit
        # propagate_nan flag, but the tl.max REDUCTION does not, hence the
        # separate "did any lane see a NaN" flag folded in after the loop.
        nan_seen = 0
        for k0 in tl.range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            m = offs < K
            v = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            av = tl.abs(v)
            amax = tl.maximum(amax, tl.max(av))
            nan_seen = tl.maximum(nan_seen, tl.max((av != av).to(tl.int32)))
        amax = tl.where(nan_seen != 0, float("nan"), amax)
        # Plain "/" here but div_rn below is NOT an oversight: a / 448.0 is a
        # single correctly-rounded IEEE divide that triton emits identically to
        # torch (verified bitwise over 4.2e6 values), whereas triton's default
        # "/" for the RECIPROCAL is free to use the fast approximate reciprocal
        # and diverges from torch's correctly-rounded Tensor.reciprocal() on
        # ~13.5% of values. Do not "unify" these into one spelling.
        scale = tl.maximum(amax / FP8_MAX, MIN_SCALE,
                           propagate_nan=tl.PropagateNan.ALL)
        tl.store(scale_ptr + row, scale)
        # div_rn: correctly rounded, matching torch's Tensor.reciprocal().
        recip = tl.math.div_rn(1.0, scale)
        if MODE == 0:
            recip = recip.to(tl.bfloat16).to(tl.float32)
        for k0 in tl.range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            m = offs < K
            v = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            p = v * recip
            if MODE == 0:
                p = p.to(tl.bfloat16).to(tl.float32)
            # propagate_nan=ALL: torch's clamp_ passes NaN through untouched.
            p = tl.minimum(
                tl.maximum(p, -FP8_MAX, propagate_nan=tl.PropagateNan.ALL),
                FP8_MAX, propagate_nan=tl.PropagateNan.ALL,
            )
            tl.store(out_ptr + base + offs,
                     p.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
                     mask=m)

    _KERNELS["epilogue"] = _epilogue_kernel
    _KERNELS["quant"] = _quant_kernel
    _triton = triton
    _tl = tl
    return True


def _key(device: torch.device, dtype: torch.dtype) -> tuple[int, torch.dtype]:
    return (device.index if device.index is not None else -1, dtype)


# ---------------------------------------------------------------------------
# Kernel launchers (raise on failure; the callers below catch and latch)
# ---------------------------------------------------------------------------

def _launch_epilogue(acc, a_scale, w_scale, bias, out_dtype):
    m, n = acc.shape
    out = torch.empty((m, n), dtype=out_dtype, device=acc.device)
    block = 1024 if n >= 1024 else _triton.next_power_of_2(n)
    grid = (m, _triton.cdiv(n, block))
    _KERNELS["epilogue"][grid](
        acc, a_scale, w_scale, bias if bias is not None else acc, out, n,
        HAS_BIAS=bias is not None, BLOCK_N=block, num_warps=4,
    )
    return out


def _launch_quant(x2):
    # Imported, never re-declared: the eager reference these kernels are probed
    # against uses THESE objects, and _probe's randn never reaches the floor, so
    # a literal copy would let a future edit to _MIN_ACT_SCALE alone pass the
    # probe and diverge on exactly the sub-floor rows the floor exists for.
    # (Deferred import; the module already does this in _probe, so no cycle.)
    from .fp8_linear import FP8_E4M3_MAX, _MIN_ACT_SCALE

    m, k = x2.shape
    out = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x2.device)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x2.device)
    if x2.dtype is torch.bfloat16:
        mode = 0
    elif x2.dtype is torch.float32:
        mode = 1
    else:
        mode = 2
    block = 1024 if k >= 1024 else _triton.next_power_of_2(k)
    _KERNELS["quant"][(m,)](
        x2, out, scale, k,
        FP8_MAX=FP8_E4M3_MAX, MIN_SCALE=_MIN_ACT_SCALE,
        MODE=mode, BLOCK_K=block, num_warps=8,
    )
    return out, scale


# ---------------------------------------------------------------------------
# Executed probe
# ---------------------------------------------------------------------------

def _bits(t: torch.Tensor) -> torch.Tensor:
    """Integer bit view of a float tensor, sized to its element.

    Float equality is strictly weaker than the contract: it calls -0.0 equal to
    +0.0 and NaN unequal to itself, so a kernel that lost a sign of zero or
    turned a NaN into a different NaN payload would pass ``torch.equal``.
    """
    return t.view({1: torch.int8, 2: torch.int16, 4: torch.int32,
                   8: torch.int64}[t.element_size()])


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _probe(device: torch.device, dtype: torch.dtype) -> bool:
    """Compile, launch, and BITWISE-compare against the eager chain. Never raises."""
    from .fp8_linear import _eager_epilogue, _eager_quantize_activation

    try:
        m, n, k = 8, 48, 96
        # A PRIVATE generator, never torch.manual_seed / the default generator:
        # this probe fires lazily inside somebody else's forward pass, and
        # perturbing the global RNG stream would silently change their sampling.
        # (Caught exactly that way -- it moved tmp/fp8_scaled_mm_numerics.py's
        # bf16 worst case from 1.2656e-03 to 1.6310e-03 by reshuffling its data.)
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        rnd = lambda *s, dt=torch.float32: torch.randn(  # noqa: E731
            *s, device=device, dtype=dt, generator=gen
        )
        acc = rnd(m, n) * 30.0
        a_s = torch.rand(m, 1, device=device, dtype=torch.float32, generator=gen) * 1e-2 + 1e-4
        w_s = torch.rand(1, n, device=device, dtype=torch.float32, generator=gen) * 1e-2 + 1e-4
        bias = rnd(n, dt=dtype)
        x = rnd(m, k, dt=dtype)

        # A non-finite accumulator too: the epilogue is pure mul/add so it has no
        # NaN-sensitive op today, but the check costs nothing and pins that down.
        acc_nf = acc.clone()
        acc_nf[0, 0] = float("nan")
        acc_nf[1, 1] = float("inf")
        acc_nf[2, 2] = float("-inf")
        for src in (acc, acc_nf):
            for b in (None, bias):
                got = _launch_epilogue(src, a_s, w_s, b, dtype)
                want = _eager_epilogue(src.clone(), a_s, w_s, b, dtype)
                if not _bit_equal(got, want):
                    _report(f"epi{_key(device, dtype)}",
                            f"fused epilogue differs from the eager chain on "
                            f"{device}/{dtype} (bias={b is not None}); using the eager path")
                    return False

        # NaN/Inf rows are probed, not just randn: triton's tl.max/tl.maximum
        # default to PropagateNan.NONE where torch.amax/clamp_ propagate, so a
        # regression there would turn a NaN row into finite +-448 garbage --
        # invisible to a randn-only probe.
        x_nf = x.clone()
        x_nf[0, k // 2] = float("nan")     # one NaN in an otherwise normal row
        x_nf[1, :] = float("nan")          # all-NaN row (hits the scale floor)
        x_nf[2, k // 3] = float("inf")     # +inf -> inf scale
        x_nf[3, :] = 0.0                   # all-zero row (hits the scale floor)
        for src in (x, x_nf):
            q_got, s_got = _launch_quant(src)
            q_want, s_want = _eager_quantize_activation(src)
            if not _bit_equal(s_got, s_want) or not torch.equal(
                q_got.view(torch.uint8), q_want.view(torch.uint8)
            ):
                _report(f"quant{_key(device, dtype)}",
                        f"fused quantizer differs from the eager chain on "
                        f"{device}/{dtype}; using the eager path")
                return False
    except Exception as exc:
        _report(f"probe{_key(device, dtype)}",
                f"fused kernels unusable on {device}/{dtype} "
                f"({type(exc).__name__}: {exc}); using the eager path")
        return False
    _report(f"ok{_key(device, dtype)}",
            f"fused FP8 epilogue + activation quantizer active on {device}/{dtype} "
            f"(bitwise-identical to the eager chain)")
    return True


def _ensure(device: torch.device, dtype: torch.dtype) -> bool:
    """Cached ``_probe``. Returns False for every non-CUDA device."""
    if not _FUSED_ENABLED or device.type != "cuda":
        return False
    key = _key(device, dtype)
    try:
        return _FUSED_OK[key]
    except KeyError:
        pass
    if not _import_triton():
        with _FUSED_LOCK:
            return _FUSED_OK.setdefault(key, False)
    ok = _probe(device, dtype)
    with _FUSED_LOCK:
        return _FUSED_OK.setdefault(key, ok)


def _is_alloc_failure(exc: BaseException) -> bool:
    """True for an allocation shortage, however it is spelled.

    Not just ``torch.cuda.OutOfMemoryError``: triton's launcher and the CUDA
    driver report a shortage as a plain ``RuntimeError`` too, and latching the
    kernel off for the whole process on a transient shortage (with no re-arm
    short of a restart -- ``set_scaled_mm_enabled`` deliberately does not clear
    this latch) is exactly the failure ``fp8_linear._is_allocation_failure``
    exists to avoid on the GEMM path. Same marker list, reused not copied.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    try:
        from .fp8_linear import _is_allocation_failure
    except Exception:
        return False
    return _is_allocation_failure(exc)


def _latch_off(device: torch.device, dtype: torch.dtype, what: str, exc: BaseException) -> None:
    key = _key(device, dtype)
    with _FUSED_LOCK:
        _FUSED_OK[key] = False
    _report(f"rt{what}{key}",
            f"fused {what} failed at runtime on {device}/{dtype} "
            f"({type(exc).__name__}: {exc}); using the eager path from now on")


# ---------------------------------------------------------------------------
# Public entry points: return None when the caller must use the eager chain
# ---------------------------------------------------------------------------

def fused_epilogue(acc, a_scale, w_scale, bias, out_dtype):
    """``((acc * a_scale) * w_scale [+ bias]).to(out_dtype)`` in one kernel.

    ``acc`` must be a contiguous 2-D float32 tensor, ``a_scale`` ``(m, 1)`` and
    ``w_scale`` ``(1, n)`` contiguous float32, ``bias`` a contiguous ``(n,)``
    tensor of any float dtype (widened to float32 in-kernel, exactly as
    ``bias.to(torch.float32)`` does) or None. Returns None if the fused path is
    not usable, in which case the caller runs the eager chain.
    """
    if not _ensure(acc.device, out_dtype):
        return None
    if acc.dtype is not torch.float32 or acc.ndim != 2 or not acc.is_contiguous():
        return None
    if bias is not None and not bias.is_contiguous():
        return None
    # The kernel indexes the scales directly (a_scale[row], w_scale[col]), so a
    # stride-0 expanded or non-float32 scale from a future caller would read out
    # of bounds / misread rather than fall back. Holds at today's only call site.
    m, n = acc.shape
    for scale, need in ((a_scale, m), (w_scale, n)):
        if (scale.dtype is not torch.float32 or not scale.is_contiguous()
                or scale.numel() < need):
            return None
    try:
        return _launch_epilogue(acc, a_scale, w_scale, bias, out_dtype)
    except Exception as exc:
        if _is_alloc_failure(exc):
            # Transient and shape-specific -- do NOT latch; let the caller's own
            # OOM handling decide (it falls the whole layer back to dequant).
            raise
        _latch_off(acc.device, out_dtype, "epilogue", exc)
        return None


def fused_quantize(x2):
    """Per-token e4m3 quantization of a contiguous 2-D activation, in one kernel.

    Returns ``(x_fp8, scale)`` bitwise-identical to the eager
    ``_quantize_activation``, or None if the fused path is not usable.
    """
    if x2.ndim != 2 or not x2.is_contiguous():
        return None
    if not _ensure(x2.device, x2.dtype):
        return None
    try:
        return _launch_quant(x2)
    except Exception as exc:
        if _is_alloc_failure(exc):
            raise
        _latch_off(x2.device, x2.dtype, "quantizer", exc)
        return None
