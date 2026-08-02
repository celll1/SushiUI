"""Fused Triton kernels for the INT8 W8A8 fast path's non-GEMM stages.

``Int8Linear`` wraps ``torch._int_mm`` in two elementwise passes -- a per-token
activation quantizer over ``(m, k)`` and an epilogue over ``(m, n)`` -- and
Phase 0 measured that WITHOUT fusing them the whole path is worthless: eager int8
came out at 1.515x over bf16, statistically tied with the shipped fp8 fused path
(1.550x), while fused int8 reached 2.561x. These kernels are therefore a
REQUIREMENT of the design, not an optimization on top of it.

* the epilogue -- ``acc.float().mul_(a_scale).mul_(w_scale)[.add_(bias)]
  .to(dtype)`` -- is 5 kernels over an ``(m, n)`` buffer with a float32
  temporary, where one fused pass reads the int32 accumulator once and writes the
  output once.
* the activation quantizer -- ``abs / amax / div / clamp / reciprocal / to(f32) /
  mul / round / clamp / to(int8)`` -- is ~10 kernels over an ``(m, k)`` buffer
  including a full float32 widening of the input, where one fused pass reads the
  input once and writes int8 + one scale per row.

BITWISE EQUALITY IS THE CONTRACT, not "close enough". The eager ordering in
``int8_linear`` is deliberate and the committed numerics depend on it. Phase 0
already demonstrated the hazard on this exact problem: a ``torch.compile``-produced
int8 epilogue gave 0.010739 where eager gave 0.011114 -- a DIFFERENT function,
not a rounding wobble. Five places therefore deviate from the obvious kernel:

1. The epilogue keeps the ``(acc * a_scale) * w_scale`` ASSOCIATION (not
   ``acc * (a_scale * w_scale)``, which is what a textbook int8 epilogue does and
   what inductor produced above).
2. With a bias, the final multiply is emitted as ``mul.rn.f32`` inline asm so the
   NVPTX backend cannot contract multiply-then-add into an FMA. An FMA would be
   *more* accurate and therefore *different*.
3. The quantizer takes both the ``amax / 127`` divide and the reciprocal with
   ``div_rn`` (correctly rounded, matching torch); triton's default ``/`` is free
   to use the fast approximate reciprocal and diverges from torch's correctly
   rounded ``Tensor.reciprocal()`` on ~13.5% of values.
4. Rounding to the integer grid is ``libdevice.rint`` -- round-half-to-EVEN,
   which is what ``torch.round`` does. ``libdevice.round`` is half-away-from-zero
   and would differ on every exact ``.5``, which on a 127-level grid is not rare.
5. NaN. Every min/max carries ``propagate_nan=tl.PropagateNan.ALL``, and the row
   amax carries a separate "a NaN was seen" flag because the ``tl.max`` REDUCTION
   has no such option. Triton defaults to ``PropagateNan.NONE`` while
   ``torch.amax``/``torch.clamp_`` propagate; the fp8 kernels shipped that bug
   once and silently laundered a NaN row into a finite full-scale row. Here the
   NaN must reach the SCALE (which makes the epilogue emit a NaN row, the loud
   behaviour) while the int8 PAYLOAD gets an explicit 0 -- matching what CUDA's
   float->int conversion does in torch, rather than relying on LLVM's undefined
   ``fptosi`` of NaN. The NaN PAYLOAD of the scale is the single exemption from
   bitwise equality in this module -- see ``_scale_equal`` for why torch makes
   that unavoidable and why it is sound.

Availability is decided by an EXECUTED probe (``_ensure``): the kernels are
compiled, launched on tiny tensors including NaN/Inf/zero/denormal/outlier rows,
and their output compared BITWISE (on integer bit views) against the eager
reference. A successful ``import triton`` proves nothing. Any failure, at probe
time or call time, latches the key off for the process and the caller silently
runs the eager chain.

Set ``SUSHI_INT8_FUSED=0`` to disable the fused path entirely.
"""

from __future__ import annotations

import os
import threading

import torch


# Off switch, read once at import (matches SUSHI_FP8_FUSED's shape).
_FUSED_ENABLED = os.environ.get("SUSHI_INT8_FUSED", "1") != "0"

# (device index, dtype) -> True (probed and bitwise-exact) / False (latched off).
_FUSED_OK: dict[tuple[int, torch.dtype], bool] = {}
_FUSED_LOCK = threading.Lock()
_FUSED_REPORTED: set[str] = set()

# Populated by _import_triton() on first use; None = not imported yet,
# False = unavailable.
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
        print(f"[Int8Fused] {message}")


def _import_triton() -> bool:
    """Import triton and define the kernels. Idempotent; never raises."""
    global _triton, _tl
    if _triton is not None:
        return _triton is not False
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
    except Exception as exc:
        _triton = False
        _report("import", f"triton unavailable ({type(exc).__name__}: {exc}); "
                          "INT8 fast path uses the eager epilogue/quantizer")
        return False

    @triton.jit
    def _epilogue_kernel(acc_ptr, as_ptr, ws_ptr, b_ptr, out_ptr, N,
                         HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr):
        """out = ((acc.float() * a_scale[row]) * w_scale[col] [+ bias[col]]).to(dtype)

        One read of the int32 accumulator, one write of the output.
        """
        row = tl.program_id(0)
        col = tl.program_id(1)
        offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        base = row.to(tl.int64) * N + offs
        acc = tl.load(acc_ptr + base, mask=mask, other=0).to(tl.float32)
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
                      INT8_MAX: tl.constexpr, RECIP_INT8_MAX: tl.constexpr,
                      MIN_SCALE: tl.constexpr, BLOCK_K: tl.constexpr):
        """Per-token int8 quantization: row amax, then scale + round + convert.

        No per-input-dtype MODE constexpr (unlike the fp8 quantizer): the eager
        int8 chain widens EVERY input dtype to float32 before the multiply, so
        one code path serves bf16/fp16/fp32 and the load's ``.to(tl.float32)``
        is exact for all three.
        """
        row = tl.program_id(0)
        base = row.to(tl.int64) * K
        amax = 0.0
        # NaN BOOKKEEPING: torch.amax/clamp_ PROPAGATE NaN; triton's tl.max /
        # tl.maximum default to PropagateNan.NONE. tl.maximum takes an explicit
        # flag, but the tl.max REDUCTION does not -- hence the separate "did any
        # lane see a NaN" flag folded in after the loop.
        nan_seen = 0
        for k0 in tl.range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            m = offs < K
            v = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            av = tl.abs(v)
            amax = tl.maximum(amax, tl.max(av))
            nan_seen = tl.maximum(nan_seen, tl.max((av != av).to(tl.int32)))
        # A canonical quiet NaN (0x7fc00000). torch's amax invents its own
        # payload here (0x7fffffff on CUDA) and tl.max has already discarded the
        # data's, so the PAYLOAD is deliberately not matched -- only the fact of
        # being NaN, which is the whole contract. See _scale_equal.
        amax = tl.where(nan_seen != 0, float("nan"), amax)
        # A MULTIPLY by the pre-rounded 1/127, not a divide -- see
        # int8_linear._RECIP_INT8_MAX. (Triton's div_rn here would be a true
        # correctly-rounded divide and would differ from torch's CUDA scalar
        # division by 1 ulp on ~1 row in 8, which then flips exact-.5 roundings
        # downstream. Measured, not hypothetical.)
        scale = tl.maximum(amax * RECIP_INT8_MAX, MIN_SCALE,
                           propagate_nan=tl.PropagateNan.ALL)
        tl.store(scale_ptr + row, scale)
        # div_rn: correctly rounded, matching torch's Tensor.reciprocal() (which
        # IS a true divide -- unlike the scalar division above).
        recip = libdevice.div_rn(1.0, scale)
        for k0 in tl.range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            m = offs < K
            v = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            # rint = round-half-to-EVEN, which is torch.round. libdevice.round
            # is half-away-from-zero and would differ on every exact .5.
            p = libdevice.rint(v * recip)
            # propagate_nan=ALL: torch's clamp_ passes NaN through untouched.
            p = tl.minimum(
                tl.maximum(p, -INT8_MAX, propagate_nan=tl.PropagateNan.ALL),
                INT8_MAX, propagate_nan=tl.PropagateNan.ALL,
            )
            # NaN -> 0 EXPLICITLY. torch's float->int8 cast of NaN yields 0 on
            # CUDA; LLVM's fptosi of NaN is poison, so the kernel must not be
            # allowed to invent its own answer. The NaN is not lost: it already
            # reached the SCALE above, and the epilogue turns the whole row NaN.
            p = tl.where(p != p, 0.0, p)
            tl.store(out_ptr + base + offs, p.to(tl.int8), mask=m)

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
    # against uses THESE objects, so a literal copy would let a future edit to
    # _MIN_ACT_SCALE alone pass the probe and diverge on exactly the sub-floor
    # rows the floor exists for. (Deferred import; _probe does this too.)
    from .int8_linear import INT8_MAX, _MIN_ACT_SCALE, _RECIP_INT8_MAX

    m, k = x2.shape
    out = torch.empty((m, k), dtype=torch.int8, device=x2.device)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x2.device)
    block = 1024 if k >= 1024 else _triton.next_power_of_2(k)
    _KERNELS["quant"][(m,)](
        x2, out, scale, k,
        INT8_MAX=INT8_MAX, RECIP_INT8_MAX=_RECIP_INT8_MAX,
        MIN_SCALE=_MIN_ACT_SCALE, BLOCK_K=block, num_warps=8,
    )
    return out, scale


# ---------------------------------------------------------------------------
# Executed probe
# ---------------------------------------------------------------------------

def _bits(t: torch.Tensor) -> torch.Tensor:
    """Integer bit view of a tensor, sized to its element.

    Float equality is strictly weaker than the contract: it calls -0.0 equal to
    +0.0 and NaN unequal to itself, so a kernel that lost a sign of zero or
    turned a NaN into a different NaN payload would pass ``torch.equal``.
    """
    return t.view({1: torch.int8, 2: torch.int16, 4: torch.int32,
                   8: torch.int64}[t.element_size()])


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _scale_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise scale comparison, with the NaN PAYLOAD (and only that) exempted.

    The ONE documented relaxation of the bitwise contract in this module, and it
    is forced by torch, not chosen for convenience: ``torch.amax`` over an
    all-NaN CUDA row returns ``0x7fffffff``, which is neither the NaN that was in
    the data (``0x7fc00000``) nor anything a kernel could reconstruct -- ``tl.max``
    has already dropped the NaN by the time we know one was there. Requiring the
    payload to match would mean hard-coding an undocumented torch reduction
    artifact and re-latching the whole fast path off the first time torch changes
    it.

    What IS still required, exactly: NaN in the SAME positions in both, and
    bit-identical everywhere else. That is the entire contract the caller
    depends on -- a NaN scale multiplies its whole output row to NaN in the
    epilogue, which is the loud behaviour, and every NaN payload does that
    identically. This is emphatically NOT the fp8 kernels' old bug, where a NaN
    became a FINITE full-scale value and a blown-up activation was laundered into
    plausible noise; the ``propagate_nan`` flags and the ``nan_seen`` reduction
    flag above are what prevent that, and this helper still catches their loss
    (a lost NaN shows up as a NaN-position mismatch).
    """
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    if not torch.equal(nan_a, nan_b):
        return False
    keep = ~nan_a
    return torch.equal(_bits(a)[keep], _bits(b)[keep])


def _probe(device: torch.device, dtype: torch.dtype) -> bool:
    """Compile, launch, and BITWISE-compare against the eager chain. Never raises."""
    from .int8_linear import _eager_epilogue, _eager_quantize_activation

    try:
        m, n, k = 8, 48, 96
        # A PRIVATE generator, never torch.manual_seed / the default generator:
        # this probe fires lazily inside somebody else's forward pass, and
        # perturbing the global RNG stream would silently change their sampling.
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        rnd = lambda *s, dt=torch.float32: torch.randn(  # noqa: E731
            *s, device=device, dtype=dt, generator=gen
        )
        # int32 accumulators of a realistic magnitude, plus the extremes.
        acc = (rnd(m, n) * 3.0e5).to(torch.int32)
        acc[0, 0] = 0
        acc[0, 1] = 2 ** 31 - 1
        acc[0, 2] = -(2 ** 31)
        acc[0, 3] = 2 ** 24 + 1      # first integer float32 cannot represent
        a_s = torch.rand(m, 1, device=device, dtype=torch.float32, generator=gen) * 1e-2 + 1e-4
        w_s = torch.rand(1, n, device=device, dtype=torch.float32, generator=gen) * 1e-2 + 1e-4
        bias = rnd(n, dt=dtype)
        x = rnd(m, k, dt=dtype)

        for b in (None, bias):
            got = _launch_epilogue(acc, a_s, w_s, b, dtype)
            want = _eager_epilogue(acc, a_s, w_s, b, dtype)
            if not _bit_equal(got, want):
                _report(f"epi{_key(device, dtype)}",
                        f"fused epilogue differs from the eager chain on "
                        f"{device}/{dtype} (bias={b is not None}); using the eager path")
                return False

        # Hostile activation rows, not just randn: the NaN/zero/denormal/outlier
        # cases are exactly the ones a randn-only probe cannot see, and the
        # rounding rule (half-to-even) is only exercised by exact .5 values.
        x_nf = x.clone()
        x_nf[0, k // 2] = float("nan")     # one NaN in an otherwise normal row
        x_nf[1, :] = float("nan")          # all-NaN row
        x_nf[2, k // 3] = float("inf")     # +inf -> inf scale, zero reciprocal
        x_nf[3, :] = 0.0                   # all-zero row (hits the scale floor)
        x_nf[4, :] = torch.finfo(dtype).tiny / 4  # denormals (sub-floor scale)
        x_nf[5, 0] = 1.0e4                 # one huge outlier dominating its row
        # A row whose scaled values land exactly on .5 boundaries: with amax at
        # column 0, every other column is a clean fraction of 127 -> exact .5 at
        # the odd half-steps. This is what separates rint from round.
        half = torch.arange(k, device=device, dtype=torch.float32)
        x_nf[6, :] = ((half % 254.0) - 127.0) / 254.0
        x_nf[6, 0] = 1.0
        for src in (x, x_nf):
            q_got, s_got = _launch_quant(src)
            q_want, s_want = _eager_quantize_activation(src)
            if not _scale_equal(s_got, s_want) or not torch.equal(q_got, q_want):
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
            f"fused INT8 epilogue + activation quantizer active on {device}/{dtype} "
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

    Latching the kernel off for the whole process on a transient shortage (with
    no re-arm short of a restart) is exactly the failure
    ``int8_linear._is_allocation_failure`` exists to avoid on the GEMM path.
    Same helper, reused not copied.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    try:
        from .int8_linear import _is_allocation_failure
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
    """``((acc.float() * a_scale) * w_scale [+ bias]).to(out_dtype)`` in one kernel.

    ``acc`` must be a contiguous 2-D int32 tensor, ``a_scale`` ``(m, 1)`` and
    ``w_scale`` ``(1, n)`` contiguous float32, ``bias`` a contiguous ``(n,)``
    tensor of any float dtype (widened to float32 in-kernel, exactly as
    ``bias.to(torch.float32)`` does) or None. Returns None if the fused path is
    not usable, in which case the caller runs the eager chain.
    """
    if not _ensure(acc.device, out_dtype):
        return None
    if acc.dtype is not torch.int32 or acc.ndim != 2 or not acc.is_contiguous():
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
    """Per-token int8 quantization of a contiguous 2-D activation, in one kernel.

    Returns ``(x_int8, scale)`` bitwise-identical to the eager
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
