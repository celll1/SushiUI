"""The quantized dequant path is pinned BITWISE to one arithmetic definition.

WHAT IS PINNED, and why bitwise rather than "close".

``Int8Linear`` and ``Fp8Linear`` fall back to ``_dequant_forward`` whenever the
W8A8 fast path cannot serve a call, which is the overwhelming majority of calls
(both fast paths are opt-in per process) AND the whole of training: LoRA over a
quantized base trains through this exact function. The definition of the layer's
output in this repo is

    w = codes.to(x.dtype) * scale.to(x.dtype).unsqueeze(1)
    F.linear(x, w, bias.to(x.dtype))

``Int8Linear`` ships that expression in its PROMOTED spelling, ``codes * s``:
an integer tensor times a float tensor promotes to the float dtype, so torch
does the widening inside the multiply and writes one ``(out, in)`` buffer
instead of two. That is an optimization only because it is bitwise identical --
every int8 code in [-128, 127] is exactly representable in bf16 (8 mantissa
bits), fp16 (11) and fp32 (24), so the widening rounds nothing in either
spelling and the single remaining rounding is the multiply's. Measured on an
RTX 6000 Ada (sm_89, torch 2.10, interleaved A/B minima): 20.4 -> 15.1 us for a
2048x2048 bf16 dequantize, 34.0 -> 24.6 us for the whole ``_dequant_forward`` at
one token, and a wash (0.99x) once the GEMM dominates at 4096 tokens.

``Fp8Linear`` deliberately does NOT fold the cast: ``float8_e4m3fn`` has no
promoting multiply at all. ``PromotedSpellingTest`` pins both halves of that
asymmetry, so neither "why is fp8 still doing two kernels" nor "torch changed
its promotion rule" can pass silently.

The tests compare INTEGER BIT VIEWS (``.view(int16/int32)``) with NaN placement
compared separately, not ``allclose``: a tolerance-based test would accept a
reordering or a fused variant that changes the last bit of every weight in the
model, which is exactly the class of change this file exists to refuse. Anyone
rewriting ``_dequant_forward`` into a form that is not bitwise equal to the
definition above fails here.

Coverage: {CPU, CUDA if present} x {int8, e4m3 codes} x {bf16, fp16, fp32} x
{bias, no bias} x hostile scales (zero, fp16-subnormal, float32.tiny, huge,
+Inf) x hostile activations (normal, zeros, NaN, +-Inf, huge, fp16-subnormal).
CUDA arms are skipped, not failed, on a machine without a GPU -- the dtype
matrix is the point, and the production dtypes are covered on CPU regardless.
"""

import itertools
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ""))

from core.models.ideogram4.vendor.fp8_linear import (  # noqa: E402
    FP8_WEIGHT_DTYPE,
    Fp8Linear,
)
from core.models.ideogram4.vendor.int8_linear import (  # noqa: E402
    INT8_WEIGHT_DTYPE,
    Int8Linear,
)

OUT, IN, TOKENS = 24, 32, 12

COMPUTE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# Integer bit-view dtype per float dtype. Comparing these is the whole point:
# two floats are equal here only if every bit of their representation matches.
BIT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}

E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0**-9
FP16_MIN_SUBNORMAL = 5.96e-8


def devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    """True only if a and b have the same dtype, NaN placement and raw bits."""
    if a.dtype is not b.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype}"
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    if not torch.equal(nan_a, nan_b):
        return False, f"NaN placement differs ({int(nan_a.sum())} vs {int(nan_b.sum())})"
    view = BIT_VIEW[a.dtype]
    keep = ~nan_a
    left, right = a[keep].view(view), b[keep].view(view)
    if not torch.equal(left, right):
        return False, f"{int((left != right).sum())} differing bit patterns"
    return True, ""


def reference_dequant_forward(
    codes: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    x: torch.Tensor,
) -> torch.Tensor:
    """THE definition, written out longhand and independently of the modules."""
    w = codes.to(x.dtype) * scale.to(x.dtype).unsqueeze(1)
    return F.linear(x, w, bias.to(x.dtype) if bias is not None else None)


def hostile_scales(device) -> list[torch.Tensor]:
    """Per-row scales chosen to break the promotion if anything can."""
    s = torch.rand(OUT, dtype=torch.float32, device=device) * 1e-2 + 1e-6
    s[0] = 0.0  # zero scale
    s[1] = FP16_MIN_SUBNORMAL  # subnormal once cast to fp16
    s[2] = 6.0e-8  # just above it, still subnormal in fp16
    s[3] = torch.finfo(torch.float32).tiny
    s[4] = 3.0e38  # +Inf once cast to fp16
    s[5] = float("inf")
    s[6] = 1.0
    s[7] = -1.0
    return [s, torch.zeros(OUT, dtype=torch.float32, device=device)]


def hostile_codes(code_dtype, device) -> list[torch.Tensor]:
    """Weights sitting on the ends of the quantization grid."""
    if code_dtype is INT8_WEIGHT_DTYPE:
        g = torch.randint(-128, 128, (OUT, IN), dtype=torch.int8, device=device)
        g[0].fill_(127)  # +int8 max
        g[1].fill_(-128)  # -int8 min
        g[2].fill_(0)
        g[3] = torch.tensor([-128, 127] * (IN // 2), dtype=torch.int8, device=device)
        return [g, torch.zeros(OUT, IN, dtype=torch.int8, device=device)]
    f = torch.randn(OUT, IN, device=device) * 100.0
    f[0].fill_(E4M3_MAX)  # e4m3 max finite
    f[1].fill_(-E4M3_MAX)
    f[2].fill_(0.0)
    f[3].fill_(E4M3_MIN_SUBNORMAL)
    return [
        f.to(FP8_WEIGHT_DTYPE),
        torch.zeros(OUT, IN, device=device).to(FP8_WEIGHT_DTYPE),
    ]


def hostile_activations(dtype, device) -> list[torch.Tensor]:
    x = torch.randn(TOKENS, IN, dtype=dtype, device=device)
    nan = x.clone()
    nan[0].fill_(float("nan"))
    inf = x.clone()
    inf[0].fill_(float("inf"))
    inf[1].fill_(float("-inf"))
    huge = x.clone()
    huge[2].fill_(60000.0 if dtype is torch.float16 else 3.0e38)
    return [
        x,
        torch.zeros(TOKENS, IN, dtype=dtype, device=device),
        nan,
        inf,
        huge,
        torch.full((TOKENS, IN), FP16_MIN_SUBNORMAL, dtype=dtype, device=device),
    ]


def build(cls, code_dtype, codes, scale, has_bias, dtype, device):
    module = cls(IN, OUT, bias=has_bias, compute_dtype=dtype).to(device)
    module.weight = codes.clone()
    module.weight_scale = scale.clone()
    bias = None
    if has_bias:
        bias = torch.randn(OUT, dtype=dtype, device=device)
        module.bias = bias.clone()
    assert module.weight.dtype is code_dtype  # buffers must survive construction
    return module, bias


class DequantForwardBitwiseTest(unittest.TestCase):
    """``_dequant_forward`` == the longhand definition, on the raw bits."""

    def _run(self, cls, code_dtype):
        checks = 0
        for device, dtype, has_bias in itertools.product(devices(), COMPUTE_DTYPES, (True, False)):
            for codes, scale in itertools.product(
                hostile_codes(code_dtype, device), hostile_scales(device)
            ):
                module, bias = build(cls, code_dtype, codes, scale, has_bias, dtype, device)
                for x in hostile_activations(dtype, device):
                    got = module._dequant_forward(x)
                    want = reference_dequant_forward(codes, scale, bias, x)
                    ok, why = bit_equal(want, got)
                    checks += 1
                    self.assertTrue(
                        ok,
                        f"{cls.__name__}._dequant_forward is not bitwise equal to "
                        f"`codes.to(x.dtype) * scale.to(x.dtype)[:, None]` on "
                        f"{device}/{dtype}/bias={has_bias}: {why}. That expression is "
                        f"the definition of this layer's output; a faster spelling of "
                        f"it may ship only while it is bit-identical.",
                    )
        self.assertGreater(checks, 0)

    def test_int8_dequant_forward_is_the_definition(self):
        self._run(Int8Linear, INT8_WEIGHT_DTYPE)

    def test_fp8_dequant_forward_is_the_definition(self):
        self._run(Fp8Linear, FP8_WEIGHT_DTYPE)

    def test_int8_non_integer_weight_buffer_still_matches(self):
        """The guard branch: a non-int8 buffer must NOT take the promoted form.

        Unreachable in production (the loaders only build these modules for int8
        checkpoints, and ``Module.to()`` skips integral tensors so the codes
        survive every dtype cast), but the branch exists because promotion of a
        float weight would widen UPWARD where the explicit cast narrows to
        ``x.dtype``. Pinned so it cannot rot into an fp32 ``w`` and an
        ``F.linear`` dtype error.
        """
        for device, dtype in itertools.product(devices(), COMPUTE_DTYPES):
            for weight_dtype in (torch.float32, torch.bfloat16):
                module = Int8Linear(IN, OUT, bias=True, compute_dtype=dtype).to(device)
                w = torch.randn(OUT, IN, dtype=weight_dtype, device=device)
                scale = torch.rand(OUT, dtype=torch.float32, device=device) + 1e-3
                bias = torch.randn(OUT, dtype=dtype, device=device)
                module.weight = w
                module.weight_scale = scale
                module.bias = bias.clone()
                x = torch.randn(TOKENS, IN, dtype=dtype, device=device)
                got = module._dequant_forward(x)
                want = reference_dequant_forward(w, scale, bias, x)
                ok, why = bit_equal(want, got)
                self.assertTrue(ok, f"{device}/{dtype}/weight={weight_dtype}: {why}")


class PromotedSpellingTest(unittest.TestCase):
    """The promotion rule itself, on both code formats."""

    def test_int8_promoted_multiply_is_bitwise_equal(self):
        """``codes * s`` == ``codes.to(dtype) * s`` for int8 codes.

        This is what makes the ``Int8Linear`` spelling legal. If a future torch
        changes integer/float promotion, this fails BEFORE anyone has to notice
        a drifted image, and the fix is to restore the explicit cast.
        """
        checks = 0
        for device, dtype in itertools.product(devices(), COMPUTE_DTYPES):
            for codes, scale in itertools.product(
                hostile_codes(INT8_WEIGHT_DTYPE, device), hostile_scales(device)
            ):
                s = scale.to(dtype).unsqueeze(1)
                two = codes.to(dtype) * s
                one = codes * s
                self.assertIs(one.dtype, dtype, "int8 * float must promote to the float dtype")
                ok, why = bit_equal(two, one)
                checks += 1
                self.assertTrue(ok, f"promoted multiply differs on {device}/{dtype}: {why}")
        self.assertGreater(checks, 0)

    def test_e4m3_promotion_is_unsupported_or_bitwise_equal(self):
        """Why ``Fp8Linear`` keeps its explicit cast.

        Measured on torch 2.10: ``float8_e4m3fn * bf16/fp16/fp32`` raises
        "Promotion for Float8 Types is not supported" on CPU and CUDA alike, so
        folding the cast there would raise on the first forward rather than save
        a kernel. Written as "unsupported OR bitwise equal" so that a torch that
        gains the promotion does not fail this test spuriously -- it would only
        fail if the promotion existed AND disagreed with the cast, which is the
        one outcome that must never be shipped.
        """
        for device, dtype in itertools.product(devices(), COMPUTE_DTYPES):
            codes = hostile_codes(FP8_WEIGHT_DTYPE, device)[0]
            s = hostile_scales(device)[0].to(dtype).unsqueeze(1)
            try:
                one = codes * s
            except RuntimeError:
                continue  # the documented, expected outcome
            two = codes.to(dtype) * s
            ok, why = bit_equal(two, one)
            self.assertTrue(
                ok,
                f"e4m3 promotion exists on {device}/{dtype} but is not bitwise equal "
                f"to the cast: {why}. Fp8Linear must keep the explicit cast.",
            )


class ForwardDispatchTest(unittest.TestCase):
    """``forward`` reaches ``_dequant_forward`` by default on both classes.

    Without this, the bitwise tests above could pass while production took a
    different path entirely. Both W8A8 fast paths are opt-in per process (they
    read an environment variable at import), and CPU tensors are refused by both
    regardless, so a CPU forward is the dequant path by construction.
    """

    def test_forward_equals_dequant_forward_on_cpu(self):
        for cls, code_dtype in ((Int8Linear, INT8_WEIGHT_DTYPE), (Fp8Linear, FP8_WEIGHT_DTYPE)):
            for dtype in COMPUTE_DTYPES:
                codes = hostile_codes(code_dtype, "cpu")[0]
                scale = hostile_scales("cpu")[0]
                module, bias = build(cls, code_dtype, codes, scale, True, dtype, "cpu")
                x = torch.randn(TOKENS, IN, dtype=dtype)
                ok, why = bit_equal(reference_dequant_forward(codes, scale, bias, x), module(x))
                self.assertTrue(ok, f"{cls.__name__} forward on {dtype}: {why}")


if __name__ == "__main__":
    unittest.main()
