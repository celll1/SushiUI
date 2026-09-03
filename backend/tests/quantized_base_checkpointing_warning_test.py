"""The quantized-base / no-gradient-checkpointing memory trap, and its warning.

WHAT IS BEING PINNED, and why it is worth a test.

``Int8Linear`` / ``Fp8Linear`` run ``_dequant_forward``:

    w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
    return F.linear(x, w, bias)

``F.linear`` SAVES its weight operand for backward, because
``grad_input = grad_output @ w``. For a bf16 ``nn.Linear`` that saved tensor is
an ALIAS of the resident parameter and costs zero extra bytes. For a quantized
Linear it is a fresh ``(out, in)`` allocation in the compute dtype, on top of the
1-byte codes -- so a quantized base costs 1 B resident + 2 B retained per weight
element where an unquantized base costs 2 B + 0.

With gradient checkpointing ON, one unit is live at a time and the quantized base
still wins. With it OFF, every layer's temporary is live at once and the whole
model materialises in the compute dtype on top of the codes: measured 426.4 MiB
against bf16's 322.2 MiB on a 28-layer 2048x2048 synthetic, and derived at 35.81
GiB against 23.88 GiB for a Krea 2 transformer.

An ``autograd.Function`` that rebuilds ``w`` in backward instead of retaining it
was built and measured against the pre-registered gate
``core/training/INT8_W8A8_TRAINING_GATE.md (G4)``. It passed the bitwise,
gradient and both memory criteria and FAILED the pre-registered step-time
ceiling, so it did not ship and the gate's own failure branch -- a factual
warning -- did. These tests therefore pin BOTH halves:

* ``SavedWeightRetentionTest`` pins the RETENTION ITSELF, so the warning cannot
  outlive the condition it describes. If a future change stops ``F.linear`` from
  saving a fresh weight here (the fix landing under a new gate, a torch change, a
  rewrite of ``_dequant_forward``), this test fails and whoever makes that change
  is sent to the warning to delete it. A warning that keeps firing about a
  condition that no longer exists is the failure mode this guards.
* ``QuantizedBaseWarningTest`` pins that the warning fires on exactly the
  condition (quantized AND not checkpointing) and on both quantized classes, and
  that it stays silent otherwise.

CPU only, no checkpoint, no CUDA: every property here is about what autograd
retains and what a predicate returns, neither of which is device-specific.
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ""))

from core.models.ideogram4.vendor.fp8_linear import Fp8Linear, quantize_weight_to_fp8  # noqa: E402
from core.models.ideogram4.vendor.int8_linear import Int8Linear, quantize_weight_to_int8  # noqa: E402
from core.adapters import count_quantized_linears  # noqa: E402
from core.training.adapters.base_adapter import (  # noqa: E402
    warn_quantized_base_without_checkpointing,
)

OUT, IN, TOKENS = 32, 24, 8


def _int8_layer(dtype=torch.float32) -> Int8Linear:
    layer = Int8Linear(IN, OUT, bias=False, compute_dtype=dtype)
    q, s = quantize_weight_to_int8(torch.randn(OUT, IN) * 0.02)
    layer.weight.data, layer.weight_scale.data = q, s
    return layer


def _fp8_layer(dtype=torch.float32) -> Fp8Linear:
    layer = Fp8Linear(IN, OUT, bias=False, compute_dtype=dtype)
    q, s = quantize_weight_to_fp8(torch.randn(OUT, IN) * 0.02)
    layer.weight.data, layer.weight_scale.data = q, s
    return layer


class _Capture:
    """Collect every tensor autograd saves under the block."""

    def __init__(self):
        self.saved = []

    def __enter__(self):
        def pack(t):
            self.saved.append(t)
            return t

        self._hooks = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc):
        return self._hooks.__exit__(*exc)


class SavedWeightRetentionTest(unittest.TestCase):
    """The fact the warning asserts: a fresh compute-dtype weight is retained."""

    def _retained_weight_bytes(self, layer, dtype):
        x = torch.randn(TOKENS, IN, dtype=dtype, requires_grad=True)
        with _Capture() as cap:
            layer(x).sum().backward()
        # F.linear saves the weight in its GEMM orientation, i.e. `w.t()`
        # ((in, out)), so both orientations count as "the weight".
        return [
            t.numel() * t.element_size()
            for t in cap.saved
            if t.dtype is dtype and tuple(t.shape) in ((OUT, IN), (IN, OUT))
        ]

    def test_quantized_linear_retains_a_dequantized_weight(self):
        for make in (_int8_layer, _fp8_layer):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(layer=make.__name__, dtype=dtype):
                    got = self._retained_weight_bytes(make(dtype), dtype)
                    self.assertEqual(
                        len(got), 1,
                        "a quantized Linear must retain exactly one dequantized "
                        "(out, in) weight per forward; if this changed, revisit "
                        "warn_quantized_base_without_checkpointing",
                    )
                    self.assertEqual(got[0], OUT * IN * torch.empty(0, dtype=dtype).element_size())

    def test_unquantized_linear_retains_only_an_alias_of_its_parameter(self):
        """The asymmetry, stated as a test rather than as a claim."""
        layer = nn.Linear(IN, OUT, bias=False, dtype=torch.float32)
        layer.weight.requires_grad_(False)
        x = torch.randn(TOKENS, IN, requires_grad=True)
        with _Capture() as cap:
            layer(x).sum().backward()
        weights = [t for t in cap.saved if tuple(t.shape) in ((OUT, IN), (IN, OUT))]
        self.assertEqual(len(weights), 1)
        self.assertEqual(
            weights[0].data_ptr(), layer.weight.data_ptr(),
            "an unquantized frozen Linear must save an ALIAS of its resident "
            "parameter (zero extra bytes), which is the baseline the quantized "
            "case is 1.5x worse than",
        )

    def test_retention_scales_with_layer_count(self):
        """N layers in one live unit retain N weights, not O(1) of them."""
        # Square layers so they stack; the point is the COUNT of retained
        # weights, not their shape.
        square = []
        for _ in range(4):
            layer = Int8Linear(IN, IN, bias=False, compute_dtype=torch.float32)
            q, s = quantize_weight_to_int8(torch.randn(IN, IN) * 0.02)
            layer.weight.data, layer.weight_scale.data = q, s
            square.append(layer)
        x = torch.randn(TOKENS, IN, requires_grad=True)
        with _Capture() as cap:
            out = x
            for layer in square:
                out = layer(out)
            out.sum().backward()
        retained = [t for t in cap.saved
                    if t.dtype is torch.float32 and tuple(t.shape) == (IN, IN)]
        self.assertEqual(
            len(retained), len(square),
            "every layer in a live unit retains its own dequantized weight -- "
            "this is why disabling gradient checkpointing multiplies the cost",
        )


class QuantizedBaseWarningTest(unittest.TestCase):
    def _model(self, make):
        return nn.Sequential(make(), nn.ReLU(), nn.Linear(OUT, OUT))

    def test_fires_only_when_quantized_and_not_checkpointing(self):
        cases = [
            (_int8_layer, False, True),
            (_int8_layer, True, False),
            (_fp8_layer, False, True),
            (_fp8_layer, True, False),
        ]
        for make, ckpt, expected in cases:
            with self.subTest(layer=make.__name__, gradient_checkpointing=ckpt):
                msg = warn_quantized_base_without_checkpointing(
                    self._model(make), gradient_checkpointing=ckpt)
                self.assertEqual(msg is not None, expected)

    def test_silent_on_an_unquantized_base(self):
        model = nn.Sequential(nn.Linear(IN, OUT), nn.Linear(OUT, OUT))
        self.assertEqual(count_quantized_linears(model), 0)
        for ckpt in (True, False):
            self.assertIsNone(
                warn_quantized_base_without_checkpointing(model, gradient_checkpointing=ckpt))

    def test_silent_on_a_missing_transformer(self):
        self.assertIsNone(
            warn_quantized_base_without_checkpointing(None, gradient_checkpointing=False))

    def test_message_states_the_layer_count_and_the_retained_byte_width(self):
        model = nn.Sequential(_int8_layer(torch.bfloat16), _fp8_layer(torch.bfloat16))
        msg = warn_quantized_base_without_checkpointing(model, gradient_checkpointing=False)
        self.assertIn("2 quantized Linear layer(s)", msg)
        self.assertIn("2 bytes per element retained", msg)

    def test_retained_byte_width_follows_the_compute_dtype(self):
        model = nn.Sequential(_int8_layer(torch.float32))
        msg = warn_quantized_base_without_checkpointing(model, gradient_checkpointing=False)
        self.assertIn("4 bytes per element retained", msg)


if __name__ == "__main__":
    unittest.main()
