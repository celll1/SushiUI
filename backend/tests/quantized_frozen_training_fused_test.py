"""The CANDIDATE frozen-base fused training forward (opt-in, default OFF).

Pins ``core/models/common/quantized_frozen_training.py`` against the
properties measured in ``INT8_CONVROT_TRAIN_FUSED`` evidence: with the path
disabled, both ``ConvRotInt8Linear`` and ``W4A8Linear`` must take the existing
``_dequant_forward`` under grad, unchanged; with it enabled, ``grad_x`` must be
BITWISE equal to the dequant path's autograd ``grad_input`` (not merely
close), no frozen operand (weight/scale/bias/sidecar) may receive a gradient,
the backward must retain nothing but aliases of the module's own resident
quantized buffers, a trainable weight must be refused loudly at enable time,
only the exact classes are eligible (no subclass), an unsupported activation
must fall back rather than raise, and a kernel failure must be fatal -- it may
never silently reroute a layer to the dequant path.

CUDA-only where the fused path itself requires CUDA (``maybe_frozen_fused_forward``
declines any non-CUDA activation by construction, so the CPU/default-off and
type-gate properties are exercised without a GPU; everything that actually
runs the fused kernel is skipped cleanly when no GPU or no comfy-kitchen is
present).
"""

import os
import sys
import unittest.mock as mock

import pytest
import torch
import torch.nn as nn

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from core.models.common.convrot_int8_linear import ConvRotInt8Linear  # noqa: E402
from core.models.common.w4a8_linear import W4A8Linear  # noqa: E402
from core.models.common import quantized_frozen_training as qft  # noqa: E402

ck = pytest.importorskip("comfy_kitchen")
from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight  # noqa: E402
import comfy_kitchen.tensor as ck_tensor  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused frozen-base path requires CUDA")

OUT, IN = 12, 256
MARKER_NUMEL = 8
COMPUTE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _convrot_layer(dtype=torch.bfloat16, device="cpu", bias=False, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(OUT, IN, dtype=torch.float32)
    qdata, scale = quantize_int8_convrot_weight(w, 256)
    layer = ConvRotInt8Linear(
        IN, OUT, bias=bias, compute_dtype=dtype,
        convrot_groupsize=256, marker_numel=MARKER_NUMEL, device=device,
    )
    layer.weight = qdata.to(device)
    layer.weight_scale = scale.reshape(-1).to(device)
    if bias:
        layer.bias = torch.randn(OUT, dtype=dtype, device=device)
    return layer


def _w4a8_layer(dtype=torch.bfloat16, device="cpu", bias=False, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(OUT, IN, dtype=torch.bfloat16)
    qdata, s_rel, s_channel, _correction, codebook = ck_tensor.quantize_w4a8_int8_weight(w)
    layer = W4A8Linear(
        IN, OUT, bias=bias, compute_dtype=dtype, group_size=16, convrot_groupsize=256,
        has_codebook=True, has_correction=False, device=device,
    )
    layer.weight = qdata.to(device)
    layer.weight_s_rel = s_rel.to(device)
    layer.weight_s_channel = s_channel.to(device)
    layer.weight_codebook = codebook.to(device)
    if bias:
        layer.bias = torch.randn(OUT, dtype=dtype, device=device)
    return layer


def _enable(layer):
    layer._frozen_training_fused = True
    layer._frozen_training_path = "test-layer"
    return layer


class _Capture:
    """Collect every tensor autograd saves under the block (same as
    quantized_base_checkpointing_warning_test.py's helper)."""

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


# ---------------------------------------------------------------------------
# 1. Default OFF
# ---------------------------------------------------------------------------

class TestDefaultOff:
    def test_env_flag_is_unset_for_this_process(self):
        assert qft.frozen_training_fused_requested() is False

    def test_convrot_class_default_is_disabled(self):
        layer = _convrot_layer()
        assert layer._frozen_training_fused is False

    def test_w4a8_class_default_is_disabled(self):
        layer = _w4a8_layer()
        assert layer._frozen_training_fused is False

    def test_convrot_default_off_never_reaches_the_fused_dispatcher(self):
        layer = _convrot_layer(dtype=torch.bfloat16)
        x = torch.randn(4, IN, dtype=torch.bfloat16, requires_grad=True)
        with mock.patch.object(
            qft, "maybe_frozen_fused_forward",
            side_effect=AssertionError("fused dispatcher must not run with the flag off"),
        ) as spy:
            out = layer(x)
        spy.assert_not_called()
        expected = layer._dequant_forward(x.detach().clone().requires_grad_(True))
        assert torch.equal(out.detach(), expected.detach())

    def test_w4a8_default_off_never_reaches_the_fused_dispatcher(self):
        layer = _w4a8_layer(dtype=torch.bfloat16)
        x = torch.randn(4, IN, dtype=torch.bfloat16, requires_grad=True)
        with mock.patch.object(
            qft, "maybe_frozen_fused_forward",
            side_effect=AssertionError("fused dispatcher must not run with the flag off"),
        ) as spy:
            out = layer(x)
        spy.assert_not_called()
        # W4A8Linear inlines its own dequant branch rather than delegating to a
        # named method, so recompute the same expression as the reference.
        weight = ck_tensor.dequantize_w4a8_int8_weight(
            layer.weight, layer.weight_s_rel, layer.weight_s_channel,
            codebook=layer.weight_codebook, correction=layer.weight_correction,
            group_size=layer.group_size, convrot_groupsize=layer.convrot_groupsize,
            output_dtype=x.dtype,
        )
        expected = torch.nn.functional.linear(x, weight, None)
        assert torch.equal(out.detach(), expected.detach())

    @CUDA
    def test_mutation_check_enabling_the_flag_is_what_the_dispatcher_guard_detects(self):
        """Negative control for the two tests above: flipping the exact flag
        they gate on must make the dispatcher fire. If this test failed to
        observe a call here, the two OFF tests above would be vacuous."""
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda"))
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        with mock.patch.object(
            qft, "maybe_frozen_fused_forward",
            side_effect=AssertionError("dispatcher invoked"),
        ) as spy:
            with pytest.raises(AssertionError):
                layer(x)
        spy.assert_called_once()


# ---------------------------------------------------------------------------
# 2. grad_x bitwise equality
# ---------------------------------------------------------------------------

@CUDA
class TestGradXBitwiseEquality:
    @pytest.mark.parametrize("dtype", COMPUTE_DTYPES)
    @pytest.mark.parametrize("bias", (True, False))
    @pytest.mark.parametrize("shape", ((4, IN), (2, 3, IN)))
    def test_convrot_grad_x_matches_dequant_forward_bitwise(self, dtype, bias, shape):
        layer = _enable(_convrot_layer(dtype=dtype, device="cuda", bias=bias))
        x_fused = torch.randn(*shape, dtype=dtype, device="cuda", requires_grad=True)
        x_dequant = x_fused.detach().clone().requires_grad_(True)

        layer(x_fused).float().sum().backward()
        layer._dequant_forward(x_dequant).float().sum().backward()

        assert torch.equal(x_fused.grad, x_dequant.grad), (
            f"grad_x differs bitwise for ConvRot dtype={dtype} bias={bias} shape={shape}, "
            f"max|delta|={(x_fused.grad - x_dequant.grad).abs().max().item()}"
        )

    @pytest.mark.parametrize("dtype", COMPUTE_DTYPES)
    @pytest.mark.parametrize("bias", (True, False))
    @pytest.mark.parametrize("shape", ((4, IN), (2, 3, IN)))
    def test_w4a8_grad_x_matches_dequant_forward_bitwise(self, dtype, bias, shape):
        layer = _w4a8_layer(dtype=dtype, device="cuda", bias=bias)
        x_fused = torch.randn(*shape, dtype=dtype, device="cuda", requires_grad=True)
        x_dequant = x_fused.detach().clone().requires_grad_(True)

        _enable(layer)
        layer(x_fused).float().sum().backward()
        layer._frozen_training_fused = False  # same instance, dequant branch this time
        layer(x_dequant).float().sum().backward()

        assert torch.equal(x_fused.grad, x_dequant.grad), (
            f"grad_x differs bitwise for W4A8 dtype={dtype} bias={bias} shape={shape}, "
            f"max|delta|={(x_fused.grad - x_dequant.grad).abs().max().item()}"
        )

    @CUDA
    def test_mutation_check_a_broken_backward_is_caught(self):
        """Negative control: flip the sign in ConvRotFrozenLinearFn.backward's
        returned grad_x and confirm the bitwise assertion above would fail.

        Exercised directly against the Function (not by editing the source
        file) by calling backward with a monkeypatched sign flip on the
        dequantized weight, which is exactly the computation the real
        backward performs.
        """
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda"))
        x_fused = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_dequant = x_fused.detach().clone().requires_grad_(True)

        real_op = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype
        with mock.patch.object(
            torch.ops.comfy_kitchen, "dequantize_int8_convrot_weight_dtype",
            side_effect=lambda *a, **k: -real_op(*a, **k),
        ):
            layer(x_fused).float().sum().backward()
        layer._dequant_forward(x_dequant).float().sum().backward()

        assert not torch.equal(x_fused.grad, x_dequant.grad), (
            "a sign-flipped backward was not detected by torch.equal -- the "
            "bitwise assertion above would be vacuous")


# ---------------------------------------------------------------------------
# 3. No weight/scale/bias/sidecar gradient
# ---------------------------------------------------------------------------

@CUDA
class TestNoFrozenOperandGradient:
    def test_convrot_module_buffers_never_accumulate_grad(self):
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda", bias=True))
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        layer(x).float().sum().backward()
        for name in ("weight", "weight_scale", "bias"):
            assert getattr(layer, name).grad is None, name

    def test_w4a8_module_buffers_never_accumulate_grad(self):
        layer = _enable(_w4a8_layer(dtype=torch.bfloat16, device="cuda", bias=True))
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        layer(x).float().sum().backward()
        for name in ("weight", "weight_s_rel", "weight_s_channel", "weight_codebook", "bias"):
            assert getattr(layer, name).grad is None, name

    def test_convrot_direct_apply_returns_none_for_every_frozen_operand_even_when_requires_grad(self):
        """Bypasses the module: proves the returned gradient tuple itself is
        None for weight/scale/bias, not merely that buffers happen not to
        require grad in production."""
        layer = _convrot_layer(dtype=torch.bfloat16, device="cuda", bias=True)
        weight = layer.weight  # int8: cannot require grad, by construction
        weight_scale = layer.weight_scale.clone().requires_grad_(True)
        bias = layer.bias.clone().requires_grad_(True)
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        out = qft.ConvRotFrozenLinearFn.apply(x, weight, weight_scale, bias, 256, "direct")
        out.float().sum().backward()

        assert x.grad is not None
        assert weight_scale.grad is None
        assert bias.grad is None

    def test_w4a8_direct_apply_returns_none_for_every_frozen_operand_even_when_requires_grad(self):
        layer = _w4a8_layer(dtype=torch.bfloat16, device="cuda", bias=True)
        weight = layer.weight
        s_rel = layer.weight_s_rel.clone().requires_grad_(True)
        s_channel = layer.weight_s_channel.clone().requires_grad_(True)
        codebook = layer.weight_codebook.clone().requires_grad_(True)
        bias = layer.bias.clone().requires_grad_(True)
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        out = qft.W4A8FrozenLinearFn.apply(
            x, weight, s_rel, s_channel, codebook, None, bias, 16, 256, "direct")
        out.float().sum().backward()

        assert x.grad is not None
        for grad in (s_rel.grad, s_channel.grad, codebook.grad, bias.grad):
            assert grad is None

    def test_mutation_check_a_leaking_backward_is_caught(self):
        """Negative control: a backward that returns a real gradient for
        weight_scale instead of None must be detected."""
        layer = _convrot_layer(dtype=torch.bfloat16, device="cuda")
        weight_scale = layer.weight_scale.clone().requires_grad_(True)
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        class _LeakyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight, weight_scale, bias, groupsize, layer_path):
                ctx.save_for_backward(weight, weight_scale)
                ctx.x_dtype = x.dtype
                ctx.groupsize = groupsize
                from comfy_kitchen import int8_linear
                return int8_linear(x, weight, weight_scale, bias=bias, out_dtype=x.dtype,
                                    convrot=True, convrot_groupsize=groupsize)

            @staticmethod
            def backward(ctx, grad_output):
                weight, weight_scale = ctx.saved_tensors
                weight_dq = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
                    weight, weight_scale.reshape(-1, 1), ctx.groupsize,
                    qft._CONVROT_DTYPE_CODES[ctx.x_dtype],
                )
                grad_x = grad_output.to(ctx.x_dtype) @ weight_dq
                # THE MUTATION: leak a real gradient for weight_scale.
                return grad_x, None, torch.zeros_like(weight_scale), None, None, None

        out = _LeakyFn.apply(x, layer.weight, weight_scale, None, 256, "leaky")
        out.float().sum().backward()
        assert weight_scale.grad is not None, (
            "the leaky backward above did not populate weight_scale.grad -- the "
            "no-gradient assertion in this class would not catch a real leak")


# ---------------------------------------------------------------------------
# 4. Retention: dequant path retains a fresh weight, fused path retains none
# ---------------------------------------------------------------------------

@CUDA
class TestRetention:
    def test_convrot_dequant_path_retains_a_fresh_activation_dtype_weight(self):
        layer = _convrot_layer(dtype=torch.bfloat16, device="cuda")
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        with _Capture() as cap:
            layer._dequant_forward(x).sum().backward()
        weight_like = [
            t for t in cap.saved
            if t.dtype is torch.bfloat16 and tuple(t.shape) in ((OUT, IN), (IN, OUT))
        ]
        assert len(weight_like) == 1
        assert weight_like[0].data_ptr() != layer.weight.data_ptr()

    def test_convrot_fused_path_saves_only_aliases_of_resident_buffers(self):
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda"))
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        resident = {"weight": layer.weight.data_ptr(), "weight_scale": layer.weight_scale.data_ptr()}
        with _Capture() as cap:
            layer(x).sum().backward()

        assert len(cap.saved) > 0, "the fused backward saved nothing at all"
        for t in cap.saved:
            assert t.data_ptr() in resident.values(), (
                f"fused backward saved a tensor ({tuple(t.shape)} {t.dtype}) that is "
                f"not an alias of a resident module buffer -- data_ptr={t.data_ptr()}")
        seen = {t.data_ptr() for t in cap.saved}
        assert resident["weight"] in seen and resident["weight_scale"] in seen

        activation_like = [
            t for t in cap.saved
            if t.dtype is torch.bfloat16 and tuple(t.shape) in ((OUT, IN), (IN, OUT))
        ]
        assert activation_like == [], "the fused path retained a dequantized activation-dtype weight"

    def test_w4a8_fused_path_saves_only_aliases_of_resident_buffers(self):
        layer = _enable(_w4a8_layer(dtype=torch.bfloat16, device="cuda"))
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        resident = {
            name: getattr(layer, name).data_ptr()
            for name in ("weight", "weight_s_rel", "weight_s_channel", "weight_codebook")
        }
        with _Capture() as cap:
            layer(x).sum().backward()

        assert len(cap.saved) > 0
        for t in cap.saved:
            assert t.data_ptr() in resident.values(), (
                f"fused backward saved a tensor ({tuple(t.shape)} {t.dtype}) that is "
                f"not an alias of a resident module buffer")
        seen = {t.data_ptr() for t in cap.saved}
        assert resident["weight"] in seen

        activation_like = [
            t for t in cap.saved
            if t.dtype is torch.bfloat16 and tuple(t.shape) in ((OUT, IN), (IN, OUT))
        ]
        assert activation_like == []

    def test_mutation_check_a_cloned_save_breaks_the_alias_assertion(self):
        """Negative control: cloning weight before save_for_backward (the
        exact regression that would silently reintroduce retention) must be
        caught by the data_ptr check above."""
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda"))
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        class _CloningFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight, weight_scale, bias, groupsize, layer_path):
                ctx.save_for_backward(weight.clone(), weight_scale)  # THE MUTATION
                ctx.x_dtype = x.dtype
                ctx.groupsize = groupsize
                from comfy_kitchen import int8_linear
                return int8_linear(x, weight, weight_scale, bias=bias, out_dtype=x.dtype,
                                    convrot=True, convrot_groupsize=groupsize)

            @staticmethod
            def backward(ctx, grad_output):
                weight, weight_scale = ctx.saved_tensors
                weight_dq = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
                    weight, weight_scale.reshape(-1, 1), ctx.groupsize,
                    qft._CONVROT_DTYPE_CODES[ctx.x_dtype],
                )
                return grad_output.to(ctx.x_dtype) @ weight_dq, None, None, None, None, None

        resident_weight_ptr = layer.weight.data_ptr()
        with _Capture() as cap:
            _CloningFn.apply(x, layer.weight, layer.weight_scale, None, 256, "cloning").sum().backward()
        weight_aliases = [t for t in cap.saved if t.data_ptr() == resident_weight_ptr]
        assert weight_aliases == [], (
            "the cloned weight should NOT be an alias of the resident buffer -- "
            "if this assertion failed, the alias check above cannot detect a "
            "retention regression")


# ---------------------------------------------------------------------------
# 5. Trainable weight refused loudly
# ---------------------------------------------------------------------------

class TestTrainableWeightRefused:
    def test_refuses_a_parameter_weight_naming_the_layer_and_the_reason(self):
        parent = nn.Module()
        parent.q_proj = _convrot_layer(device="cpu")
        parent.q_proj.weight = nn.Parameter(parent.q_proj.weight.clone(), requires_grad=False)
        with pytest.raises(RuntimeError) as excinfo:
            qft.enable_frozen_training_fused(parent)
        assert "q_proj" in str(excinfo.value)
        assert "nn.Parameter" in str(excinfo.value)

    def test_refuses_a_requires_grad_buffer_naming_the_layer_and_the_reason(self):
        parent = nn.Module()
        parent.q_proj = _convrot_layer(device="cpu")
        parent.q_proj.weight_scale = parent.q_proj.weight_scale.clone().requires_grad_(True)
        with pytest.raises(RuntimeError) as excinfo:
            qft.enable_frozen_training_fused(parent)
        assert "q_proj" in str(excinfo.value)
        assert "weight_scale" in str(excinfo.value)
        assert "requires grad" in str(excinfo.value)

    def test_refuses_any_requires_grad_parameter_on_the_layer_naming_it(self):
        parent = nn.Module()
        parent.q_proj = _convrot_layer(device="cpu")
        parent.q_proj.stray = nn.Parameter(torch.zeros(1), requires_grad=True)
        with pytest.raises(RuntimeError) as excinfo:
            qft.enable_frozen_training_fused(parent)
        assert "stray" in str(excinfo.value)
        assert "requires grad" in str(excinfo.value)

    def test_w4a8_refuses_a_parameter_weight_naming_it(self):
        parent = nn.Module()
        parent.fc = _w4a8_layer(device="cpu")
        parent.fc.weight = nn.Parameter(parent.fc.weight.clone(), requires_grad=False)
        with pytest.raises(RuntimeError) as excinfo:
            qft.enable_frozen_training_fused(parent)
        assert "fc" in str(excinfo.value)
        assert "weight" in str(excinfo.value)

    def test_w4a8_sidecar_promoted_to_a_parameter_is_refused_regardless_of_its_name(self):
        """Previously this pinned a GAP: `_frozen_violation`'s second loop
        iterated the literal names "weight"/"weight_scale"/"bias", so a W4A8
        sidecar promoted to `nn.Parameter(..., requires_grad=False)` was
        accepted. The loop now walks `named_buffers`/`named_parameters`, so the
        name list cannot rot as sidecars are added, and this test pins the
        refusal instead.
        """
        parent = nn.Module()
        parent.fc = _w4a8_layer(device="cpu")
        parent.fc.weight_s_channel = nn.Parameter(parent.fc.weight_s_channel.clone(), requires_grad=False)
        with pytest.raises(RuntimeError) as excinfo:
            qft.enable_frozen_training_fused(parent)
        assert "weight_s_channel" in str(excinfo.value)
        assert "nn.Parameter" in str(excinfo.value)
        assert parent.fc._frozen_training_fused is False

        # requires_grad=True is caught by the first (named_parameters) loop, with
        # the more specific message.
        parent2 = nn.Module()
        parent2.fc = _w4a8_layer(device="cpu")
        parent2.fc.weight_s_channel = nn.Parameter(parent2.fc.weight_s_channel.clone(), requires_grad=True)
        with pytest.raises(RuntimeError) as excinfo:
            qft.enable_frozen_training_fused(parent2)
        assert "weight_s_channel" in str(excinfo.value)
        assert "requires grad" in str(excinfo.value)

    def test_mutation_check_the_old_name_list_walk_would_miss_the_sidecar(self):
        """Negative control for the test above: the pre-fix name-list loop,
        reproduced here, accepts the promoted sidecar -- so the refusal above is
        produced by the registration walk and not by something else."""
        layer = _w4a8_layer(device="cpu")
        layer.weight_s_channel = nn.Parameter(layer.weight_s_channel.clone(), requires_grad=False)

        def _name_list_violation(module):
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    return f"parameter '{name}' requires grad"
            for name in ("weight", "weight_scale", "bias"):  # THE OLD LIST
                tensor = getattr(module, name, None)
                if tensor is None:
                    continue
                if isinstance(tensor, nn.Parameter):
                    return f"'{name}' is an nn.Parameter, not a frozen buffer"
                if tensor.requires_grad:
                    return f"'{name}' requires grad"
            return None

        assert _name_list_violation(layer) is None
        assert qft._frozen_violation(layer) is not None

    def test_a_valid_layer_is_actually_enabled(self):
        """The premise: a properly frozen layer is accepted (not merely that
        every case in this class happens to be rejected)."""
        parent = nn.Module()
        parent.q_proj = _convrot_layer(device="cpu")
        n = qft.enable_frozen_training_fused(parent, label="unit-test")
        assert n == 1
        assert parent.q_proj._frozen_training_fused is True
        assert parent.q_proj._frozen_training_path == "q_proj"

    def test_mutation_check_removing_the_parameter_guard_would_go_undetected_by_nothing_else(self):
        """Negative control: directly exercise `_frozen_violation` with the
        isinstance(Parameter) branch skipped, the way a regression that
        deleted it would behave, and confirm the case that must raise does
        not raise under that mutation -- i.e. the guard is load-bearing."""
        layer = _convrot_layer(device="cpu")
        layer.weight = nn.Parameter(layer.weight.clone(), requires_grad=False)

        def _mutated_frozen_violation(module):
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    return f"parameter '{name}' requires grad"
            for name in ("weight", "weight_scale", "bias"):
                tensor = getattr(module, name, None)
                if tensor is None:
                    continue
                # THE MUTATION: the nn.Parameter check is gone.
                if tensor.requires_grad:
                    return f"'{name}' requires grad"
            return None

        assert _mutated_frozen_violation(layer) is None, (
            "under the mutated guard a Parameter weight is accepted -- proving "
            "the real isinstance(nn.Parameter) branch is what makes the real "
            "test above raise")
        assert qft._frozen_violation(layer) is not None


# ---------------------------------------------------------------------------
# 6. Exact-type gate: a subclass is not enabled
# ---------------------------------------------------------------------------

class _ConvRotSubclass(ConvRotInt8Linear):
    pass


class _W4A8Subclass(W4A8Linear):
    pass


class TestExactTypeGate:
    def test_a_convrot_subclass_is_not_enabled(self):
        parent = nn.Module()
        parent.sub = _ConvRotSubclass(
            IN, OUT, bias=False, compute_dtype=torch.bfloat16,
            convrot_groupsize=256, marker_numel=MARKER_NUMEL,
        )
        n = qft.enable_frozen_training_fused(parent)
        assert n == 0
        assert parent.sub._frozen_training_fused is False

    def test_a_w4a8_subclass_is_not_enabled(self):
        parent = nn.Module()
        parent.sub = _W4A8Subclass(IN, OUT, bias=False, compute_dtype=torch.bfloat16)
        n = qft.enable_frozen_training_fused(parent)
        assert n == 0
        assert parent.sub._frozen_training_fused is False

    def test_mutation_check_isinstance_would_wrongly_accept_the_subclass(self):
        """Negative control: proves the base-class-only dispatch this test
        pins is not vacuously true for every possible check."""
        sub = _ConvRotSubclass(
            IN, OUT, bias=False, compute_dtype=torch.bfloat16,
            convrot_groupsize=256, marker_numel=MARKER_NUMEL,
        )
        assert isinstance(sub, tuple(qft._supported_classes())), (
            "the subclass is not even an isinstance of the supported classes -- "
            "the exact-type test above would be meaningless")
        assert type(sub) not in qft._supported_classes()


# ---------------------------------------------------------------------------
# 7. Unsupported activation falls back, does not raise
# ---------------------------------------------------------------------------

class TestUnsupportedActivationFallsBack:
    def test_maybe_frozen_fused_forward_declines_a_cpu_activation_without_raising(self):
        layer = _convrot_layer(device="cpu")
        x = torch.randn(2, IN, dtype=torch.bfloat16, device="cpu")
        assert qft.maybe_frozen_fused_forward(layer, x) is None

    @CUDA
    def test_maybe_frozen_fused_forward_declines_float64_on_cuda_without_raising(self):
        layer = _enable(_convrot_layer(device="cuda"))
        x = torch.randn(2, IN, dtype=torch.float64, device="cuda")
        assert qft.maybe_frozen_fused_forward(layer, x) is None

    def test_w4a8_maybe_frozen_fused_forward_declines_a_cpu_activation_without_raising(self):
        layer = _w4a8_layer(device="cpu")
        x = torch.randn(2, IN, dtype=torch.bfloat16, device="cpu")
        assert qft.maybe_frozen_fused_forward(layer, x) is None

    @CUDA
    def test_w4a8_maybe_frozen_fused_forward_declines_float64_on_cuda_without_raising(self):
        layer = _enable(_w4a8_layer(device="cuda"))
        x = torch.randn(2, IN, dtype=torch.float64, device="cuda")
        assert qft.maybe_frozen_fused_forward(layer, x) is None

    def test_cpu_activation_through_the_full_module_falls_back_and_matches_dequant(self):
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cpu"))
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cpu", requires_grad=True)
        out = layer(x)
        expected = layer._dequant_forward(x.detach().clone().requires_grad_(True))
        assert torch.equal(out.detach(), expected.detach())

    @CUDA
    def test_mutation_check_removing_the_dtype_check_stops_declining_float64(self):
        """Negative control, through the dispatcher: widening the supported-dtype
        tuple makes `maybe_frozen_fused_forward` STOP returning None for a
        float64 activation and dispatch to the kernel. Deleting the
        `x.dtype not in _SUPPORTED_ACTIVATION_DTYPES` clause therefore breaks the
        declination assertions above rather than leaving them vacuously green."""
        layer = _enable(_convrot_layer(device="cuda"))
        x = torch.randn(2, IN, dtype=torch.float64, device="cuda")

        assert qft.maybe_frozen_fused_forward(layer, x) is None

        with mock.patch.object(
            qft, "_SUPPORTED_ACTIVATION_DTYPES",
            qft._SUPPORTED_ACTIVATION_DTYPES + (torch.float64,),
        ):
            with mock.patch(
                "comfy_kitchen.int8_linear",
                return_value=torch.zeros(2, OUT, dtype=torch.float64, device="cuda"),
            ) as kernel:
                out = qft.maybe_frozen_fused_forward(layer, x)
        assert out is not None
        kernel.assert_called_once()


# ---------------------------------------------------------------------------
# 8. A kernel failure fails the run, does not fall back to dequant
# ---------------------------------------------------------------------------

@CUDA
class TestKernelFailureIsFatal:
    def test_convrot_kernel_failure_raises_and_names_the_layer_and_never_recovers(self):
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda"))
        layer._frozen_training_path = "blocks.3.attn.q_proj"
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        with mock.patch("comfy_kitchen.int8_linear", side_effect=RuntimeError("kernel boom")) as broken:
            with pytest.raises(RuntimeError) as first:
                layer(x)
            assert "blocks.3.attn.q_proj" in str(first.value)
            assert "Refusing to continue on the dequant path" in str(first.value)

            with pytest.raises(RuntimeError) as second:
                layer(x)
            assert "blocks.3.attn.q_proj" in str(second.value)

        assert broken.call_count == 2, (
            "the second call did not reach the fused kernel -- the layer silently "
            "rerouted to the dequant path after the first failure instead of "
            "staying fatal")

    def test_w4a8_kernel_failure_raises_and_names_the_layer_and_never_recovers(self):
        layer = _enable(_w4a8_layer(dtype=torch.bfloat16, device="cuda"))
        layer._frozen_training_path = "blocks.1.mlp.fc1"
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        with mock.patch(
            "comfy_kitchen.tensor.w4a8_int8_linear", side_effect=RuntimeError("kernel boom")
        ) as broken:
            with pytest.raises(RuntimeError) as first:
                layer(x)
            assert "blocks.1.mlp.fc1" in str(first.value)

            with pytest.raises(RuntimeError) as second:
                layer(x)
            assert "blocks.1.mlp.fc1" in str(second.value)

        assert broken.call_count == 2, (
            "the second call did not reach the fused kernel -- the layer silently "
            "rerouted to the dequant path after the first failure instead of "
            "staying fatal")

    def test_mutation_check_swallowing_the_kernel_failure_would_go_undetected_by_nothing_else(self):
        """Negative control: simulate the "catch and fall back" mutation this
        design explicitly refuses, and confirm it would make the second call
        SUCCEED (no RuntimeError, no second call to the broken kernel) -- the
        exact outcome the real test above must not observe."""
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda"))
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        def _mutated_forward(x):
            if layer._frozen_training_fused:
                try:
                    out = qft.maybe_frozen_fused_forward(layer, x)
                except Exception:
                    out = None  # THE MUTATION: swallow and fall back
                if out is not None:
                    return out
            return layer._dequant_forward(x)

        with mock.patch("comfy_kitchen.int8_linear", side_effect=RuntimeError("kernel boom")) as broken:
            out = _mutated_forward(x)  # does NOT raise under the mutation
        assert out is not None
        assert broken.call_count == 1, (
            "under the mutated (swallowing) forward, the kernel is still only "
            "invoked once and the failure is hidden -- this is precisely the "
            "silent-fallback outcome the real design refuses to allow")


# ---------------------------------------------------------------------------
# 9. Gradient checkpointing: original forward AND recompute take the same path
# ---------------------------------------------------------------------------

@CUDA
class TestGradientCheckpointing:
    """Design doc 6, "Gradient checkpointing": test the call counts directly
    rather than inferring grad mode from checkpoint internals."""

    @staticmethod
    def _paths_taken(layer, x):
        calls = []
        real_kernel = ck.int8_linear
        real_dequant = ConvRotInt8Linear._dequant_forward

        def spy_kernel(*a, **k):
            calls.append("fused")
            return real_kernel(*a, **k)

        def spy_dequant(self, inp):
            calls.append("dequant")
            return real_dequant(self, inp)

        with mock.patch("comfy_kitchen.int8_linear", side_effect=spy_kernel):
            with mock.patch.object(ConvRotInt8Linear, "_dequant_forward",
                                   autospec=True, side_effect=spy_dequant):
                out = torch.utils.checkpoint.checkpoint(
                    lambda inp: layer(inp), x, use_reentrant=False)
                out.float().sum().backward()
        return calls

    def test_flag_on_takes_the_fused_path_in_both_the_forward_and_the_recompute(self):
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda"))
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        assert self._paths_taken(layer, x) == ["fused", "fused"]

    def test_flag_off_takes_the_dequant_path_in_both_the_forward_and_the_recompute(self):
        layer = _convrot_layer(dtype=torch.bfloat16, device="cuda")
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        assert self._paths_taken(layer, x) == ["dequant", "dequant"]

    def test_checkpointed_grad_x_still_matches_the_dequant_path_bitwise(self):
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda"))
        x_fused = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_dequant = x_fused.detach().clone().requires_grad_(True)

        torch.utils.checkpoint.checkpoint(
            lambda inp: layer(inp), x_fused, use_reentrant=False).float().sum().backward()
        layer._dequant_forward(x_dequant).float().sum().backward()

        assert torch.equal(x_fused.grad, x_dequant.grad)

    def test_mutation_check_a_single_pass_would_not_produce_two_entries(self):
        """Negative control: the same spies WITHOUT checkpointing record one
        entry, so the two-entry sequences above are produced by the recompute
        and not by the spies double-counting."""
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda"))
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        calls = []
        real_kernel = ck.int8_linear

        def spy_kernel(*a, **k):
            calls.append("fused")
            return real_kernel(*a, **k)

        with mock.patch("comfy_kitchen.int8_linear", side_effect=spy_kernel):
            layer(x).float().sum().backward()
        assert calls == ["fused"]


# ---------------------------------------------------------------------------
# 10. Offload: the saved buffers must survive the layer moving
# ---------------------------------------------------------------------------

@CUDA
class TestOffloadBetweenForwardAndBackward:
    """Design doc 6, "Block swap and offload": prove the offloader does not move
    the saved buffers out from under backward.

    `LayerOffloadConductor` moves a layer with `.to(...)`, which REPLACES a
    buffer object, so the saved reference keeps the old storage alive and
    backward dequantizes the same codes it ran the forward on. The regression
    this guards is the conductor's parameter staging (pinned CPU buffers reused
    with in-place `copy_`) being extended to BUFFERS: that writes into the
    storage `ctx.saved_tensors` holds, and `grad_x` silently becomes another
    block's weight with no exception raised.
    """

    def test_grad_x_is_unchanged_when_the_layer_moves_to_cpu_before_backward(self):
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda"))
        x_fused = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_dequant = x_fused.detach().clone().requires_grad_(True)

        out = layer(x_fused)
        layer._dequant_forward(x_dequant).float().sum().backward()

        layer.to("cpu")
        assert layer.weight.device.type == "cpu"
        out.float().sum().backward()

        assert torch.equal(x_fused.grad, x_dequant.grad), (
            "grad_x changed after the layer was offloaded between forward and "
            "backward -- the saved buffers did not survive the move")

    def test_mutation_check_an_in_place_data_copy_does_corrupt_grad_x(self):
        """Negative control: an offloader that reused the buffer storage the way
        `layer_offload_conductor` reuses a parameter's (`param.data.copy_`)
        writes through the saved reference, and the assertion above catches it.

        `.data` is what makes it silent: it bypasses the autograd version
        counter, so backward dequantizes the other layer's codes and returns a
        wrong `grad_x` with no exception (see the sibling test for the plain
        `copy_` case, which does raise)."""
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda", seed=0))
        other = _convrot_layer(dtype=torch.bfloat16, device="cuda", seed=1)
        x_fused = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_dequant = x_fused.detach().clone().requires_grad_(True)

        out = layer(x_fused)
        layer._dequant_forward(x_dequant).float().sum().backward()

        layer.weight.data.copy_(other.weight)  # THE MUTATION: in-place, not a replacement
        out.float().sum().backward()

        assert not torch.equal(x_fused.grad, x_dequant.grad), (
            "writing another layer's codes into the saved storage did not change "
            "grad_x -- the offload assertion above would be vacuous")

    def test_a_plain_in_place_copy_into_a_saved_buffer_raises(self):
        """The other half of the same regression: an in-place write that does
        NOT go through `.data` bumps the saved tensor's version counter and
        backward refuses rather than returning a wrong gradient."""
        layer = _enable(_convrot_layer(dtype=torch.bfloat16, device="cuda", seed=0))
        other = _convrot_layer(dtype=torch.bfloat16, device="cuda", seed=1)
        x = torch.randn(4, IN, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        out = layer(x)
        with torch.no_grad():
            layer.weight.copy_(other.weight)
        with pytest.raises(RuntimeError, match="inplace operation"):
            out.float().sum().backward()


# ---------------------------------------------------------------------------
# 11. Enable-time branches: shape skip (not a raise) and root-path naming
# ---------------------------------------------------------------------------

class TestEnableTimeBranches:
    def test_a_convrot_groupsize_violation_is_skipped_not_raised(self):
        parent = nn.Module()
        parent.q_proj = _convrot_layer(device="cpu")
        # The constructor forbids this combination, so it is set afterwards; the
        # skip branch exists for a runtime/contract mismatch, not a fresh build.
        parent.q_proj.convrot_groupsize = 128
        assert qft.enable_frozen_training_fused(parent) == 0
        assert parent.q_proj._frozen_training_fused is False

    def test_a_w4a8_group_size_violation_is_skipped_not_raised(self):
        parent = nn.Module()
        parent.fc = _w4a8_layer(device="cpu")
        parent.fc.group_size = 24  # IN=256 is not divisible by 24
        assert qft.enable_frozen_training_fused(parent) == 0
        assert parent.fc._frozen_training_fused is False

    def test_mutation_check_the_same_layers_are_enabled_without_the_violation(self):
        """Negative control: the two layers above are enabled when their
        groupsize is left alone, so the zero counts are produced by the shape
        check and not by the layers being ineligible for another reason."""
        parent = nn.Module()
        parent.q_proj = _convrot_layer(device="cpu")
        parent.fc = _w4a8_layer(device="cpu")
        assert qft.enable_frozen_training_fused(parent) == 2

    def test_enabling_on_a_bare_layer_names_the_root_by_its_class(self):
        layer = _convrot_layer(device="cpu")
        assert qft.enable_frozen_training_fused(layer) == 1
        assert layer._frozen_training_path == "ConvRotInt8Linear"

    def test_a_bare_layer_refusal_names_the_root_by_its_class(self):
        layer = _convrot_layer(device="cpu")
        layer.weight_scale = layer.weight_scale.clone().requires_grad_(True)
        with pytest.raises(RuntimeError) as excinfo:
            qft.enable_frozen_training_fused(layer)
        assert "'ConvRotInt8Linear'" in str(excinfo.value)

    def test_mutation_check_named_modules_really_gives_the_root_an_empty_name(self):
        """Negative control: the root path is "" before the fallback renames it,
        so the assertions above are exercising that fallback."""
        layer = _convrot_layer(device="cpu")
        assert [name for name, _ in layer.named_modules()] == [""]

    def test_zero_enabled_layers_are_reported(self, capsys):
        parent = nn.Module()
        parent.plain = nn.Linear(4, 4)
        assert qft.enable_frozen_training_fused(parent, label="zero-case") == 0
        out = capsys.readouterr().out
        assert "zero-case" in out
        assert "0 layer(s)" in out

    def test_mutation_check_a_nonzero_count_prints_a_different_number(self, capsys):
        """Negative control for the line above: the count in the message tracks
        the layers actually enabled."""
        parent = nn.Module()
        parent.q_proj = _convrot_layer(device="cpu")
        assert qft.enable_frozen_training_fused(parent, label="one-case") == 1
        out = capsys.readouterr().out
        assert "1 layer(s)" in out
        assert "0 layer(s)" not in out


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v"]))
