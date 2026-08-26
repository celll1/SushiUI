"""Frozen-base fused forward for quantized ConvRot Linears in training (CANDIDATE).

Opt-in, default OFF. Implements ``docs/guides/INT8_CONVROT_TRAINING_DESIGN.md``
sections 2.1-2.3: for a Linear whose quantized weight is FROZEN, run the fused
inference kernel in forward even under autograd, and compute ``grad_x`` in
backward from a transiently rebuilt floating weight.

Two properties are the whole point, and both are structural rather than tuned:

* the autograd node saves ONLY the resident code/scale buffers, so no
  dequantized ``(out, in)`` weight is retained across the forward; and
* ``grad_x`` is the same expression ``F.linear`` on a dequantized weight
  computes, so it is bitwise equal to today's path rather than merely close.

Nothing here touches plain ``Int8Linear``/``Fp8Linear``: their W8A8 training
forward is closed by gates G3/G4 (``backend/core/training/INT8_W8A8_TRAINING_GATE.md``)
and their dequant path is a single promoted multiply, not the inverse Hadamard
that makes this trade worth measuring for ConvRot.

Block swap (MiniMax-H3 exposes it) is safe for the same reason G4 recorded:
``LayerOffloadConductor`` moves a layer with ``layer.to(...)``, which REPLACES a
buffer rather than writing into it, so a saved reference stays valid and simply
keeps that block's 1-byte codes resident until backward -- where the shipped
path already keeps 2 bytes/element resident in the same situation.

NOT DONE HERE: the artifact/base-function metadata contract of design doc 5.
An adapter trained through this path is coupled to it, and no loader refuses a
mismatch yet. That is why the whole feature sits behind an environment flag.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


# Read ONCE at import, default off. Deliberately not a config key or an API
# parameter: this path has no measured performance/quality gate yet, so there is
# nothing an operator could consent to. Precedent: SUSHI_INT8_MM,
# SUSHI_SENSENOVA_CONVROT_DEQUANT.
FROZEN_TRAINING_FUSED_ENV = "SUSHI_CONVROT_TRAIN_FUSED"
_FROZEN_TRAINING_FUSED_REQUESTED = os.environ.get(FROZEN_TRAINING_FUSED_ENV, "0") == "1"

# Backward/forward compute dtypes this path serves. The backward dtype is the
# INCOMING activation dtype; there is deliberately no separate dtype setting.
_SUPPORTED_ACTIVATION_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

_CONVROT_DTYPE_CODES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def frozen_training_fused_requested() -> bool:
    """True when ``SUSHI_CONVROT_TRAIN_FUSED=1`` was set for this process."""
    return _FROZEN_TRAINING_FUSED_REQUESTED


def _fused_failure(layer_path: str, flavour: str, exc: BaseException) -> RuntimeError:
    """The mid-run kernel failure is fatal, by design (design doc 2.3).

    Falling back to ``_dequant_forward`` for the remaining steps would fit one
    artifact against two different base functions -- the fused and dequant
    forwards compute different functions (measured ~1% relative on real ConvRot
    weights), so half a run against each is not a run.
    """
    return RuntimeError(
        f"{flavour} fused frozen-base forward failed on layer '{layer_path}': "
        f"{type(exc).__name__}: {exc}. Refusing to continue on the dequant path: "
        f"the fused and dequant forwards are different functions, so switching "
        f"base function mid-run would fit one artifact against two of them. "
        f"Unset {FROZEN_TRAINING_FUSED_ENV} to run the whole training on the "
        f"dequant path."
    )


class ConvRotFrozenLinearFn(torch.autograd.Function):
    """Fused ConvRot W8A8 forward, dequantized floating ``grad_input`` backward.

    ``weight``/``weight_scale``/``bias`` are passed as inputs purely so
    ``save_for_backward`` is legal on them; all three are frozen buffers and all
    three return ``None``. Saving a buffer does not copy its storage, so the
    saved references cost no allocation -- that is the retention property this
    whole module exists for.
    """

    @staticmethod
    def forward(ctx, x, weight, weight_scale, bias, groupsize, layer_path):
        from comfy_kitchen import int8_linear

        ctx.save_for_backward(weight, weight_scale)
        ctx.x_dtype = x.dtype
        ctx.groupsize = groupsize
        try:
            return int8_linear(
                x,
                weight,
                weight_scale,
                bias=bias,
                out_dtype=x.dtype,
                convrot=True,
                convrot_groupsize=groupsize,
            )
        except Exception as exc:
            raise _fused_failure(layer_path, "ConvRot INT8", exc) from exc

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None
        weight, weight_scale = ctx.saved_tensors
        # Straight-through in the activation-quantization sense: the fused
        # forward's literal derivative is piecewise constant. `grad_x` is the
        # exact derivative of the dequantized matmul, i.e. the current path's.
        weight_dq = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
            weight,
            weight_scale.reshape(-1, 1),
            ctx.groupsize,
            _CONVROT_DTYPE_CODES[ctx.x_dtype],
        )
        return grad_output.to(ctx.x_dtype) @ weight_dq, None, None, None, None, None


class W4A8FrozenLinearFn(torch.autograd.Function):
    """Fused packed-W4A8 forward, dequantized floating ``grad_input`` backward.

    Same contract as ``ConvRotFrozenLinearFn``; W4A8 simply carries more frozen
    sidecars (relative and per-channel scales, an optional codebook and an
    optional correction), all of which are saved by reference and none of which
    receives a gradient.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        s_rel,
        s_channel,
        codebook,
        correction,
        bias,
        group_size,
        convrot_groupsize,
        layer_path,
    ):
        from comfy_kitchen.tensor import w4a8_int8_linear

        ctx.save_for_backward(weight, s_rel, s_channel, codebook, correction)
        ctx.x_dtype = x.dtype
        ctx.group_size = group_size
        ctx.convrot_groupsize = convrot_groupsize
        try:
            return w4a8_int8_linear(
                x,
                weight,
                s_rel,
                s_channel,
                codebook=codebook,
                correction=correction,
                bias=bias,
                group_size=group_size,
                convrot_groupsize=convrot_groupsize,
                out_dtype=x.dtype,
            )
        except Exception as exc:
            raise _fused_failure(layer_path, "W4A8 INT8", exc) from exc

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,) * 10
        from comfy_kitchen.tensor import dequantize_w4a8_int8_weight

        weight, s_rel, s_channel, codebook, correction = ctx.saved_tensors
        weight_dq = dequantize_w4a8_int8_weight(
            weight,
            s_rel,
            s_channel,
            codebook=codebook,
            correction=correction,
            group_size=ctx.group_size,
            convrot_groupsize=ctx.convrot_groupsize,
            output_dtype=ctx.x_dtype,
        )
        return (grad_output.to(ctx.x_dtype) @ weight_dq,) + (None,) * 9


def _supported_classes() -> "tuple[type, ...]":
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from core.models.common.w4a8_linear import W4A8Linear

    return (ConvRotInt8Linear, W4A8Linear)


def _frozen_violation(module: nn.Module) -> "str | None":
    """Name the first thing on ``module`` that would make this path illegal.

    A trainable weight must never reach the fused kernel: it would receive no
    gradient at all (the function returns None for it), so the run would look
    healthy while optimizing nothing. SenseNova's full fine-tune materializes
    its trainable half into real bf16 ``nn.Parameter`` Linears, which changes the
    module TYPE and so is already excluded by the dispatch rule; this check
    catches every other way a caller could get here.
    """
    for name, param in module.named_parameters(recurse=False):
        if param.requires_grad:
            return f"parameter '{name}' requires grad"
    # Walked by registration, not by a name list: a W4A8 sidecar (or any sidecar
    # added later) must be covered without editing this function.
    for name, tensor in module.named_buffers(recurse=False):
        if tensor.requires_grad:
            return f"'{name}' requires grad"
    promoted = next(iter(module.named_parameters(recurse=False)), None)
    if promoted is not None:
        return f"'{promoted[0]}' is an nn.Parameter, not a frozen buffer"
    return None


def _shape_violation(module: nn.Module) -> "str | None":
    """Validate the per-flavour shape/groupsize contract the kernel assumes."""
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from core.models.common.w4a8_linear import W4A8Linear

    if type(module) is ConvRotInt8Linear:
        groupsize = int(module.convrot_groupsize)
        if groupsize != 256 or module.in_features % 256:
            return (
                f"ConvRot requires K divisible by groupsize 256, got "
                f"K={module.in_features}, groupsize={groupsize}"
            )
        return None
    if type(module) is W4A8Linear:
        group_size = int(module.group_size)
        if module.in_features % 2 or module.in_features % group_size:
            return (
                f"W4A8 requires K divisible by 2 and by group_size={group_size}, "
                f"got K={module.in_features}"
            )
        return None
    return f"{type(module).__name__} has no fused frozen-base forward"


def enable_frozen_training_fused(module: nn.Module, *, label: str = "") -> int:
    """Opt every eligible frozen quantized Linear under ``module`` into the path.

    EXPLICIT and per-instance, deliberately distinct from ``_force_dequant``,
    ``_allow_int8_mm``, grad mode and any inference env variable (design doc 7.2
    forbids overloading all four). Returns the number of layers enabled.

    Raises rather than skipping when a matched module's weight is trainable: a
    caller asking for this on a trainable weight has a wrong mental model, and a
    silent skip would hide it until the artifact came out untrained.
    """
    supported = _supported_classes()
    enabled = 0
    for path, child in module.named_modules():
        if type(child) not in supported:
            continue
        # named_modules() names the root "", which is what a caller passing a
        # single Linear gets; the path exists to identify the layer in a failure
        # message, so it must never be empty.
        path = path or type(child).__name__
        violation = _frozen_violation(child)
        if violation is not None:
            raise RuntimeError(
                f"Cannot enable the fused frozen-base training forward on "
                f"'{path}' ({type(child).__name__}): {violation}. This path "
                f"returns no weight gradient, so a trainable weight routed "
                f"through it would silently never move. Materialize the "
                f"trainable half to floating parameters instead."
            )
        shape_violation = _shape_violation(child)
        if shape_violation is not None:
            # Not fatal: the dispatch rule sends unsupported shapes to today's
            # dequant path, which serves them correctly.
            continue
        child._frozen_training_fused = True
        child._frozen_training_path = path
        enabled += 1
    if label:
        # Printed for 0 too: a call that matches nothing is the difference between
        # measuring this path and measuring the dequant path under its name.
        detail = (
            "backward computes grad_input from a rebuilt floating weight in the "
            "activation dtype" if enabled else
            f"no eligible {'/'.join(c.__name__ for c in supported)} was found "
            f"(a module whose shape the kernel does not serve is skipped)"
        )
        print(
            f"[QuantFrozenTraining] {label}: fused frozen-base forward enabled on "
            f"{enabled} layer(s) of {type(module).__name__}; {detail}"
        )
    return enabled


def maybe_frozen_fused_forward(module: nn.Module, x: torch.Tensor) -> "torch.Tensor | None":
    """Run the fused frozen-base forward, or None to use the dequant path.

    Implements the design doc's dispatch rule. Conditions 1, 2, 3 and 5 are
    settled at enable time; 4 (the activation) is a property of the call, so it
    is checked here. Returning None is only ever a pre-dispatch decision about
    dtype/device -- a KERNEL failure raises, it does not fall back.
    """
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from core.models.common.w4a8_linear import W4A8Linear

    if not x.is_cuda or x.dtype not in _SUPPORTED_ACTIVATION_DTYPES:
        return None
    path = getattr(module, "_frozen_training_path", "") or type(module).__name__

    if type(module) is ConvRotInt8Linear:
        return ConvRotFrozenLinearFn.apply(
            x,
            module.weight,
            module.weight_scale,
            module.bias,
            int(module.convrot_groupsize),
            path,
        )
    if type(module) is W4A8Linear:
        return W4A8FrozenLinearFn.apply(
            x,
            module.weight,
            module.weight_s_rel,
            module.weight_s_channel,
            module.weight_codebook,
            module.weight_correction,
            module.bias,
            int(module.group_size),
            int(module.convrot_groupsize),
            path,
        )
    return None
