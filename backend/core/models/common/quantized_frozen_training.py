"""Frozen-base fused forward for quantized Linears in training.

Opt-in, default OFF. Implements ``docs/guides/INT8_CONVROT_TRAINING_DESIGN.md``
sections 2.1-2.3: for a Linear whose quantized weight is FROZEN, run the fused
inference kernel in forward even under autograd, and compute ``grad_x`` in
backward from a floating weight. The transient path rebuilds that weight one
layer at a time; an optional speed path retains a non-persistent BF16 cache.

The generic candidate path has two structural properties:

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


class ConvRotCachedBackwardLinearFn(torch.autograd.Function):
    """ConvRot INT8 forward with a cached floating weight for ``grad_input``."""

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        weight_scale,
        bias,
        backward_weight,
        groupsize,
        layer_path,
    ):
        from comfy_kitchen import int8_linear

        ctx.save_for_backward(backward_weight)
        ctx.x_dtype = x.dtype
        ctx.backward_dtype = backward_weight.dtype
        try:
            return int8_linear(
                x,
                weight,
                weight_scale,
                bias=bias,
                out_dtype=backward_weight.dtype,
                convrot=True,
                convrot_groupsize=groupsize,
            )
        except Exception as exc:
            raise _fused_failure(layer_path, "ConvRot INT8", exc) from exc

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,) * 8
        (backward_weight,) = ctx.saved_tensors
        grad_input = (
            grad_output.to(ctx.backward_dtype) @ backward_weight
        ).to(ctx.x_dtype)
        return grad_input, None, None, None, None, None, None, None


class ConvRotPrefetchBackwardLinearFn(torch.autograd.Function):
    """ConvRot INT8 forward with block-scoped lazy BF16 backward weights."""

    @staticmethod
    def forward(ctx, x, weight, weight_scale, bias, groupsize, layer_path, cache):
        from comfy_kitchen import int8_linear

        ctx.x_dtype = x.dtype
        ctx.backward_dtype = cache.dtype
        ctx.layer_path = layer_path
        ctx.cache = cache
        try:
            return int8_linear(
                x,
                weight,
                weight_scale,
                bias=bias,
                out_dtype=cache.dtype,
                convrot=True,
                convrot_groupsize=groupsize,
            )
        except Exception as exc:
            raise _fused_failure(layer_path, "ConvRot INT8", exc) from exc

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,) * 7
        backward_weight = ctx.cache.acquire(
            ctx.layer_path, grad_output.device, ctx.backward_dtype
        )
        grad_input = grad_output.to(ctx.backward_dtype) @ backward_weight
        return grad_input.to(ctx.x_dtype), None, None, None, None, None, None


class ConvRotBlockPrefetchCache:
    """Bounded GPU cache for ConvRot BF16 grad-input weights.

    The first backward use in block N materializes N and schedules N-1 on a
    dedicated CUDA stream. Qwen's backward traverses transformer blocks in
    reverse order, including checkpoint recomputation, so the N-1 dequant can
    overlap N's attention/MLP backward. Non-block Linears use a small permanent
    cache and are reported separately.
    """

    def __init__(self, module: nn.Module, *, dtype: torch.dtype, cache_blocks: int,
                 prefetch_depth: int):
        if cache_blocks < 1:
            raise ValueError("ConvRot backward cache_blocks must be at least 1")
        if prefetch_depth < 0 or prefetch_depth >= cache_blocks:
            raise ValueError(
                "ConvRot backward prefetch_depth must be >=0 and smaller than cache_blocks"
            )
        self.dtype = dtype
        self.dtype_code = _CONVROT_DTYPE_CODES[dtype]
        self.cache_blocks = int(cache_blocks)
        self.prefetch_depth = int(prefetch_depth)
        self.blocks: dict[int, list[tuple[str, nn.Module]]] = {}
        self.outside: dict[str, nn.Module] = {}
        self.weights: dict[str, torch.Tensor] = {}
        self.resident_blocks: list[int] = []
        self.events: dict[int, torch.cuda.Event] = {}
        self.block_bytes: dict[int, int] = {}
        self.outside_bytes = 0
        self.dequant_calls = 0
        self.wait_count = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self._last_acquired_block = None

        device = None
        for path, child in module.named_modules():
            from core.models.common.convrot_int8_linear import ConvRotInt8Linear

            if type(child) is not ConvRotInt8Linear or _shape_violation(child) is not None:
                continue
            if child.weight.device.type != "cuda":
                raise RuntimeError(
                    f"Cannot build ConvRot prefetch metadata for '{path}' on "
                    f"{child.weight.device}; move the transformer to CUDA first."
                )
            device = child.weight.device
            parts = path.split(".")
            block = None
            if len(parts) >= 2 and parts[0] == "transformer_blocks":
                try:
                    block = int(parts[1])
                except ValueError:
                    block = None
            if block is None:
                self.outside[path] = child
            else:
                self.blocks.setdefault(block, []).append((path, child))
                self.block_bytes[block] = self.block_bytes.get(block, 0) + (
                    child.out_features * child.in_features
                    * torch.empty((), dtype=dtype).element_size()
                )
        if device is None:
            raise RuntimeError("No eligible CUDA ConvRot layers found for prefetch cache")
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

        # Outside-block projections have no stable reverse block boundary. They
        # are few and remain resident; the repeating 32-block body is bounded.
        with torch.cuda.stream(self.stream):
            for path, child in self.outside.items():
                value = self._dequant(child)
                self.weights[path] = value
                self.outside_bytes += value.numel() * value.element_size()
        self.outside_ready = torch.cuda.Event()
        self.outside_ready.record(self.stream)

    def _dequant(self, child: nn.Module) -> torch.Tensor:
        self.dequant_calls += 1
        return torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
            child.weight,
            child.weight_scale.reshape(-1, 1),
            int(child.convrot_groupsize),
            self.dtype_code,
        )

    def _schedule(self, block: int) -> None:
        if block in self.events or block not in self.blocks:
            return
        with torch.cuda.stream(self.stream):
            for path, child in self.blocks[block]:
                self.weights[path] = self._dequant(child)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.events[block] = event
        self.resident_blocks.append(block)

    def _evict_except(self, keep: set[int]) -> None:
        for block in list(self.resident_blocks):
            if block in keep:
                continue
            for path, _child in self.blocks.get(block, ()):
                self.weights.pop(path, None)
            self.events.pop(block, None)
            self.resident_blocks.remove(block)

    def acquire(self, path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if device != self.device or dtype != self.dtype:
            raise RuntimeError(
                f"ConvRot prefetch cache expects {self.device}/{self.dtype}, got "
                f"{device}/{dtype} for '{path}'"
            )
        current_stream = torch.cuda.current_stream(device)
        if path in self.outside:
            current_stream.wait_event(self.outside_ready)
            weight = self.weights[path]
            weight.record_stream(current_stream)
            return weight

        parts = path.split(".")
        block = int(parts[1])
        first_in_block = block != self._last_acquired_block
        if first_in_block:
            was_scheduled = block in self.events
            self._schedule(block)
            current_stream.wait_event(self.events[block])
            self.wait_count += 1
            if was_scheduled:
                self.prefetch_hits += 1
            else:
                self.prefetch_misses += 1
            self._last_acquired_block = block
            future = {
                candidate
                for candidate in range(block - 1, block - 1 - self.prefetch_depth, -1)
                if candidate in self.blocks
            }
            self._evict_except({block, *future})
            for candidate in sorted(future, reverse=True):
                self._schedule(candidate)
            # The configured ring is a hard Python residency bound. CUDA's
            # allocator may retain freed blocks, which peak metrics report.
            while len(self.resident_blocks) > self.cache_blocks:
                victim = self.resident_blocks[0]
                if victim == block:
                    break
                self._evict_except(set(self.resident_blocks[1:]))

        weight = self.weights[path]
        weight.record_stream(current_stream)
        return weight

    @property
    def resident_limit_bytes(self) -> int:
        largest = sorted(self.block_bytes.values(), reverse=True)[:self.cache_blocks]
        return self.outside_bytes + sum(largest)


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


def enable_frozen_training_cached_backward(
    module: nn.Module,
    *,
    dtype: torch.dtype,
    label: str = "",
) -> tuple[int, int]:
    """Use ConvRot INT8 forward and retain one floating weight for backward.

    The cache is a non-persistent buffer: it follows explicit module moves but
    is neither saved nor treated as the base model's authoritative weight.
    Returns ``(layer_count, cache_bytes)``.
    """
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear

    if dtype not in _SUPPORTED_ACTIVATION_DTYPES:
        raise ValueError(f"Unsupported ConvRot backward cache dtype: {dtype}")
    dtype_code = _CONVROT_DTYPE_CODES[dtype]
    enabled = 0
    cache_bytes = 0
    for path, child in module.named_modules():
        if type(child) is not ConvRotInt8Linear:
            continue
        path = path or type(child).__name__
        violation = _frozen_violation(child)
        if violation is not None:
            raise RuntimeError(
                f"Cannot enable cached ConvRot training on '{path}': {violation}. "
                "The cached path is only valid for a frozen base weight."
            )
        shape_violation = _shape_violation(child)
        if shape_violation is not None:
            continue
        if child.weight.device.type != "cuda":
            raise RuntimeError(
                f"Cannot build the ConvRot backward cache for '{path}' on "
                f"{child.weight.device}; move the transformer to CUDA first."
            )
        backward_weight = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
            child.weight,
            child.weight_scale.reshape(-1, 1),
            int(child.convrot_groupsize),
            dtype_code,
        )
        child.register_buffer(
            "_frozen_training_backward_weight",
            backward_weight,
            persistent=False,
        )
        child._frozen_training_fused = True
        child._frozen_training_cached_backward = True
        child._frozen_training_path = path
        enabled += 1
        cache_bytes += backward_weight.numel() * backward_weight.element_size()
    if label:
        print(
            f"[QuantFrozenTraining] {label}: ConvRot INT8 forward with cached "
            f"{dtype} backward weights on {enabled} layer(s), "
            f"cache={cache_bytes / 1024**3:.3f} GiB"
        )
    return enabled, cache_bytes


def enable_frozen_training_prefetch_backward(
    module: nn.Module,
    *,
    dtype: torch.dtype,
    cache_blocks: int = 2,
    prefetch_depth: int = 1,
    label: str = "",
) -> tuple[int, int, ConvRotBlockPrefetchCache]:
    """Use INT8 forward with a bounded, asynchronously prefetched BF16 cache."""
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear

    if dtype not in _SUPPORTED_ACTIVATION_DTYPES:
        raise ValueError(f"Unsupported ConvRot backward cache dtype: {dtype}")
    eligible = []
    for path, child in module.named_modules():
        if type(child) is not ConvRotInt8Linear:
            continue
        path = path or type(child).__name__
        violation = _frozen_violation(child)
        if violation is not None:
            raise RuntimeError(
                f"Cannot enable prefetched ConvRot training on '{path}': "
                f"{violation}. The cache is valid only for frozen base weights."
            )
        if _shape_violation(child) is None:
            eligible.append((path, child))
    cache = ConvRotBlockPrefetchCache(
        module,
        dtype=dtype,
        cache_blocks=cache_blocks,
        prefetch_depth=prefetch_depth,
    )
    for path, child in eligible:
        child._frozen_training_fused = True
        child._frozen_training_prefetch_backward = True
        child._frozen_training_prefetch_cache = cache
        child._frozen_training_path = path
    if label:
        print(
            f"[QuantFrozenTraining] {label}: ConvRot INT8 forward with "
            f"{cache_blocks}-block {dtype} backward cache, prefetch_depth="
            f"{prefetch_depth}, layers={len(eligible)}, resident_limit="
            f"{cache.resident_limit_bytes / 1024**3:.3f} GiB "
            f"(outside={cache.outside_bytes / 1024**3:.3f} GiB)"
        )
    return len(eligible), cache.resident_limit_bytes, cache


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
        prefetch_cache = getattr(module, "_frozen_training_prefetch_cache", None)
        if getattr(module, "_frozen_training_prefetch_backward", False):
            if prefetch_cache is None:
                raise RuntimeError(
                    f"ConvRot backward prefetch cache is missing for layer '{path}'"
                )
            return ConvRotPrefetchBackwardLinearFn.apply(
                x,
                module.weight,
                module.weight_scale,
                module.bias,
                int(module.convrot_groupsize),
                path,
                prefetch_cache,
            )
        backward_weight = getattr(module, "_frozen_training_backward_weight", None)
        if getattr(module, "_frozen_training_cached_backward", False):
            if backward_weight is None:
                raise RuntimeError(
                    f"ConvRot backward cache is missing for layer '{path}'"
                )
            if backward_weight.device != x.device:
                raise RuntimeError(
                    f"ConvRot backward cache for layer '{path}' is "
                    f"on {backward_weight.device}, but activation is on {x.device}"
                )
            return ConvRotCachedBackwardLinearFn.apply(
                x,
                module.weight,
                module.weight_scale,
                module.bias,
                backward_weight,
                int(module.convrot_groupsize),
                path,
            )
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
