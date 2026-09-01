"""Gradient-norm accumulation for the fused backward pass.

Under fused backward the per-parameter hooks apply the update and clear
``param.grad`` immediately, so by the time the trainer asks for gradient norms
there is nothing left to measure and every norm it reports is 0.0. The hooks
therefore record each gradient's squared norm here, before clearing it.

The squares stay on the device: one ``.item()``-equivalent sync per step (in
``squared_norms``), not one per parameter.

This module does not classify anything. It keys by ``id(param)`` so the
trainer's existing component classification (LoRA adapter components, module
walks for full fine-tuning) keeps deciding which bucket each parameter is in.
"""

from typing import Dict, Iterable, List, Optional

import torch

ACCUMULATOR_ATTR = "_sushiui_fused_grad_norm"
OBSERVER_ATTR = "_sushiui_fused_grad_observer"


class FusedGradNormAccumulator:
    """``id(param) -> sum of squared gradient elements`` for one step."""

    def __init__(self):
        self._squares: Dict[int, torch.Tensor] = {}
        self.enabled = False

    def begin_step(self, enabled: bool) -> None:
        """Start a step, dropping whatever the previous one accumulated.

        ``enabled`` is the gate: the norms are only read on steps that log them,
        and recording on the others would cost a reduction per parameter for a
        number nothing reads.
        """
        self._squares.clear()
        self.enabled = bool(enabled)

    def record(self, param: torch.nn.Parameter) -> None:
        """Record ``param``'s current gradient. Call before clearing it."""
        grad = param.grad
        if grad is None:
            return
        # FP32 accumulation, but WITHOUT materialising the gradient in fp32: a
        # BF16 grad squared in BF16 loses most of the mantissa, so the dtype=
        # argument is load-bearing -- it upcasts inside the reduction. The old
        # `grad.detach().float().pow(2).sum()` allocated two full-size temporaries
        # per parameter, which at 8.1B trainable params is ~113 GB of memory
        # traffic per step for a number that is only ever charted.
        # Result is a 0-dim device tensor, so no sync here.
        square = torch.linalg.vector_norm(grad.detach(), ord=2, dtype=torch.float32).pow(2)
        previous = self._squares.get(id(param))
        self._squares[id(param)] = square if previous is None else previous + square

    def has(self, param: torch.nn.Parameter) -> bool:
        return id(param) in self._squares

    def squared_norms(self) -> Dict[int, float]:
        """``id(param) -> ||grad||^2``, with one device->host sync per device."""
        if not self._squares:
            return {}
        by_device: Dict[torch.device, List[int]] = {}
        for key, tensor in self._squares.items():
            by_device.setdefault(tensor.device, []).append(key)
        out: Dict[int, float] = {}
        for keys in by_device.values():
            values = torch.stack([self._squares[key] for key in keys]).tolist()
            out.update(zip(keys, values))
        return out


def attach_grad_norm_accumulator(optimizer, accumulator: Optional[FusedGradNormAccumulator]) -> None:
    setattr(optimizer, ACCUMULATOR_ATTR, accumulator)


def get_grad_norm_accumulator(optimizer) -> Optional[FusedGradNormAccumulator]:
    return getattr(optimizer, ACCUMULATOR_ATTR, None)


def record_fused_grad_norm(optimizer, param: torch.nn.Parameter) -> None:
    """Record ``param``'s gradient on ``optimizer``'s accumulator, if gated on."""
    accumulator = getattr(optimizer, ACCUMULATOR_ATTR, None)
    if accumulator is not None and accumulator.enabled:
        accumulator.record(param)


def attach_grad_observer(optimizer, observer) -> None:
    """Give ``optimizer``'s fused hooks a second reader of each gradient.

    Carried on the optimizer for the same reason the norm accumulator is: the
    per-parameter hooks are built in four different places (the trainer's own
    loop, fused optimizer groups, and each ring-buffer optimizer's own
    registration), and only the optimizer is in scope in all of them. A probe
    that instead reached for the trainer would have to be wired into each one
    separately -- which is exactly how the timestep-cosine probe came to be
    armed but never fed: the ring-buffer path registers its hooks itself and
    returns before the trainer's loop runs.

    ``observer`` needs one method, ``observe(param)``, and must not raise.
    Passing None detaches.
    """
    setattr(optimizer, OBSERVER_ATTR, observer)


def record_fused_grad_observation(optimizer, param: torch.nn.Parameter) -> None:
    """Hand ``param``'s gradient to ``optimizer``'s observer, if one is attached.

    Called from the same point as :func:`record_fused_grad_norm` -- before the
    hook applies its update and clears the gradient, which is the only window
    in which it exists. When nothing is attached this is one attribute lookup.
    """
    observer = getattr(optimizer, OBSERVER_ATTR, None)
    if observer is not None:
        observer.observe(param)


def squared_norms_from_grads(params: Iterable[torch.nn.Parameter]) -> Dict[int, float]:
    """The same measurement, taken from live gradients (non-fused path).

    Shares ``record``/``squared_norms`` so both paths compute the same quantity
    the same way, rather than agreeing by coincidence.
    """
    accumulator = FusedGradNormAccumulator()
    accumulator.enabled = True
    for param in params:
        accumulator.record(param)
    return accumulator.squared_norms()
