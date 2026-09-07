"""Per-parameter outlier clipping for the fused backward pass.

``max_grad_norm`` is a GLOBAL norm: it needs every gradient in hand before it can
scale any of them. The fused backward pass applies each parameter's update as
soon as that parameter's gradient exists, so the global norm is not knowable in
time and the trainer sets ``max_grad_norm=0.0`` there -- no clipping at all.

What IS knowable inside the hook is the parameter's own complete gradient. This
clips against that parameter's own running scale rather than an absolute number
the operator would have to guess, because the failure it addresses is relative:
a gradient thousands of times the usual one. It is not a global-norm clip and
does not replace ``max_grad_norm`` on the non-fused path.

It matters for a sign-based optimizer too. Lion's step size is already bounded
by the learning rate, but its MOMENTUM is not: ``m <- b2*m + (1-b2)*g`` at
b2=0.99 carries an outlier for the several hundred steps it takes to decay, and
the sign of every update in that window follows it.

Two rules the hot path is built on:

* nothing may leave the device. A ``.item()`` here is one host sync per
  parameter per step, inside the backward pass;
* the gradient is read as few times as possible. Measured: 25 aten ops per
  parameter per step, of which ONE touches the whole gradient (the in-place
  scale) when ``fused_grad_norm``'s accumulator is enabled and its reduction can
  be reused, TWO when it is not. The rest are 0-dim scalar ops -- cheap in
  bandwidth, one kernel launch each. At 588 trained Linears that is ~15k extra
  launches per step, which is why the feature is off by default.
"""

from typing import Dict, List, Optional

import torch

CLIPPER_ATTR = "_sushiui_fused_grad_clip"

#: EMA decay for a parameter's running gradient scale. Not a served parameter:
#: the scale it tracks moves on the timescale of training, not of a run's
#: configuration, and a second knob here would only be another number to guess.
_SCALE_DECAY = 0.99

#: Denominator floor, so a parameter whose gradient is exactly zero produces a
#: ratio of 0 rather than a NaN.
_EPS = 1e-12


class FusedGradClipper:
    """Clips each parameter's gradient to ``factor`` x its own running scale.

    One instance per run, attached to the optimizer (the only object in scope at
    all four fused hook sites -- see ``fused_grad_norm.attach_grad_observer``).
    """

    def __init__(self, factor: float, warmup_steps: int,
                 decay: float = _SCALE_DECAY):
        if not factor > 0:
            raise ValueError(
                f"a fused gradient clip needs a positive factor, got {factor!r} "
                f"(0 means the feature is off and no clipper is built)")
        self.factor = float(factor)
        # The running scales live only in this process: they are not written to
        # the checkpoint, so every resume re-learns them and the first
        # `warmup_steps` updates of each parameter after a restart are unclipped.
        self.warmup_steps = max(0, int(warmup_steps))
        self.decay = float(decay)
        # id(param) -> (running scale, updates seen). The scale is a 0-dim
        # device tensor so the hot path never leaves the device.
        self._scale: Dict[int, torch.Tensor] = {}
        self._seen: Dict[int, int] = {}
        # id(param) -> a stable small integer, so the worst offender can be
        # named from an argmax without a per-parameter host sync.
        self._index: Dict[int, int] = {}
        self._names: List[str] = []
        self._clipped: Optional[torch.Tensor] = None
        self._worst_ratio: Optional[torch.Tensor] = None
        self._worst_index: Optional[torch.Tensor] = None

    def reset_scales(self) -> None:
        """Forget every running scale, restarting warmup for all parameters.

        For a caller that has just changed what the gradients mean -- ReLoRA
        merges its adapters into the base and re-initialises them.
        """
        self._scale.clear()
        self._seen.clear()

    def name_parameters(self, names_by_id: Dict[int, str]) -> None:
        """Give the clipper the parameter names it reports offenders by."""
        for key, name in names_by_id.items():
            if key not in self._index:
                self._index[key] = len(self._names)
                self._names.append(name)

    def _slot(self, param: torch.nn.Parameter) -> int:
        key = id(param)
        index = self._index.get(key)
        if index is None:
            index = len(self._names)
            self._index[key] = index
            self._names.append(f"<unnamed parameter {index}>")
        return index

    def _reset_counters(self, device: torch.device) -> None:
        self._clipped = torch.zeros((), dtype=torch.float32, device=device)
        self._worst_ratio = torch.zeros((), dtype=torch.float32, device=device)
        self._worst_index = torch.full((), -1.0, dtype=torch.float32, device=device)

    @staticmethod
    def _recorded_norm(optimizer, param) -> Optional[torch.Tensor]:
        """This step's norm from the grad-norm accumulator, if it took one.

        Both readers run at the same four hook sites and the recorder runs
        first, so on a step it is enabled the reduction has already happened;
        repeating it is a second full pass over the gradient for no new
        information. Returns None when the accumulator is off for this step.
        """
        if optimizer is None:
            return None
        from .fused_grad_norm import get_grad_norm_accumulator

        accumulator = get_grad_norm_accumulator(optimizer)
        if accumulator is None or not accumulator.enabled:
            return None
        square = accumulator.squared(param)
        return None if square is None else square.sqrt()

    def clip(self, param: torch.nn.Parameter, optimizer=None) -> None:
        """Scale ``param.grad`` down if it is an outlier for this parameter.

        Called from the fused hooks with the gradient complete and before the
        update. In place, and device-resident throughout.
        """
        grad = param.grad
        if grad is None:
            return
        key = id(param)
        norm = self._recorded_norm(optimizer, param)
        if norm is None:
            # fp32 inside the reduction: a bf16 gradient squared in bf16 loses
            # most of its mantissa (the same reason fused_grad_norm passes
            # dtype=).
            norm = torch.linalg.vector_norm(grad.detach(), ord=2,
                                            dtype=torch.float32)
        scale = self._scale.get(key)
        if scale is None:
            scale = norm.detach().clone()
            self._scale[key] = scale
            self._seen[key] = 1
            return
        seen = self._seen[key] = self._seen.get(key, 0) + 1
        if seen <= self.warmup_steps:
            # Still learning what "usual" is for this parameter. Clipping
            # against a scale built from a handful of gradients would be
            # clipping against noise.
            scale.mul_(self.decay).add_(norm, alpha=1.0 - self.decay)
            return

        if self._clipped is None:
            self._reset_counters(norm.device)
        previous = scale.clone()
        threshold = previous.mul(self.factor)
        ratio = norm / previous.clamp_min(_EPS)
        # <= 1 leaves the gradient untouched; the multiply is unconditional so
        # no branch here can force a host sync.
        factor = (threshold / norm.clamp_min(_EPS)).clamp(max=1.0)
        grad.mul_(factor.to(grad.dtype))

        was_clipped = (factor < 1.0).to(torch.float32)
        self._clipped.add_(was_clipped)
        spike = ratio.mul_(was_clipped)
        beat = (spike > self._worst_ratio).to(torch.float32)
        self._worst_index.mul_(1.0 - beat).add_(beat, alpha=float(self._slot(param)))
        torch.maximum(self._worst_ratio, spike, out=self._worst_ratio)
        # A clipped step contributes the PREVIOUS scale, not the threshold it was
        # cut to. Feeding the threshold back in would multiply the scale by
        # (decay + (1-decay)*factor) on every clipped step -- 1.07x at the
        # defaults -- so a sustained burst would walk the bar up behind itself
        # and defeat the clip in a few hundred steps. This way a burst leaves the
        # scale flat, while an unclipped gradient still tracks it in both
        # directions.
        contribution = previous.sub_(norm).mul_(was_clipped).add_(norm)
        scale.mul_(self.decay).add_(contribution, alpha=1.0 - self.decay)

    def take_step_summary(self) -> Optional[Dict[str, object]]:
        """What this step clipped, and reset. One host sync, or none.

        Returns None when nothing was clipped, which is the case on every step
        of a healthy run.
        """
        if self._clipped is None:
            return None
        counters = torch.stack([self._clipped, self._worst_ratio, self._worst_index])
        self._clipped = None
        self._worst_ratio = None
        self._worst_index = None
        clipped, worst_ratio, worst_index = counters.tolist()
        if clipped <= 0:
            return None
        index = int(worst_index)
        return {
            "clipped_parameters": int(clipped),
            "worst_ratio": float(worst_ratio),
            "worst_parameter": (self._names[index]
                                if 0 <= index < len(self._names) else None),
        }


def attach_fused_grad_clipper(optimizer, clipper: Optional[FusedGradClipper]) -> None:
    setattr(optimizer, CLIPPER_ATTR, clipper)


def get_fused_grad_clipper(optimizer) -> Optional[FusedGradClipper]:
    return getattr(optimizer, CLIPPER_ATTR, None)


def apply_fused_grad_clip(optimizer, param: torch.nn.Parameter) -> None:
    """Clip ``param``'s gradient, if ``optimizer`` carries a clipper.

    Called from every fused hook site AFTER the norm is recorded, so the charted
    gradient norm stays the one the backward pass actually produced -- a spike
    the operator needs to see, not the value that was let through.
    """
    clipper = getattr(optimizer, CLIPPER_ATTR, None)
    if clipper is not None:
        clipper.clip(param, optimizer)
