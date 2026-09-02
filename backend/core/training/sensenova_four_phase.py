"""Split SenseNova backward at the prefix K/V boundary for MoT eviction.

The understanding forward first produces detached, grad-requiring K/V leaves;
after the generation backward, phase 3 recomputes that forward and propagates
the captured K/V gradients. Recompute equivalence requires zero attention
dropout. The default repeats this cycle per MNT iteration.

The optional shared window reuses one boundary across an MNT window and updates
the understanding half once with the summed or averaged boundary gradient. It
therefore changes the generation:understanding update ratio to N:1. Both modes
require fused backward. See SENSENOVA_TRAINING_DESIGN.md 8.3.2 and 8.3.5.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch

GRAD_REDUCTIONS = ("sum", "mean")


class SenseNovaFourPhaseBackward:
    """One trainer's boundary cut. Installed as ``trainer.sensenova_four_phase``."""

    def __init__(
        self,
        trainer: Any,
        *,
        shared_window: bool = False,
        reduction: str = "sum",
    ):
        reduction = str(reduction or "sum").strip().lower()
        if reduction not in GRAD_REDUCTIONS:
            raise ValueError(
                f"SenseNova four-phase gradient reduction must be one of "
                f"{', '.join(GRAD_REDUCTIONS)}, got {reduction!r}"
            )
        self.trainer = trainer
        self.shared_window = bool(shared_window)
        self.reduction = reduction
        self._current: Tuple[Any, List[torch.Tensor]] | None = None
        self._pending: List[Tuple[Any, List[torch.Tensor]]] = []
        self._window_size: int | None = None
        self._window_backwards = 0
        self._window_aborted = False
        self._phase_three_ran = False
        self.dropped_backwards = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def window_aborted(self) -> bool:
        """A shared window was discarded mid-flight; the batch cannot continue."""
        return self.shared_window and self._window_aborted

    @property
    def phase_three_ran(self) -> bool:
        """Whether phase 3 followed the last generation backward."""
        return self._phase_three_ran

    def cut(self, cache: Any, inputs: Any) -> Any:
        """Return a cache of grad-requiring leaves and keep what phase 3 needs."""
        from core.training.ops.sensenova_ops import (
            _TrainingPrefixCache,
            _TrainingPrefixLayer,
        )

        if self._current is not None:
            raise RuntimeError(
                "SenseNova four-phase: a previous boundary cut was never captured; "
                "every prefix must be followed by its generation backward"
            )
        leaves: List[torch.Tensor] = []
        layers = []
        for layer in cache.layers:
            keys = layer.keys.detach().requires_grad_(True)
            values = layer.values.detach().requires_grad_(True)
            leaves.extend((keys, values))
            layers.append(_TrainingPrefixLayer(keys, values))
        self._current = (inputs, leaves)
        self._window_size = None
        self._window_backwards = 0
        self._window_aborted = False
        self._phase_three_ran = False
        return _TrainingPrefixCache(layers, packed=getattr(cache, "packed", None))

    # -- shared window ------------------------------------------------------

    def begin_window(self, size: int) -> None:
        """Declare how many generation backwards share the current cut."""
        if not self.shared_window:
            return
        size = int(size)
        if size < 1:
            raise ValueError(
                f"SenseNova four-phase window size must be >= 1, got {size}"
            )
        if self._window_backwards:
            raise RuntimeError(
                "SenseNova four-phase: a new MNT window opened while the previous "
                f"one still had {self._window_backwards} uncaptured generation "
                "backward(s); their understanding gradient would be lost"
            )
        self._window_size = size

    def is_final_iteration(self) -> bool:
        """Whether the next generation backward closes the window."""
        if not self.shared_window or self._window_size is None:
            return True
        return self._window_backwards >= self._window_size - 1

    def after_generation_backward(self) -> None:
        """Called once per generation backward, immediately after it returns."""
        self._phase_three_ran = False
        if not self.shared_window:
            self.capture()
            self.flush()
            self._phase_three_ran = True
            return
        if self._window_aborted or self._current is None:
            # The window was discarded mid-flight (see ``discard``); this
            # iteration's generation update landed and its understanding
            # gradient has nowhere to go.
            self.dropped_backwards += 1
            return
        if self._window_size is None:
            raise RuntimeError(
                "SenseNova four-phase: a shared-prefix generation backward ran "
                "before begin_window() declared the MNT window size, so nothing "
                "knows when phase 3 is due"
            )
        self._window_backwards += 1
        if self._window_backwards >= self._window_size:
            self.capture()
            self.flush()
            self._phase_three_ran = True

    def capture(self) -> None:
        """Take the boundary gradient the generation backward(s) produced."""
        if self._current is None:
            return
        inputs, leaves = self._current
        if self.shared_window:
            if self._window_backwards == 0:
                raise RuntimeError(
                    "SenseNova four-phase: the shared boundary cut was captured "
                    "without a single generation backward having read it; the "
                    "understanding half would receive nothing"
                )
            if (
                self._window_size is not None
                and self._window_backwards != self._window_size
            ):
                raise RuntimeError(
                    "SenseNova four-phase: the shared boundary cut was read by "
                    f"{self._window_backwards} generation backward(s) but the MNT "
                    f"window declared {self._window_size}; one iteration's "
                    "understanding gradient is missing or double-counted"
                )
        self._current = None
        if all(leaf.grad is None for leaf in leaves):
            raise RuntimeError(
                "SenseNova four-phase: the generation backward left no gradient on "
                "any of the boundary K/V leaves. The understanding half would "
                "receive nothing while the loss fell normally"
            )
        grads = [
            leaf.grad if leaf.grad is not None else torch.zeros_like(leaf)
            for leaf in leaves
        ]
        for leaf in leaves:
            leaf.grad = None
        count = self._window_backwards
        self._window_backwards = 0
        self._window_size = None
        if self.shared_window and self.reduction == "mean" and count > 1:
            for grad in grads:
                grad.div_(count)
        self._pending.append((inputs, grads))

    def flush(self) -> None:
        """Phase 3: recompute each pending understanding forward and back it up."""
        if not self._pending:
            return
        if self._current is not None:
            raise RuntimeError(
                "SenseNova four-phase: an uncaptured boundary cut is outstanding at "
                "the optimizer step; its understanding gradient would be lost"
            )
        from core.training.ops.sensenova_ops import _build_trainable_prefix

        trainer = self.trainer
        evictor = getattr(trainer, "sensenova_phase_evictor", None)
        if evictor is not None:
            evictor.enter_und_backward()
            evictor.assert_understanding_resident()
        pending, self._pending = self._pending, []
        for inputs, grads in pending:
            cache = _build_trainable_prefix(trainer, trainer.transformer, inputs)
            tensors = [
                tensor
                for layer in cache.layers
                for tensor in (layer.keys, layer.values)
            ]
            if len(tensors) != len(grads):
                raise RuntimeError(
                    "SenseNova four-phase: the recomputed prefix has "
                    f"{len(tensors)} boundary tensors, the captured gradient "
                    f"{len(grads)}"
                )
            torch.autograd.backward(tensors, grad_tensors=grads)
            del cache, tensors

    def discard(self) -> int:
        """Drop pending state and return the lost understanding backwards."""
        dropped = self._window_backwards
        self._current = None
        self._pending = []
        self._window_backwards = 0
        self._window_size = None
        self._window_aborted = bool(self.shared_window)
        self._phase_three_ran = False
        self.dropped_backwards += dropped
        return dropped


def understanding_deferred_parameters(trainer: Any) -> Sequence[torch.nn.Parameter]:
    """Return the evicted understanding parameters deferred by a shared window."""
    four_phase = getattr(trainer, "sensenova_four_phase", None)
    if four_phase is None or not four_phase.shared_window:
        return ()
    evictor = getattr(trainer, "sensenova_phase_evictor", None)
    if evictor is None:
        raise RuntimeError(
            "SenseNova four-phase shared prefix has no phase evictor to take the "
            "deferred parameter set from"
        )
    deferred: List[torch.nn.Parameter] = []
    for module in evictor.understanding_modules:
        for parameter in module._parameters.values():
            if parameter is not None and parameter.requires_grad:
                deferred.append(parameter)
    return deferred


def install_four_phase_backward(
    trainer: Any,
    *,
    shared_window: bool | None = None,
    reduction: str | None = None,
) -> SenseNovaFourPhaseBackward:
    if shared_window is None:
        shared_window = bool(
            getattr(trainer, "sensenova_four_phase_shared_prefix", False)
        )
    if reduction is None:
        reduction = getattr(trainer, "sensenova_four_phase_grad_reduction", "sum")
    context = SenseNovaFourPhaseBackward(
        trainer, shared_window=shared_window, reduction=reduction
    )
    trainer.sensenova_four_phase = context
    return context
