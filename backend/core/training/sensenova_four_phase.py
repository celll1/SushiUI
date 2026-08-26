"""Split SenseNova's single ``loss.backward()`` in two at the prefix KV cache.

SENSENOVA_TRAINING_DESIGN.md 8.3.2. Training the understanding half and evicting
it are incompatible inside ONE backward -- there is no coordinate in the middle
of ``loss.backward()`` at which a weight swap can be inserted. They are not
incompatible in principle: the two halves meet only at the prefix KV cache
(50.5 MiB at 258 tokens, measured in Phase 0), so cutting the graph there costs
that much again in boundary gradient and yields two backwards that can be run
under different residency.

Per optimizer step::

    prefix        und resident   understanding forward (no grad -- see below),
                                 boundary K/V handed on as grad-requiring leaves
    denoise       gen resident   generation forward + backward, terminating in
                                 the leaves' .grad
    und_backward  und resident   recompute the understanding forward WITH grad,
                                 then autograd.backward(recomputed, kv_grad)

The design table calls for a grad-carrying phase 1. This runs phase 1 under
``no_grad`` instead: phase 3 recomputes the forward regardless, so a phase-1
graph would be built and then dropped unused, and its activations are the
residency the split exists to avoid. Numerically the two are the same forward --
same function, same inputs, same weights, and ``attention_dropout`` is asserted
zero by ``assert_understanding_training_supported`` precisely so a recompute
reproduces its own forward (the same property gradient checkpointing already
depends on inside each phase).

MNT > 1, two modes.

DEFAULT (per-iteration). Each MNT iteration runs its own complete cycle: cut,
generation backward, capture, phase 3. Exact, and the cost is the two weight
round trips per iteration rather than per step, announced by
``warn_four_phase_mnt_cost``.

SHARED WINDOW (``sensenova_four_phase_shared_prefix``, opt-in, default off).
One phase 1, one ``cut()`` and one set of boundary leaves per MNT window; all N
generation graphs read the SAME leaves, so autograd accumulates into
``leaf.grad`` natively and the boundary costs one gradient buffer regardless of
N. Phase 3 runs once, at the end of the window. The understanding half stays on
CPU for the whole window, so transitions drop from 2N to 2 per batch -- the loop
shape a FROZEN understanding half already has.

Its exactness premise -- that the understanding weights are invariant across
the window -- HOLDS, and does so for a reason the mechanism enforces rather than
assumes: this route is fused-backward-only, so a parameter moves exactly when
its own post-accumulate-grad hook fires, and the understanding parameters
receive gradient from nothing but phase 3 (the generation forward reads the
boundary K/V as detached leaves). No phase 3, no understanding update, so the
weights the single deferred backward reads are bit-identically the ones its
phase-1 forward read.

What it changes is what is TRAINED, which is why it is a setting and not a
silent optimisation: the understanding half takes ONE update per window from the
window's summed (or averaged, per ``reduction``) boundary gradient instead of N
sequential updates, its Adafactor ``state['step']`` advances once per window, and
the generation:understanding update ratio becomes N:1.

Requires the fused-backward route. Phase 3 leaves the generation half on CPU, so
a subsequent ``optimizer.step()`` would meet CUDA gradients on CPU parameters;
under fused backward there is no such call, each half being stepped by its own
post-accumulate hooks while it is resident.

GRADIENT NORMALISATION is selectable because ``.grad`` accumulates: ``sum``
(the default) is the exact gradient of the window's SUMMED loss, ``mean``
divides by the number of generation backwards the window ran, giving the
gradient of its averaged loss. Neither reproduces the generation half, which
takes N separate updates from N separate per-iteration gradients.
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
        """Whether the LAST generation backward was followed by phase 3.

        Distinct from ``is_final_iteration``, which answers about the NEXT
        backward: at the step seam iteration N-2 satisfies that predicate while
        the evictor is still in ``denoise``. This is what decides which half the
        seam may assert resident.
        """
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
        return _TrainingPrefixCache(layers)

    # -- shared window ------------------------------------------------------

    def begin_window(self, size: int) -> None:
        """Declare how many generation backwards will read this cut's leaves.

        Independent of the counter it is checked against: the trainer names the
        MNT count, ``capture`` counts the backwards that actually ran, and a
        mismatch is what replaces the per-iteration cut/backward pairing.
        """
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
        """Whether the NEXT generation backward closes the window.

        Off the shared route every iteration is its own window, so this is
        True and the caller's per-iteration expectations are unchanged.
        """
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
        """Drop everything outstanding -- a skipped batch, or teardown.

        Returns the number of generation backwards whose understanding gradient
        is being thrown away. Per-iteration that is at most the current
        iteration's, the asymmetry a skip has always had. Under a shared window
        it is every backward the window has run so far, whose GENERATION updates
        already landed, so the caller announces the count rather than letting the
        asymmetry widen silently. Phase 3 cannot be run here instead: discard's
        callers are a CUDA error and teardown.
        """
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
    """The parameters a shared window defers, for the update census.

    Empty unless the shared route is armed. Taken from the modules the EVICTOR
    stages to CPU for the denoise phase, so the census's reduced expectation set
    and the half that is physically absent during the generation backward are
    one set rather than two that agree.
    """
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
