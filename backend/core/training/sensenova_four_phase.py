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

MNT > 1. The design says to accumulate the boundary gradient and run phase 3
once. The SHAPES permit it: SenseNova is B1 by contract
(``_collate_sensenova_b1_prefix`` allows exactly one prefix) and the MNT loop
iterates TIMESTEPS for that one item, re-encoding ``captions[0]`` every
iteration (``_sensenova_mnt_conditioning``), so every boundary K/V in a window
has the same shape and would add.

What does not hold under MNT is the design's exactness premise, that the
understanding weights are invariant across the window. The MNT loop steps the
optimizer once per iteration -- that is why it re-encodes the prefix at all
rather than retaining the graph -- so an accumulated gradient flushed at the end
of the window would be back-propagated against understanding weights that
earlier iterations had already moved. (Under gradient ACCUMULATION the premise
would hold, since nothing steps until the window closes; that route is refused
here for an unrelated reason.) Phase 3 therefore runs with its own backward, per
iteration, which is exact: the recomputed forward reads the same understanding
weights its own phase-1 forward did, the generation hooks that fire in between
touching no understanding weight.

The cost is the two weight round trips per MNT iteration rather than per step,
announced on the training_log channel by ``warn_four_phase_mnt_cost`` rather
than refused. The update-frequency asymmetry the design records is unchanged.
The pending list below is kept because it is the honest shape of the deferral
the design asks for, and because ``flush`` is also reachable from the
optimizer-step seam; it never holds more than one entry on any route that can
arm four-phase today.

Requires the fused-backward route. Phase 3 leaves the generation half on CPU, so
a subsequent ``optimizer.step()`` would meet CUDA gradients on CPU parameters;
under fused backward there is no such call, each half being stepped by its own
post-accumulate hooks while it is resident.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch


class SenseNovaFourPhaseBackward:
    """One trainer's boundary cut. Installed as ``trainer.sensenova_four_phase``."""

    def __init__(self, trainer: Any):
        self.trainer = trainer
        self._current: Tuple[Any, List[torch.Tensor]] | None = None
        self._pending: List[Tuple[Any, List[torch.Tensor]]] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

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
        return _TrainingPrefixCache(layers)

    def capture(self) -> None:
        """Take the boundary gradient the generation backward just produced."""
        if self._current is None:
            return
        inputs, leaves = self._current
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

    def discard(self) -> None:
        """Drop everything outstanding -- a skipped batch, or teardown."""
        self._current = None
        self._pending = []


def install_four_phase_backward(trainer: Any) -> SenseNovaFourPhaseBackward:
    context = SenseNovaFourPhaseBackward(trainer)
    trainer.sensenova_four_phase = context
    return context
