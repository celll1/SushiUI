"""Training-only SenseNova MoT phase eviction state machine.

Three states (``full`` / ``prefix`` / ``denoise``) serve a frozen understanding
half. A TRAINED understanding half needs a fourth, ``und_backward``, because the
single ``loss.backward()`` the three-state machine assumes has no coordinate at
which a weight swap can be inserted (SENSENOVA_TRAINING_DESIGN.md 8.3.2). The
split cuts the graph at the prefix KV cache and runs two backwards, so the
sequence per step becomes::

    prefix        und resident   understanding forward, boundary K/V kept as leaves
    denoise       gen resident   generation forward + backward down to leaf .grad
    und_backward  und resident   recompute the understanding forward, then
                                 autograd.backward(recomputed, grad_tensors=kv_grad)

``und_backward -> prefix`` is a no-op: the understanding half is already resident
and the next step's prefix phase wants exactly that, which saves one of the three
round trips the naive cycle would pay.

LAYER SELECTION (8.4 re-derived for this case). ``select_mot_weight_modules``
keys on PERSISTENCE -- a module is selected if it owns a Parameter OR a
persistent buffer -- and that stays correct here rather than reverting to a
``parameters()`` rule now that U-2-1 gives the trained half real Parameters
again. Two reasons, and only the second is new:

  * the frozen half is still ``Int8Linear``, which owns no Parameter at all, and
    it is the half being evicted. A ``parameters()`` rule selects RMSNorm and
    nothing else, which is how this classifier shipped inert twice.
  * persistence also carries ``Int8Linear``'s scale buffers. Selecting the
    quantized weight without its scale would not merely lose a saving, it would
    split one module's tensors across two devices.

TRANSFER ORDER. A two-sided transition is interleaved PAIRWISE -- stage one
outgoing module to pinned CPU, then move its incoming twin to the device, and
repeat -- rather than running the whole outgoing half to CPU and only then
loading the whole incoming half. The batched order holds both halves in pinned
host memory at once, because the incoming half's pinned tensors are released
only when ``_move_modules_to_device`` reassigns ``parameter.data``, which the
batched order defers until every outgoing module is already staged.
Interleaved, the ledger-measured pinned high-water on the synthetic tree
(``sensenova_mot_staging_highwater_test.py``) is one half plus one module;
torch's caching host allocator reuses freed pinned blocks by SIZE CLASS rather
than byte range, so cross-pair reuse only converges once a layer's distinct
tensor sizes have each been freed once, and whatever peak is reached is never
returned to the OS, so it stays sticky for the run. That bound covers the
steady-state phase cycle only: ``teardown()`` and a failed transition's
best-effort recovery deliberately put BOTH halves back on CPU (there is no
GPU left to keep either one on), so their host high-water is the two-halves
peak this interleave otherwise avoids, not the bound above (see
``_best_effort_cpu``). The run-121 observation that motivated this change and
the bf16 arithmetic extrapolated from it are recorded in
SENSENOVA_TRAINING_DESIGN.md 8.5, not repeated here.

The device invariant is unchanged and for the same reason: within a pair the
d2h still precedes the h2d, and the pairs carry identical tensor signatures, so
the incoming half grows on device exactly as fast as the outgoing one shrinks.
Device residency stays at one half throughout, never their sum.

What IS new for four-phase is that the evicted half now carries gradients, so
``_assert_grad_free`` refuses to move a half whose Parameters still hold
``.grad``: under fused backward the hook nulls each gradient as it applies it, so
a surviving ``.grad`` at a phase boundary means the optimizer did not run over
that parameter, and moving it would silently detach the gradient from its weight.

ASYNC H2D -- not attempted. Transfers are synchronous today: neither
``_move_modules_to_device`` nor ``mot_cpu_staging._stage_tensor`` passes
``non_blocking``. The real obstacle is a cost trade-off, not a correctness
barrier: ``non_blocking`` h2d only pays off from PINNED host memory, which
puts it in direct conflict with ``sensenova_mot_pageable_staging`` below --
the two would compete for the same host memory, not compose. Separately, an
overlap window would hold a transient extra module on-device; the synthetic
tree's own ledger already prices that at one module against one half (~1.6%,
a ratio from that tree, not a device measurement -- see
``sensenova_mot_staging_highwater_test.py``), and a failed transition's
recovery would need the in-flight copy tracked and synchronized before
touching it. Neither is disqualifying; both are unresolved.

PAGEABLE STAGING -- opt-in, off by default
(``sensenova_mot_pageable_staging``). Trades the pinned pool's sticky
high-water (torch's caching host allocator never returns a pinned block to
the OS) for host RAM the OS can reclaim, at an unmeasured transfer-time cost.
Refused without ``sensenova_mot_phase_eviction``, since with the evictor off
nothing here ever runs. Read from ``trainer.config`` rather than a promoted
attribute, so this flag needed no change to ``BaseTrainer``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from torch import nn

from core.models.sensenova.mot_cpu_staging import stage_modules_to_pinned_cpu
from core.models.sensenova.mot_weight_selector import select_mot_weight_modules

_PIN_FAILURE_MESSAGE = (
    "[SenseNova] Training MoT eviction could not pin CPU staging "
    "memory ({exc}); continuing with blocking pageable copies."
)


def _move_modules_to_cpu(
    modules: Iterable[nn.Module], *, warn_once: Dict[str, bool], pageable: bool = False
) -> None:
    stage_modules_to_pinned_cpu(
        modules, warn_once=warn_once, warn_message=_PIN_FAILURE_MESSAGE,
        pageable=pageable,
    )


def _module_already_staged_cpu(module: nn.Module, *, pageable: bool = False) -> bool:
    """True iff every owned tensor is already staged CPU for the CURRENT
    staging mode -- the same condition ``_stage_tensor`` short-circuits on,
    checked once per module instead of once per tensor so ``_best_effort_cpu``
    skips a module outright rather than entering it and no-op'ing tensor by
    tensor. Under ``pageable=True`` "staged" means CPU regardless of pin state
    (pageable staging never re-pins a tensor it finds already on CPU); under
    the default it additionally requires ``is_pinned()``."""
    def _staged(tensor) -> bool:
        if tensor.device.type != "cpu":
            return False
        return True if pageable else tensor.is_pinned()

    for parameter in module._parameters.values():
        if parameter is not None and not _staged(parameter.data):
            return False
    for name, buffer in module._buffers.items():
        if buffer is None or name in module._non_persistent_buffers_set:
            continue
        if not _staged(buffer):
            return False
    return True


def _move_modules_to_device(modules: Iterable[nn.Module], device: Any) -> None:
    for module in modules:
        for parameter in module._parameters.values():
            if parameter is not None:
                parameter.data = parameter.data.to(device)
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and name not in module._non_persistent_buffers_set:
                module._buffers[name] = buffer.to(device)


class SenseNovaTrainingPhaseEvictor:
    """Keep only the phase-active MoT half resident while training."""

    def __init__(
        self, transformer: nn.Module, device: Any, *, four_phase: bool = False,
        pageable_staging: bool = False,
    ):
        selection = select_mot_weight_modules(
            transformer,
            require_exact_symmetry=True,
            allow_understanding_adapters=four_phase,
        )
        self._gen_modules = selection.gen_modules
        self._und_modules = selection.und_modules
        self._pairs = selection.pairs
        self._gen_unpaired = selection.gen_unpaired
        self._und_unpaired = selection.und_unpaired
        self.transformer = transformer
        self.device = device
        self.four_phase = bool(four_phase)
        # sensenova_mot_pageable_staging: see this module's PAGEABLE STAGING
        # note. Off by default, which reproduces today's pinned-only behavior
        # exactly (every _move_modules_to_cpu call below defaults pageable=False).
        self._pageable = bool(pageable_staging)
        self.state = "full"
        self._warn_once: Dict[str, bool] = {}
        self._assert_pairing_covers_both_halves()
        self._evict_generation_plan = self._evict_plan("generation")
        self._swap_generation_plan = self._swap_plan("generation")
        self._swap_understanding_plan = self._swap_plan("understanding")

    def _assert_pairing_covers_both_halves(self) -> None:
        """Refuse to run at all unless every selected module is either paired or
        a declared per-side extra (see this module's TRANSFER ORDER for why the
        batched fallback this refuses is not acceptable); a partially
        interleaved transition would be worse still, stranding a module on the
        wrong device for a phase that then reads it."""
        for half, modules, paired, extras in (
            ("generation", self._gen_modules, [p[0] for p in self._pairs], self._gen_unpaired),
            ("understanding", self._und_modules, [p[1] for p in self._pairs], self._und_unpaired),
        ):
            covered = {id(module) for module in (*paired, *extras)}
            if len(paired) + len(extras) != len(modules) or covered != {
                id(module) for module in modules
            }:
                raise RuntimeError(
                    f"SenseNova eviction cannot interleave the {half} half: "
                    f"{len(modules)} selected module(s) against {len(paired)} paired "
                    f"+ {len(extras)} unpaired. Refusing the batched order, which "
                    f"holds both halves in host memory at once"
                )

    @property
    def understanding_modules(self) -> tuple:
        """The half staged to CPU for the denoise phase.

        Public because the shared-window census derives its deferred parameter
        set from it: "deferred" and "absent during the generation backward" must
        be one set, not two.
        """
        return tuple(self._und_modules)

    def _best_effort_cpu(self) -> Exception | None:
        first_error = None
        for module in (*self._gen_modules, *self._und_modules):
            if _module_already_staged_cpu(module, pageable=self._pageable):
                continue
            try:
                _move_modules_to_cpu(
                    (module,), warn_once=self._warn_once, pageable=self._pageable
                )
            except Exception as exc:
                first_error = first_error or exc
        try:
            self.transformer.to("cpu")
        except Exception as exc:
            first_error = first_error or exc
        return first_error

    def _assert_grad_free(self, modules, half: str) -> None:
        """Refuse to evict a half that still owns gradients.

        Only meaningful when the half is trained. Under fused backward the
        post-accumulate hook applies and then nulls each gradient, so a ``.grad``
        surviving to a phase boundary means the optimizer never ran over that
        parameter -- the exact "a half is never updated while the loss falls"
        failure 8.3 names. Staging it to CPU anyway would leave a CUDA gradient
        attached to a CPU weight, which the next hook cannot use either.
        """
        for module in modules:
            for name, parameter in module._parameters.items():
                if parameter is None or parameter.grad is None:
                    continue
                raise RuntimeError(
                    f"SenseNova four-phase eviction cannot stage the {half} half to "
                    f"CPU while {name} still holds a gradient: the optimizer has not "
                    f"consumed it, so this half would be moved without being updated"
                )

    def _half(self, half: str):
        return self._gen_modules if half == "generation" else self._und_modules

    def _evict_plan(self, half: str):
        """One-sided: stage a half to CPU with nothing coming back the other way."""
        return tuple(("d2h", (module,), half) for module in self._half(half))

    def _swap_plan(self, evicted: str):
        """Two-sided, interleaved pair by pair (see this module's TRANSFER ORDER).

        Unpaired extras cannot interleave -- they have no twin whose transfer
        pays for theirs. Outgoing extras go FIRST (staging only ever shrinks
        device residency) and incoming extras LAST.
        """
        if evicted == "generation":
            out_at, in_at = 0, 1
            out_extras, in_extras = self._gen_unpaired, self._und_unpaired
            out_half, in_half = "generation", "understanding"
        else:
            out_at, in_at = 1, 0
            out_extras, in_extras = self._und_unpaired, self._gen_unpaired
            out_half, in_half = "understanding", "generation"
        plan = [("d2h", (module,), out_half) for module in out_extras]
        for pair in self._pairs:
            plan.append(("d2h", (pair[out_at],), out_half))
            plan.append(("h2d", (pair[in_at],), in_half))
        plan.extend(("h2d", (module,), in_half) for module in in_extras)
        return tuple(plan)

    def _transition(self, operations, next_state: str) -> None:
        if self.state == "failed":
            raise RuntimeError("SenseNova eviction cannot reuse a failed transfer state")
        try:
            if self.four_phase:
                # Pre-flight over the WHOLE outgoing half, before any module
                # moves: the interleave would otherwise strand half a swap on a
                # gradient found late in the sweep.
                for half in dict.fromkeys(
                    half for operation, _, half in operations if operation == "d2h"
                ):
                    self._assert_grad_free(self._half(half), half)
            for operation, modules, _half in operations:
                if operation == "d2h":
                    _move_modules_to_cpu(
                        modules, warn_once=self._warn_once, pageable=self._pageable
                    )
                else:
                    _move_modules_to_device(modules, self.device)
        except Exception:
            self.state = "failed"
            self._best_effort_cpu()
            raise
        self.state = next_state

    def enter_prefix(self) -> None:
        if self.state == "failed":
            raise RuntimeError("SenseNova eviction cannot reuse a failed transfer state")
        if self.state == "prefix":
            return
        if self.state == "und_backward":
            # The design's saved round trip: the understanding half is already
            # resident from phase 3, and phase 0 of the next step wants it there.
            self.state = "prefix"
            return
        if self.state == "full":
            operations = self._evict_generation_plan
        elif self.state == "denoise":
            operations = self._swap_generation_plan
        else:
            raise RuntimeError(f"Invalid SenseNova eviction state: {self.state}")
        self._transition(operations, "prefix")

    def enter_denoise(self) -> None:
        if self.state == "failed":
            raise RuntimeError("SenseNova eviction cannot reuse a failed transfer state")
        if self.state == "denoise":
            return
        if self.state != "prefix":
            raise RuntimeError("SenseNova denoise phase requires a completed prefix phase")
        self._transition(self._swap_understanding_plan, "denoise")

    def enter_und_backward(self) -> None:
        """Phase 3: bring the understanding half back for its own backward."""
        if not self.four_phase:
            raise RuntimeError(
                "SenseNova und_backward phase requires the four-phase evictor; the "
                "three-state machine has no coordinate at which the understanding "
                "half can be made resident again inside one backward"
            )
        if self.state == "failed":
            raise RuntimeError("SenseNova eviction cannot reuse a failed transfer state")
        if self.state == "und_backward":
            return
        if self.state != "denoise":
            raise RuntimeError(
                "SenseNova und_backward phase requires a completed denoise phase, "
                f"got {self.state}"
            )
        self._transition(self._swap_generation_plan, "und_backward")

    def _assert_half_resident(self, modules, half: str, states) -> None:
        if self.state not in states:
            wanted = " or ".join(states)
            raise RuntimeError(
                f"SenseNova {half} work requires {wanted} state, got {self.state}"
            )
        expected = torch.device(self.device)

        def on_expected_device(tensor) -> bool:
            return tensor.device.type == expected.type and (
                expected.index is None or tensor.device.index == expected.index
            )

        for module in modules:
            for parameter in module._parameters.values():
                if parameter is None:
                    continue
                if not on_expected_device(parameter):
                    raise RuntimeError(f"SenseNova {half} parameter is not GPU-resident")
                if parameter.grad is not None and parameter.grad.device != parameter.device:
                    raise RuntimeError(f"SenseNova {half} gradient is on the wrong device")
            for name, buffer in module._buffers.items():
                if (
                    buffer is not None
                    and name not in module._non_persistent_buffers_set
                    and not on_expected_device(buffer)
                ):
                    raise RuntimeError(f"SenseNova {half} buffer is not GPU-resident")

    def assert_generation_resident(self) -> None:
        self._assert_half_resident(self._gen_modules, "generation", ("denoise",))

    def assert_understanding_resident(self) -> None:
        self._assert_half_resident(
            self._und_modules, "understanding", ("prefix", "und_backward")
        )

    def teardown(self) -> None:
        if self.state == "closed":
            return
        if self.state not in ("full", "prefix", "denoise", "und_backward", "failed"):
            raise RuntimeError(f"Invalid SenseNova eviction state: {self.state}")
        error = self._best_effort_cpu()
        self.state = "closed"
        if error is not None:
            raise RuntimeError(
                "SenseNova eviction teardown could not normalize all weights to CPU"
            ) from error


def install_training_phase_eviction(trainer: Any) -> SenseNovaTrainingPhaseEvictor:
    # ``sensenova_four_phase_eviction`` is set on EVERY trainer by
    # BaseTrainer.__init__, and LoRATrainer calls this installer too. Gate on the
    # method here as well as in train_runner: the four-phase selector relaxes the
    # symmetry backstop, and that backstop exists for exactly the case where the
    # front-line check did not run (a hand-built config, a probe, direct YAML).
    from api.param_defaults import TRAINING_DEFAULTS
    from core.training.ops.training_method import is_full_finetune

    four_phase = bool(
        getattr(trainer, "sensenova_four_phase_eviction", False)
    ) and is_full_finetune(trainer)
    # Read off trainer.config (the raw dict BaseTrainer.__init__ copies from
    # train_config) rather than a promoted trainer attribute: unlike the two
    # flags above, nothing else on the trainer needs to branch on this one, so
    # there is no reason to widen BaseTrainer's __init__ for a staging-mode
    # sub-option of eviction.
    pageable_staging = bool(
        getattr(trainer, "config", {}).get(
            "sensenova_mot_pageable_staging",
            TRAINING_DEFAULTS["sensenova_mot_pageable_staging"],
        )
    )
    evictor = SenseNovaTrainingPhaseEvictor(
        trainer.transformer, trainer.device, four_phase=four_phase,
        pageable_staging=pageable_staging,
    )
    trainer.sensenova_phase_evictor = evictor
    return evictor
