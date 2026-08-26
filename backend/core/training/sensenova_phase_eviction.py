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

OVERLAPPED TRANSFER -- opt-in, off by default
(``sensenova_mot_overlap_transfer``). By default both legs of a swap are
host-blocking, so the transfer term is ``d2h + h2d``. PCIe is full duplex and
the two directions have independent copy engines, so a pair's outgoing and
incoming legs can run concurrently on two side streams; the arithmetic ceiling
is ``max(d2h, h2d)``. What the flag actually buys is UNMEASURED -- it ships off
for that reason.

Six things make it safe rather than merely faster:

  * ``record_stream`` on every d2h source. ``stage_modules_to_pinned_cpu``
    reassigns ``parameter.data``, dropping the model's reference to the CUDA
    source immediately; with the copy still in flight the caching allocator
    would otherwise be free to hand that block to the concurrent h2d
    destination. The ``sources`` list keeps the tensor alive across that
    reassignment so the ``record_stream`` call is still legal when it lands.
  * the pinned h2d source is held in the in-flight record until its event has
    been waited on. Torch may event-guard a pinned ``non_blocking`` h2d, but that
    is not verified here, and the caching HOST allocator would otherwise be free
    to re-hand the block to the next ``_stage_tensor``.
  * the h2d DESTINATION is allocated on the DEFAULT stream, and only the copy
    into it is issued on the side stream (see ``_move_modules_to_device``).
    Torch's device caching allocator partitions free blocks by OWNING STREAM, so
    a destination requested inside the side stream's context could never be
    handed a block the outgoing half just freed on the default stream -- every
    incoming module would ``cudaMalloc`` fresh and the window bound below would
    be a whole half instead of a few modules. It also leaves the destination
    default-stream-owned, so the next phase's compute reads a block whose
    lifetime that stream already controls.
  * ``_transition`` synchronizes the device BEFORE it issues anything. Under
    overlap that is a CORRECTNESS barrier and not only the timing one section
    8.6 introduced it as: the side streams read, and free, weights the preceding
    phase's still-queued compute may still be writing, and the join below only
    makes the default stream wait on the side streams, never the reverse.
  * every stream is joined before ``_transition`` returns, which is also what
    orders the side-stream writes before the next phase's compute reads them.
    ``assert_generation_resident`` and its twin check DEVICE PLACEMENT only and
    would pass on a queued-but-unfinished copy. A transition that raises drains
    the same way before it unwinds, so no pinned block is returned to the
    caching host allocator with its copy still in flight.
  * the window is ``_OVERLAP_WINDOW_PAIRS`` deep, so the transient extra device
    residency is bounded by that many modules rather than by a whole half.

Refused together with ``sensenova_mot_pageable_staging`` (see
``install_training_phase_eviction``) rather than silently degraded, and dropped
to the serial path for the rest of the run if a pinned allocation ever fails --
an async copy out of pageable host memory is bounce-buffered and effectively
host-synchronous, so it would pay every correctness cost above for nothing.

TRANSFER ACCOUNTING. Every transition tallies seconds and bytes per direction
into plain attributes, drained once per step by the train loop (see
``drain_transfer_stats``). Section 8.6 of SENSENOVA_TRAINING_DESIGN.md states
this loop's per-iteration transfer volume as ARITHMETIC; these counters are the
measurement it never had. The seconds change UNIT of measurement between the two
modes -- host wall time around a blocking copy when serial, CUDA event time on
the side stream when overlapped, where the two directions run concurrently and
their sum therefore exceeds the transition's wall. ``overlap_active`` says which
mode produced them and is charted as ``sn_swap_overlap``. It is derived from
what each transition ACTUALLY RAN and AND-ed across the drain window, so a step
that straddles a mid-run downgrade -- part event milliseconds, part host wall
seconds -- reports serial rather than claiming a unit half its total is not in.

PAGEABLE STAGING -- opt-in, off by default
(``sensenova_mot_pageable_staging``). Trades the pinned pool's sticky
high-water (torch's caching host allocator never returns a pinned block to
the OS) for host RAM the OS can reclaim, at an unmeasured transfer-time cost.
Refused without ``sensenova_mot_phase_eviction``, since with the evictor off
nothing here ever runs. Read from ``trainer.config`` rather than a promoted
attribute, so this flag needed no change to ``BaseTrainer``.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, Iterable, List, NamedTuple, Optional

import torch
from torch import nn

from core.models.sensenova.mot_cpu_staging import stage_modules_to_pinned_cpu
from core.models.sensenova.mot_weight_selector import select_mot_weight_modules

_PIN_FAILURE_MESSAGE = (
    "[SenseNova] Training MoT eviction could not pin CPU staging "
    "memory ({exc}); continuing with blocking pageable copies."
)

_OVERLAP_PAGEABLE_REFUSAL = (
    "sensenova_mot_overlap_transfer and sensenova_mot_pageable_staging cannot "
    "be combined: pageable staging hands the copies ordinary host memory, and "
    "cudaMemcpyAsync out of pageable memory is staged through a driver bounce "
    "buffer and is effectively host-synchronous, so the overlap would pay its "
    "correctness cost (record_stream on every freed device block, pinned-source "
    "lifetime, a transient extra module on device) for no concurrency at all. "
    "Enable one of the two."
)

_OVERLAP_DOWNGRADE_MESSAGE = (
    "[SenseNova] MoT overlapped transfer disabled for the rest of this run: a "
    "pinned staging allocation failed, and an async copy against pageable host "
    "memory is bounce-buffered and effectively host-synchronous. Continuing on "
    "the serial transfer path."
)

_OVERLAP_NON_CUDA_MESSAGE = (
    "[SenseNova] MoT overlapped transfer is inert on this run: the evictor's "
    "device is not a live CUDA one, so there are no independent copy engines to "
    "issue the two directions on. Continuing on the serial transfer path."
)

# Pairs of (outgoing, incoming) modules allowed in flight at once. Each pair
# admitted ahead of its twin's completion is one module of transient extra
# device residency: at bf16 the largest single MoT weight is 0.09375 GiB, so
# four pairs bound it at 0.375 GiB, ~2.5% of a half. Small enough to budget as
# real, deep enough to keep both copy engines fed while the host retires the
# oldest pair. The bound is only a bound because the incoming destination is
# allocated on the DEFAULT stream (see _move_modules_to_device): a side-stream
# allocation cannot be handed a default-stream-owned free block at all, so the
# growth would be a whole half rather than this window.
_OVERLAP_WINDOW_PAIRS = 4


def _move_modules_to_cpu(
    modules: Iterable[nn.Module], *, warn_once: Dict[str, bool], pageable: bool = False,
    non_blocking: bool = False, sources: Optional[List[torch.Tensor]] = None,
) -> None:
    stage_modules_to_pinned_cpu(
        modules, warn_once=warn_once, warn_message=_PIN_FAILURE_MESSAGE,
        pageable=pageable, non_blocking=non_blocking, sources=sources,
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


def _move_modules_to_device(
    modules: Iterable[nn.Module], device: Any, *, non_blocking: bool = False,
    sources: Optional[List[torch.Tensor]] = None, streams=None, stream=None,
) -> None:
    """``sources`` collects the host tensors each copy reads FROM. Under
    ``non_blocking`` the reassignment below drops the model's reference to a
    pinned block whose copy is still in flight; the caller holds these until the
    copy's event has been waited on.

    ``streams``/``stream`` split the move in two. The destination is allocated
    HERE, under whatever stream is current (the default one), and only the copy
    into it is issued on ``stream``. Issuing the allocation on the side stream
    instead would defeat the whole point: torch's device caching allocator
    partitions free blocks by owning stream, so a side-stream request can never
    be handed a block the outgoing half just freed on the default stream and
    each incoming module would ``cudaMalloc`` fresh. ``record_stream`` on the
    destination is the other half of that split -- the default-stream owner must
    respect the side stream's write when it eventually frees the block.

    Without ``streams`` this is ``Tensor.to``, which short-circuits an
    already-resident tensor; the split path keeps that short-circuit by taking
    it only for a tensor that is actually on the host.
    """
    def _collect(tensor):
        if sources is not None and tensor.device.type == "cpu":
            sources.append(tensor)

    plan: List[tuple] = []

    def _place(tensor):
        _collect(tensor)
        if streams is None or tensor.device.type != "cpu":
            return tensor.to(device, non_blocking=non_blocking)
        destination = torch.empty_like(tensor, device=device)
        plan.append((destination, tensor))
        return destination

    for module in modules:
        for parameter in module._parameters.values():
            if parameter is not None:
                parameter.data = _place(parameter.data)
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and name not in module._non_persistent_buffers_set:
                module._buffers[name] = _place(buffer)
    if not plan:
        return
    with streams.stream_context(stream):
        for destination, source in plan:
            destination.copy_(source, non_blocking=non_blocking)
            destination.record_stream(stream)


class _InFlight(NamedTuple):
    """One issued-but-unretired copy. ``keepalive`` is the h2d leg's pinned
    source (see this module's OVERLAP note); the d2h leg keeps nothing, having
    ``record_stream``-ed its device source instead."""

    operation: str
    start: Any
    end: Any
    keepalive: tuple


class _TransferStreams:
    """The two side streams and their events, isolated behind one object so the
    overlap path is reachable from the CPU-only synthetic tree by injection.

    ``cuda`` is that seam one level down: the evictor injects a whole fake
    ``_TransferStreams``, while a test of THIS class injects a fake ``torch.cuda``
    into the real one, so the routing and join wiring is covered rather than
    replaced."""

    def __init__(self, device: Any, *, cuda=torch.cuda):
        self.device = device
        self._cuda = cuda
        self.d2h = cuda.Stream(device=device)
        self.h2d = cuda.Stream(device=device)

    def stream_for(self, operation: str):
        return self.d2h if operation == "d2h" else self.h2d

    def stream_context(self, stream):
        return self._cuda.stream(stream)

    def record_event(self, stream):
        event = self._cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def join(self) -> None:
        current = self._cuda.current_stream(self.device)
        current.wait_stream(self.d2h)
        current.wait_stream(self.h2d)


def _make_transfer_streams(device: Any) -> Optional[_TransferStreams]:
    """None on any device that is not a live CUDA one, which is what makes the
    flag a clean no-op rather than a crash on a CPU/meta evictor."""
    try:
        resolved = torch.device(device)
    except (TypeError, ValueError, RuntimeError):
        return None
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return _TransferStreams(resolved)


class SenseNovaTrainingPhaseEvictor:
    """Keep only the phase-active MoT half resident while training."""

    def __init__(
        self, transformer: nn.Module, device: Any, *, four_phase: bool = False,
        pageable_staging: bool = False, overlap_transfer: bool = False,
        streams_factory=_make_transfer_streams,
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
        # sensenova_mot_overlap_transfer: see this module's OVERLAPPED TRANSFER
        # note. The installer refuses this pair with a user-facing message; this
        # backstop is for a hand-built evictor that never went through it.
        if pageable_staging and overlap_transfer:
            raise ValueError(_OVERLAP_PAGEABLE_REFUSAL)
        self._overlap = bool(overlap_transfer)
        self._streams_factory = streams_factory
        self._streams = None
        self._overlap_downgraded = False
        # None until a transition has run. AND-ed across the drain window; see
        # this module's TRANSFER ACCOUNTING note for why it is derived from what
        # ran rather than from what was configured.
        self._overlap_ran: Optional[bool] = None
        self._overlap_this_transition = False
        self.state = "full"
        self._warn_once: Dict[str, bool] = {}
        self.d2h_seconds = 0.0
        self.h2d_seconds = 0.0
        self.d2h_bytes = 0
        self.h2d_bytes = 0
        try:
            self._device_obj = torch.device(device)
        except (TypeError, ValueError, RuntimeError):
            self._device_obj = None
        self._sync_device = (
            self._device_obj
            if self._device_obj is not None
            and self._device_obj.type == "cuda"
            and torch.cuda.is_available()
            else None
        )
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
    def overlap_active(self) -> bool:
        """True iff EVERY transition since the last drain ran the two-stream
        path to completion. False before the first transition, and False for a
        step that straddled a downgrade."""
        return bool(self._overlap_ran)

    @property
    def understanding_modules(self) -> tuple:
        """The half staged to CPU for the denoise phase.

        Public because the shared-window census derives its deferred parameter
        set from it: "deferred" and "absent during the generation backward" must
        be one set, not two.
        """
        return tuple(self._und_modules)

    def _best_effort_cpu(self) -> Exception | None:
        # Before the FIRST predicate call, not just before the first copy:
        # _module_already_staged_cpu checks device and pin flag, never content,
        # so a pinned buffer whose d2h has not landed reads as already staged and
        # is skipped -- a corrupt CPU half. self.transformer.to("cpu") below would
        # race in-flight copies for the same reason. A no-op on the serial path.
        self._sync_transfers()
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

    def _sync(self) -> None:
        if self._sync_device is not None:
            torch.cuda.synchronize(self._sync_device)

    def _sync_transfers(self) -> None:
        """Join the overlap side streams, then the device. Ordered that way so a
        stream that is still queueing work is drained by the device sync too."""
        if self._streams is not None:
            self._streams.join()
        self._sync()

    def _overlap_streams(self):
        """The side streams, or None to run serially: the flag is off, a pin
        failure has downgraded the run, or the device is not a live CUDA one."""
        if not self._overlap or self._overlap_downgraded:
            return None
        if self._streams is None:
            self._streams = self._streams_factory(self.device)
            if self._streams is None:
                self._downgrade_overlap(_OVERLAP_NON_CUDA_MESSAGE)   # never retry
        return self._streams

    def _downgrade_overlap(self, message: str = _OVERLAP_DOWNGRADE_MESSAGE) -> None:
        """Serial for the rest of the run, said once. Also clears the CURRENT
        transition's flag: its seconds are now part event milliseconds and part
        host wall, so it must not be charted as an overlapped one."""
        self._overlap_this_transition = False
        if not self._overlap_downgraded:
            self._overlap_downgraded = True
            print(message)

    def _will_copy(self, tensor, operation: str) -> bool:
        """Whether this operation actually copies ``tensor``, mirroring the
        short-circuits in ``_stage_tensor`` (d2h) and ``Tensor.to`` (h2d) so an
        already-staged / already-resident tensor is charged zero bytes."""
        if operation == "d2h":
            if tensor.device.type != "cpu":
                return True
            return not (self._pageable or tensor.is_pinned())
        if self._device_obj is None:
            return True
        return tensor.device.type != self._device_obj.type or (
            self._device_obj.index is not None
            and tensor.device.index != self._device_obj.index
        )

    def _pending_bytes(self, modules, operation: str) -> int:
        total = 0
        for module in modules:
            for parameter in module._parameters.values():
                if parameter is None or not self._will_copy(parameter.data, operation):
                    continue
                total += parameter.data.numel() * parameter.data.element_size()
            for name, buffer in module._buffers.items():
                if buffer is None or name in module._non_persistent_buffers_set:
                    continue
                if self._will_copy(buffer, operation):
                    total += buffer.numel() * buffer.element_size()
        return total

    def drain_transfer_stats(self) -> Dict[str, Any]:
        """Return this step's transfer totals and reset them.

        Drained ONCE per step by the train loop: ``log_extra_metric`` overwrites
        rather than accumulates, and a four-phase step contains two swaps. The
        evictor deliberately has no trainer backref, so the pull is one-way.

        ``overlap_active`` travels WITH the totals rather than being read off
        the evictor afterwards: it is the unit label for the two seconds, and
        reading it after the reset would label them by the next step's mode.
        """
        stats = {
            "d2h_seconds": self.d2h_seconds,
            "h2d_seconds": self.h2d_seconds,
            "d2h_bytes": self.d2h_bytes,
            "h2d_bytes": self.h2d_bytes,
            "overlap_active": self.overlap_active,
        }
        self.d2h_seconds = 0.0
        self.h2d_seconds = 0.0
        self.d2h_bytes = 0
        self.h2d_bytes = 0
        self._overlap_ran = None
        return stats

    def _charge(self, operation: str, seconds: float, moved: int) -> None:
        if operation == "d2h":
            self.d2h_seconds += seconds
            self.d2h_bytes += moved
        else:
            self.h2d_seconds += seconds
            self.h2d_bytes += moved

    def _run_serial(self, operations) -> None:
        """No per-operation sync: both primitives are host-blocking here
        (``pinned.copy_(cuda, non_blocking=False)`` and ``Tensor.to`` both end in
        a stream synchronize), so one would add a device-wide barrier per module
        -- ~250 per four-phase step -- and charge its own latency to the
        transfer bucket it is supposed to be measuring. The barrier that IS
        load-bearing is the leading one in ``_transition``."""
        for operation, modules, _half in operations:
            moved = self._pending_bytes(modules, operation)
            started = time.perf_counter()
            if operation == "d2h":
                _move_modules_to_cpu(
                    modules, warn_once=self._warn_once, pageable=self._pageable
                )
            else:
                _move_modules_to_device(modules, self.device)
            self._charge(operation, time.perf_counter() - started, moved)

    def _retire(self, entry: _InFlight) -> None:
        """Wait for one issued copy and charge its CUDA-event time. Waiting on
        the HOST (not just ordering a stream) is what bounds the window: the
        outgoing block is not reusable, and the pinned source below not
        re-handable, until the copy has actually landed."""
        entry.end.synchronize()
        self._charge(entry.operation, entry.start.elapsed_time(entry.end) / 1000.0, 0)

    def _run_overlapped(self, operations, streams) -> None:
        inflight: deque = deque()
        try:
            for index, (operation, modules, _half) in enumerate(operations):
                if self._overlap_downgraded:
                    while inflight:
                        self._retire(inflight.popleft())
                    streams.join()
                    self._run_serial(tuple(operations)[index:])
                    return
                while len(inflight) >= 2 * _OVERLAP_WINDOW_PAIRS:
                    self._retire(inflight.popleft())
                self._charge(operation, 0.0, self._pending_bytes(modules, operation))
                stream = streams.stream_for(operation)
                sources: List[torch.Tensor] = []
                if operation == "d2h":
                    # The destination is HOST memory here, so the whole leg runs
                    # under the side stream's context.
                    with streams.stream_context(stream):
                        start = streams.record_event(stream)
                        _move_modules_to_cpu(
                            modules, warn_once=self._warn_once, pageable=False,
                            non_blocking=True, sources=sources,
                        )
                        end = streams.record_event(stream)
                    for source in sources:
                        if source.is_cuda:
                            source.record_stream(stream)
                    sources = []   # record_stream replaces the keepalive here
                else:
                    # NOT under the stream context: the destination is DEVICE
                    # memory and must be allocated on the default stream, or the
                    # window bound is a whole half. _move_modules_to_device
                    # enters the context for the copies alone. The events bracket
                    # that stream's own work either way.
                    start = streams.record_event(stream)
                    _move_modules_to_device(
                        modules, self.device, non_blocking=True, sources=sources,
                        streams=streams, stream=stream,
                    )
                    end = streams.record_event(stream)
                inflight.append(_InFlight(operation, start, end, tuple(sources)))
                if self._warn_once.get("pin_failed"):
                    self._downgrade_overlap()
            while inflight:
                self._retire(inflight.popleft())
            streams.join()
        except BaseException:
            # `inflight` and every keepalive in it are released as this frame
            # unwinds, handing pinned blocks back to the caching host allocator
            # while their copies may still be reading them. Drain first --
            # _best_effort_cpu's own sync happens strictly later.
            self._sync_transfers()
            raise

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
            # Leading sync, load-bearing twice over and NOT removable as
            # redundant. Serially it is attribution: the first blocking copy
            # would otherwise absorb the tail of the preceding phase's
            # still-queued compute and inflate the d2h bucket (the mistake
            # behind the retracted number in SENSENOVA_TRAINING_DESIGN.md 8.3.2),
            # and it adds no net wait, since the copy blocks on that same work
            # one statement later. Under overlap it is CORRECTNESS: the side
            # streams read and free weights that compute may still be writing,
            # and the join at the end of the transition only makes the default
            # stream wait on the side streams, never the reverse.
            self._sync()
            streams = self._overlap_streams()
            self._overlap_this_transition = streams is not None
            if streams is None:
                self._run_serial(operations)
            else:
                self._run_overlapped(operations, streams)
            # Derived from what RAN: _downgrade_overlap clears the flag mid-plan.
            self._overlap_ran = (
                self._overlap_this_transition
                if self._overlap_ran is None
                else (self._overlap_ran and self._overlap_this_transition)
            )
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
    # train_config) rather than promoted trainer attributes: unlike the two
    # flags above, nothing else on the trainer branches on these two, so there
    # is no reason to widen BaseTrainer's __init__ for transfer-mode
    # sub-options of eviction.
    config = getattr(trainer, "config", {})
    pageable_staging = bool(
        config.get(
            "sensenova_mot_pageable_staging",
            TRAINING_DEFAULTS["sensenova_mot_pageable_staging"],
        )
    )
    overlap_transfer = bool(
        config.get(
            "sensenova_mot_overlap_transfer",
            TRAINING_DEFAULTS["sensenova_mot_overlap_transfer"],
        )
    )
    if pageable_staging and overlap_transfer:
        raise ValueError(_OVERLAP_PAGEABLE_REFUSAL)
    evictor = SenseNovaTrainingPhaseEvictor(
        trainer.transformer, trainer.device, four_phase=four_phase,
        pageable_staging=pageable_staging, overlap_transfer=overlap_transfer,
    )
    trainer.sensenova_phase_evictor = evictor
    return evictor
