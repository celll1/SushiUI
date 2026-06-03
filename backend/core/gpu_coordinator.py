"""GPU coordination between image-generation requests and background training.

When an image-generation request arrives while a tagger (or other) trainer
is occupying the GPU, this coordinator:

  1. pauses every registered trainer at its next batch boundary,
  2. probes free VRAM and decides whether to offload trainer state
     (DRAM / disk / split) or just pause-in-place,
  3. yields the GPU to the generation request,
  4. on release, signals trainers to restore state and resume.

The coordinator uses threading events as the sole cross-thread channel —
the trainer thread itself does all device-movement work, so we avoid the
classic ``RuntimeError: cannot move CUDA tensor from another thread``
pitfall.

Usage from generation endpoint:

    async with gpu_coordinator.generation_slot(estimated_peak_gb=10.0):
        result = await asyncio.to_thread(pipeline_manager.generate_txt2img, ...)
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Literal, Optional, Protocol

import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------

OffloadMode = Literal["none", "dram", "disk", "split"]


@dataclass
class OffloadDecision:
    """How to free GPU for an incoming generation request.

    none  — keep trainer state on GPU, just pause the loop
    dram  — move model + optimizer state to pinned/regular DRAM
    disk  — move both to a torch.save file under swap_dir
    split — model → DRAM (small), optimizer state → disk (large)
    """
    mode: OffloadMode
    swap_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Trainer-side protocol
# ---------------------------------------------------------------------------

class TrainerHandle(Protocol):
    """Trainer-side interface for the coordinator.

    All methods marked "trainer-thread-only" MUST be invoked from the
    thread that owns the trainer's model/optimizer.  The coordinator
    NEVER calls these directly — it sets ``pause_event`` and the trainer
    invokes them itself when it reaches the next batch boundary.
    """
    # Coordinator → trainer
    pause_event: threading.Event
    # Trainer → coordinator (raised when offload completes, lowered on cycle reset)
    resumed_event: threading.Event
    # Trainer → coordinator (raised when restore completes)
    restored_event: threading.Event
    # Decision the trainer should follow on the next pause (set by coordinator)
    pending_decision: Optional[OffloadDecision]
    # For disk-tier swap location
    output_dir: str

    def estimate_state_bytes(self) -> int: ...
    def trainer_label(self) -> str: ...


# ---------------------------------------------------------------------------
# Coordinator singleton
# ---------------------------------------------------------------------------

class GPUCoordinator:
    """Refcounted serialization point for GPU access.

    Multiple concurrent generation requests share a single pause cycle
    (the trainer is paused once, all gens run in parallel sharing the
    freed VRAM, and trainer resumes only after the LAST generation
    releases its slot).

    Grace-period optimisation
    -------------------------
    After the last generation slot is released, the trainer is NOT
    resumed immediately.  Instead a short timer (``resume_grace_sec``)
    is started.  If another generation arrives before the timer fires,
    the trainer is already paused — the offload/restore round-trip is
    skipped entirely, saving several seconds of state movement for
    back-to-back generations.  Only when the timer fires with no pending
    generation is the trainer actually restored and resumed.

    The coordinator is thread-safe but its `generation_slot()` context
    manager is async — it offloads the blocking trainer-pause wait to
    a default executor so the FastAPI event loop stays responsive.
    """

    def __init__(self, resume_grace_sec: float = 5.0):
        self._lock = threading.RLock()
        self._handles: List[TrainerHandle] = []
        self._active_generations = 0
        self._resume_grace_sec = resume_grace_sec
        # Trainers currently in the paused state (offloaded or just paused).
        # Non-empty either while actively generating OR during grace period.
        self._currently_paused: List[TrainerHandle] = []
        self._resume_timer: Optional[threading.Timer] = None

    # -- registration ----------------------------------------------------

    def register_trainer(self, h: TrainerHandle) -> None:
        with self._lock:
            if h not in self._handles:
                self._handles.append(h)
                print(f"[GPUCoordinator] Registered trainer: {h.trainer_label()}")

    def unregister_trainer(self, h: TrainerHandle) -> None:
        with self._lock:
            if h in self._handles:
                self._handles.remove(h)
            # If this trainer was paused (grace period), remove it and cancel
            # the timer when the list becomes empty to avoid a spurious resume.
            if h in self._currently_paused:
                self._currently_paused.remove(h)
                if not self._currently_paused and self._resume_timer is not None:
                    self._resume_timer.cancel()
                    self._resume_timer = None
            print(f"[GPUCoordinator] Unregistered trainer: {h.trainer_label()}")

    def is_paused(self) -> bool:
        with self._lock:
            return self._active_generations > 0

    def get_active_tagger_handle(self):
        """Return the first registered TaggerTrainerHandle that can currently
        run inference (model on CUDA, processor attached), or None."""
        with self._lock:
            handles = list(self._handles)
        for h in handles:
            if hasattr(h, "can_predict") and h.can_predict():
                return h
        return None

    # -- decision logic --------------------------------------------------

    def _free_vram_gb(self) -> float:
        """Free CUDA memory in GB.  Returns +inf when CUDA unavailable
        so the 'no offload needed' branch is taken (the only sensible
        default when generation will run on CPU)."""
        if not torch.cuda.is_available():
            return float("inf")
        try:
            torch.cuda.empty_cache()
            free_b, _ = torch.cuda.mem_get_info()
            return free_b / 1e9
        except Exception:
            return 0.0

    def _dram_available_gb(self) -> float:
        if not _HAS_PSUTIL:
            return float("inf")   # assume plenty (DRAM tier is safest default)
        return psutil.virtual_memory().available / 1e9

    def _decide_for_handle(self, h: TrainerHandle, gen_peak_gb: float) -> OffloadDecision:
        """Conditional-auto policy:

          - if (free_vram - 2 GB safety) >= gen_peak: no offload needed
          - else if DRAM available >= state + 4 GB safety: DRAM
          - else: split (params -> DRAM, optimizer -> disk)
        """
        free_gb = self._free_vram_gb()
        state_gb = h.estimate_state_bytes() / 1e9
        if free_gb >= gen_peak_gb + 2.0:
            return OffloadDecision("none")
        dram_avail_gb = self._dram_available_gb()
        if dram_avail_gb >= state_gb + 4.0:
            return OffloadDecision("dram")
        # need disk
        swap_dir = os.path.join(h.output_dir, ".gpu_swap")
        try:
            os.makedirs(swap_dir, exist_ok=True)
        except OSError as e:
            print(f"[GPUCoordinator] WARNING: could not create swap dir {swap_dir}: {e}; "
                  f"falling back to DRAM (may OOM)")
            return OffloadDecision("dram")
        # Split: small things (params) → DRAM, big things (optimizer) → disk
        return OffloadDecision("split", swap_dir=swap_dir)

    # -- pause / resume cycle -------------------------------------------

    def _begin_pause_cycle(self, gen_peak_gb: float, timeout: float) -> List[TrainerHandle]:
        """Set pause_event on all registered trainers and wait for them
        to acknowledge (resumed_event).  Returns the list of trainers we
        successfully paused (so they can be resumed later)."""
        with self._lock:
            handles = list(self._handles)
            if not handles:
                return []

        paused: List[TrainerHandle] = []
        for h in handles:
            decision = self._decide_for_handle(h, gen_peak_gb)
            h.pending_decision = decision
            print(f"[GPUCoordinator] Pausing {h.trainer_label()}: "
                  f"decision={decision.mode}"
                  + (f" swap_dir={decision.swap_dir}" if decision.swap_dir else "")
                  + f" (gen_peak={gen_peak_gb:.1f}GB)")
            h.resumed_event.clear()
            h.restored_event.clear()
            h.pause_event.set()

        # Wait for each to ack offload completion.  Soft warn at half-timeout,
        # hard give up at full timeout (caller proceeds anyway).
        deadline = time.monotonic() + timeout
        for h in handles:
            remaining = max(0.0, deadline - time.monotonic())
            if h.resumed_event.wait(timeout=remaining):
                paused.append(h)
                print(f"[GPUCoordinator] {h.trainer_label()} acknowledged pause")
            else:
                print(f"[GPUCoordinator] ERROR: {h.trainer_label()} did not pause "
                      f"within {timeout:.1f}s — proceeding without offload "
                      f"(generation may OOM)")
                # leave pause_event set; trainer may catch up later and offload
        return paused

    def _end_pause_cycle(self, paused: List[TrainerHandle], timeout: float = 30.0):
        """Clear pause_event so trainers restore state and resume."""
        for h in paused:
            print(f"[GPUCoordinator] Resuming {h.trainer_label()}")
            h.pause_event.clear()
        # Wait for restore acks (best-effort)
        deadline = time.monotonic() + timeout
        for h in paused:
            remaining = max(0.0, deadline - time.monotonic())
            if not h.restored_event.wait(timeout=remaining):
                print(f"[GPUCoordinator] WARNING: {h.trainer_label()} did not "
                      f"restore within {timeout:.1f}s — continuing anyway")

    # -- grace-period timer ---------------------------------------------

    def _start_grace_timer(self) -> None:
        """Start (or restart) the post-generation grace timer.

        Must be called with self._lock held.
        When the timer fires, it resumes all currently-paused trainers
        provided no new generation has arrived in the interim.
        """
        if self._resume_timer is not None:
            self._resume_timer.cancel()

        def _fire() -> None:
            with self._lock:
                if self._active_generations > 0:
                    # A new generation started and should have cancelled us;
                    # this is a harmless race — just bail out.
                    return
                self._resume_timer = None
                to_resume = list(self._currently_paused)
                self._currently_paused.clear()
            if to_resume:
                labels = [h.trainer_label() for h in to_resume]
                print(f"[GPUCoordinator] Grace period expired — resuming {labels}")
                self._end_pause_cycle(to_resume, timeout=30.0)

        t = threading.Timer(self._resume_grace_sec, _fire)
        t.daemon = True
        t.start()
        self._resume_timer = t
        print(f"[GPUCoordinator] Grace period started ({self._resume_grace_sec}s)")

    # -- public async context manager -----------------------------------

    @asynccontextmanager
    async def generation_slot(self, estimated_peak_gb: float, timeout: float = 60.0):
        """Acquire the GPU for image generation.  Refcounted.

        On entry, if a previous grace period is still running (trainers
        already paused) the timer is cancelled and the existing paused
        state is reused — no offload/restore round-trip.

        Args:
            estimated_peak_gb: Conservative estimate of peak VRAM the
                generation will need.  Drives the offload decision.
            timeout: Seconds to wait for trainers to acknowledge pause
                before proceeding anyway (with a log warning).
        """
        loop = asyncio.get_event_loop()
        need_pause = False

        with self._lock:
            self._active_generations += 1
            if self._active_generations == 1:
                if self._currently_paused:
                    # Grace period active — reuse existing paused state.
                    if self._resume_timer is not None:
                        self._resume_timer.cancel()
                        self._resume_timer = None
                    labels = [h.trainer_label() for h in self._currently_paused]
                    print(f"[GPUCoordinator] Grace-period reuse: {labels} "
                          f"already paused, skipping offload/restore")
                else:
                    need_pause = True

        try:
            if need_pause:
                new_paused = await loop.run_in_executor(
                    None, self._begin_pause_cycle, estimated_peak_gb, timeout
                )
                with self._lock:
                    self._currently_paused = new_paused
            yield
        finally:
            start_grace = False
            with self._lock:
                self._active_generations -= 1
                if self._active_generations == 0 and self._currently_paused:
                    start_grace = True
                    self._start_grace_timer()
            # _start_grace_timer was called under the lock above; nothing else needed.


# Module-level singleton.
gpu_coordinator = GPUCoordinator()
