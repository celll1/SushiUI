"""Process-local lifecycle gate for the live inference model."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional


class ModelStateBusyError(RuntimeError):
    """Raised when a model mutation would overlap another lifecycle activity."""


class ModelStateCoordinator:
    def __init__(self) -> None:
        # A Condition (not a bare lock) so a background job can wait for the
        # gate to clear instead of failing the moment anything else is busy.
        self._lock = threading.Condition(threading.RLock())
        self._active_generations = 0
        self._activities: Dict[str, int] = {}
        self._mutation: Optional[str] = None

    def begin_generation(self) -> None:
        with self._lock:
            if self._mutation is not None:
                raise ModelStateBusyError(
                    f"Cannot start generation while {self._mutation} is changing the loaded model."
                )
            self._active_generations += 1

    def end_generation(self) -> None:
        with self._lock:
            self._active_generations = max(0, self._active_generations - 1)
            self._lock.notify_all()

    def begin_activity(self, name: str) -> None:
        with self._lock:
            if self._mutation is not None:
                raise ModelStateBusyError(
                    f"Cannot start {name} while {self._mutation} is changing the loaded model."
                )
            self._activities[name] = self._activities.get(name, 0) + 1

    def end_activity(self, name: str) -> None:
        with self._lock:
            count = self._activities.get(name, 0)
            if count <= 1:
                self._activities.pop(name, None)
            else:
                self._activities[name] = count - 1
            self._lock.notify_all()

    def _blockers(self, include_activities: bool) -> List[str]:
        """Caller must hold self._lock."""
        reasons: List[str] = []
        if self._mutation is not None:
            reasons.append(self._mutation)
        if self._active_generations:
            count = self._active_generations
            reasons.append(f"{count} generation request{'s' if count != 1 else ''}")
        if include_activities and self._activities:
            reasons.extend(sorted(self._activities))
        return reasons

    @contextmanager
    def mutation(
        self,
        name: str,
        *,
        wait_timeout: Optional[float] = None,
        wait_for_activities: bool = True,
        on_wait: Optional[Callable[[List[str]], None]] = None,
    ):
        """Take exclusive ownership of the live model's lifecycle.

        wait_timeout=None fails immediately when anything else is busy -- the
        right shape for an interactive request that has to answer now. A
        background job with a status channel should pass a timeout so it queues
        behind in-flight work instead of dying on a single concurrent
        generation, and on_wait to report what it is queued behind.

        wait_for_activities=False ignores registered activities. Subprocess
        training never touches the in-process model, so a job that only needs
        the weights to hold still does not have to outlast a training run.
        """
        deadline = None if wait_timeout is None else time.monotonic() + wait_timeout
        reported: Optional[List[str]] = None
        with self._lock:
            while True:
                reasons = self._blockers(wait_for_activities)
                if not reasons:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    remaining = 0
                if deadline is None or remaining == 0:
                    raise ModelStateBusyError(
                        f"Cannot start {name}; blocked by: " + ", ".join(reasons) + "."
                    )
                if on_wait is not None and reasons != reported:
                    reported = reasons
                    on_wait(reasons)
                # Bounded wait: end_generation/end_activity notify, but a
                # cap keeps the timeout honest if a notify is ever missed.
                self._lock.wait(min(remaining, 0.5))
            self._mutation = name
        try:
            yield
        finally:
            with self._lock:
                if self._mutation == name:
                    self._mutation = None
                self._lock.notify_all()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "active_generations": self._active_generations,
                "activities": sorted(self._activities),
                "mutation": self._mutation,
            }


model_state_coordinator = ModelStateCoordinator()
