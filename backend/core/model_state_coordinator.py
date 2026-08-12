"""Process-local lifecycle gate for the live inference model."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Optional


class ModelStateBusyError(RuntimeError):
    """Raised when a model mutation would overlap another lifecycle activity."""


class ModelStateCoordinator:
    def __init__(self) -> None:
        self._lock = threading.RLock()
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

    @contextmanager
    def mutation(self, name: str):
        with self._lock:
            reasons = []
            if self._mutation is not None:
                reasons.append(self._mutation)
            if self._active_generations:
                reasons.append(f"{self._active_generations} generation request(s)")
            if self._activities:
                reasons.extend(sorted(self._activities))
            if reasons:
                raise ModelStateBusyError(
                    f"Cannot start {name} while " + ", ".join(reasons) + " is active."
                )
            self._mutation = name
        try:
            yield
        finally:
            with self._lock:
                if self._mutation == name:
                    self._mutation = None

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "active_generations": self._active_generations,
                "activities": sorted(self._activities),
                "mutation": self._mutation,
            }


model_state_coordinator = ModelStateCoordinator()
