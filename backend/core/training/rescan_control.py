"""
In-process controller for skipping the *currently rescanning* dataset during a
training pre-flight rescan (image-generation LoRA/Full-FT and tagger training).

The pre-flight rescan runs inside the main backend process (not the training
subprocess): for image-gen it runs in the ``start_training_run`` async handler,
for tagger in a background ``_run`` thread.  A skip request arrives as a normal
HTTP call on the event loop and only needs to flip an in-memory flag, which the
rescan's directory walkers / registration loop poll cooperatively.

Skip is *per current dataset*: it aborts the dataset being rescanned right now
and the loop then continues to the next dataset (and finally to training).
Already-committed changes from a partially-completed rescan are kept as-is.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple


class RescanSkipped(Exception):
    """Raised by a cooperatively-cancellable walker when the current dataset's
    rescan has been skipped via the controller."""


class RescanSkipController:
    """Thread-safe registry of the currently-rescanning dataset per run, plus a
    skip flag the rescan polls.  Keyed by ``(scope, str(run_id))`` where scope
    is ``"training"`` or ``"tagger"``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (scope, run_id) -> {"dataset_id": int, "skip": bool}
        self._current: Dict[Tuple[str, str], Dict] = {}

    def begin(self, scope: str, run_id, dataset_id: int) -> None:
        """Mark *dataset_id* as the dataset now being rescanned for this run."""
        with self._lock:
            self._current[(scope, str(run_id))] = {
                "dataset_id": int(dataset_id),
                "skip": False,
            }

    def end(self, scope: str, run_id) -> None:
        """Clear the current-dataset registration for this run."""
        with self._lock:
            self._current.pop((scope, str(run_id)), None)

    def request_skip(self, scope: str, run_id, dataset_id: Optional[int] = None) -> bool:
        """Flag the current dataset to be skipped.

        When *dataset_id* is given, the skip only applies if it matches the
        dataset currently being rescanned (avoids racing a skip meant for a
        dataset that already finished).  Returns True if a matching active
        rescan was flagged.
        """
        with self._lock:
            cur = self._current.get((scope, str(run_id)))
            if cur is None:
                return False
            if dataset_id is not None and int(dataset_id) != cur["dataset_id"]:
                return False
            cur["skip"] = True
            return True

    def should_skip(self, scope: str, run_id) -> bool:
        """Poll: True once a skip has been requested for the current dataset."""
        with self._lock:
            cur = self._current.get((scope, str(run_id)))
            return bool(cur and cur["skip"])

    def current_dataset(self, scope: str, run_id) -> Optional[int]:
        with self._lock:
            cur = self._current.get((scope, str(run_id)))
            return cur["dataset_id"] if cur else None


# Module-level singleton shared by the routes and the walkers.
rescan_skip_controller = RescanSkipController()
