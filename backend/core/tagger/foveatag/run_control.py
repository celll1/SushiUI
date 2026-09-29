"""Thread-safe lifecycle controls shared by FoveaTag training stages."""

from __future__ import annotations

import threading


class FoveaRunControl:
    """Request a checkpointed stop from a backend-owned training thread."""

    def __init__(self) -> None:
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()
