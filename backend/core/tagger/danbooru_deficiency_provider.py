"""
Thread-safe holder for low-F1 (deficient) existing-vocabulary tags that the
tagger trainer wants the Danbooru augmentation worker to collect extra samples
for.

Unlike the surveyor (which discovers *new* tags by polling Danbooru), this
provider is a passive sink: the training thread periodically computes the
worst-F1 existing vocab tags from its per-tag metrics accumulator and pushes
their normalized names here via :meth:`set_targets`.  The Danbooru sampler
worker reads them via :meth:`get_targets` and issues collection queries.

Mirrors the surveyor's ``get_approved()`` consumer API so the sampler can treat
both feeds uniformly.
"""

from __future__ import annotations

import threading
from typing import List, Set


class DanbooruDeficiencyProvider:
    """Holds the current set of low-F1 target tags (normalized names)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._targets: Set[str] = set()

    # -- Producer API (training thread) --------------------------------

    def set_targets(self, normalized_tags: List[str]) -> None:
        """Replace the current target set with ``normalized_tags``.

        Called periodically by the trainer after recomputing worst-F1 tags.
        Replacement (not union) so tags whose F1 has recovered drop out.
        """
        with self._lock:
            self._targets = {t for t in normalized_tags if t}

    # -- Consumer API (sampler worker) ---------------------------------

    def get_targets(self) -> Set[str]:
        """Return a snapshot of the current low-F1 target tags."""
        with self._lock:
            return set(self._targets)
