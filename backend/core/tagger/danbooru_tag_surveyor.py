"""
Periodic surveyor that discovers newly created high-frequency Danbooru tags
for vocabulary expansion during tagger training.

The surveyor runs in a background daemon thread and maintains a set of
"approved" tags: tags created within the configured lookback window whose
post_count meets the threshold AND are absent from the current vocabulary.

The training thread queries approved_tags via get_approved() and, after adding
them to the vocabulary, calls mark_added() so they are no longer re-proposed.
"""

from __future__ import annotations

import datetime
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from .danbooru_client import DanbooruClient
from .tag_vocabulary import normalize_tag

if TYPE_CHECKING:
    from .tag_vocabulary import TagVocabulary

# Danbooru tag category codes
_CAT_GENERAL   = 0
_CAT_ARTIST    = 1
_CAT_COPYRIGHT = 3
_CAT_CHARACTER = 4
_CAT_META      = 5

# Map category code → vocabulary category name
_CAT_NAME: Dict[int, str] = {
    _CAT_GENERAL:   "General",
    _CAT_ARTIST:    "Artist",
    _CAT_COPYRIGHT: "Copyright",
    _CAT_CHARACTER: "Character",
    _CAT_META:      "Meta",
}


class DanbooruTagSurveyor:
    """Background thread that periodically fetches new high-count Danbooru tags.

    A tag is added to the approved set when:
      - Its ``created_at`` falls within the last ``lookback_days`` days
      - Its ``post_count`` >= ``min_count``
      - Its category is in ``categories``
      - It is NOT already in the current vocabulary

    The vocabulary reference is held by pointer, so changes made by
    ``expand_vocab_and_head`` are immediately visible here.
    """

    def __init__(
        self,
        vocabulary: "TagVocabulary",
        categories: Optional[List[int]] = None,
        min_count: int = 200,
        lookback_days: int = 90,
        survey_interval: float = 3600.0,
        api_interval: float = 1.4,
        dl_speed_kbps: int = 500,
    ) -> None:
        self._vocabulary      = vocabulary
        self._categories      = categories if categories is not None else [
            _CAT_GENERAL, _CAT_COPYRIGHT, _CAT_CHARACTER
        ]
        self._min_count       = min_count
        self._lookback_days   = lookback_days
        self._survey_interval = survey_interval

        self._client = DanbooruClient(api_interval=api_interval, dl_speed_kbps=dl_speed_kbps)

        self._approved: Set[str] = set()   # normalized tag names
        self._lock = threading.Lock()

        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._survey_loop, name="DanbooruSurveyor", daemon=True
        )
        self._thread.start()
        print(
            f"[DanbooruSurveyor] Started: min_count={self._min_count}, "
            f"lookback={self._lookback_days}d, interval={self._survey_interval}s"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Consumer API (called from training thread)
    # ------------------------------------------------------------------

    def get_approved(self) -> Set[str]:
        """Return a snapshot of currently approved new tags."""
        with self._lock:
            return set(self._approved)

    def mark_added(self, tags: List[str]) -> None:
        """Remove tags that have been added to the vocabulary."""
        normalized = {normalize_tag(t) for t in tags}
        with self._lock:
            self._approved -= normalized

    # ------------------------------------------------------------------
    # Background survey
    # ------------------------------------------------------------------

    def _survey_loop(self) -> None:
        # Run immediately on start, then every survey_interval seconds.
        while not self._stop.is_set():
            try:
                self._run_survey()
            except Exception as exc:
                print(f"[DanbooruSurveyor] Survey error: {exc}")
            # Interruptible sleep
            elapsed = 0.0
            while elapsed < self._survey_interval and not self._stop.is_set():
                time.sleep(min(10.0, self._survey_interval - elapsed))
                elapsed += 10.0

    def _run_survey(self) -> None:
        cutoff = (
            datetime.date.today() - datetime.timedelta(days=self._lookback_days)
        ).isoformat()

        new_approved: Set[str] = set()

        for category in self._categories:
            page = 1
            while True:
                tags = self._client.fetch_tags(
                    created_after=cutoff,
                    min_count=self._min_count,
                    category=category,
                    page=page,
                )
                if not tags:
                    break
                for entry in tags:
                    norm = normalize_tag(entry.get("name", ""))
                    if norm and norm not in self._vocabulary.tag_to_idx:
                        new_approved.add(norm)
                if len(tags) < 200:
                    break
                page += 1

        if new_approved:
            with self._lock:
                added_now = new_approved - self._approved
                self._approved |= new_approved
            if added_now:
                cat_label = ",".join(
                    _CAT_NAME.get(c, str(c)) for c in self._categories
                )
                print(
                    f"[DanbooruSurveyor] {len(added_now)} new approved tag(s) "
                    f"(categories={cat_label}, min_count={self._min_count}, "
                    f"lookback={self._lookback_days}d). "
                    f"Total approved: {len(self._approved)}"
                )
