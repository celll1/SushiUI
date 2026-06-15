"""
Danbooru online augmentation for tagger training.

DanbooruSampleBuffer runs a daemon thread that pre-fetches images from Danbooru
and converts them to tensors.  Samples are stored as raw tag lists; labels are
built at injection time in MixedDataLoader so that vocabulary expansions are
reflected immediately.

When vocabulary expansion is enabled, the buffer also drives *dynamic* queries:
tags discovered by the surveyor are queried directly so the freshly-grown heads
receive positive samples.  Within one epoch each dynamic tag is collected at
most once (tracked by post_id — no tensor caching, so no DRAM growth); a tag
whose posts are fully collected is skipped for the rest of the epoch.  At each
epoch boundary MixedDataLoader.__iter__ calls reset_download_cycle() so the next
epoch collects them again — mirroring the base dataset, which re-reads every
image once per epoch.

MixedDataLoader wraps a DataLoader and interleaves pure-Danbooru batches:
  1. Each base batch is yielded as (batch, is_injection=False).
  2. Every injection_interval base batches, a pure-Danbooru batch of size
     injection_batch_size is yielded as (batch, is_injection=True) — but
     only if the buffer has enough samples; otherwise skipped.
  3. The training loop must skip scheduler.step() and global_step += 1 on
     injection batches so LR / resume reproducibility match the base loader.
"""

from __future__ import annotations

import collections
import queue
import random
import threading
import time
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from .danbooru_client import DanbooruClient
from .tag_vocabulary import (
    QUALITY_TAG_GROUPS,
    RATING_TAGS,
    TagVocabulary,
    normalize_tag,
)
from .tagger_dataset import tagger_collate_fn

# Danbooru post field → category code, for co-occurrence vocab discovery.
# (general=0, artist=1, copyright=3, character=4, meta=5)
_COOC_CATEGORY_FIELDS = (
    (0, "tag_string_general"),
    (1, "tag_string_artist"),
    (3, "tag_string_copyright"),
    (4, "tag_string_character"),
    (5, "tag_string_meta"),
)


# ---------------------------------------------------------------------------
# Standalone label / mask builder (mirrors TaggerDataset._build_label_and_mask)
# ---------------------------------------------------------------------------

def _build_label_and_mask_standalone(
    tags: List[str],
    vocabulary: TagVocabulary,
    quality_masking_mode: str = "intra_group",
    alias_resolver: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (label, loss_mask) tensors from a tag list without a Dataset instance."""
    if alias_resolver is not None:
        tags = [alias_resolver.resolve(t) for t in tags]
    else:
        tags = [normalize_tag(t) for t in tags]

    voc = vocabulary
    num_tags = voc.num_tags
    label     = torch.zeros(num_tags, dtype=torch.float32)
    loss_mask = torch.ones(num_tags,  dtype=torch.float32)

    tag_set = set(tags)
    for tag in tag_set:
        if tag in voc.tag_to_idx:
            label[voc.tag_to_idx[tag]] = 1.0

    has_rating = any(normalize_tag(r) in tag_set for r in RATING_TAGS)
    if not has_rating:
        for idx in voc.rating_indices:
            loss_mask[idx] = 0.0

    present_groups: set = set()
    for group_name, gtags in QUALITY_TAG_GROUPS.items():
        if any(normalize_tag(t) in tag_set for t in gtags):
            present_groups.add(group_name)

    if not present_groups:
        for group_indices in voc.quality_indices.values():
            for idx in group_indices:
                loss_mask[idx] = 0.0
    elif quality_masking_mode == "intra_group":
        for group_name in present_groups:
            for idx in voc.quality_indices[group_name]:
                if label[idx] == 0.0:
                    loss_mask[idx] = 0.0
    # "cross_group": leave all loss_mask[*] = 1

    return label, loss_mask


# ---------------------------------------------------------------------------
# DanbooruSampleBuffer
# ---------------------------------------------------------------------------

class DanbooruSampleBuffer:
    """Background daemon thread that pre-fetches Danbooru images as tensors.

    Each buffered sample is a tuple (pixel_values, pixel_attention_mask,
    spatial_shapes, raw_tags) — labels are NOT pre-computed so they stay
    consistent after vocabulary expansions.

    The training loop drains the queue without blocking — if the fetch thread
    cannot keep up, training continues on the local dataset alone.
    """

    def __init__(
        self,
        tag_queries: List[str],
        vocabulary: TagVocabulary,
        processor: AutoProcessor,
        is_naflex: bool,
        quality_masking_mode: str = "intra_group",
        alias_resolver: Any = None,
        max_posts_per_query: int = 200,
        min_score: int = 0,
        buffer_size: int = 32,
        api_interval: float = 1.4,
        dl_speed_kbps: int = 500,
        expander: Any = None,
        surveyor: Any = None,
        deficiency_provider: Any = None,
        weight_static: float = 1.0,
        weight_new_tag: float = 1.0,
        weight_low_f1: float = 1.0,
        low_f1_min_posts: int = 50,
        cooc_expand_enable: bool = False,
        cooc_min_count: int = 50,
        cooc_categories: Optional[List[int]] = None,
        initial_dynamic_tags: Optional[Any] = None,
        max_dynamic_tags: int = 0,
        weight_cooc: float = 0.1,
        cooc_collect_per_epoch: int = 50,
        cooc_order_random: bool = True,
        initial_cooc_active_tags: Optional[List[str]] = None,
        query_expand: bool = False,
        query_min_count: int = 200,
        query_categories: Optional[List[int]] = None,
        query_top_k: int = 50,
        query_max_expanded: int = 0,
        query_resolve_interval: float = 3600.0,
        initial_query_tags: Optional[Any] = None,
        query_collect_per_epoch: int = 0,
        new_tag_collect_per_epoch: int = 0,
        low_f1_collect_per_epoch: int = 0,
    ) -> None:
        self._tag_queries      = list(tag_queries)
        self._vocabulary       = vocabulary
        self._processor        = processor
        self._is_naflex        = is_naflex
        self._quality_masking  = quality_masking_mode
        self._alias_resolver   = alias_resolver
        self._max_posts        = max_posts_per_query
        self._min_score        = min_score
        self._buffer_size      = buffer_size
        self._expander         = expander
        self._surveyor         = surveyor
        self._deficiency_provider = deficiency_provider
        # Co-occurrence discovery: add unknown tags that appear in collected
        # posts >= cooc_min_count times, filtered to cooc_categories. Unlike the
        # surveyor (which only finds *recently created* tags) this catches tags
        # that are old on Danbooru but simply absent from the training vocab
        # (e.g. a new character's copyright/general tags). API-free: the category
        # comes from the post's tag_string_<category> fields.
        self._cooc_enable      = bool(cooc_expand_enable)
        self._cooc_min_count   = max(1, int(cooc_min_count))
        self._cooc_categories  = set(cooc_categories if cooc_categories is not None else [0, 3, 4])
        self._cooc_lock        = threading.Lock()
        self._cooc_counts: Dict[str, int] = {}   # normalized unknown tag → co-occurrence count
        self._cooc_proposed: Set[str] = set()    # already handed to the expander
        # Recency-ordered names of promoted co-occurrence tags, for UI display.
        # Bounded; the set above remains the authoritative dedup membership test.
        self._cooc_promoted_order: collections.deque = collections.deque(maxlen=200)
        # Collection-path weights for weighted random query selection.  Paths
        # with no available queries are excluded and the remaining weights are
        # renormalized at selection time.
        self._weight_static    = max(0.0, float(weight_static))
        self._weight_new_tag   = max(0.0, float(weight_new_tag))
        self._weight_low_f1    = max(0.0, float(weight_low_f1))
        # Co-occurrence ACTIVE collection: once a tag is promoted by cooc, also
        # collect it directly (its own posts, order:random for diversity) up to a
        # balanced per-epoch quota so it gets trained across epochs — but only
        # lightly (low weight), since collecting a cooc copyright/general tag from
        # its own posts inevitably re-presents its companions; a small quota +
        # random sampling avoids over-reinforcing that co-occurrence.
        self._weight_cooc      = max(0.0, float(weight_cooc))
        self._cooc_collect_per_epoch = max(0, int(cooc_collect_per_epoch))
        self._cooc_order_random = bool(cooc_order_random)
        self._cooc_active_collect = self._weight_cooc > 0 and self._cooc_collect_per_epoch > 0
        # Denormalized cooc tags to actively collect (seeded from a persisted
        # snapshot on resume so collection continues across resumes).
        self._cooc_active_tags: List[str] = [t for t in (initial_cooc_active_tags or []) if t]
        self._cooc_active_seen: Set[str] = {normalize_tag(t) for t in self._cooc_active_tags}
        # Per-tag collected count this epoch (under _cycle_lock); reset each epoch.
        self._cooc_collected: Dict[str, int] = {}
        # A low-F1 tag is only collected when Danbooru can supply at least this
        # many posts for it (page-1 fetch count); otherwise it is marked
        # unavailable and skipped (genuinely rare tags we cannot augment).
        self._low_f1_min_posts = int(low_f1_min_posts)

        self._client  = DanbooruClient(api_interval=api_interval, dl_speed_kbps=dl_speed_kbps)
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Dynamic queries: Danbooru tag names (underscored) discovered by the
        # surveyor.  We accumulate these into a *persistent* list so we keep
        # downloading a new tag's posts even after it has been added to the
        # vocabulary (the surveyor drops it from its approved set on add, but
        # the freshly-grown head still needs positive samples).
        #
        # On resume the previously-expanded tags are already in the vocabulary,
        # so the surveyor would NOT re-discover them — they would silently stop
        # being collected. ``initial_dynamic_tags`` re-seeds the list from a
        # persisted snapshot so collection (and thus learning) of expanded tags
        # continues across resumes regardless of vocab membership.
        #
        # LRU bound: the list grows monotonically (surveyor + resume seeds), so
        # when max_dynamic_tags > 0 the least-recently-*collected* tag is evicted
        # once the cap is exceeded. ``_dynamic_last_used`` (tag → wall-clock) is
        # the recency key, persisted alongside the tags so the LRU order survives
        # resume. A regressed evicted tag is re-collected by the low-F1 path.
        self._max_dynamic_tags = int(max_dynamic_tags)
        self._dynamic_last_used: Dict[str, float] = {}
        _seed = initial_dynamic_tags or []
        if isinstance(_seed, dict):
            self._dynamic_tags: List[str] = [t for t in _seed.keys() if t]
            for t in self._dynamic_tags:
                try:
                    self._dynamic_last_used[t] = float(_seed[t])
                except (TypeError, ValueError):
                    self._dynamic_last_used[t] = time.time()
        else:  # list (fresh run or legacy snapshot format)
            self._dynamic_tags = [t for t in _seed if t]
            _now0 = time.time()
            for t in self._dynamic_tags:
                self._dynamic_last_used[t] = _now0
        self._dynamic_seen: Set[str] = {normalize_tag(t) for t in self._dynamic_tags}

        # Low-F1 queries: existing-vocab Danbooru tag names (underscored) fed by
        # the trainer's deficiency provider (worst per-tag F1).  Like dynamic
        # tags they are collected per-epoch; additionally each is gated by a
        # one-time Danbooru availability check (>= low_f1_min_posts) — tags that
        # are too rare on Danbooru to augment go into _low_f1_unavailable.
        self._low_f1_tags: List[str] = []
        self._low_f1_unavailable: Set[str] = set()   # below min_posts (persistent)

        # Query mode: per-tag collection pool for tags resolved from the user's
        # queries (name_matches). Symmetric to the surveyor's dynamic list but
        # discovered by query resolution rather than recency. When query_expand
        # is on, the Query path collects these PER-TAG (round-robin, bounded by
        # weight_static) so a wildcard resolving to N tags contributes N
        # collection units — not one. Persisted to danbooru_query_tags.json and
        # re-seeded on resume.
        self._query_expand = bool(query_expand)
        # Build the query resolver with the buffer's own rate-limited client so
        # resolution and post collection share a single Danbooru API budget.
        self._resolver = None
        if self._query_expand:
            try:
                from .danbooru_query_resolver import QueryResolver
                self._resolver = QueryResolver(
                    client=self._client,
                    min_count=int(query_min_count),
                    categories=(query_categories if query_categories is not None else [0, 3, 4]),
                    top_k=int(query_top_k),
                )
            except Exception as _qre:
                print(f"[DanbooruSampler] QueryResolver init failed: {_qre}")
                self._resolver = None
        self._query_max_expanded = int(query_max_expanded)
        self._query_resolve_interval = max(0.0, float(query_resolve_interval))
        self._last_query_resolve = 0.0   # 0 → resolve on first worker pass
        _qseed = initial_query_tags or []
        self._query_last_used: Dict[str, float] = {}
        if isinstance(_qseed, dict):
            self._query_tags: List[str] = [t for t in _qseed.keys() if t]
            for t in self._query_tags:
                try:
                    self._query_last_used[t] = float(_qseed[t])
                except (TypeError, ValueError):
                    self._query_last_used[t] = time.time()
        else:
            self._query_tags = [t for t in _qseed if t]
            _nowq = time.time()
            for t in self._query_tags:
                self._query_last_used[t] = _nowq
        self._query_seen: Set[str] = {normalize_tag(t) for t in self._query_tags}
        # Cumulative NEW tags added to the vocab via query resolution (run-wide cap).
        self._expanded_via_query: Set[str] = set()
        # Per-tag per-epoch collection caps (0 = unlimited) for the query /
        # new_tag / low_f1 paths. Bounds how many posts a single high-post_count
        # tag contributes per epoch so it does not monopolise the injected
        # batches. _collect_count is the shared per-tag counter (reset each epoch);
        # cooc keeps its own counter (_cooc_collected).
        self._query_collect_per_epoch = max(0, int(query_collect_per_epoch))
        self._new_tag_collect_per_epoch = max(0, int(new_tag_collect_per_epoch))
        self._low_f1_collect_per_epoch = max(0, int(low_f1_collect_per_epoch))
        self._collect_count: Dict[str, int] = {}

        # Per-epoch download cycle (memory-free dedup; only post_ids stored).
        # Within one epoch each dynamic tag is collected at most once, mirroring
        # how the base dataset reads each image exactly once per epoch.  A tag
        # whose posts are all collected is marked "exhausted" and skipped for the
        # rest of the epoch; reset_download_cycle() (called at each epoch start)
        # clears these so the next epoch collects them again.
        self._cycle_lock = threading.Lock()
        self._downloaded_ids: Set[int] = set()
        self._exhausted_tags: Set[str] = set()
        self._cycle_gen = 0   # bumped on each reset; guards cross-epoch exhaustion

        # Metrics (thread-safe via _metrics_lock)
        self._metrics_lock = threading.Lock()
        self._tag_freq: Dict[str, int] = {}
        # Per-targeted-new-tag collected sample count.  Keyed by the normalized
        # tag name (matching _tag_freq), counting how many posts were gathered
        # for each surveyor-approved new/deficient tag.  Lets the UI surface
        # *which* new tags augmentation is actively collecting, instead of the
        # ever-dominant common tags (1girl, solo, …) in _tag_freq.
        self._dynamic_tag_freq: Dict[str, int] = {}
        # Per-targeted-low-F1-tag collected sample count (same keying as above).
        self._low_f1_tag_freq: Dict[str, int] = {}
        # Per-cooc-tag actively-collected sample count.
        self._cooc_tag_freq: Dict[str, int] = {}
        # Query mode: per-resolved-tag collected count (expand mode, per-tag path)
        # and per-query-string collected count (legacy per-string static path).
        self._query_tag_freq: Dict[str, int] = {}
        self._static_query_freq: Dict[str, int] = {}
        self._recent_posts: collections.deque = collections.deque(maxlen=100)
        self._total_collected = 0
        self._total_injected_batches = 0
        self._buffer_starvation = 0
        self._total_dynamic_collected = 0
        self._total_low_f1_collected = 0
        self._total_cooc_proposed = 0   # unknown co-occurring tags handed to the expander
        self._total_cooc_collected = 0  # samples actively collected for cooc tags
        self._total_query_collected = 0   # per-tag collected for resolved query tags
        self._total_static_collected = 0  # per-string collected for legacy static queries

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="DanbooruSampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def reset_download_cycle(self) -> None:
        """Begin a new collection cycle (called at each epoch boundary).

        Clears the per-epoch downloaded-id and exhausted-tag sets so that every
        discovered new tag is collected once again in the new epoch — matching
        the base dataset, which re-reads each image once per epoch.
        """
        with self._cycle_lock:
            self._downloaded_ids.clear()
            self._exhausted_tags.clear()
            self._cooc_collected.clear()   # reset per-tag cooc quota for the new epoch
            self._collect_count.clear()    # reset per-tag query/new_tag/low_f1 quota
            self._cycle_gen += 1

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    def get_nowait(self) -> Optional[Tuple]:
        """Return a buffered sample (pv, pam, ss, raw_tags) or None."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def drain_batch(self, n: int) -> Optional[List[Tuple]]:
        """Return n samples if available, else None (no partial drain).

        Used by the interrupt-batch scheme: only yield a Danbooru batch when
        we can fill it completely.  If fewer than n samples are buffered, we
        leave them in the queue and skip this injection slot.
        """
        if self._queue.qsize() < n:
            with self._metrics_lock:
                self._buffer_starvation += 1
            return None
        items = []
        for _ in range(n):
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                # Race: someone else drained. Put back what we have.
                for it in items:
                    try:
                        self._queue.put_nowait(it)
                    except queue.Full:
                        pass
                with self._metrics_lock:
                    self._buffer_starvation += 1
                return None
        with self._metrics_lock:
            self._total_injected_batches += 1
        return items

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of collection metrics (thread-safe)."""
        # _dynamic_tags is mutated by the worker under _cycle_lock; read its
        # length under the same lock rather than racing the worker's append.
        with self._cycle_lock:
            dynamic_tags_count = len(self._dynamic_tags)
            low_f1_tags_count = len(self._low_f1_tags)
            low_f1_unavailable_count = len(self._low_f1_unavailable)
            cooc_active_count = len(self._cooc_active_tags)
            query_tags_count = len(self._query_tags)
            query_expanded_count = len(self._expanded_via_query)
        with self._cooc_lock:
            cooc_pending_count = len(self._cooc_counts)
            cooc_promoted_count = len(self._cooc_proposed)
            # Most-recently promoted first; bounded snapshot for the UI.
            cooc_proposed_tags = list(reversed(self._cooc_promoted_order))
        with self._metrics_lock:
            top_tags = sorted(self._tag_freq.items(), key=lambda x: -x[1])[:100]
            top_dynamic_tags = sorted(self._dynamic_tag_freq.items(), key=lambda x: -x[1])[:100]
            top_low_f1_tags = sorted(self._low_f1_tag_freq.items(), key=lambda x: -x[1])[:100]
            top_cooc_tags = sorted(self._cooc_tag_freq.items(), key=lambda x: -x[1])[:100]
            top_query_tags = sorted(self._query_tag_freq.items(), key=lambda x: -x[1])[:100]
            top_static_queries = sorted(self._static_query_freq.items(), key=lambda x: -x[1])[:100]
            return {
                "total_collected":         self._total_collected,
                "total_injected_batches":  self._total_injected_batches,
                "buffer_starvation_count": self._buffer_starvation,
                "buffer_capacity":         self._buffer_size,
                "buffer_current":          self._queue.qsize(),
                "unique_tags_seen":        len(self._tag_freq),
                "dynamic_tags_count":      dynamic_tags_count,
                "total_dynamic_collected": self._total_dynamic_collected,
                "dynamic_unique_tags_collected": len(self._dynamic_tag_freq),
                "low_f1_tags_count":       low_f1_tags_count,
                "low_f1_unavailable_count": low_f1_unavailable_count,
                "total_low_f1_collected":  self._total_low_f1_collected,
                "low_f1_unique_tags_collected": len(self._low_f1_tag_freq),
                "cooc_pending_count":      cooc_pending_count,
                "cooc_promoted_count":     cooc_promoted_count,
                "total_cooc_proposed":     self._total_cooc_proposed,
                "cooc_proposed_tags":      cooc_proposed_tags,
                "cooc_active_count":       cooc_active_count,
                "total_cooc_collected":    self._total_cooc_collected,
                "cooc_unique_tags_collected": len(self._cooc_tag_freq),
                "top_cooc_tags":           [{"tag": t, "count": c} for t, c in top_cooc_tags],
                "top_tags":                [{"tag": t, "count": c} for t, c in top_tags],
                "top_dynamic_tags":        [{"tag": t, "count": c} for t, c in top_dynamic_tags],
                "top_low_f1_tags":         [{"tag": t, "count": c} for t, c in top_low_f1_tags],
                # Query mode (per-tag, expand) + legacy per-string static counts.
                "query_tags_count":        query_tags_count,
                "query_expanded_count":    query_expanded_count,
                "total_query_collected":   self._total_query_collected,
                "query_unique_tags_collected": len(self._query_tag_freq),
                "top_query_tags":          [{"tag": t, "count": c} for t, c in top_query_tags],
                "total_static_collected":  self._total_static_collected,
                "top_static_queries":      [{"tag": t, "count": c} for t, c in top_static_queries],
                "recent_posts":            list(self._recent_posts),
            }

    def snapshot_dynamic_tags(self) -> Dict[str, float]:
        """Return ``{tag: last_used}`` for the dynamic (new-tag) query list, so it
        (and its LRU recency) can be re-seeded on resume (see
        ``initial_dynamic_tags``)."""
        with self._cycle_lock:
            return {t: self._dynamic_last_used.get(t, 0.0) for t in self._dynamic_tags}

    def snapshot_cooc_active_tags(self) -> List[str]:
        """Return the cooc active-collection query list, so it can be re-seeded
        on resume (see ``initial_cooc_active_tags``) and active collection of
        co-occurrence-promoted tags continues across resumes."""
        with self._cycle_lock:
            return list(self._cooc_active_tags)

    def snapshot_query_tags(self) -> Dict[str, float]:
        """Return ``{tag: last_used}`` for the resolved Query collection pool, so
        it (and its recency) can be re-seeded on resume (see
        ``initial_query_tags``). Lets per-tag collection of query-resolved tags
        continue across resumes without re-hitting the tags API."""
        with self._cycle_lock:
            return {t: self._query_last_used.get(t, 0.0) for t in self._query_tags}

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_query(q: str) -> str:
        """Translate convenience prefix '!tag' → Danbooru '-tag' (exclude)."""
        parts = []
        for tok in q.split():
            if tok.startswith("!"):
                parts.append("-" + tok[1:])
            else:
                parts.append(tok)
        return " ".join(parts)

    @staticmethod
    def _denormalize_tag(norm: str) -> str:
        """Convert a vocabulary-normalized tag back to a Danbooru query token.

        normalize_tag() lowercases and turns underscores into spaces; Danbooru
        tag search expects underscores. Parentheses stay literal (URL-encoded
        downstream by fetch_posts). This recovers the queryable name for the
        overwhelming majority of character/copyright tags.
        """
        return norm.strip().replace(" ", "_")

    def _refresh_dynamic_tags(self) -> None:
        """Pull newly-approved tags from the surveyor into the persistent
        dynamic-query list. Idempotent; safe to call every iteration."""
        if self._surveyor is None:
            return
        try:
            approved = self._surveyor.get_approved()
        except Exception:
            return
        _now = time.time()
        for norm in approved:
            if norm in self._dynamic_seen:
                continue
            self._dynamic_seen.add(norm)
            dq = self._denormalize_tag(norm)
            if dq:
                # _dynamic_tags is read by get_metrics()/_next_query under
                # _cycle_lock; append under the same lock to avoid a race.
                with self._cycle_lock:
                    self._dynamic_tags.append(dq)
                    self._dynamic_last_used[dq] = _now  # fresh tag — protected from eviction
                    self._evict_dynamic_lru_locked()

    def _evict_dynamic_lru_locked(self) -> None:
        """Evict least-recently-collected dynamic tags down to the cap.
        Caller must hold _cycle_lock. No-op when max_dynamic_tags <= 0."""
        cap = self._max_dynamic_tags
        if cap <= 0:
            return
        while len(self._dynamic_tags) > cap:
            # Least-recently-used = smallest last_used timestamp.
            victim = min(self._dynamic_tags, key=lambda t: self._dynamic_last_used.get(t, 0.0))
            try:
                self._dynamic_tags.remove(victim)
            except ValueError:
                break
            self._dynamic_last_used.pop(victim, None)
            self._dynamic_seen.discard(normalize_tag(victim))
            self._exhausted_tags.discard(victim)

    def _refresh_query_tags(self) -> None:
        """Resolve the user's queries to concrete tags (throttled) and add them
        to the per-tag Query collection pool; vocab-absent tags are also proposed
        to the expander. No-op unless query_expand is on with a resolver.

        Makes Danbooru tags-API calls (rate-limited by the client), gated to once
        per ``query_resolve_interval`` so it does not starve post collection.
        """
        if not self._query_expand or self._resolver is None or not self._tag_queries:
            return
        now = time.time()
        if self._last_query_resolve != 0.0 and \
                (now - self._last_query_resolve) < self._query_resolve_interval:
            return
        self._last_query_resolve = now

        known = set(self._vocabulary.tag_to_idx.keys())
        n_pool = 0
        new_for_vocab: Set[str] = set()
        for raw_query in self._tag_queries:
            if self._stop.is_set():
                return
            try:
                resolved = self._resolver.resolve_query(raw_query)
            except Exception as exc:
                print(f"[DanbooruSampler] query resolve error for {raw_query!r}: {exc}")
                continue
            for norm, _count, _cat in resolved:
                if norm in self._query_seen:
                    continue
                is_new = norm not in known
                # Run-wide cap applies ONLY to NEW vocab tags (existing tags added
                # for per-tag collection do not count against the expansion budget).
                if is_new and self._query_max_expanded > 0 and \
                        len(self._expanded_via_query) >= self._query_max_expanded:
                    continue
                dq = self._denormalize_tag(norm)
                if not dq:
                    continue
                self._query_seen.add(norm)
                with self._cycle_lock:
                    self._query_tags.append(dq)
                    self._query_last_used[dq] = now
                n_pool += 1
                if is_new:
                    new_for_vocab.add(norm)
                    self._expanded_via_query.add(norm)
        if new_for_vocab and self._expander is not None:
            self._expander.propose(new_for_vocab)
        if n_pool or new_for_vocab:
            print(f"[DanbooruSampler] Query resolution: +{n_pool} pool tag(s), "
                  f"{len(new_for_vocab)} new to vocab "
                  f"(query pool={len(self._query_tags)}, "
                  f"expanded-via-query={len(self._expanded_via_query)})")

    def _refresh_low_f1_tags(self) -> None:
        """Sync the low-F1 query list to the deficiency provider's current set.

        Unlike dynamic (new) tags — which persist because a freshly-grown head
        keeps needing samples — low-F1 targets are *rebuilt* each refresh so a
        tag whose F1 has recovered (and was dropped by the trainer) stops being
        collected. Availability (_low_f1_unavailable) persists across rebuilds
        since Danbooru post counts don't change during a run.
        """
        if self._deficiency_provider is None:
            return
        try:
            targets = self._deficiency_provider.get_targets()
        except Exception:
            return
        dq_list = [dq for dq in (self._denormalize_tag(n) for n in targets) if dq]
        with self._cycle_lock:
            self._low_f1_tags = dq_list

    def _next_query(
        self, static_idx: int, dyn_idx: int, lowf1_idx: int, cooc_idx: int, query_idx: int
    ) -> Optional[Tuple[str, str]]:
        """Choose the next query via weighted random path selection.

        Returns ``(query_string, kind)`` where ``kind`` is one of
        ``"static"`` / ``"query"`` / ``"new_tag"`` / ``"low_f1"`` / ``"cooc"``, or
        ``None`` when nothing is available this epoch (worker idles until
        reset_download_cycle()).

        The Query path uses weight ``_weight_static``. When ``query_expand`` is on
        it collects the resolved tags PER-TAG from ``_query_tags`` (kind="query",
        per-epoch deduped) so a wildcard that resolved to N tags contributes N
        collection units. When off it collects the raw queries PER-STRING
        (kind="static", unbounded), preserving the legacy behaviour and the
        full per-query metatag semantics.

        Each path contributes its weight only when it has at least one available
        item; weights are renormalized over the available paths.
        """
        with self._cycle_lock:
            active_dyn = [t for t in self._dynamic_tags if t not in self._exhausted_tags]
            active_lowf1 = [
                t for t in self._low_f1_tags
                if t not in self._exhausted_tags and t not in self._low_f1_unavailable
            ]
            active_cooc = [
                t for t in self._cooc_active_tags
                if t not in self._exhausted_tags
                and self._cooc_collected.get(t, 0) < self._cooc_collect_per_epoch
            ] if self._cooc_active_collect else []
            active_query = [
                t for t in self._query_tags if t not in self._exhausted_tags
            ] if self._query_expand else []

        paths: List[Tuple[str, float]] = []
        if self._weight_static > 0:
            if self._query_expand:
                if active_query:
                    paths.append(("query", self._weight_static))
            elif self._tag_queries:
                paths.append(("static", self._weight_static))
        if active_dyn and self._weight_new_tag > 0:
            paths.append(("new_tag", self._weight_new_tag))
        if active_lowf1 and self._weight_low_f1 > 0:
            paths.append(("low_f1", self._weight_low_f1))
        if active_cooc and self._weight_cooc > 0:
            paths.append(("cooc", self._weight_cooc))
        if not paths:
            return None

        total_w = sum(w for _, w in paths)
        r = random.random() * total_w
        acc = 0.0
        chosen = paths[-1][0]
        for name, w in paths:
            acc += w
            if r <= acc:
                chosen = name
                break

        if chosen == "static":
            raw = self._tag_queries[static_idx % len(self._tag_queries)]
            return self._translate_query(raw), "static"
        if chosen == "query":
            return active_query[query_idx % len(active_query)], "query"
        if chosen == "new_tag":
            return active_dyn[dyn_idx % len(active_dyn)], "new_tag"
        if chosen == "cooc":
            return active_cooc[cooc_idx % len(active_cooc)], "cooc"
        return active_lowf1[lowf1_idx % len(active_lowf1)], "low_f1"

    def _worker(self) -> None:
        static_idx = 0
        dyn_idx = 0
        lowf1_idx = 0
        cooc_idx = 0
        query_idx = 0
        while not self._stop.is_set():
            self._refresh_dynamic_tags()
            self._refresh_low_f1_tags()
            self._refresh_query_tags()

            choice = self._next_query(static_idx, dyn_idx, lowf1_idx, cooc_idx, query_idx)
            if choice is None:
                # Nothing to fetch: all feeds empty/exhausted for this epoch.
                # Idle until reset_download_cycle() or new targets arrive.
                time.sleep(2.0)
                continue

            query, kind = choice
            if kind == "new_tag":
                dyn_idx += 1
            elif kind == "low_f1":
                lowf1_idx += 1
            elif kind == "cooc":
                cooc_idx += 1
            elif kind == "query":
                query_idx += 1
            else:
                static_idx += 1

            # new_tag / low_f1 / cooc / query are per-epoch "collected" paths: dedup
            # post_ids and exhaust the tag once fully collected (cooc also exhausts
            # when it hits its per-epoch quota). static (per-string) is unbounded.
            is_collected = kind in ("new_tag", "low_f1", "cooc", "query")

            # Per-tag per-epoch collection cap (0 = unlimited). cooc uses its own
            # quota (_cooc_collected); the others share _collect_count.
            if kind == "query":
                _per_tag_cap = self._query_collect_per_epoch
            elif kind == "new_tag":
                _per_tag_cap = self._new_tag_collect_per_epoch
            elif kind == "low_f1":
                _per_tag_cap = self._low_f1_collect_per_epoch
            else:
                _per_tag_cap = 0

            # Snapshot the epoch generation: exhaustion decisions computed below
            # must only apply to THIS epoch. If reset_download_cycle() runs (epoch
            # boundary) before we mark a tag exhausted, the generation changes and
            # we discard the decision — otherwise a tag could be wrongly skipped
            # for the whole next epoch.
            with self._cycle_lock:
                start_gen = self._cycle_gen

            try:
                # Cooc active collection queries the tag's own posts with
                # order:random so each visit (and each epoch) samples a different,
                # diverse slice — not just the original co-occurrence context —
                # which weakens spurious companion co-occurrence.
                _fetch_query = query
                if kind == "cooc" and self._cooc_order_random:
                    _fetch_query = f"{query} order:random"
                posts = self._client.fetch_posts(_fetch_query, page=1, min_score=self._min_score)

                # Low-F1 availability gate: only augment tags Danbooru can supply
                # >= low_f1_min_posts for. Reuses the page-1 fetch (no extra API
                # call); note the count is post-score-filter (min_score), so a
                # high min_score raises the effective availability bar. Only a
                # non-empty-but-short result is a reliable "too rare" signal — an
                # empty result may be a transient error/rate limit, so it is left
                # to the per-epoch exhaustion path below. Once blacklisted a tag
                # is excluded from future selection.
                if kind == "low_f1" and 0 < len(posts) < self._low_f1_min_posts:
                    with self._cycle_lock:
                        self._low_f1_unavailable.add(query)
                    time.sleep(0.3)
                    continue

                if not posts:
                    if is_collected:
                        # No posts (or transient error) — don't hammer this tag
                        # again this epoch.
                        with self._cycle_lock:
                            if self._cycle_gen == start_gen:
                                self._exhausted_tags.add(query)
                    time.sleep(2.0)
                    continue

                if self._max_posts < len(posts):
                    posts = posts[:self._max_posts]
                random.shuffle(posts)

                dedup_skipped = 0
                for post in posts:
                    if self._stop.is_set():
                        return
                    # Per-tag per-epoch cap: stop this visit once the tag has hit
                    # its quota and exhaust it for the rest of the epoch (next epoch
                    # re-collects from scratch after reset_download_cycle()).
                    if _per_tag_cap > 0:
                        with self._cycle_lock:
                            if self._collect_count.get(query, 0) >= _per_tag_cap:
                                if self._cycle_gen == start_gen:
                                    self._exhausted_tags.add(query)
                                break
                    pid = int(post.get("id", 0) or 0)
                    if is_collected:
                        with self._cycle_lock:
                            if pid in self._downloaded_ids:
                                dedup_skipped += 1
                                continue  # already collected this epoch
                    sample = self._process_post(post)
                    if sample is None:
                        continue
                    try:
                        self._queue.put(sample, timeout=10.0)
                    except queue.Full:
                        continue  # backpressure — drop, retry on next visit
                    if is_collected:
                        # Mark collected only after a successful enqueue, so a
                        # dropped sample can be retried before the tag exhausts.
                        with self._cycle_lock:
                            self._downloaded_ids.add(pid)
                            if kind == "new_tag":
                                # Refresh LRU recency: this tag is still productive.
                                self._dynamic_last_used[query] = time.time()
                            elif kind == "query":
                                # Refresh recency so the resume snapshot reflects use.
                                self._query_last_used[query] = time.time()
                            elif kind == "cooc":
                                # Per-epoch quota: once reached, exhaust the tag
                                # for the rest of this epoch (balanced collection).
                                _cc = self._cooc_collected.get(query, 0) + 1
                                self._cooc_collected[query] = _cc
                                if self._cooc_collect_per_epoch > 0 and _cc >= self._cooc_collect_per_epoch:
                                    self._exhausted_tags.add(query)
                            # Per-tag per-epoch cap for query / new_tag / low_f1.
                            if _per_tag_cap > 0 and kind in ("new_tag", "low_f1", "query"):
                                _pc = self._collect_count.get(query, 0) + 1
                                self._collect_count[query] = _pc
                                if _pc >= _per_tag_cap:
                                    self._exhausted_tags.add(query)
                        _nt = normalize_tag(query) if query else ""
                        with self._metrics_lock:
                            if kind == "new_tag":
                                self._total_dynamic_collected += 1
                                if _nt:
                                    self._dynamic_tag_freq[_nt] = self._dynamic_tag_freq.get(_nt, 0) + 1
                            elif kind == "query":
                                self._total_query_collected += 1
                                if _nt:
                                    self._query_tag_freq[_nt] = self._query_tag_freq.get(_nt, 0) + 1
                            elif kind == "cooc":
                                self._total_cooc_collected += 1
                                if _nt:
                                    self._cooc_tag_freq[_nt] = self._cooc_tag_freq.get(_nt, 0) + 1
                            else:  # low_f1
                                self._total_low_f1_collected += 1
                                if _nt:
                                    self._low_f1_tag_freq[_nt] = self._low_f1_tag_freq.get(_nt, 0) + 1
                    elif kind == "static":
                        # Legacy per-query-string collection (query_expand off):
                        # count posts per query string for the UI "Queries" view.
                        with self._metrics_lock:
                            self._total_static_collected += 1
                            self._static_query_freq[query] = self._static_query_freq.get(query, 0) + 1

                # A collected tag is exhausted for this epoch once every post it
                # returned was already collected (pure dedup pass). Decode errors
                # and backpressure drops do NOT exhaust it — those posts are
                # retried on a later visit until genuinely collected. Guard on the
                # epoch generation so a reset mid-fetch doesn't carry the decision
                # into the next epoch.
                if is_collected and len(posts) > 0 and dedup_skipped == len(posts):
                    with self._cycle_lock:
                        if self._cycle_gen == start_gen:
                            self._exhausted_tags.add(query)

            except Exception as exc:
                print(f"[DanbooruSampler] Worker error: {exc}")
                time.sleep(5.0)

    def _update_cooc(self, post: dict, known: Set[str]) -> None:
        """Count unknown tags (category taken from the post's tag_string_<cat>
        fields) and, once a tag has co-occurred >= cooc_min_count times across
        collected posts, hand it to the vocab expander. Catches vocab-absent
        tags regardless of their Danbooru creation date (unlike the surveyor).
        """
        to_propose: Set[str] = set()
        with self._cooc_lock:
            for cat_code, field in _COOC_CATEGORY_FIELDS:
                if cat_code not in self._cooc_categories:
                    continue
                raw = post.get(field) or ""
                for tok in raw.split():
                    norm = normalize_tag(tok)
                    if not norm or norm in known or norm in self._cooc_proposed:
                        continue
                    c = self._cooc_counts.get(norm, 0) + 1
                    if c >= self._cooc_min_count:
                        self._cooc_proposed.add(norm)
                        self._cooc_promoted_order.append(norm)
                        self._cooc_counts.pop(norm, None)  # promoted — free memory
                        to_propose.add(norm)
                    else:
                        self._cooc_counts[norm] = c
        if to_propose:
            self._expander.propose(to_propose)
            with self._metrics_lock:
                self._total_cooc_proposed += len(to_propose)
            # Register promoted cooc tags for ACTIVE collection (so they get
            # trained across epochs, not just from incidental co-occurrence).
            if self._cooc_active_collect:
                with self._cycle_lock:
                    for _norm in to_propose:
                        if _norm in self._cooc_active_seen:
                            continue
                        _dq = self._denormalize_tag(_norm)
                        if _dq:
                            self._cooc_active_tags.append(_dq)
                            self._cooc_active_seen.add(_norm)

    def _process_post(self, post: dict) -> Optional[Tuple]:
        result = self._client.download_inmemory(post)
        if result is None:
            return None
        img_bytes, _ext, raw_tags = result

        # Decode image
        try:
            img = Image.open(BytesIO(img_bytes))
            img.load()
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
        except (OSError, Image.DecompressionBombError) as exc:
            print(f"[DanbooruSampler] Image decode error (post {post.get('id')}): {exc}")
            return None

        # Run vision processor
        try:
            inputs = self._processor(images=[img], return_tensors="pt")
        except Exception as exc:
            print(f"[DanbooruSampler] Processor error (post {post.get('id')}): {exc}")
            return None

        if self._is_naflex:
            pixel_values         = inputs["pixel_values"].squeeze(0)
            pixel_attention_mask = inputs["pixel_attention_mask"].squeeze(0)
            spatial_shapes       = inputs["spatial_shapes"].squeeze(0)
        else:
            pixel_values         = inputs["pixel_values"].squeeze(0)
            pixel_attention_mask = torch.zeros(0, dtype=torch.int32)
            spatial_shapes       = torch.zeros(0, dtype=torch.int64)

        # Propose unknown approved tags to vocab expander (if configured)
        if self._expander is not None:
            known = set(self._vocabulary.tag_to_idx.keys())
            # 1) Surveyor-approved (recently created) tags appearing in this post.
            if self._surveyor is not None:
                approved = self._surveyor.get_approved()
                new_approved = ({normalize_tag(t) for t in raw_tags if t} & approved) - known
                if new_approved:
                    self._expander.propose(new_approved)
            # 2) Co-occurrence discovery (created-at independent; category from
            #    the post's tag_string_<category> fields).
            if self._cooc_enable:
                self._update_cooc(post, known)

        # Record metrics
        with self._metrics_lock:
            self._total_collected += 1
            for t in raw_tags:
                nt = normalize_tag(t) if t else ""
                if nt:
                    self._tag_freq[nt] = self._tag_freq.get(nt, 0) + 1
            preview_tags = [normalize_tag(t) for t in raw_tags[:20] if t]
            self._recent_posts.append({
                "post_id":    int(post.get("id", 0) or 0),
                "tags":       preview_tags,
                "tag_count":  len([t for t in raw_tags if t]),
                "timestamp":  time.time(),
            })

        return (pixel_values, pixel_attention_mask, spatial_shapes, raw_tags)


# ---------------------------------------------------------------------------
# MixedDataLoader
# ---------------------------------------------------------------------------

class MixedDataLoader:
    """DataLoader wrapper that interleaves pure-Danbooru batches.

    The base loader's batches are passed through untouched (only label-padded
    after vocabulary expansion).  Every ``injection_interval`` base batches,
    we attempt to drain a full Danbooru batch from the buffer and yield it
    as an "injection batch".

    Yielded items are 2-tuples: ``(batch, is_injection: bool)``.

    The training loop must:
      - Always do ``optimizer.step()``.
      - Skip ``scheduler.step()`` and ``global_step += 1`` when is_injection=True.
      - This way, LR phase remains aligned with the base loader's progress
        and resume reproducibility is preserved.

    If the buffer cannot supply ``injection_batch_size`` samples, the
    injection slot is skipped silently (training continues on base batches).

    Delegates ``.dataset``, ``.num_workers``, ``.batch_size`` to the base
    loader so the trainer's resume-loader construction works correctly.
    ``__len__`` returns the base loader's length (injection batches are
    "bonus" updates and don't count toward epoch progress).
    """

    def __init__(
        self,
        base_loader: DataLoader,
        buffer: DanbooruSampleBuffer,
        injection_interval: int = 4,
        injection_batch_size: Optional[int] = None,
        expander: Any = None,
        expansion_callback: Optional[Callable[[List[str]], None]] = None,
        vocabulary: Optional[TagVocabulary] = None,
        quality_masking_mode: str = "intra_group",
        alias_resolver: Any = None,
    ) -> None:
        self.base_loader          = base_loader
        self._buffer              = buffer
        self._injection_interval  = max(1, int(injection_interval))
        # Fall back to base_loader.batch_size when not specified
        if injection_batch_size is None or injection_batch_size <= 0:
            self._injection_batch_size = int(base_loader.batch_size or 1)
        else:
            self._injection_batch_size = int(injection_batch_size)
        self._expander            = expander
        self._expansion_callback  = expansion_callback
        self._vocabulary          = vocabulary
        self._quality_masking     = quality_masking_mode
        self._alias_resolver      = alias_resolver

    # Proxy attributes used by the trainer
    @property
    def dataset(self):
        return self.base_loader.dataset

    @property
    def num_workers(self):
        return self.base_loader.num_workers

    @property
    def batch_size(self):
        return self.base_loader.batch_size

    def __len__(self) -> int:
        return len(self.base_loader)

    def rewrap(self, new_base_loader: DataLoader) -> "MixedDataLoader":
        """Return a new MixedDataLoader over *new_base_loader*, sharing this
        instance's Danbooru buffer, expander, vocabulary and injection settings.

        Used on mid-epoch resume: the trainer rebuilds a plain base DataLoader
        that skips the already-processed batches, then re-wraps it here so the
        interrupt-batch injection continues for the resumed epoch.  Without this
        the resumed epoch would run on the bare base loader and Danbooru
        injection would silently stop until the next epoch boundary.
        """
        return MixedDataLoader(
            new_base_loader,
            buffer=self._buffer,
            injection_interval=self._injection_interval,
            injection_batch_size=self._injection_batch_size,
            expander=self._expander,
            expansion_callback=self._expansion_callback,
            vocabulary=self._vocabulary,
            quality_masking_mode=self._quality_masking,
            alias_resolver=self._alias_resolver,
        )

    def _build_injection_batch(self) -> Optional[Tuple]:
        """Drain a full Danbooru batch, build labels, and collate. None if
        buffer is insufficient or all samples failed to collate.
        """
        items = self._buffer.drain_batch(self._injection_batch_size)
        if items is None:
            return None
        voc = self._vocabulary if self._vocabulary is not None else self._buffer._vocabulary
        built = []
        for (pv_d, pam_d, ss_d, raw_tags) in items:
            lbl, lm = _build_label_and_mask_standalone(
                raw_tags, voc,
                quality_masking_mode=self._quality_masking,
                alias_resolver=self._alias_resolver,
            )
            built.append((pv_d, pam_d, ss_d, lbl, lm))
        return tagger_collate_fn(built)

    def __iter__(self):
        # New epoch: let the buffer collect each new tag once again this epoch
        # (mirrors the base dataset re-reading every image per epoch).
        if hasattr(self._buffer, "reset_download_cycle"):
            self._buffer.reset_download_cycle()
        base_step = 0
        for batch in self.base_loader:
            if batch is None:
                yield (None, False)
                continue

            # ① Vocabulary expansion check (training thread)
            if self._expander is not None and self._expander.has_pending():
                new_tags = self._expander.consume_pending()
                if self._expansion_callback is not None:
                    self._expansion_callback(new_tags)

            pv, pam, ss, labels, loss_masks = batch

            # ② Pad base-loader batch labels to current vocabulary size
            if self._vocabulary is not None:
                current_n = self._vocabulary.num_tags
                if labels.shape[1] < current_n:
                    pad = current_n - labels.shape[1]
                    labels     = F.pad(labels,     (0, pad), value=0.0)
                    loss_masks = F.pad(loss_masks, (0, pad), value=1.0)

            yield ((pv, pam, ss, labels, loss_masks), False)
            base_step += 1

            # ③ Interrupt batch: every injection_interval base batches,
            #    attempt to yield a pure-Danbooru batch.
            if base_step % self._injection_interval == 0:
                inj = self._build_injection_batch()
                if inj is not None:
                    yield (inj, True)
