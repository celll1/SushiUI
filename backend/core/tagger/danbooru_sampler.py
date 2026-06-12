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
        # Collection-path weights for weighted random query selection.  Paths
        # with no available queries are excluded and the remaining weights are
        # renormalized at selection time.
        self._weight_static    = max(0.0, float(weight_static))
        self._weight_new_tag   = max(0.0, float(weight_new_tag))
        self._weight_low_f1    = max(0.0, float(weight_low_f1))
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
        self._dynamic_tags: List[str] = []
        self._dynamic_seen: Set[str] = set()

        # Low-F1 queries: existing-vocab Danbooru tag names (underscored) fed by
        # the trainer's deficiency provider (worst per-tag F1).  Like dynamic
        # tags they are collected per-epoch; additionally each is gated by a
        # one-time Danbooru availability check (>= low_f1_min_posts) — tags that
        # are too rare on Danbooru to augment go into _low_f1_unavailable.
        self._low_f1_tags: List[str] = []
        self._low_f1_unavailable: Set[str] = set()   # below min_posts (persistent)

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
        self._recent_posts: collections.deque = collections.deque(maxlen=100)
        self._total_collected = 0
        self._total_injected_batches = 0
        self._buffer_starvation = 0
        self._total_dynamic_collected = 0
        self._total_low_f1_collected = 0

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
        with self._metrics_lock:
            top_tags = sorted(self._tag_freq.items(), key=lambda x: -x[1])[:100]
            top_dynamic_tags = sorted(self._dynamic_tag_freq.items(), key=lambda x: -x[1])[:100]
            top_low_f1_tags = sorted(self._low_f1_tag_freq.items(), key=lambda x: -x[1])[:100]
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
                "top_tags":                [{"tag": t, "count": c} for t, c in top_tags],
                "top_dynamic_tags":        [{"tag": t, "count": c} for t, c in top_dynamic_tags],
                "top_low_f1_tags":         [{"tag": t, "count": c} for t, c in top_low_f1_tags],
                "recent_posts":            list(self._recent_posts),
            }

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
        self, static_idx: int, dyn_idx: int, lowf1_idx: int
    ) -> Optional[Tuple[str, str]]:
        """Choose the next query via weighted random path selection.

        Returns ``(query_string, kind)`` where ``kind`` is one of
        ``"static"`` / ``"new_tag"`` / ``"low_f1"``, or ``None`` when nothing is
        available this epoch (worker idles until reset_download_cycle()).

        Each path contributes its configured weight only when it has at least
        one available query; weights are renormalized over the available paths.
        Dynamic/low-F1 tags already exhausted this epoch (and, for low-F1, tags
        below the Danbooru availability threshold) are excluded.
        """
        with self._cycle_lock:
            active_dyn = [t for t in self._dynamic_tags if t not in self._exhausted_tags]
            active_lowf1 = [
                t for t in self._low_f1_tags
                if t not in self._exhausted_tags and t not in self._low_f1_unavailable
            ]

        paths: List[Tuple[str, float]] = []
        if self._tag_queries and self._weight_static > 0:
            paths.append(("static", self._weight_static))
        if active_dyn and self._weight_new_tag > 0:
            paths.append(("new_tag", self._weight_new_tag))
        if active_lowf1 and self._weight_low_f1 > 0:
            paths.append(("low_f1", self._weight_low_f1))
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
        if chosen == "new_tag":
            return active_dyn[dyn_idx % len(active_dyn)], "new_tag"
        return active_lowf1[lowf1_idx % len(active_lowf1)], "low_f1"

    def _worker(self) -> None:
        static_idx = 0
        dyn_idx = 0
        lowf1_idx = 0
        while not self._stop.is_set():
            self._refresh_dynamic_tags()
            self._refresh_low_f1_tags()

            choice = self._next_query(static_idx, dyn_idx, lowf1_idx)
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
            else:
                static_idx += 1

            # new_tag and low_f1 are per-epoch "collected" paths: dedup post_ids
            # and exhaust the tag once fully collected. static is unbounded.
            is_collected = kind in ("new_tag", "low_f1")

            # Snapshot the epoch generation: exhaustion decisions computed below
            # must only apply to THIS epoch. If reset_download_cycle() runs (epoch
            # boundary) before we mark a tag exhausted, the generation changes and
            # we discard the decision — otherwise a tag could be wrongly skipped
            # for the whole next epoch.
            with self._cycle_lock:
                start_gen = self._cycle_gen

            try:
                posts = self._client.fetch_posts(query, page=1, min_score=self._min_score)

                # Low-F1 availability gate: only augment tags Danbooru can supply
                # >= low_f1_min_posts for. Reuses the page-1 fetch (no extra API
                # call). Only a non-empty-but-short result is a reliable "too
                # rare" signal — an empty result may be a transient error/rate
                # limit, so it is left to the per-epoch exhaustion path below.
                # Once blacklisted a tag is excluded from future selection.
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
                        _nt = normalize_tag(query) if query else ""
                        with self._metrics_lock:
                            if kind == "new_tag":
                                self._total_dynamic_collected += 1
                                if _nt:
                                    self._dynamic_tag_freq[_nt] = self._dynamic_tag_freq.get(_nt, 0) + 1
                            else:  # low_f1
                                self._total_low_f1_collected += 1
                                if _nt:
                                    self._low_f1_tag_freq[_nt] = self._low_f1_tag_freq.get(_nt, 0) + 1

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
        if self._expander is not None and self._surveyor is not None:
            voc = self._vocabulary
            approved = self._surveyor.get_approved()
            normalized = {normalize_tag(t) for t in raw_tags if t}
            new_approved = (normalized & approved) - set(voc.tag_to_idx.keys())
            if new_approved:
                self._expander.propose(new_approved)

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
