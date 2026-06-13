"""
Online Danbooru augmentation for IMAGE-GENERATION training (LoRA / Full FT).

This is the diffusion-side counterpart of the tagger's Danbooru augmentation.
Unlike the tagger — which has a fixed classification head and therefore grows
its vocabulary — image-generation conditioning accepts arbitrary text, so there
is NO vocabulary expansion here.  The mechanism reduces to: fetch extra training
images from Danbooru (by user-supplied static queries and/or auto-detected
under-represented tags) and feed them into training as ordinary samples.

Design (efficiency):
  - A background thread does only CPU/IO work: query selection, API fetch,
    in-memory image download, decode, and caption construction.  It never
    touches the GPU (the VAE / text-encoder are owned by the training thread
    and, in swap modes, are off-GPU during training).
  - Collected items are kept fully IN MEMORY as COMPRESSED bytes (no disk
    writes, no full-resolution decode held in the buffer).  The buffer is
    bounded; the training loop decodes one image at a time at encode time and
    frees the bytes immediately after, so CPU RAM stays bounded to roughly
    buffer_size compressed images (a few MB each).  VRAM is unaffected.
  - The training loop splices full same-bucket Danbooru batches into the
    epoch's batch list every N base batches (interrupt-batch injection).  Their
    latents/embeddings are encoded by the existing swap-buffer refill cycle
    (when VAE/TE are already resident on GPU), so no per-step encoder swap is
    incurred.

The two query paths:
  - static     : user-supplied Danbooru tag queries (newline separated).
  - deficiency : tags that are under-represented in the training dataset,
                 auto-detected from the dataset caption tag-frequency histogram
                 (and/or a user-supplied explicit list).  No per-tag F1 (there
                 is no such metric in diffusion training) — purely data-balance.
"""

from __future__ import annotations

import io
import math
import queue
import random
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from core.tagger.danbooru_client import DanbooruClient, _RATING_MAP


# Danbooru tag category codes
_CAT_GENERAL = 0
_CAT_ARTIST = 1
_CAT_COPYRIGHT = 3
_CAT_CHARACTER = 4
_CAT_META = 5

# Per-category source field on a post dict, in the order they are emitted into
# the constructed caption (general → character → copyright → artist → meta).
_CAPTION_CATEGORY_FIELDS: Tuple[Tuple[int, str], ...] = (
    (_CAT_GENERAL, "tag_string_general"),
    (_CAT_CHARACTER, "tag_string_character"),
    (_CAT_COPYRIGHT, "tag_string_copyright"),
    (_CAT_ARTIST, "tag_string_artist"),
    (_CAT_META, "tag_string_meta"),
)

# Danbooru category code → dataset-convention category NAME (matches the names
# used in tag_data / caption_config so per-category shuffle/dropout lines up).
_CATEGORY_CODE_TO_NAME: Dict[int, str] = {
    _CAT_GENERAL: "General",
    _CAT_ARTIST: "Artist",
    _CAT_COPYRIGHT: "Copyright",
    _CAT_CHARACTER: "Character",
    _CAT_META: "Meta",
}

_ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _normalize_for_count(tag: str) -> str:
    """Lowercase + underscores→spaces, mirroring the tagger's normalize_tag so
    dataset-caption tags and Danbooru tags compare consistently."""
    return tag.strip().lower().replace("_", " ")


# ----------------------------------------------------------------------------
# Auto deficiency: dataset tag-frequency analysis
# ----------------------------------------------------------------------------

class DatasetTagFrequencyAnalyzer:
    """Build a tag-frequency histogram from the training dataset captions and
    surface under-represented tags as Danbooru collection queries.

    Deficiency for image-gen is defined by DATASET FREQUENCY (how many training
    images carry the tag), not by any model-quality metric.  Tags appearing in
    fewer than ``min_count`` images — capped to the ``top_k`` rarest — are
    returned as queries so augmentation can rebalance the data.
    """

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}

    def add_caption_tags(self, tags: Sequence[str]) -> None:
        """Accumulate one image's tags (deduplicated within the image)."""
        seen = set()
        for t in tags:
            n = _normalize_for_count(t)
            if not n or n in seen:
                continue
            seen.add(n)
            self._counts[n] = self._counts.get(n, 0) + 1

    @property
    def total_unique_tags(self) -> int:
        return len(self._counts)

    def deficient_queries(
        self,
        min_count: int,
        top_k: int,
        exclude_substrings: Sequence[str] = ("rating:", "score:"),
    ) -> List[str]:
        """Return Danbooru queries (underscored tag names) for tags whose image
        count is below ``min_count``, the rarest ``top_k`` first.

        Tags are emitted in ascending count order so the most-deficient tags are
        prioritised by the caller's pagination budget.
        """
        cand = [
            (tag, c)
            for tag, c in self._counts.items()
            if c < min_count
            and not any(sub in tag for sub in exclude_substrings)
        ]
        cand.sort(key=lambda x: x[1])  # rarest first
        if top_k > 0:
            cand = cand[:top_k]
        # Danbooru queries use underscores for spaces
        return [tag.replace(" ", "_") for tag, _c in cand]


# ----------------------------------------------------------------------------
# Background collector
# ----------------------------------------------------------------------------

class _ReadyItem:
    """One collected image ready for injection.  Holds the COMPRESSED image
    bytes (not a decoded PIL) so the buffer stays bounded to a few MB/image;
    the training loop decodes lazily, one at a time, at encode time.

    ``tag_data`` is the per-category tag list (``[{"tag","category"}]``); the
    final caption string is built per-epoch in the training loop so the same
    shuffle / dropout the dataset uses can be applied.  ``tags`` is the flat
    name list kept only for metrics display."""

    __slots__ = ("post_id", "image_bytes", "tag_data", "bucket_w", "bucket_h", "tags")

    def __init__(self, post_id: int, image_bytes: bytes, tag_data: List[Dict[str, str]],
                 bucket_w: int, bucket_h: int, tags: List[str]) -> None:
        self.post_id = post_id
        self.image_bytes = image_bytes
        self.tag_data = tag_data
        self.bucket_w = bucket_w
        self.bucket_h = bucket_h
        self.tags = tags


class DanbooruImageCollector:
    """Background-threaded Danbooru image collector for diffusion training.

    Produces decoded, bucket-assigned PIL images grouped by bucket so the
    training loop can drain full same-bucket batches (collate-compatible).
    """

    def __init__(
        self,
        *,
        static_queries: List[str],
        deficiency_queries: Optional[List[str]] = None,
        bucket_resolutions: Sequence[Tuple[int, int]],
        weight_static: float = 1.0,
        weight_deficiency: float = 1.0,
        min_score: int = 0,
        max_posts_per_query: int = 200,
        api_interval: float = 1.4,
        dl_speed_kbps: int = 500,
        buffer_size: int = 64,
        include_rating_tag: bool = False,
        max_caption_tags: int = 0,
    ) -> None:
        self._static_queries = [q.strip() for q in static_queries if q.strip()]
        self._deficiency_queries = [q.strip() for q in (deficiency_queries or []) if q.strip()]
        self._bucket_resolutions = list(bucket_resolutions) or [(1024, 1024)]
        self._weight_static = max(0.0, float(weight_static))
        self._weight_deficiency = max(0.0, float(weight_deficiency))
        self._min_score = int(min_score)
        self._max_posts_per_query = max(1, int(max_posts_per_query))
        self._buffer_size = max(1, int(buffer_size))
        self._include_rating_tag = bool(include_rating_tag)
        self._max_caption_tags = max(0, int(max_caption_tags))

        self._client = DanbooruClient(api_interval=api_interval, dl_speed_kbps=dl_speed_kbps)

        # Ready items grouped by bucket key "WxH" → deque of _ReadyItem.
        self._lock = threading.Lock()
        self._buckets: Dict[str, deque] = {}
        self._ready_count = 0

        # Per-query pagination state and per-epoch dedup.
        self._query_pages: Dict[str, int] = {}
        self._cycle_lock = threading.Lock()
        self._downloaded_ids: set = set()
        self._exhausted_queries: set = set()

        # Metrics
        self._metrics_lock = threading.Lock()
        self._total_collected = 0
        self._total_injected_batches = 0
        self._buffer_starvation = 0
        self._tag_freq: Dict[str, int] = {}
        self._static_collected = 0
        self._deficiency_collected = 0
        self._recent_posts: deque = deque(maxlen=100)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="DanbooruImageCollector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def reset_download_cycle(self) -> None:
        """Begin a new collection cycle (called at each epoch boundary) so every
        query may collect again this epoch (mirrors the base dataset re-reading
        each image once per epoch)."""
        with self._cycle_lock:
            self._downloaded_ids.clear()
            self._exhausted_queries.clear()
            self._query_pages.clear()

    def update_deficiency_queries(self, queries: List[str]) -> None:
        """Replace the auto-detected deficiency query list (thread-safe)."""
        with self._cycle_lock:
            self._deficiency_queries = [q.strip() for q in queries if q.strip()]

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def has_full_batch(self, n: int) -> bool:
        with self._lock:
            return any(len(dq) >= n for dq in self._buckets.values())

    def drain_batch(self, n: int) -> Optional[List[_ReadyItem]]:
        """Return ``n`` ready items from a SINGLE bucket (all-or-nothing), or
        None if no bucket currently holds ``n`` items.  Picking from one bucket
        guarantees the injected batch is collate-compatible (uniform latent
        shape)."""
        with self._lock:
            # Prefer the fullest bucket to keep memory turning over.
            best_key = None
            best_len = 0
            for key, dq in self._buckets.items():
                if len(dq) >= n and len(dq) > best_len:
                    best_key = key
                    best_len = len(dq)
            if best_key is None:
                with self._metrics_lock:
                    self._buffer_starvation += 1
                return None
            dq = self._buckets[best_key]
            items = [dq.popleft() for _ in range(n)]
            self._ready_count -= n
        with self._metrics_lock:
            self._total_injected_batches += 1
        return items

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            buffer_current = self._ready_count
            bucket_dist = {k: len(v) for k, v in self._buckets.items()}
        with self._metrics_lock:
            top_tags = sorted(self._tag_freq.items(), key=lambda x: -x[1])[:100]
            return {
                "enabled": True,
                "total_collected": self._total_collected,
                "total_injected_batches": self._total_injected_batches,
                "buffer_starvation_count": self._buffer_starvation,
                "buffer_capacity": self._buffer_size,
                "buffer_current": buffer_current,
                "unique_tags_seen": len(self._tag_freq),
                "static_collected": self._static_collected,
                "deficiency_collected": self._deficiency_collected,
                "deficiency_query_count": len(self._deficiency_queries),
                "bucket_distribution": bucket_dist,
                "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
                "recent_posts": list(self._recent_posts),
            }

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _pick_path(self) -> Optional[str]:
        """Weighted random choice of collection path among those with at least
        one non-exhausted query.  Returns 'static' / 'deficiency' / None."""
        with self._cycle_lock:
            static_avail = any(q not in self._exhausted_queries for q in self._static_queries)
            defic_avail = any(q not in self._exhausted_queries for q in self._deficiency_queries)
        paths: List[Tuple[str, float]] = []
        if static_avail and self._weight_static > 0:
            paths.append(("static", self._weight_static))
        if defic_avail and self._weight_deficiency > 0:
            paths.append(("deficiency", self._weight_deficiency))
        if not paths:
            return None
        total = sum(w for _, w in paths)
        r = random.random() * total
        upto = 0.0
        for name, w in paths:
            upto += w
            if r <= upto:
                return name
        return paths[-1][0]

    def _pick_query(self, path: str) -> Optional[str]:
        with self._cycle_lock:
            pool = self._static_queries if path == "static" else list(self._deficiency_queries)
            avail = [q for q in pool if q not in self._exhausted_queries]
        if not avail:
            return None
        return random.choice(avail)

    def _worker(self) -> None:
        while not self._stop.is_set():
            # Back off when the buffer is full — keep memory bounded.
            with self._lock:
                full = self._ready_count >= self._buffer_size
            if full:
                if self._stop.wait(0.5):
                    break
                continue

            path = self._pick_path()
            if path is None:
                # Nothing left to collect this cycle; idle until reset.
                if self._stop.wait(1.0):
                    break
                continue

            query = self._pick_query(path)
            if query is None:
                continue

            try:
                self._collect_one_page(path, query)
            except Exception as exc:  # noqa: BLE001 — never let the worker die
                print(f"[DanbooruImageCollector] collect error for {query!r}: {exc}")
                if self._stop.wait(1.0):
                    break

    def _collect_one_page(self, path: str, query: str) -> None:
        with self._cycle_lock:
            page = self._query_pages.get(query, 1)
        posts = self._client.fetch_posts(query, page=page, min_score=self._min_score)
        with self._cycle_lock:
            self._query_pages[query] = page + 1
            if not posts or page * 200 >= self._max_posts_per_query:
                self._exhausted_queries.add(query)
        if not posts:
            return

        for post in posts:
            if self._stop.is_set():
                return
            with self._lock:
                if self._ready_count >= self._buffer_size:
                    return
            pid = post.get("id")
            if pid is None:
                continue
            with self._cycle_lock:
                if pid in self._downloaded_ids:
                    continue
                self._downloaded_ids.add(pid)

            item = self._process_post(post)
            if item is None:
                continue

            key = f"{item.bucket_w}x{item.bucket_h}"
            with self._lock:
                self._buckets.setdefault(key, deque()).append(item)
                self._ready_count += 1
            with self._metrics_lock:
                self._total_collected += 1
                if path == "static":
                    self._static_collected += 1
                else:
                    self._deficiency_collected += 1
                for t in item.tags:
                    n = _normalize_for_count(t)
                    if n:
                        self._tag_freq[n] = self._tag_freq.get(n, 0) + 1
                self._recent_posts.append({
                    "post_id": pid,
                    "tag_count": len(item.tags),
                    "tags": item.tags[:20],
                    "path": path,
                })

    def _process_post(self, post: dict) -> Optional[_ReadyItem]:
        file_ext = (post.get("file_ext") or "").lower()
        if file_ext not in _ALLOWED_EXTENSIONS:
            return None
        result = self._client.download_inmemory(post)
        if result is None:
            return None
        img_bytes, _ext, _flat_tags = result
        # Header-only size read (no full pixel decode) so the buffer holds only
        # the compressed bytes — keeps CPU RAM bounded.  Full decode + any mode
        # conversion happens later in the training loop (same as a normal
        # on-disk image), one image at a time.
        try:
            with Image.open(io.BytesIO(img_bytes)) as _im:
                w, h = _im.size
        except (OSError, Image.DecompressionBombError) as exc:
            print(f"[DanbooruImageCollector] header read error (post {post.get('id')}): {exc}")
            return None
        if not w or not h:
            return None

        bucket_w, bucket_h = self._assign_bucket(int(w), int(h))
        tag_data, tags = self._build_tag_data(post)
        return _ReadyItem(
            post_id=int(post.get("id")),
            image_bytes=img_bytes,
            tag_data=tag_data,
            bucket_w=bucket_w,
            bucket_h=bucket_h,
            tags=tags,
        )

    def _assign_bucket(self, w: int, h: int) -> Tuple[int, int]:
        """Pick the configured bucket whose aspect ratio is closest to the
        image.  The actual resize/crop happens later in encode_image()."""
        if h <= 0 or w <= 0:
            return self._bucket_resolutions[0]
        ar = w / h
        best = self._bucket_resolutions[0]
        best_d = float("inf")
        for (bw, bh) in self._bucket_resolutions:
            if bh <= 0:
                continue
            d = abs(math.log((bw / bh) / ar))
            if d < best_d:
                best_d = d
                best = (bw, bh)
        return best

    def _build_tag_data(self, post: dict) -> Tuple[List[Dict[str, str]], List[str]]:
        """Build the per-category tag_data list ([{"tag","category"}]) from the
        post's tag_string_<category> fields, plus the flat name list (metrics).

        Category names match the dataset convention (General/Character/
        Copyright/Artist/Meta/Rating) so the training loop can run the same
        per-category shuffle / dropout via process_caption_with_tag_data().
        """
        tag_data: List[Dict[str, str]] = []
        flat: List[str] = []
        if self._include_rating_tag:
            rating_short = post.get("rating")
            if rating_short in _RATING_MAP:
                _r = _RATING_MAP[rating_short]
                tag_data.append({"tag": _r, "category": "Rating"})
                flat.append(_r)
        for cat_code, field in _CAPTION_CATEGORY_FIELDS:
            cname = _CATEGORY_CODE_TO_NAME.get(cat_code, "General")
            for tok in (post.get(field) or "").split():
                t = tok.replace("_", " ")
                tag_data.append({"tag": t, "category": cname})
                flat.append(t)
        if self._max_caption_tags > 0 and len(tag_data) > self._max_caption_tags:
            tag_data = tag_data[: self._max_caption_tags]
            flat = flat[: self._max_caption_tags]
        return tag_data, flat
