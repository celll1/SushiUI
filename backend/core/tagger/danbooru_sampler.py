"""
Danbooru online augmentation for tagger training.

DanbooruSampleBuffer runs a daemon thread that pre-fetches images from Danbooru
and converts them to tensors.  Samples are stored as raw tag lists; labels are
built at injection time in MixedDataLoader so that vocabulary expansions are
reflected immediately.

MixedDataLoader wraps a DataLoader and:
  1. Checks VocabExpander for pending new tags and calls expansion_callback.
  2. Pads base-loader batch labels/loss_masks to the current vocabulary size.
     New-tag columns get label=0, loss_mask=1 (genuine negatives).
  3. Drains up to max_inject_per_batch Danbooru samples, builds their labels
     using the current vocabulary, and re-collates the combined batch.
"""

from __future__ import annotations

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

        self._client  = DanbooruClient(api_interval=api_interval, dl_speed_kbps=dl_speed_kbps)
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

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

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    def get_nowait(self) -> Optional[Tuple]:
        """Return a buffered sample (pv, pam, ss, raw_tags) or None."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        query_idx = 0
        while not self._stop.is_set():
            query = self._tag_queries[query_idx % len(self._tag_queries)]
            query_idx += 1

            try:
                posts = self._client.fetch_posts(query, page=1, min_score=self._min_score)
                if not posts:
                    time.sleep(2.0)
                    continue

                random.shuffle(posts)
                for post in posts:
                    if self._stop.is_set():
                        return
                    sample = self._process_post(post)
                    if sample is None:
                        continue
                    try:
                        self._queue.put(sample, timeout=10.0)
                    except queue.Full:
                        pass  # buffer full — drop and keep fetching

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
            new_approved = normalized & approved - set(voc.tag_to_idx.keys())
            if new_approved:
                self._expander.propose(new_approved)

        return (pixel_values, pixel_attention_mask, spatial_shapes, raw_tags)


# ---------------------------------------------------------------------------
# MixedDataLoader
# ---------------------------------------------------------------------------

class MixedDataLoader:
    """DataLoader wrapper that injects Danbooru samples into each batch.

    On every batch yielded by the base loader:
      1. Check VocabExpander for pending new tags; call expansion_callback if any.
      2. Pad base-loader labels/loss_masks to the current vocabulary size.
         New tag columns: label=0 (negative), loss_mask=1 (train on negatives).
      3. Drain up to ``max_inject_per_batch`` items from the buffer, build labels
         with the current vocabulary, and re-collate the combined batch.

    Delegates ``.dataset``, ``.num_workers``, ``.batch_size`` to the base loader
    so the trainer's resume-loader construction works correctly.
    """

    def __init__(
        self,
        base_loader: DataLoader,
        buffer: DanbooruSampleBuffer,
        max_inject_per_batch: int = 1,
        expander: Any = None,
        expansion_callback: Optional[Callable[[List[str]], None]] = None,
        vocabulary: Optional[TagVocabulary] = None,
        quality_masking_mode: str = "intra_group",
        alias_resolver: Any = None,
    ) -> None:
        self.base_loader          = base_loader
        self._buffer              = buffer
        self._max_inject          = max_inject_per_batch
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

    def __iter__(self):
        for batch in self.base_loader:
            if batch is None:
                yield batch
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

            # ③ Drain Danbooru buffer and build labels at injection time
            injections = []
            for _ in range(self._max_inject):
                s = self._buffer.get_nowait()
                if s is None:
                    break
                pv_d, pam_d, ss_d, raw_tags = s
                voc = self._vocabulary if self._vocabulary is not None else self._buffer._vocabulary
                lbl, lm = _build_label_and_mask_standalone(
                    raw_tags, voc,
                    quality_masking_mode=self._quality_masking,
                    alias_resolver=self._alias_resolver,
                )
                injections.append((pv_d, pam_d, ss_d, lbl, lm))

            if not injections:
                yield (pv, pam, ss, labels, loss_masks)
                continue

            # ④ Re-collate base items + injections
            B = pv.shape[0]
            items = [(pv[i], pam[i], ss[i], labels[i], loss_masks[i]) for i in range(B)]
            merged = tagger_collate_fn(items + injections)
            yield merged if merged is not None else (pv, pam, ss, labels, loss_masks)
