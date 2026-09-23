"""Phase 1 records from datasets.db: DatasetItem + tags-format DatasetCaption (design §5)."""

from __future__ import annotations

import hashlib
import os
import time
import unicodedata
from array import array
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from .style_dataset import stable_item_key
from .style_labels import StyleLabelResolver, StyleLabels

_ROWS_PER_FETCH = 5000
_MAX_EXAMPLES = 5
_MAX_COLLISIONS_REPORTED = 10
_GATHER_ROWS = 1 << 16


@dataclass(frozen=True)
class StyleDatasetConfig:
    dataset_id: int
    caption_types: tuple[str, ...]  # empty: every tags-format caption


def parse_dataset_configs(configs: Any) -> tuple[StyleDatasetConfig, ...]:
    """`[{dataset_id, caption_types}]`, the tagger's shape; a missing or empty caption_types means all
    caption types, as in `DatasetConfigItem`."""
    if not isinstance(configs, (list, tuple)) or not configs:
        raise ValueError("dataset_configs must be a non-empty list of {dataset_id, caption_types}")
    parsed = []
    for entry in configs:
        if not isinstance(entry, dict):
            raise ValueError(f"dataset_configs entry must be an object, got {entry!r}")
        dataset_id = entry.get("dataset_id")
        if isinstance(dataset_id, bool) or not isinstance(dataset_id, int):
            raise ValueError(f"dataset_configs entry needs an integer dataset_id, got {dataset_id!r}")
        types = entry.get("caption_types") or []
        if not isinstance(types, (list, tuple)) or not all(isinstance(t, str) and t for t in types):
            raise ValueError(f"caption_types of dataset {dataset_id} must be a list of names, got {types!r}")
        parsed.append(StyleDatasetConfig(dataset_id, tuple(dict.fromkeys(types))))
    ids = [c.dataset_id for c in parsed]
    if len(set(ids)) != len(ids):
        raise ValueError(f"dataset_configs lists a dataset more than once: {ids}")
    return tuple(parsed)


def dataset_relative_path(image_path: str, dataset_root: str) -> Optional[str]:
    """`image_path` relative to the dataset root after `abspath` on both, '/'-separated (the key form);
    None when the image is not under the root."""
    try:
        relative = os.path.relpath(os.path.abspath(image_path), os.path.abspath(dataset_root))
    except ValueError:  # another drive
        return None
    relative = relative.replace(os.sep, "/")
    if os.altsep:
        relative = relative.replace(os.altsep, "/")
    if relative in (".", "..") or relative.startswith("../") or os.path.isabs(relative):
        return None
    return relative


@dataclass(frozen=True, eq=False)
class ItemPaths:
    """Columnar (item_id, item_key, dataset_id, image_path) sorted by item_id; paths live in one UTF-8
    buffer so millions of items cost no per-item Python objects and pickle cheaply to workers."""

    item_ids: np.ndarray
    item_keys: np.ndarray
    dataset_ids: np.ndarray
    offsets: np.ndarray
    blob: np.ndarray

    def __len__(self) -> int:
        return int(self.item_ids.size)

    def path(self, i: int) -> str:
        return self.blob[self.offsets[i]:self.offsets[i + 1]].tobytes().decode("utf-8")

    def take(self, item_ids: np.ndarray) -> "ItemPaths":
        """Rows for `item_ids` in the given order; every id must be present."""
        item_ids = np.asarray(item_ids, dtype=np.int64)
        pos = np.searchsorted(self.item_ids, item_ids)
        if pos.size and (pos.max() >= self.item_ids.size or np.any(self.item_ids[pos] != item_ids)):
            raise KeyError("item ids missing from the path table")
        return self._gather(pos)

    def _gather(self, pos: np.ndarray) -> "ItemPaths":
        starts = self.offsets[pos]
        lengths = self.offsets[pos + 1] - starts
        offsets = np.zeros(pos.size + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        blob = np.empty(int(offsets[-1]), dtype=np.uint8)
        # Chunked: a byte-level index over the whole buffer would be 8x its size.
        for lo in range(0, pos.size, _GATHER_ROWS):
            hi = min(lo + _GATHER_ROWS, pos.size)
            base, end = int(offsets[lo]), int(offsets[hi])
            index = np.repeat(starts[lo:hi] - offsets[lo:hi], lengths[lo:hi]) + np.arange(base, end, dtype=np.int64)
            blob[base:end] = self.blob[index]
        return ItemPaths(self.item_ids[pos], self.item_keys[pos], self.dataset_ids[pos], offsets, blob)


class _ItemPathsBuilder:
    def __init__(self) -> None:
        self.ids, self.keys, self.datasets = array("q"), array("q"), array("q")
        self.offsets = array("q", [0])
        self.blob = bytearray()

    def append(self, item_id: int, item_key: int, dataset_id: int, path: str) -> int:
        self.ids.append(item_id)
        self.keys.append(item_key)
        self.datasets.append(dataset_id)
        self.blob += path.encode("utf-8")
        self.offsets.append(len(self.blob))
        return len(self.ids) - 1

    def path(self, row: int) -> str:
        return self.blob[self.offsets[row]:self.offsets[row + 1]].decode("utf-8")

    def finish(self) -> ItemPaths:
        ids = np.frombuffer(self.ids, dtype=np.int64)
        unsorted = ItemPaths(ids, np.frombuffer(self.keys, dtype=np.int64),
                             np.frombuffer(self.datasets, dtype=np.int64),
                             np.frombuffer(self.offsets, dtype=np.int64), np.frombuffer(self.blob, dtype=np.uint8))
        order = np.argsort(ids, kind="stable")
        if order.size > 1 and np.any(ids[order][1:] == ids[order][:-1]):
            raise ValueError("an item was ingested twice")
        return unsorted._gather(order)


def normalized_absolute_path(image_path: str) -> str:
    """The form under which two items name the same file: absolute, NFC, and case- and separator-folded
    where the OS folds them (`os.path.normcase`)."""
    return os.path.normcase(unicodedata.normalize("NFC", os.path.abspath(image_path)))


def _path_hash(normalized: str) -> int:
    digest = hashlib.blake2b(normalized.encode("utf-8", "surrogatepass"), digest_size=8, person=b"style-path")
    return int.from_bytes(digest.digest(), "little", signed=True)


def _path_fingerprint(normalized: str) -> tuple[int, int]:
    digest = hashlib.blake2b(normalized.encode("utf-8", "surrogatepass"), digest_size=16,
                             person=b"style-pinpath").digest()
    return int.from_bytes(digest[:8], "little", signed=True), int.from_bytes(digest[8:], "little", signed=True)


class _PinnedPaths:
    """Normalised path -> the pinned item that owns it (lowest dataset_id among pinned copies), keyed by a
    128-bit path fingerprint so no path strings are held for millions of pinned items."""

    def __init__(self, high: array, low: array, owners: array) -> None:
        high, low, owners = (np.frombuffer(a, dtype=np.int64) for a in (high, low, owners))
        order = np.lexsort((low, high))  # stable: equal paths keep their dataset_id order
        high, low, owners = high[order], low[order], owners[order]
        first = np.ones(high.size, dtype=bool)
        first[1:] = (high[1:] != high[:-1]) | (low[1:] != low[:-1])
        self._high, self._low, self._owners = high[first], low[first], owners[first]

    def owner(self, normalized: str) -> Optional[int]:
        high, low = _path_fingerprint(normalized)
        i = int(np.searchsorted(self._high, high))
        while i < self._high.size and self._high[i] == high:
            if self._low[i] == low:
                return int(self._owners[i])
            i += 1
        return None


class _PathDeduper:
    """Records of earlier datasets as sorted 64-bit path hashes with their path-table rows. A hash hit is
    confirmed against the stored path, so a hash collision keeps both items."""

    def __init__(self) -> None:
        self._hashes = np.zeros(0, dtype=np.int64)
        self._rows = np.zeros(0, dtype=np.int64)
        self._pending_hashes, self._pending_rows = array("q"), array("q")

    def find(self, path_hash: int, normalized: str, builder: _ItemPathsBuilder) -> Optional[int]:
        if not self._hashes.size:
            return None
        i = int(np.searchsorted(self._hashes, path_hash))
        while i < self._hashes.size and self._hashes[i] == path_hash:
            row = int(self._rows[i])
            if normalized_absolute_path(builder.path(row)) == normalized:
                return row
            i += 1
        return None

    def add(self, path_hash: int, row: int) -> None:
        self._pending_hashes.append(path_hash)
        self._pending_rows.append(row)

    def end_dataset(self) -> None:
        """Items of one dataset are not compared with each other: an exact repeated path is refused by the
        item-key check, but a case or separator variant of one path within a dataset is not detected."""
        hashes = np.concatenate([self._hashes, np.frombuffer(self._pending_hashes, dtype=np.int64)])
        rows = np.concatenate([self._rows, np.frombuffer(self._pending_rows, dtype=np.int64)])
        order = np.argsort(hashes, kind="stable")
        self._hashes, self._rows = hashes[order], rows[order]
        self._pending_hashes, self._pending_rows = array("q"), array("q")


class DuplicateItemKeyError(ValueError):
    def __init__(self, message: str, collisions: list[list[tuple[int, int, str]]]) -> None:
        super().__init__(message)
        self.collisions = collisions  # per key: [(dataset_id, item_id, image_path), ...]


def check_unique_item_keys(table: ItemPaths) -> None:
    keys = table.item_keys
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    dup = np.flatnonzero(sorted_keys[1:] == sorted_keys[:-1])
    if not dup.size:
        return
    duplicated = np.unique(sorted_keys[dup])
    collisions = []
    for key in duplicated[:_MAX_COLLISIONS_REPORTED]:
        rows = order[np.searchsorted(sorted_keys, key, "left"):np.searchsorted(sorted_keys, key, "right")]
        collisions.append([(int(table.dataset_ids[r]), int(table.item_ids[r]), table.path(int(r))) for r in rows])
    lines = "\n".join("  " + " | ".join(f"dataset {d}, item {i}: {p}" for d, i, p in group) for group in collisions)
    more = f"\n  ... and {duplicated.size - len(collisions)} more colliding keys" \
        if duplicated.size > len(collisions) else ""
    raise DuplicateItemKeyError(
        f"{duplicated.size} item key(s) are shared by several items; an item key is (dataset id, image path "
        f"relative to the dataset root), so these rows name the same image. Remove the duplicate rows and "
        f"restart:\n{lines}{more}", collisions)


@dataclass
class IngestStats:
    datasets: int = 0
    items_scanned: int = 0
    items_missing_image: int = 0
    items_outside_root: int = 0
    items_without_tag_caption: int = 0
    items_duplicate_path: int = 0
    records: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class StyleIngest:
    """`records()` streams Phase 1 records (item_id, item_key, labels) for `build_style_index`; when it
    is exhausted, `paths` holds the path table of every record, item keys have been checked unique and
    every configured dataset has yielded a record or only duplicates.

    Captions follow `TaggerDataset._build_samples` given a caption_types filter: `is_tags_format` captions,
    restricted to that dataset's caption_types when non-empty, all selected captions of an item
    contributing; items whose image file is missing are skipped. Unlike here, the tagger run passes
    caption_types=None, so it ignores each dataset_configs entry's caption_types.

    Datasets are read in dataset_id order. An image that several items name (same
    `normalized_absolute_path`, e.g. under nested dataset roots) is kept once, since the copies' item keys
    differ: it would train twice and could land on both sides of the image split. The copy kept is the one
    whose item key is in `pinned_keys` (the saved split snapshot; the lowest dataset_id among several),
    else the lowest dataset_id. Keeping the pinned copy keeps a trained image under the key it trained with,
    whose hash put it on the train side."""

    def __init__(self, session, configs: Sequence[StyleDatasetConfig], resolver: StyleLabelResolver,
                 progress: Optional[Callable[[int, int, str], None]] = None,
                 pinned_keys: Optional[np.ndarray] = None) -> None:
        self._session = session
        self._configs = tuple(configs)
        self.resolver = resolver
        self._progress = progress
        self._pinned = np.unique(np.asarray(pinned_keys if pinned_keys is not None else [], dtype=np.int64))
        self._pinned_paths: Optional[_PinnedPaths] = None
        self.stats = IngestStats()
        self.paths: Optional[ItemPaths] = None
        self.skipped_examples: dict[int, list[str]] = {}  # dataset_id -> first skipped items
        self.duplicate_examples: list[str] = []
        self._deduper = _PathDeduper() if len(self._configs) > 1 else None

    def _skip(self, dataset_id: int, what: str) -> None:
        examples = self.skipped_examples.setdefault(dataset_id, [])
        if len(examples) < _MAX_EXAMPLES:
            examples.append(what)

    def records(self) -> Iterator[tuple[int, int, StyleLabels]]:
        from sqlalchemy import and_
        from database.models import Dataset, DatasetCaption, DatasetItem

        builder = _ItemPathsBuilder()
        stats = self.stats
        last_emit = 0.0
        configs = sorted(self._configs, key=lambda c: c.dataset_id)
        if self._deduper is not None and self._pinned.size:
            self._pinned_paths = self._pinned_path_table(configs)
        for n, config in enumerate(configs):
            dataset = self._session.query(Dataset.id, Dataset.path).filter(Dataset.id == config.dataset_id).first()
            if dataset is None:
                raise ValueError(f"dataset {config.dataset_id} does not exist")
            if not dataset.path:
                raise ValueError(f"dataset {config.dataset_id} has no root path, so its item keys are undefined")
            stats.datasets += 1
            total = self._session.query(DatasetItem.id).filter(DatasetItem.dataset_id == config.dataset_id).count()
            on = [DatasetCaption.item_id == DatasetItem.id, DatasetCaption.is_tags_format == True]  # noqa: E712
            if config.caption_types:
                on.append(DatasetCaption.caption_type.in_(config.caption_types))
            rows = (self._session.query(DatasetItem.id, DatasetItem.image_path, DatasetCaption.id,
                                        DatasetCaption.tag_data, DatasetCaption.content)
                    .outerjoin(DatasetCaption, and_(*on))
                    .filter(DatasetItem.dataset_id == config.dataset_id)
                    .order_by(DatasetItem.id, DatasetCaption.id)
                    .yield_per(_ROWS_PER_FETCH))
            current: Optional[tuple[int, str]] = None
            captions: list[tuple[Any, Optional[str]]] = []
            done = 0
            records_before, duplicates_before = stats.records, stats.items_duplicate_path
            for item_id, image_path, caption_id, tag_data, content in _with_sentinel(rows):
                if current is not None and item_id != current[0]:
                    record = self._record(config.dataset_id, dataset.path, current, captions, builder)
                    done += 1
                    if record is not None:
                        yield record
                    captions = []
                    if self._progress is not None and time.monotonic() - last_emit >= 0.3:
                        last_emit = time.monotonic()
                        self._progress(done, total, f"Loading dataset {config.dataset_id} ({n + 1}/"
                                                    f"{len(self._configs)}): {done:,}/{total:,} items")
                if item_id is None:
                    break
                current = (int(item_id), image_path)
                if caption_id is not None:
                    captions.append((tag_data, content))
            if stats.records == records_before and stats.items_duplicate_path == duplicates_before:
                raise ValueError(f"dataset {config.dataset_id} yielded no usable item ({done} scanned; check the "
                                 f"image files, the dataset root and caption_types {list(config.caption_types)}): "
                                 f"{self.skipped_examples.get(config.dataset_id, [])}")
            if self._deduper is not None:
                self._deduper.end_dataset()
        self.paths = builder.finish()
        check_unique_item_keys(self.paths)

    def _pinned_path_table(self, configs: Sequence[StyleDatasetConfig]) -> _PinnedPaths:
        """Pre-pass over item paths (no file checks): the owning pinned item of each pinned path. Only an item
        with a selected tags-format caption can own a path, since the main pass drops any other."""
        from sqlalchemy import and_, exists
        from database.models import Dataset, DatasetCaption, DatasetItem

        high, low, owners = array("q"), array("q"), array("q")
        pinned = self._pinned
        for n, config in enumerate(configs):
            root = self._session.query(Dataset.path).filter(Dataset.id == config.dataset_id).scalar()
            if not root:
                continue  # refused by the main pass
            if self._progress is not None:
                self._progress(n, len(configs), f"Locating trained images in dataset {config.dataset_id}")
            captioned = [DatasetCaption.item_id == DatasetItem.id, DatasetCaption.is_tags_format == True]  # noqa: E712
            if config.caption_types:
                captioned.append(DatasetCaption.caption_type.in_(config.caption_types))
            rows = (self._session.query(DatasetItem.id, DatasetItem.image_path)
                    .filter(DatasetItem.dataset_id == config.dataset_id, exists().where(and_(*captioned)))
                    .order_by(DatasetItem.id).yield_per(_ROWS_PER_FETCH))
            ids: list[int] = []
            paths: list[str] = []
            keys: list[int] = []

            def flush() -> None:
                if not keys:
                    return
                k = np.asarray(keys, dtype=np.int64)
                pos = np.minimum(np.searchsorted(pinned, k), pinned.size - 1)
                for i in np.flatnonzero(pinned[pos] == k):
                    h, lo = _path_fingerprint(normalized_absolute_path(paths[i]))
                    high.append(h)
                    low.append(lo)
                    owners.append(ids[i])
                ids.clear()
                paths.clear()
                keys.clear()

            for item_id, image_path in rows:
                relative = dataset_relative_path(image_path, root) if image_path else None
                if relative is None:
                    continue
                ids.append(int(item_id))
                paths.append(image_path)
                keys.append(stable_item_key(config.dataset_id, relative))
                if len(keys) >= _ROWS_PER_FETCH:
                    flush()
            flush()
        return _PinnedPaths(high, low, owners)

    def _drop_duplicate(self, item_id: int, dataset_id: int, image_path: str, of: str) -> None:
        self.stats.items_duplicate_path += 1
        if len(self.duplicate_examples) < _MAX_EXAMPLES:
            self.duplicate_examples.append(f"item {item_id} of dataset {dataset_id} ({image_path!r}) is {of}")

    def _record(self, dataset_id: int, root: str, item: tuple[int, str], captions: list,
                builder: _ItemPathsBuilder) -> Optional[tuple[int, int, StyleLabels]]:
        item_id, image_path = item
        stats = self.stats
        stats.items_scanned += 1
        if not image_path or not os.path.isfile(image_path):
            stats.items_missing_image += 1
            self._skip(dataset_id, f"item {item_id}: missing image {image_path!r}")
            return None
        relative = dataset_relative_path(image_path, root)
        if relative is None:
            stats.items_outside_root += 1
            self._skip(dataset_id, f"item {item_id}: {image_path!r} is outside dataset root {root!r}")
            return None
        if not captions:
            stats.items_without_tag_caption += 1
            return None
        if self._deduper is not None:
            normalized = normalized_absolute_path(image_path)
            path_hash = _path_hash(normalized)
            owner = self._pinned_paths.owner(normalized) if self._pinned_paths is not None else None
            if owner is not None:
                if owner != item_id:  # the owner itself is never a duplicate: other copies all drop here
                    self._drop_duplicate(item_id, dataset_id, image_path, f"item {owner}, trained on by this run")
                    return None
            else:
                original = self._deduper.find(path_hash, normalized, builder)
                if original is not None:
                    self._drop_duplicate(item_id, dataset_id, image_path,
                                         f"item {builder.ids[original]} of dataset {builder.datasets[original]}")
                    return None
        labels = self.resolver.resolve_captions(captions)
        key = stable_item_key(dataset_id, relative)
        row = builder.append(item_id, key, dataset_id, image_path)
        if self._deduper is not None:
            self._deduper.add(path_hash, row)
        stats.records += 1
        return item_id, key, labels


def _with_sentinel(rows: Iterable) -> Iterator[tuple]:
    yield from rows
    yield None, None, None, None, None


def build_label_resolver(root_dir: str, use_tag_aliases: bool, use_gelbooru_categories: bool,
                         signature_tags: Sequence[str], non_copyright_tags: Sequence[str]) -> StyleLabelResolver:
    """The production resolver: taglist manager, comma-tag resolver and alias table as the tagger builds them.
    A missing alias table is refused rather than skipped: without it artist names, and so the split, change."""
    from core.tagger.comma_tag_resolver import CommaTagResolver
    from core.tagger.tag_alias_resolver import TagAliasResolver, alias_table_path
    from core.training.tag_group_utils import get_tag_group_manager

    manager = get_tag_group_manager(enable_gelbooru=use_gelbooru_categories)
    comma = CommaTagResolver.build_from_taglist_cache(root_dir, use_gelbooru=use_gelbooru_categories)
    print(f"[StyleTraining] Comma-tag resolver: {len(comma)} comma-containing tags")
    aliases = None
    if use_tag_aliases:
        path = alias_table_path(root_dir)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"style_use_tag_aliases is on but the alias table {path} does not exist")
        aliases = TagAliasResolver.load(path)
        print(f"[StyleTraining] Loaded {len(aliases)} tag aliases from {path}")
    return StyleLabelResolver(manager, comma, aliases, signature_tags, non_copyright_tags)
