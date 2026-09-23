"""Phase 1 records from datasets.db: DatasetItem + tags-format DatasetCaption (design §5)."""

from __future__ import annotations

import os
import time
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

    def append(self, item_id: int, item_key: int, dataset_id: int, path: str) -> None:
        self.ids.append(item_id)
        self.keys.append(item_key)
        self.datasets.append(dataset_id)
        self.blob += path.encode("utf-8")
        self.offsets.append(len(self.blob))

    def finish(self) -> ItemPaths:
        ids = np.frombuffer(self.ids, dtype=np.int64)
        unsorted = ItemPaths(ids, np.frombuffer(self.keys, dtype=np.int64),
                             np.frombuffer(self.datasets, dtype=np.int64),
                             np.frombuffer(self.offsets, dtype=np.int64), np.frombuffer(self.blob, dtype=np.uint8))
        order = np.argsort(ids, kind="stable")
        if order.size > 1 and np.any(ids[order][1:] == ids[order][:-1]):
            raise ValueError("an item was ingested twice")
        return unsorted._gather(order)


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
    records: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class StyleIngest:
    """`records()` streams Phase 1 records (item_id, item_key, labels) for `build_style_index`; when it
    is exhausted, `paths` holds the path table of every record, item keys have been checked unique and
    every configured dataset has yielded a record.

    Captions follow `TaggerDataset._build_samples` given a caption_types filter: `is_tags_format` captions,
    restricted to that dataset's caption_types when non-empty, all selected captions of an item
    contributing; items whose image file is missing are skipped. Unlike here, the tagger run passes
    caption_types=None, so it ignores each dataset_configs entry's caption_types."""

    def __init__(self, session, configs: Sequence[StyleDatasetConfig], resolver: StyleLabelResolver,
                 progress: Optional[Callable[[int, int, str], None]] = None) -> None:
        self._session = session
        self._configs = tuple(configs)
        self.resolver = resolver
        self._progress = progress
        self.stats = IngestStats()
        self.paths: Optional[ItemPaths] = None
        self.skipped_examples: list[str] = []

    def _skip(self, what: str) -> None:
        if len(self.skipped_examples) < _MAX_EXAMPLES:
            self.skipped_examples.append(what)

    def records(self) -> Iterator[tuple[int, int, StyleLabels]]:
        from sqlalchemy import and_
        from database.models import Dataset, DatasetCaption, DatasetItem

        builder = _ItemPathsBuilder()
        stats = self.stats
        last_emit = 0.0
        for n, config in enumerate(self._configs):
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
            records_before = stats.records
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
            if stats.records == records_before:
                raise ValueError(f"dataset {config.dataset_id} yielded no usable item ({done} scanned; check the "
                                 f"image files, the dataset root and caption_types {list(config.caption_types)}): "
                                 f"{self.skipped_examples}")
        self.paths = builder.finish()
        check_unique_item_keys(self.paths)

    def _record(self, dataset_id: int, root: str, item: tuple[int, str], captions: list,
                builder: _ItemPathsBuilder) -> Optional[tuple[int, int, StyleLabels]]:
        item_id, image_path = item
        stats = self.stats
        stats.items_scanned += 1
        if not image_path or not os.path.isfile(image_path):
            stats.items_missing_image += 1
            self._skip(f"item {item_id}: missing image {image_path!r}")
            return None
        relative = dataset_relative_path(image_path, root)
        if relative is None:
            stats.items_outside_root += 1
            self._skip(f"item {item_id}: {image_path!r} is outside dataset root {root!r}")
            return None
        if not captions:
            stats.items_without_tag_caption += 1
            return None
        labels = self.resolver.resolve_captions(captions)
        key = stable_item_key(dataset_id, relative)
        builder.append(item_id, key, dataset_id, image_path)
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
