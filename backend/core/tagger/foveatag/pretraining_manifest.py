"""Scalable path manifests and deterministic domain sampling for Phase 2."""

from __future__ import annotations

import hashlib
import heapq
import json
import mmap
import os
from pathlib import Path
import sqlite3
import struct
from typing import Any, Mapping, Optional, Sequence

from torch.utils.data import Dataset

from .pretraining import FoveaPretrainingSample, load_pretraining_sample
from .tiled_pyramid import TiledPyramidSource, tiled_pyramid_cache_path
from .views import ViewConfig


MANIFEST_FORMAT_VERSION = 1
_UINT64_SIZE = struct.calcsize("<Q")
_UINT64_MASK = (1 << 64) - 1


class IndexedPathManifest:
    """Random-access JSONL paths backed by a compact mmap offset index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.index_path = self.path.with_suffix(f"{self.path.suffix}.idx")
        if not self.path.is_file() or not self.index_path.is_file():
            raise FileNotFoundError(f"indexed path manifest is incomplete: {self.path}")
        size = self.index_path.stat().st_size
        if size % _UINT64_SIZE:
            raise ValueError(f"manifest index size is invalid: {self.index_path}")
        self._length = size // _UINT64_SIZE
        self._path_handle = None
        self._index_handle = None
        self._index_map = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> str:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        self._ensure_open()
        offset = struct.unpack_from("<Q", self._index_map, index * _UINT64_SIZE)[0]
        self._path_handle.seek(offset)
        return str(json.loads(self._path_handle.readline()))

    def close(self) -> None:
        if self._index_map is not None:
            self._index_map.close()
        if self._index_handle is not None:
            self._index_handle.close()
        if self._path_handle is not None:
            self._path_handle.close()
        self._path_handle = self._index_handle = self._index_map = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_path_handle"] = None
        state["_index_handle"] = None
        state["_index_map"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._index_map is not None:
            return
        self._path_handle = self.path.open("rb")
        self._index_handle = self.index_path.open("rb")
        if not self._length:
            raise ValueError(f"indexed path manifest is empty: {self.path}")
        self._index_map = mmap.mmap(self._index_handle.fileno(), 0, access=mmap.ACCESS_READ)


class ManifestPretrainingDataset(Dataset):
    def __init__(
        self,
        manifest: IndexedPathManifest,
        *,
        view_config: ViewConfig,
        seed: int,
        max_source_pixels: Optional[int],
        pyramid_cache_root: Optional[str | Path] = None,
    ) -> None:
        if not len(manifest):
            raise ValueError("manifest must not be empty")
        self.manifest = manifest
        self.view_config = view_config
        self.seed = int(seed)
        self.max_source_pixels = max_source_pixels
        self.pyramid_cache_root = (
            None if pyramid_cache_root is None else str(pyramid_cache_root)
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Optional[FoveaPretrainingSample]:
        seed = _splitmix64(self.seed + self.epoch * len(self) + index) & 0x7FFFFFFF
        return load_pretraining_sample(
            self.manifest[index],
            self.view_config,
            seed=seed,
            max_source_pixels=self.max_source_pixels,
            pyramid_cache_root=getattr(self, "pyramid_cache_root", None),
        )


class DomainBalancedPretrainingDataset(Dataset):
    """Sample mmap manifests with stateless deterministic domain weighting."""

    def __init__(
        self,
        manifests: Mapping[int, IndexedPathManifest],
        domain_weights: Mapping[int, float],
        *,
        epoch_samples: int,
        view_config: ViewConfig,
        seed: int,
        max_source_pixels: Optional[int],
        pyramid_cache_root: Optional[str | Path] = None,
    ) -> None:
        if epoch_samples <= 0:
            raise ValueError("epoch_samples must be positive")
        if set(manifests) != set(domain_weights) or not manifests:
            raise ValueError("manifests and domain_weights must contain the same domains")
        if any(len(manifest) == 0 for manifest in manifests.values()):
            raise ValueError("domain manifests must not be empty")
        if any(weight <= 0 for weight in domain_weights.values()):
            raise ValueError("domain weights must be positive")
        self.domain_ids = tuple(sorted(manifests))
        self.manifests = dict(manifests)
        total = sum(float(domain_weights[domain]) for domain in self.domain_ids)
        cumulative = 0.0
        thresholds = []
        for domain in self.domain_ids:
            cumulative += float(domain_weights[domain]) / total
            thresholds.append(cumulative)
        thresholds[-1] = 1.0
        self.thresholds = tuple(thresholds)
        self.epoch_samples = int(epoch_samples)
        self.view_config = view_config
        self.seed = int(seed)
        self.max_source_pixels = max_source_pixels
        self.pyramid_cache_root = (
            None if pyramid_cache_root is None else str(pyramid_cache_root)
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_samples

    def selection(self, index: int) -> tuple[int, int, int]:
        if index < 0 or index >= self.epoch_samples:
            raise IndexError(index)
        key = _splitmix64(self.seed + self.epoch * self.epoch_samples + index)
        fraction = key / float(1 << 64)
        position = next(
            position for position, threshold in enumerate(self.thresholds) if fraction < threshold
        )
        domain = self.domain_ids[position]
        item_key = _splitmix64(key)
        item_index = item_key % len(self.manifests[domain])
        sample_seed = _splitmix64(item_key) & 0x7FFFFFFF
        return domain, item_index, sample_seed

    def __getitem__(self, index: int) -> Optional[FoveaPretrainingSample]:
        domain, item_index, sample_seed = self.selection(index)
        return load_pretraining_sample(
            self.manifests[domain][item_index],
            self.view_config,
            seed=sample_seed,
            max_source_pixels=self.max_source_pixels,
            pyramid_cache_root=getattr(self, "pyramid_cache_root", None),
        )


def build_pretraining_manifests(
    database_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_ids: Sequence[int],
    validation_counts: Mapping[int, int],
    excluded_paths: Sequence[str] = (),
    minimum_side: int = 256,
    max_source_pixels: Optional[int] = 40_000_000,
    pyramid_cache_root: Optional[str | Path] = None,
    seed: int = 0,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Build exact validation and train path manifests without loading paths in RAM."""
    database_path = Path(database_path).resolve()
    output_dir = Path(output_dir)
    domains = tuple(sorted(set(int(value) for value in dataset_ids)))
    if not domains or minimum_side <= 0:
        raise ValueError("dataset_ids and minimum_side must be non-empty and positive")
    if set(validation_counts) != set(domains):
        raise ValueError("validation_counts must specify every dataset")
    if any(int(validation_counts[domain]) <= 0 for domain in domains):
        raise ValueError("validation counts must be positive")
    if max_source_pixels is not None and max_source_pixels <= 0:
        raise ValueError("max_source_pixels must be positive when provided")
    stat = database_path.stat()
    excluded = {_path_key(path) for path in excluded_paths}
    cached_source_keys: set[str] = set()
    pyramid_cache_contract = None
    if pyramid_cache_root is not None:
        cache_index, resolved_cache_root = _pyramid_cache_index(pyramid_cache_root)
        cached_source_keys = _relevant_cached_source_keys(
            database_path,
            domains=domains,
            minimum_side=minimum_side,
            max_source_pixels=max_source_pixels,
            available_keys=set(cache_index),
        )
        entries = []
        for key in sorted(cached_source_keys):
            source, digest = cache_index[key]
            source.validate_source(source.source_path)
            entries.append(f"{key}\0{digest}")
        pyramid_cache_contract = {
            "root": str(resolved_cache_root),
            "count": len(entries),
            "entries_sha256": _sequence_hash(entries),
        }
    request = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "database_path": str(database_path),
        "database_size": stat.st_size,
        "database_mtime_ns": stat.st_mtime_ns,
        "dataset_ids": list(domains),
        "validation_counts": {str(key): int(validation_counts[key]) for key in domains},
        "minimum_side": int(minimum_side),
        "max_source_pixels": max_source_pixels,
        "seed": int(seed),
        "excluded_path_count": len(excluded),
        "excluded_path_hash": _sequence_hash(sorted(excluded)),
    }
    if pyramid_cache_contract is not None:
        request["pyramid_cache"] = pyramid_cache_contract
    metadata_path = output_dir / "manifest_metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("request") == request and _manifest_set_is_complete(output_dir, existing):
            return existing
        if reuse_existing:
            saved_request = dict(existing.get("request") or {})
            current_request = dict(request)
            for key in ("database_size", "database_mtime_ns"):
                saved_request.pop(key, None)
                current_request.pop(key, None)
            if saved_request != current_request:
                raise ValueError("existing pretraining manifest contract does not match resume")
            if not _manifest_set_is_complete(output_dir, existing):
                raise ValueError("existing pretraining manifests are incomplete")
            return existing
    elif reuse_existing:
        raise ValueError("resume requires existing pretraining manifests")

    output_dir.mkdir(parents=True, exist_ok=True)
    uri = database_path.as_uri() + "?mode=ro"
    result: dict[str, Any] = {"request": request, "domains": {}}
    with sqlite3.connect(uri, uri=True) as connection:
        for domain in domains:
            validation_count = int(validation_counts[domain])
            source_size_filter = _source_size_filter_counts(
                connection,
                domain,
                minimum_side=minimum_side,
                max_source_pixels=max_source_pixels,
                cached_source_keys=cached_source_keys,
            )
            selected = heapq.nsmallest(
                validation_count,
                _ranked_paths(
                    _eligible_rows(
                        connection,
                        domain,
                        minimum_side=minimum_side,
                        max_source_pixels=max_source_pixels,
                        cached_source_keys=cached_source_keys,
                    ),
                    excluded=excluded,
                    seed=seed,
                    domain=domain,
                ),
            )
            if len(selected) != validation_count:
                raise ValueError(
                    f"dataset {domain} has fewer than {validation_count} eligible validation paths"
                )
            validation_keys = {key for _, key in selected}
            train_path = output_dir / f"dataset_{domain}_train.jsonl"
            validation_path = output_dir / f"dataset_{domain}_validation.jsonl"
            train_writer = _IndexedManifestWriter(train_path)
            validation_writer = _IndexedManifestWriter(validation_path)
            eligible = excluded_count = 0
            try:
                for row in _eligible_rows(
                    connection,
                    domain,
                    minimum_side=minimum_side,
                    max_source_pixels=max_source_pixels,
                    cached_source_keys=cached_source_keys,
                ):
                    path = str(row[0])
                    key = _path_key(path)
                    if key in excluded:
                        excluded_count += 1
                        continue
                    eligible += 1
                    if key in validation_keys:
                        validation_writer.write(path)
                    else:
                        train_writer.write(path)
                train_info = train_writer.finish()
                validation_info = validation_writer.finish()
            except BaseException:
                train_writer.abort()
                validation_writer.abort()
                raise
            if validation_info["count"] != validation_count:
                raise ValueError(f"dataset {domain} validation paths are not unique")
            result["domains"][str(domain)] = {
                "source_size_filter": source_size_filter,
                "eligible_after_exclusion": eligible,
                "excluded": excluded_count,
                "train": train_info,
                "validation": validation_info,
            }
    _write_json_atomic(metadata_path, result)
    return result


def open_domain_manifests(
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    split: str,
) -> dict[int, IndexedPathManifest]:
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    output_dir = Path(output_dir)
    return {
        int(domain): IndexedPathManifest(output_dir / details[split]["path"])
        for domain, details in metadata["domains"].items()
    }


def _eligible_rows(
    connection: sqlite3.Connection,
    domain: int,
    *,
    minimum_side: int,
    max_source_pixels: Optional[int],
    cached_source_keys: set[str],
):
    condition = "dataset_id = ? AND width >= ? AND height >= ?"
    parameters: tuple[int, ...] = (domain, minimum_side, minimum_side)
    if max_source_pixels is not None:
        condition += " AND width * height <= ?"
        parameters += (max_source_pixels,)
    rows = connection.execute(
        f"SELECT image_path FROM dataset_items WHERE {condition} ORDER BY id",
        parameters,
    )
    yield from rows
    if max_source_pixels is None or not cached_source_keys:
        return
    oversized = connection.execute(
        "SELECT image_path FROM dataset_items "
        "WHERE dataset_id = ? AND width >= ? AND height >= ? "
        "AND width * height > ? ORDER BY id",
        (domain, minimum_side, minimum_side, max_source_pixels),
    )
    for row in oversized:
        if _path_key(str(row[0])) in cached_source_keys:
            yield row


def _source_size_filter_counts(
    connection: sqlite3.Connection,
    domain: int,
    *,
    minimum_side: int,
    max_source_pixels: Optional[int],
    cached_source_keys: set[str],
) -> dict[str, Optional[int]]:
    minimum_eligible = int(connection.execute(
        "SELECT COUNT(*) FROM dataset_items "
        "WHERE dataset_id = ? AND width >= ? AND height >= ?",
        (domain, minimum_side, minimum_side),
    ).fetchone()[0])
    if max_source_pixels is None:
        after_cap = minimum_eligible
        over_cap = 0
    else:
        after_cap = int(connection.execute(
            "SELECT COUNT(*) FROM dataset_items "
            "WHERE dataset_id = ? AND width >= ? AND height >= ? "
            "AND width * height <= ?",
            (domain, minimum_side, minimum_side, max_source_pixels),
        ).fetchone()[0])
        over_cap = minimum_eligible - after_cap
    cached_over_cap = 0
    if max_source_pixels is not None and cached_source_keys:
        rows = connection.execute(
            "SELECT image_path FROM dataset_items "
            "WHERE dataset_id = ? AND width >= ? AND height >= ? "
            "AND width * height > ?",
            (domain, minimum_side, minimum_side, max_source_pixels),
        )
        cached_over_cap = sum(
            _path_key(str(row[0])) in cached_source_keys for row in rows
        )
    return {
        "minimum_side_eligible": minimum_eligible,
        "max_source_pixels": max_source_pixels,
        "over_pixel_cap": over_cap,
        "eligible_after_pixel_cap": after_cap,
        "cached_over_pixel_cap": cached_over_cap,
        "eligible_after_pixel_policy": after_cap + cached_over_cap,
    }


def _pyramid_cache_index(
    cache_root: str | Path,
) -> tuple[dict[str, tuple[TiledPyramidSource, str]], Path]:
    root = Path(cache_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"pyramid cache root is missing: {root}")
    sources = {}
    for manifest in sorted(root.glob("*/*/manifest.json")):
        source = TiledPyramidSource(manifest.parent)
        expected = tiled_pyramid_cache_path(root, source.source_path).resolve()
        if manifest.parent.resolve() != expected:
            raise ValueError(f"pyramid cache is outside its deterministic location: {manifest}")
        key = _path_key(str(source.source_path))
        if key in sources:
            raise ValueError(f"pyramid cache contains duplicate source: {source.source_path}")
        sources[key] = (source, _file_sha256(manifest))
    return sources, root


def _relevant_cached_source_keys(
    database_path: Path,
    *,
    domains: Sequence[int],
    minimum_side: int,
    max_source_pixels: Optional[int],
    available_keys: set[str],
) -> set[str]:
    if max_source_pixels is None or not available_keys:
        return set()
    relevant = set()
    uri = database_path.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        for domain in domains:
            rows = connection.execute(
                "SELECT image_path FROM dataset_items "
                "WHERE dataset_id = ? AND width >= ? AND height >= ? "
                "AND width * height > ?",
                (domain, minimum_side, minimum_side, max_source_pixels),
            )
            relevant.update(
                key for row in rows if (key := _path_key(str(row[0]))) in available_keys
            )
    return relevant


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranked_paths(rows, *, excluded: set[str], seed: int, domain: int):
    for row in rows:
        key = _path_key(str(row[0]))
        if key not in excluded:
            yield _stable_rank_key(key, seed=seed, domain=domain), key


class _IndexedManifestWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.index_path = path.with_suffix(f"{path.suffix}.idx")
        self.temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        self.temporary_index = self.index_path.with_suffix(f"{self.index_path.suffix}.tmp")
        self.path_handle = self.temporary_path.open("wb")
        self.index_handle = self.temporary_index.open("wb")
        self.digest = hashlib.sha256()
        self.count = 0

    def write(self, path: str) -> None:
        encoded = (json.dumps(path, ensure_ascii=False) + "\n").encode("utf-8")
        self.index_handle.write(struct.pack("<Q", self.path_handle.tell()))
        self.path_handle.write(encoded)
        self.digest.update(encoded)
        self.count += 1

    def finish(self) -> dict[str, Any]:
        self.path_handle.close()
        self.index_handle.close()
        os.replace(self.temporary_path, self.path)
        os.replace(self.temporary_index, self.index_path)
        return {
            "path": self.path.name,
            "index_path": self.index_path.name,
            "count": self.count,
            "bytes": self.path.stat().st_size,
            "sha256": self.digest.hexdigest(),
        }

    def abort(self) -> None:
        self.path_handle.close()
        self.index_handle.close()
        self.temporary_path.unlink(missing_ok=True)
        self.temporary_index.unlink(missing_ok=True)


def _manifest_set_is_complete(output_dir: Path, metadata: Mapping[str, Any]) -> bool:
    try:
        for details in metadata["domains"].values():
            for split in ("train", "validation"):
                value = details[split]
                path = output_dir / value["path"]
                index_path = output_dir / value["index_path"]
                if not path.is_file() or index_path.stat().st_size != value["count"] * _UINT64_SIZE:
                    return False
                if "bytes" in value and path.stat().st_size != value["bytes"]:
                    return False
        return True
    except (KeyError, OSError, TypeError):
        return False


def _stable_rank(path: str, *, seed: int, domain: int) -> bytes:
    return _stable_rank_key(_path_key(path), seed=seed, domain=domain)


def _stable_rank_key(path_key: str, *, seed: int, domain: int) -> bytes:
    value = f"{seed}\0{domain}\0{path_key}".encode("utf-8", errors="surrogatepass")
    return hashlib.blake2b(value, digest_size=16).digest()


def _path_key(path: str) -> str:
    # Paths are database identities here; resolving every M: entry would touch
    # remote/storage metadata millions of times during manifest construction.
    return os.path.normcase(os.path.abspath(path))


def _sequence_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
