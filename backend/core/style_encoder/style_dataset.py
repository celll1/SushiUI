"""Artist index, hash split and confound score (design §5, §7)."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import asdict, dataclass, fields
from typing import Iterable, Optional

import numpy as np

from .style_labels import StyleLabels

# Bump when either split hash below changes. Name changes from alias/comma/taglist updates are
# absorbed by pinning sampled items (build_style_index).
NAME_NORMALIZATION_VERSION = 1
_INDEX_HASH_DOMAIN = b"style-index-v3"

TRAIN, VAL_IMAGE, VAL_ARTIST = 0, 1, 2


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def fields_from_state(cls, state: dict, what: str) -> dict:
    missing = [f.name for f in fields(cls) if f.name not in state]
    if missing:
        raise ValueError(f"saved {what} state is missing {missing}")
    return {f.name: state[f.name] for f in fields(cls)}


@dataclass(frozen=True)
class SplitParams:
    val_artist_ratio: float
    val_image_ratio: float
    min_images_per_artist: int
    seed: int
    name_normalization_version: int

    def __post_init__(self) -> None:
        for name in ("val_artist_ratio", "val_image_ratio"):
            value = getattr(self, name)
            if not _is_number(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value!r}")
        if not _is_int(self.min_images_per_artist) or self.min_images_per_artist < 1:
            raise ValueError(f"min_images_per_artist must be an integer >= 1, got {self.min_images_per_artist!r}")
        if not _is_int(self.seed) or self.seed < 0:
            raise ValueError(f"split seed must be an integer >= 0, got {self.seed!r}")
        if not _is_int(self.name_normalization_version):
            raise ValueError("name_normalization_version must be an integer")

    @classmethod
    def create(cls, val_artist_ratio: float, val_image_ratio: float, min_images_per_artist: int,
               seed: int) -> "SplitParams":
        return cls(val_artist_ratio, val_image_ratio, min_images_per_artist, seed, NAME_NORMALIZATION_VERSION)

    def to_state(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state(cls, state: dict) -> "SplitParams":
        return cls(**fields_from_state(cls, state, "split parameter"))


def stable_item_key(dataset_id: int, relative_path: str) -> int:
    """Signed 64-bit key from (dataset_id, relative image path). DatasetItem.id is not stable: SQLite
    reuses the highest ids after deletes."""
    path = unicodedata.normalize("NFC", relative_path.replace("\\", "/"))
    digest = hashlib.blake2b(digest_size=8, person=b"style-item")
    for part in (str(int(dataset_id)).encode("ascii"), path.encode("utf-8")):
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return int.from_bytes(digest.digest(), "little", signed=True)


def artist_split_unit(name: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}\0{name}".encode("utf-8"), digest_size=8,
                             person=b"style-artist").digest()
    return (int.from_bytes(digest, "little") >> 11) / 2.0 ** 53


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def splitmix64(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.uint64) + _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


def image_split_units(item_keys: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64 of (item_key xor seed word); vectorized because it runs over every indexed item."""
    seed_word = int.from_bytes(hashlib.blake2b(str(seed).encode("utf-8"), digest_size=8,
                                               person=b"style-image").digest(), "little")
    z = splitmix64(np.ascontiguousarray(item_keys, dtype=np.int64).view(np.uint64) ^ np.uint64(seed_word))
    return (z >> np.uint64(11)).astype(np.float64) / 2.0 ** 53


def csr_group(keys: np.ndarray, values: np.ndarray, n_keys: int) -> tuple[np.ndarray, np.ndarray]:
    """indptr [n_keys + 1] and `values` sorted by (key, value)."""
    order = np.lexsort((values, keys))
    indptr = np.zeros(n_keys + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=n_keys), out=indptr[1:])
    return indptr, values[order]


def gather_csr(indptr: np.ndarray, values: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(entry_row, entry_value) for every CSR entry of `rows`."""
    counts = indptr[rows + 1] - indptr[rows]
    total = int(counts.sum())
    entry_rows = np.repeat(rows, counts)
    offsets = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    return entry_rows, values[np.repeat(indptr[rows], counts) + offsets]


class _Interner:
    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def __call__(self, name: str) -> int:
        found = self.ids.get(name)
        if found is None:
            found = self.ids[name] = len(self.ids)
        return found

    def finalize(self) -> tuple[tuple[str, ...], np.ndarray]:
        names = list(self.ids)
        order = sorted(range(len(names)), key=names.__getitem__)
        remap = np.empty(len(names), dtype=np.int64)
        remap[order] = np.arange(len(names), dtype=np.int64)
        return tuple(names[i] for i in order), remap


@dataclass(frozen=True)
class IndexStats:
    records: int
    items_without_artist: int
    items_with_multiple_artists: int
    indexed_items: int
    artists: int
    val_artists: int
    eligible_train_artists: int
    train_items: int
    train_items_of_ineligible_artists: int
    val_image_items: int
    val_artist_items: int
    pinned_items: int
    forced_train_artists: int
    forced_train_items: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, eq=False)
class StyleIndex:
    """Rows are indexed items sorted by item_id; names are interned in sorted order, so the
    structure is a pure function of the record set, not of input order."""

    split_params: SplitParams
    item_ids: np.ndarray
    item_keys: np.ndarray
    item_artist: np.ndarray
    item_split: np.ndarray
    has_signature: np.ndarray
    artist_names: tuple[str, ...]
    copyright_names: tuple[str, ...]
    character_names: tuple[str, ...]
    artist_is_val: np.ndarray
    artist_train_count: np.ndarray
    artist_eligible: np.ndarray
    artist_indptr: np.ndarray
    artist_items: np.ndarray
    item_copyright_indptr: np.ndarray
    item_copyrights: np.ndarray
    item_character_indptr: np.ndarray
    item_characters: np.ndarray
    copyright_pair_indptr: np.ndarray
    pair_copyright: np.ndarray
    pair_artist: np.ndarray
    pair_indptr: np.ndarray
    pair_items: np.ndarray
    index_hash: str
    stats: IndexStats

    def sampled_keys(self) -> np.ndarray:
        """Sorted item_keys the sampler can draw: TRAIN split of eligible artists."""
        rows = (self.item_split == TRAIN) & self.artist_eligible[self.item_artist]
        return np.sort(self.item_keys[rows])

    def content_ids(self, row: int) -> np.ndarray:
        """Character ∪ Copyright ids; characters are offset by the copyright count."""
        cps = self.item_copyrights[self.item_copyright_indptr[row]:self.item_copyright_indptr[row + 1]]
        chars = self.item_characters[self.item_character_indptr[row]:self.item_character_indptr[row + 1]]
        return np.concatenate([cps.astype(np.int64), chars.astype(np.int64) + len(self.copyright_names)])


def build_style_index(records: Iterable[tuple[int, int, StyleLabels]], split_params: SplitParams,
                      pinned_keys: np.ndarray, force_train_keys: Optional[np.ndarray] = None) -> StyleIndex:
    """Records are (item_id, item_key, labels). An artist owning a pinned key (an item sampled
    earlier in the run) stays on the train side even if its name hashes to held-out."""
    artists, copyrights, characters = _Interner(), _Interner(), _Interner()
    item_ids: list[int] = []
    item_keys: list[int] = []
    artist_tmp: list[int] = []
    signature: list[bool] = []
    cp_counts: list[int] = []
    cp_flat: list[int] = []
    ch_counts: list[int] = []
    ch_flat: list[int] = []
    n_records = no_artist = multi_artist = 0
    for item_id, item_key, labels in records:
        n_records += 1
        if len(labels.artists) != 1:
            if labels.artists:
                multi_artist += 1
            else:
                no_artist += 1
            continue
        (artist,) = labels.artists
        item_ids.append(int(item_id))
        item_keys.append(int(item_key))
        artist_tmp.append(artists(artist))
        signature.append(bool(labels.has_signature))
        cp_counts.append(len(labels.copyrights))
        cp_flat.extend(copyrights(name) for name in labels.copyrights)
        ch_counts.append(len(labels.characters))
        ch_flat.extend(characters(name) for name in labels.characters)

    n = len(item_ids)
    ids = np.asarray(item_ids, dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    duplicate = np.flatnonzero(ids[1:] == ids[:-1])
    if duplicate.size:
        raise ValueError(f"duplicate item_id in style index: {int(ids[duplicate[0]])}")
    keys = np.asarray(item_keys, dtype=np.int64)[order]
    sorted_keys = np.sort(keys)
    duplicate = np.flatnonzero(sorted_keys[1:] == sorted_keys[:-1])
    if duplicate.size:
        raise ValueError(f"duplicate item_key in style index: {int(sorted_keys[duplicate[0]])}")
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)

    artist_names, artist_remap = artists.finalize()
    copyright_names, cp_remap = copyrights.finalize()
    character_names, ch_remap = characters.finalize()
    n_artists = len(artist_names)
    item_artist = artist_remap[np.asarray(artist_tmp, dtype=np.int64)][order].astype(np.int32)
    has_signature = np.asarray(signature, dtype=bool)[order]

    def item_csr(counts: list[int], flat: list[int], remap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = inverse[np.repeat(np.arange(n, dtype=np.int64), np.asarray(counts, dtype=np.int64))]
        cols = remap[np.asarray(flat, dtype=np.int64)].astype(np.int32)
        return csr_group(rows, cols, n)

    item_copyright_indptr, item_copyrights = item_csr(cp_counts, cp_flat, cp_remap)
    item_character_indptr, item_characters = item_csr(ch_counts, ch_flat, ch_remap)

    artist_is_val = np.fromiter(
        (artist_split_unit(name, split_params.seed) < split_params.val_artist_ratio for name in artist_names),
        dtype=bool, count=n_artists)
    pinned_rows = np.isin(keys, np.asarray(pinned_keys, dtype=np.int64))
    pinned_artists = np.zeros(n_artists, dtype=bool)
    pinned_artists[item_artist[pinned_rows]] = True
    forced = pinned_artists & artist_is_val
    artist_is_val &= ~forced
    image_val = image_split_units(keys, split_params.seed) < split_params.val_image_ratio
    if force_train_keys is not None:
        image_val[np.isin(keys, force_train_keys)] = False
    item_split = np.where(artist_is_val[item_artist], VAL_ARTIST,
                          np.where(image_val, VAL_IMAGE, TRAIN)).astype(np.int8)
    artist_train_count = np.bincount(item_artist[item_split == TRAIN], minlength=n_artists)
    artist_eligible = ~artist_is_val & (artist_train_count >= split_params.min_images_per_artist)

    artist_indptr, artist_items = csr_group(item_artist, np.arange(n, dtype=np.int64), n_artists)

    e_rows, e_cps = gather_csr(item_copyright_indptr, item_copyrights, np.arange(n, dtype=np.int64))
    e_artists = item_artist[e_rows]
    pair_order = np.lexsort((e_rows, e_artists, e_cps))
    s_cps, s_artists = e_cps[pair_order], e_artists[pair_order]
    pair_items = e_rows[pair_order]
    starts = np.flatnonzero(np.r_[True, (s_cps[1:] != s_cps[:-1]) | (s_artists[1:] != s_artists[:-1])]) \
        if s_cps.size else np.zeros(0, dtype=np.int64)
    pair_copyright = s_cps[starts]
    pair_artist = s_artists[starts]
    pair_indptr = np.append(starts, s_cps.size).astype(np.int64)
    copyright_pair_indptr = np.searchsorted(pair_copyright, np.arange(len(copyright_names) + 1), side="left")

    digest = hashlib.blake2b(_INDEX_HASH_DOMAIN, digest_size=16)

    def put(data: bytes) -> None:
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)

    put(ids.astype("<i8").tobytes())
    put(keys.astype("<i8").tobytes())
    put(item_artist.astype("<i4").tobytes())
    put(item_split.astype("i1").tobytes())
    # Copyrights and characters drive batch composition, so exact resume must see them too.
    for indptr, values in ((item_copyright_indptr, item_copyrights), (item_character_indptr, item_characters)):
        put(indptr.astype("<i8").tobytes())
        put(values.astype("<i4").tobytes())
    for names in (artist_names, copyright_names, character_names):
        digest.update(len(names).to_bytes(8, "little"))
        for name in names:
            put(name.encode("utf-8"))

    train_rows = item_split == TRAIN
    eligible_rows = artist_eligible[item_artist]
    stats = IndexStats(
        records=n_records,
        items_without_artist=no_artist,
        items_with_multiple_artists=multi_artist,
        indexed_items=n,
        artists=n_artists,
        val_artists=int(artist_is_val.sum()),
        eligible_train_artists=int(artist_eligible.sum()),
        train_items=int((train_rows & eligible_rows).sum()),
        train_items_of_ineligible_artists=int((train_rows & ~eligible_rows).sum()),
        val_image_items=int((item_split == VAL_IMAGE).sum()),
        val_artist_items=int((item_split == VAL_ARTIST).sum()),
        pinned_items=int(pinned_rows.sum()),
        forced_train_artists=int(forced.sum()),
        forced_train_items=int(forced[item_artist].sum()),
    )
    return StyleIndex(
        split_params=split_params, item_ids=ids, item_keys=keys, item_artist=item_artist, item_split=item_split,
        has_signature=has_signature, artist_names=artist_names, copyright_names=copyright_names,
        character_names=character_names, artist_is_val=artist_is_val,
        artist_train_count=artist_train_count, artist_eligible=artist_eligible,
        artist_indptr=artist_indptr, artist_items=artist_items,
        item_copyright_indptr=item_copyright_indptr, item_copyrights=item_copyrights,
        item_character_indptr=item_character_indptr, item_characters=item_characters,
        copyright_pair_indptr=copyright_pair_indptr, pair_copyright=pair_copyright,
        pair_artist=pair_artist, pair_indptr=pair_indptr, pair_items=pair_items,
        index_hash=digest.hexdigest(), stats=stats,
    )


def confound_scores(index: StyleIndex) -> np.ndarray:
    """s_a = c_a * e_a per artist (design §7), over all indexed items (train and val).

    c_a = |a's items with c*| / |a's items|, e_a = |a's items with c*| / |items with c*|, where c* is
    a's most frequent copyright (ties: lowest copyright id); 0 for artists with no copyright at all.
    """
    scores = np.zeros(len(index.artist_names), dtype=np.float64)
    if index.pair_artist.size == 0:
        return scores
    pair_count = np.diff(index.pair_indptr)
    copyright_total = np.bincount(index.item_copyrights, minlength=len(index.copyright_names))
    artist_total = np.diff(index.artist_indptr)
    order = np.lexsort((index.pair_copyright, -pair_count, index.pair_artist))
    sorted_artists = index.pair_artist[order]
    best = order[np.flatnonzero(np.r_[True, sorted_artists[1:] != sorted_artists[:-1]])]
    artist = index.pair_artist[best]
    count = pair_count[best].astype(np.float64)
    scores[artist] = (count / artist_total[artist]) * (count / copyright_total[index.pair_copyright[best]])
    return scores
