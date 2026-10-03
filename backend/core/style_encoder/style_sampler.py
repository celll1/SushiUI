"""P x K artist batch sampler over the train split (design §6)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Iterator, Optional, Sequence

import numpy as np

from .style_dataset import TRAIN, StyleIndex, _is_int, _is_number, csr_group, fields_from_state, gather_csr

# Bump when batch composition changes for the same (index, params, seed, epoch, step).
SAMPLER_VERSION = 1
# Rejection draws per pick before the exact O(range) fallback; both sample the same distribution.
_REJECTION_DRAWS_PER_PICK = 8


@dataclass(frozen=True)
class SamplerParams:
    artists_per_batch: int
    images_per_artist: int
    same_copyright_batch_ratio: float
    artist_sampling_power: float
    seed: int
    sampler_version: int
    numpy_version: str

    def __post_init__(self) -> None:
        for name in ("artists_per_batch", "images_per_artist"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        ratio = self.same_copyright_batch_ratio
        if not _is_number(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError(f"same_copyright_batch_ratio must be in [0, 1], got {ratio!r}")
        power = self.artist_sampling_power
        if not _is_number(power) or not math.isfinite(power):
            raise ValueError(f"artist_sampling_power must be a finite number, got {power!r}")
        if not _is_int(self.seed) or self.seed < 0:
            raise ValueError(f"sampler seed must be an integer >= 0, got {self.seed!r}")
        if not _is_int(self.sampler_version):
            raise ValueError("sampler_version must be an integer")
        if not isinstance(self.numpy_version, str):
            raise ValueError("numpy_version must be a string")

    @classmethod
    def create(cls, artists_per_batch: int, images_per_artist: int, same_copyright_batch_ratio: float,
               artist_sampling_power: float, seed: int) -> "SamplerParams":
        return cls(artists_per_batch, images_per_artist, same_copyright_batch_ratio, artist_sampling_power,
                   seed, SAMPLER_VERSION, np.__version__)

    def to_state(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state(cls, state: dict) -> "SamplerParams":
        return cls(**fields_from_state(cls, state, "sampler parameter"))


@dataclass(frozen=True, eq=False)
class StyleBatch:
    epoch: int
    step: int
    artists: np.ndarray
    rows: np.ndarray
    copyright: Optional[int]


def _draw_distinct(rng: np.random.Generator, lo: int, hi: int, count: int,
                   cum: Optional[np.ndarray] = None, exclude: Sequence[int] = ()) -> list[int]:
    """`count` distinct ints from [lo, hi) without replacement, weighted by the inclusive prefix
    sums `cum` (uniform when None)."""
    seen = set(exclude)
    picked: list[int] = []
    width = hi - lo
    if cum is not None:
        base = float(cum[lo - 1]) if lo > 0 else 0.0
        total = float(cum[hi - 1]) - base
        window = cum[lo:hi]
    attempts = _REJECTION_DRAWS_PER_PICK * count
    while len(picked) < count and attempts > 0:
        attempts -= 1
        if cum is None:
            value = lo + int(rng.integers(width))
        else:
            value = lo + min(int(np.searchsorted(window, base + rng.random() * total, side="right")), width - 1)
        if value not in seen:
            seen.add(value)
            picked.append(value)
    if len(picked) < count:
        rest = np.array([v for v in range(lo, hi) if v not in seen], dtype=np.int64)
        need = count - len(picked)
        if rest.size < need:
            raise RuntimeError(f"cannot draw {count} distinct values from a range of {width}")
        p = None
        if cum is not None:
            weights = np.diff(np.concatenate(([base], window)))[rest - lo]
            p = weights / weights.sum()
        picked.extend(int(v) for v in rng.choice(rest, size=need, replace=False, p=p))
    return picked


class StyleBatchSampler:
    """Batches are a pure function of (seed, epoch, step), so resuming at a position needs no RNG
    state beyond those three integers."""

    def __init__(self, index: StyleIndex, artists_per_batch: int, images_per_artist: int,
                 same_copyright_batch_ratio: float, artist_sampling_power: float, seed: int) -> None:
        self.params = SamplerParams.create(artists_per_batch, images_per_artist, same_copyright_batch_ratio,
                                           artist_sampling_power, seed)
        min_images = index.split_params.min_images_per_artist
        if images_per_artist > min_images:
            raise ValueError(f"images_per_artist ({images_per_artist}) must not exceed "
                             f"min_images_per_artist ({min_images})")
        self.index = index
        self._p = int(artists_per_batch)
        self._k = int(images_per_artist)
        self._ratio = float(same_copyright_batch_ratio)
        self._power = float(artist_sampling_power)
        self._seed = int(seed)

        self._eligible = np.flatnonzero(index.artist_eligible)
        n_eligible = self._eligible.size
        if n_eligible < self._p:
            raise ValueError(f"only {n_eligible} eligible train artists; artists_per_batch is {self._p}")
        local = np.full(len(index.artist_names), -1, dtype=np.int64)
        local[self._eligible] = np.arange(n_eligible, dtype=np.int64)
        rows = np.flatnonzero((index.item_split == TRAIN) & index.artist_eligible[index.item_artist])
        row_local = np.full(index.item_ids.size, -1, dtype=np.int64)
        row_local[rows] = local[index.item_artist[rows]]

        self._artist_row_indptr, self._artist_rows = csr_group(row_local[rows], rows, n_eligible)
        artist_weights = np.diff(self._artist_row_indptr).astype(np.float64) ** self._power
        self._artist_cum = np.cumsum(artist_weights)
        self.train_images = int(rows.size)
        self.epoch_length = self.train_images // (self._p * self._k)
        if self.epoch_length == 0:
            raise ValueError(f"{self.train_images} train images cannot fill one batch of "
                             f"{self._p} x {self._k}")

        # Character groups per artist; a character-less item is its own group so those items are
        # not collapsed into one.
        ch_rows, ch_ids = gather_csr(index.item_character_indptr, index.item_characters, rows)
        bare = rows[index.item_character_indptr[rows + 1] == index.item_character_indptr[rows]]
        g_rows = np.concatenate([ch_rows, bare])
        g_keys = np.concatenate([ch_ids.astype(np.int64), len(index.character_names) + bare])
        g_artists = row_local[g_rows]
        order = np.lexsort((g_rows, g_keys, g_artists))
        g_rows, g_keys, g_artists = g_rows[order], g_keys[order], g_artists[order]
        starts = np.flatnonzero(np.r_[True, (g_artists[1:] != g_artists[:-1]) | (g_keys[1:] != g_keys[:-1])])
        self._group_rows = g_rows
        self._group_indptr = np.append(starts, g_rows.size).astype(np.int64)
        self._group_cum = np.cumsum(np.diff(self._group_indptr).astype(np.float64))
        self._artist_group_indptr = np.searchsorted(g_artists[starts], np.arange(n_eligible + 1), side="left")

        cp_rows, cp_ids = gather_csr(index.item_copyright_indptr, index.item_copyrights, rows)
        cp_artists = row_local[cp_rows]
        order = np.lexsort((cp_rows, cp_artists, cp_ids))
        cp_rows, cp_ids, cp_artists = cp_rows[order], cp_ids[order], cp_artists[order]
        self._cp_rows = cp_rows
        if cp_rows.size:
            starts = np.flatnonzero(np.r_[True, (cp_ids[1:] != cp_ids[:-1]) | (cp_artists[1:] != cp_artists[:-1])])
        else:
            starts = np.zeros(0, dtype=np.int64)
        ends = np.append(starts[1:], cp_rows.size)
        keep = (ends - starts) >= self._k
        pair_lo, pair_hi = starts[keep], ends[keep]
        pair_cp, pair_artist = cp_ids[pair_lo], cp_artists[pair_lo]
        cps, counts = np.unique(pair_cp, return_counts=True)
        qualifying = counts >= self._p
        select = np.repeat(qualifying, counts)
        self._q_copyrights = cps[qualifying].astype(np.int64)
        self._pair_lo, self._pair_hi = pair_lo[select], pair_hi[select]
        self._pair_artist = pair_artist[select]
        self._cp_pair_indptr = np.append(0, np.cumsum(counts[qualifying])).astype(np.int64)

    @property
    def qualifying_copyrights(self) -> np.ndarray:
        return self._q_copyrights

    def batch(self, epoch: int, step: int) -> StyleBatch:
        if not _is_int(epoch) or epoch < 0:
            raise ValueError(f"epoch must be an integer >= 0, got {epoch!r}")
        if not _is_int(step) or not 0 <= step < self.epoch_length:
            raise IndexError(f"step {step!r} outside epoch of length {self.epoch_length}")
        rng = np.random.default_rng(np.random.SeedSequence([self._seed, int(epoch), int(step)]))
        if rng.random() < self._ratio and self._q_copyrights.size:
            return self._same_copyright_batch(rng, epoch, step)
        locals_ = _draw_distinct(rng, 0, self._eligible.size, self._p, cum=self._artist_cum)
        rows = np.array([self._diverse_rows(rng, a) for a in locals_], dtype=np.int64)
        return StyleBatch(epoch, step, self._eligible[locals_], rows, None)

    def epoch_batches(self, epoch: int, start_step: int) -> Iterator[StyleBatch]:
        for step in range(start_step, self.epoch_length):
            yield self.batch(epoch, step)

    def _same_copyright_batch(self, rng: np.random.Generator, epoch: int, step: int) -> StyleBatch:
        q = int(rng.integers(self._q_copyrights.size))
        pairs = _draw_distinct(rng, int(self._cp_pair_indptr[q]), int(self._cp_pair_indptr[q + 1]), self._p)
        rows = np.array([
            self._cp_rows[_draw_distinct(rng, int(self._pair_lo[p]), int(self._pair_hi[p]), self._k)]
            for p in pairs
        ], dtype=np.int64)
        artists = self._eligible[self._pair_artist[pairs]]
        return StyleBatch(epoch, step, artists, rows, int(self._q_copyrights[q]))

    def _diverse_rows(self, rng: np.random.Generator, artist: int) -> list[int]:
        glo, ghi = int(self._artist_group_indptr[artist]), int(self._artist_group_indptr[artist + 1])
        groups = _draw_distinct(rng, glo, ghi, min(self._k, ghi - glo), cum=self._group_cum)
        picked: list[int] = []
        seen: set[int] = set()
        for group in groups:
            lo, hi = int(self._group_indptr[group]), int(self._group_indptr[group + 1])
            for _ in range(_REJECTION_DRAWS_PER_PICK):
                row = int(self._group_rows[lo + int(rng.integers(hi - lo))])
                if row not in seen:
                    seen.add(row)
                    picked.append(row)
                    break
        if len(picked) < self._k:
            alo, ahi = int(self._artist_row_indptr[artist]), int(self._artist_row_indptr[artist + 1])
            own = self._artist_rows[alo:ahi]
            taken = [alo + int(np.searchsorted(own, row)) for row in picked]
            extra = _draw_distinct(rng, alo, ahi, self._k - len(picked), exclude=taken)
            picked.extend(int(self._artist_rows[pos]) for pos in extra)
        return picked
