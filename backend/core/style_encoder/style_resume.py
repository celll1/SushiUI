"""Resume decision for style training (design §9 table)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Union

import numpy as np

from .style_dataset import TRAIN, SplitParams, StyleIndex
from .style_sampler import SamplerParams, StyleBatchSampler

_SNAPSHOT_VERSION = 2


class SplitParamsMismatch(ValueError):
    pass


def _npz_path(path: Union[str, os.PathLike]) -> str:
    path = os.fspath(path)
    return path if path.endswith(".npz") else path + ".npz"


@dataclass(frozen=True, eq=False)
class SplitSnapshot:
    """item_keys ever sampled in this run (sorted, unique int64). Only grows: removed items stay, so
    a trained image re-added under a held-out artist name is still pinned to train."""

    keys: np.ndarray

    def __post_init__(self) -> None:
        if self.keys.dtype != np.int64 or self.keys.ndim != 1:
            raise ValueError("split snapshot keys must be a 1-D int64 array")
        if self.keys.size > 1 and not (self.keys[1:] > self.keys[:-1]).all():
            raise ValueError("split snapshot keys must be sorted and unique")

    @classmethod
    def empty(cls) -> "SplitSnapshot":
        return cls(np.zeros(0, dtype=np.int64))

    def merged_with(self, index: StyleIndex) -> "SplitSnapshot":
        return SplitSnapshot(np.union1d(self.keys, index.sampled_keys()).astype(np.int64))

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Writes `path`, adding `.npz` when missing (as `load` does)."""
        np.savez_compressed(_npz_path(path), version=np.int64(_SNAPSHOT_VERSION), keys=self.keys)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "SplitSnapshot":
        with np.load(_npz_path(path), allow_pickle=False) as data:
            missing = [name for name in ("version", "keys") if name not in data.files]
            if missing:
                raise ValueError(f"split snapshot {os.fspath(path)!r} is missing {missing}")
            version = int(data["version"])
            if version != _SNAPSHOT_VERSION:
                raise ValueError(f"unsupported split snapshot version {version}")
            return cls(data["keys"])


@dataclass(frozen=True)
class ResumeKey:
    index_hash: str
    split: SplitParams
    sampler: SamplerParams

    @classmethod
    def of(cls, sampler: StyleBatchSampler) -> "ResumeKey":
        return cls(sampler.index.index_hash, sampler.index.split_params, sampler.params)

    def to_state(self) -> dict:
        return {"index_hash": self.index_hash, "split": self.split.to_state(), "sampler": self.sampler.to_state()}

    @classmethod
    def from_state(cls, state: dict) -> "ResumeKey":
        return cls(str(state["index_hash"]), SplitParams.from_state(state["split"]),
                   SamplerParams.from_state(state["sampler"]))


@dataclass(frozen=True, eq=False)
class ResumePlan:
    mode: str
    epoch: int
    step_in_epoch: int
    next_snapshot: SplitSnapshot


def start_snapshot(index: StyleIndex) -> SplitSnapshot:
    """Snapshot to save for a fresh run (built with empty pinned_keys)."""
    return SplitSnapshot.empty().merged_with(index)


def plan_resume(saved: ResumeKey, saved_snapshot: SplitSnapshot, sampler: StyleBatchSampler,
                saved_epoch: int, saved_step_in_epoch: int) -> ResumePlan:
    """`sampler.index` must be built with pinned_keys=saved_snapshot.keys. `saved_step_in_epoch` is the
    next batch to draw. Save `next_snapshot`, never `saved_snapshot`."""
    current = ResumeKey.of(sampler)
    if saved.split != current.split:
        diff = {f.name: (getattr(saved.split, f.name), getattr(current.split, f.name))
                for f in fields(SplitParams) if getattr(saved.split, f.name) != getattr(current.split, f.name)}
        raise SplitParamsMismatch(
            f"Style split parameters changed since the checkpoint (saved, current): {diff}. Resuming would "
            f"move existing artists between train and held-out validation; start a new run instead.")
    index = sampler.index
    pinned_rows = np.isin(index.item_keys, saved_snapshot.keys)
    if (index.item_split[pinned_rows] != TRAIN).any():
        raise ValueError("style index was not built with pinned_keys from the saved split snapshot")
    next_snapshot = saved_snapshot.merged_with(index)
    epoch_length = sampler.epoch_length
    # The full numpy version is compared: NEP 19 gives no stream guarantee across releases.
    if saved.index_hash == current.index_hash and saved.sampler == current.sampler:
        if not 0 <= saved_step_in_epoch <= epoch_length:
            raise ValueError(f"saved step {saved_step_in_epoch} outside epoch of length {epoch_length}")
        if saved_step_in_epoch < epoch_length:
            return ResumePlan("exact", saved_epoch, saved_step_in_epoch, next_snapshot)
        return ResumePlan("exact", saved_epoch + 1, 0, next_snapshot)
    # An epoch that had not started yet is not skipped.
    return ResumePlan("epoch_boundary", saved_epoch + (1 if saved_step_in_epoch > 0 else 0), 0, next_snapshot)
