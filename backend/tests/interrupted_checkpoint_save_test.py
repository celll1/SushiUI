"""A save killed mid-shard must not read as the newest checkpoint.

`ShardedSafetensorsWriter` writes `<stem>-partNNNNN.tmp.safetensors` and renames
them to `-NNNNN-of-NNNNN` only in `close()`, once the shard count is known. A
process killed between the first flush and that rename leaves the part files on
disk (`abort()` runs only when an exception unwinds). Those names match
`*_step_*.safetensors` and are NOT shard members by the `-NNNNN-of-NNNNN`
pattern, so before this guard a resume selected a fraction of one interrupted
save as its base -- observed on run127: 2 of 9 shards, 1012 missing keys, and a
refusal from the quantized-base gate that named the wrong cause.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.base_trainer import (  # noqa: E402
    _checkpoint_step_from_name,
    _interrupted_save_steps,
    _is_provisional_shard,
    _list_checkpoint_entries,
)


def _touch(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"0")
    return path


def test_a_provisional_shard_is_not_a_checkpoint_entry(tmp_path):
    _touch(tmp_path, "run_step_000100.safetensors")
    _touch(tmp_path, "run_step_000200-part00000.tmp.safetensors")
    _touch(tmp_path, "run_step_000200-part00001.tmp.safetensors")

    entries = _list_checkpoint_entries(tmp_path)
    assert [p.name for p in entries] == ["run_step_000100.safetensors"]
    latest = max(entries, key=lambda p: _checkpoint_step_from_name(p.name) or 0)
    assert _checkpoint_step_from_name(latest.name) == 100


def test_a_finished_sharded_save_is_still_found(tmp_path):
    """The rename happened: the index is the entry, its members are not."""
    _touch(tmp_path, "run_step_000200.safetensors.index.json")
    _touch(tmp_path, "run_step_000200-00001-of-00002.safetensors")
    _touch(tmp_path, "run_step_000200-00002-of-00002.safetensors")

    entries = _list_checkpoint_entries(tmp_path)
    assert [p.name for p in entries] == ["run_step_000200.safetensors.index.json"]


def test_the_interrupted_step_is_reported_with_its_shard_count(tmp_path):
    _touch(tmp_path, "run_step_000100.safetensors")
    _touch(tmp_path, "run_step_000200-part00000.tmp.safetensors")
    _touch(tmp_path, "run_step_000200-part00001.tmp.safetensors")
    _touch(tmp_path, "run_step_000300-part00000.tmp.safetensors")

    assert _interrupted_save_steps(tmp_path) == {200: 2, 300: 1}
    assert _interrupted_save_steps(tmp_path / "nothing-here") == {}


def test_only_the_writing_time_name_is_provisional():
    assert _is_provisional_shard("run_step_000200-part00000.tmp.safetensors")
    assert not _is_provisional_shard("run_step_000200-00001-of-00009.safetensors")
    assert not _is_provisional_shard("run_step_000200.safetensors")
    assert not _is_provisional_shard("run_step_000200.safetensors.index.json")
