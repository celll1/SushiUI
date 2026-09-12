"""Checkpoint sidecar and Danbooru resume-state contracts."""

import json
from pathlib import Path

from core.tagger.tagger_trainer import (
    _prune_step_checkpoints,
    _save_checkpoint_artifacts,
    _save_danbooru_runtime_state,
)


class _Vocabulary:
    num_tags = 2
    idx_to_tag = {0: "a", 1: "b"}

    def to_dict(self):
        return {"idx_to_tag": {"0": "a", "1": "b"}}


class _TagMetrics:
    has_data = True

    def save(self, path, **kwargs):
        Path(path).write_text(json.dumps(kwargs, default=str), encoding="utf-8")


class _OodAccumulator:
    n_seen = 10

    def finalize(self, path):
        Path(path).write_text("reference", encoding="utf-8")

    def save_reservoir(self, path):
        Path(path).write_text("reservoir", encoding="utf-8")


class _DanbooruBuffer:
    def get_metrics(self):
        return {"downloaded": 3}

    def snapshot_dynamic_tags(self):
        return ["new_tag"]

    def snapshot_cooc_active_tags(self):
        return ["cooc_tag"]

    def snapshot_query_tags(self):
        return ["query_tag"]

    def snapshot_epoch_progress(self):
        return {"collected": 7}


def test_checkpoint_artifacts_keep_the_complete_companion_set(tmp_path):
    _save_checkpoint_artifacts(
        _TagMetrics(),
        _OodAccumulator(),
        _Vocabulary(),
        str(tmp_path),
        "step_000010",
        epoch_boundary=False,
        save_tag_metrics=True,
        hard_lo=0.25,
        hard_hi=0.75,
        calib_method="jeffreys",
        calib_eps=0.5,
        calib_prior_strength=10.0,
        save_ood_reference=True,
    )

    assert {path.name for path in tmp_path.iterdir()} == {
        "step_000010_vocabulary.json",
        "step_000010_tag_metrics.npz",
        "step_000010_ood_ref.npz",
        "latest_ood_reservoir.npz",
    }


def test_danbooru_runtime_state_keeps_all_resume_files(tmp_path):
    _save_danbooru_runtime_state(_DanbooruBuffer(), str(tmp_path), epoch=4)

    assert json.loads((tmp_path / "danbooru_metrics.json").read_text(encoding="utf-8")) == {"downloaded": 3}
    assert json.loads((tmp_path / "danbooru_dynamic_tags.json").read_text(encoding="utf-8")) == ["new_tag"]
    assert json.loads((tmp_path / "danbooru_cooc_active_tags.json").read_text(encoding="utf-8")) == ["cooc_tag"]
    assert json.loads((tmp_path / "danbooru_query_tags.json").read_text(encoding="utf-8")) == ["query_tag"]
    assert json.loads((tmp_path / "danbooru_epoch_progress.json").read_text(encoding="utf-8")) == {
        "collected": 7,
        "epoch": 4,
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_pruning_removes_the_complete_step_bundle(tmp_path):
    suffixes = (
        ".safetensors",
        "_metadata.json",
        "_state.json",
        "_optimizer.pt",
        "_vocabulary.json",
        "_tag_metrics.npz",
        "_ood_ref.npz",
    )
    for step in (10, 20):
        name = f"step_{step:06d}"
        for suffix in suffixes:
            (tmp_path / f"{name}{suffix}").write_text("x", encoding="utf-8")

    _prune_step_checkpoints(str(tmp_path), keep_last_n=1)

    assert not list(tmp_path.glob("step_000010*"))
    assert {path.name for path in tmp_path.glob("step_000020*")} == {
        f"step_000020{suffix}" for suffix in suffixes
    }
