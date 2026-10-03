"""Style resume decision and split pinning (design §9; T10)."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_dataset import (
    TRAIN, SplitParams, artist_split_unit, build_style_index, stable_item_key,
)
from core.style_encoder.style_labels import StyleLabelResolver
from core.style_encoder.style_resume import (
    ResumeKey, SplitParamsMismatch, SplitSnapshot, plan_resume, start_snapshot,
)
from core.style_encoder.style_sampler import StyleBatchSampler
from core.style_encoder.style_views import ViewParams
from core.tagger.tag_alias_resolver import TagAliasResolver

SPLIT = SplitParams.create(val_artist_ratio=0.3, val_image_ratio=0.2, min_images_per_artist=2, seed=7)
VIEWS = ViewParams.create(mask_ratio_min=0.3, mask_ratio_max=0.75, max_num_patches=16, seed=3)


class _AllArtists:
    def get_tag_group(self, tag):
        return "Artist"


def _index(items, aliases=None, pins=None, split=SPLIT):
    resolver = StyleLabelResolver(_AllArtists(), None, TagAliasResolver(aliases or {}), (), ("original",))
    records = [(i, stable_item_key(1, f"{i}.png"), resolver.resolve(None, name)) for i, name in items]
    return build_style_index(records, split, np.zeros(0, dtype=np.int64) if pins is None else pins.keys)


def _sampler(index, **overrides):
    values = dict(artists_per_batch=2, images_per_artist=2, same_copyright_batch_ratio=0.5,
                  artist_sampling_power=0.5, seed=3)
    values.update(overrides)
    return StyleBatchSampler(index, **values)


def _alias(name):
    return name.replace(" ", "_")


def _names(count):
    names = [f"artist {a}" for a in range(count)]
    held_out = [n for n in names if artist_split_unit(n, SPLIT.seed) < SPLIT.val_artist_ratio]
    train = [n for n in names if n not in held_out]
    return train, held_out, [(i, n) for i, n in enumerate(n for n in train + held_out for _ in range(4))]


def test_resume_decision():
    _, _, items = _names(20)
    sampler = _sampler(_index(items))
    snapshot = start_snapshot(sampler.index)
    saved = ResumeKey.from_state(ResumeKey.of(sampler, VIEWS).to_state())
    length = sampler.epoch_length

    def plan(step=1, key=saved):
        result = plan_resume(key, snapshot, sampler, VIEWS, 3, step)
        return result.mode, result.epoch, result.step_in_epoch

    assert plan() == ("exact", 3, 1)
    assert plan(step=length) == ("exact", 4, 0)
    assert plan(key=replace(saved, index_hash="other")) == ("epoch_boundary", 4, 0)
    assert plan(step=0, key=replace(saved, index_hash="other")) == ("epoch_boundary", 3, 0)
    for changed in (dict(artists_per_batch=3), dict(images_per_artist=1), dict(same_copyright_batch_ratio=0.25),
                    dict(artist_sampling_power=1.0), dict(seed=4),
                    dict(sampler_version=saved.sampler.sampler_version + 1),
                    dict(numpy_version=saved.sampler.numpy_version + ".post1")):
        assert plan(key=replace(saved, sampler=replace(saved.sampler, **changed))) == ("epoch_boundary", 4, 0)
    for changed in (dict(mask_ratio_min=0.2), dict(mask_ratio_max=0.5), dict(max_num_patches=32), dict(seed=4),
                    dict(views_version=VIEWS.views_version + 1)):
        assert plan(key=replace(saved, views=replace(saved.views, **changed))) == ("epoch_boundary", 4, 0)
    for changed in (dict(val_artist_ratio=0.25), dict(val_image_ratio=0.1), dict(min_images_per_artist=3),
                    dict(seed=8), dict(name_normalization_version=0)):
        with pytest.raises(SplitParamsMismatch, match=next(iter(changed))):
            plan(key=replace(saved, split=replace(saved.split, **changed)))


def test_sampled_artist_renamed_into_held_out_name_is_pinned_to_train():
    train, held_out, items = _names(40)
    first = _sampler(_index(items))
    snapshot = start_snapshot(first.index)
    assert np.isin(first.index.item_keys[first.index.item_artist == first.index.artist_names.index(train[0])],
                   snapshot.keys).any()

    aliases = {_alias(train[0]): _alias(held_out[0])}
    with pytest.raises(ValueError, match="pinned_keys"):
        plan_resume(ResumeKey.of(first, VIEWS), snapshot, _sampler(_index(items, aliases)), VIEWS, 1, 1)

    after = _index(items, aliases, pins=snapshot)
    target = after.artist_names.index(held_out[0])
    assert not after.artist_is_val[target]
    assert (after.stats.forced_train_artists, after.stats.forced_train_items) == (1, 8)
    plan = plan_resume(ResumeKey.of(first, VIEWS), snapshot, _sampler(after), VIEWS, 1, 1)
    assert plan.mode == "epoch_boundary"
    assert np.array_equal(np.intersect1d(plan.next_snapshot.keys, snapshot.keys), snapshot.keys)


def test_never_sampled_artist_is_not_pinned():
    train, held_out, items = _names(40)
    small = next(f"small artist {k}" for k in range(1000)
                 if artist_split_unit(f"small artist {k}", SPLIT.seed) >= SPLIT.val_artist_ratio)
    items = items + [(10_000, small)]
    snapshot = start_snapshot(_index(items))
    assert stable_item_key(1, "10000.png") not in snapshot.keys

    after = _index(items, {_alias(small): _alias(held_out[0])}, pins=snapshot)
    assert after.artist_is_val[after.artist_names.index(held_out[0])]
    assert after.stats.forced_train_artists == 0


def test_removed_sampled_item_is_pinned_when_re_added(tmp_path):
    train, held_out, items = _names(40)
    first = _sampler(_index(items))
    snapshot = start_snapshot(first.index)
    gone = next(i for i, _ in items if stable_item_key(1, f"{i}.png") in snapshot.keys)
    without = [it for it in items if it[0] != gone]

    second = _sampler(_index(without, pins=snapshot))
    path = tmp_path / "split_snapshot"
    plan_resume(ResumeKey.of(first, VIEWS), snapshot, second, VIEWS, 1, 1).next_snapshot.save(path)
    assert (tmp_path / "split_snapshot.npz").exists()
    history = SplitSnapshot.load(path)
    assert stable_item_key(1, f"{gone}.png") in history.keys

    third_index = _index(without + [(gone, held_out[0])], pins=history)
    third = _sampler(third_index)
    plan = plan_resume(ResumeKey.of(second, VIEWS), history, third, VIEWS, 1, 1)
    row = int(np.flatnonzero(third_index.item_ids == gone)[0])
    assert third_index.item_split[row] == TRAIN
    assert not third_index.artist_is_val[third_index.artist_names.index(held_out[0])]
    assert plan.mode == "epoch_boundary"


def test_snapshot_load_rejects_incomplete_files(tmp_path):
    np.savez_compressed(tmp_path / "no_version.npz", keys=np.zeros(0, dtype=np.int64))
    np.savez_compressed(tmp_path / "no_keys.npz", version=np.int64(2))
    for name in ("no_version", "no_keys"):
        with pytest.raises(ValueError, match="missing"):
            SplitSnapshot.load(tmp_path / name)
