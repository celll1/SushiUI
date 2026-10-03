"""Style index split stability, item keys and confound score (design §5, §7; T4, T5)."""

from fractions import Fraction
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_dataset import (
    TRAIN, VAL_ARTIST, VAL_IMAGE, SplitParams, build_style_index, confound_scores, stable_item_key,
)
from core.style_encoder.style_labels import StyleLabels
from core.style_encoder.style_sampler import SAMPLER_VERSION, SamplerParams


def _labels(artists, copyrights=(), characters=()):
    return StyleLabels(frozenset(artists), frozenset(characters), frozenset(copyrights), False)


def _params(**overrides):
    values = dict(val_artist_ratio=0.3, val_image_ratio=0.2, min_images_per_artist=2, seed=7)
    values.update(overrides)
    return SplitParams.create(**values)


def _build(records, params):
    keyed = [(i, stable_item_key(1, f"images/{i}.png"), labels) for i, labels in records]
    return build_style_index(keyed, params, pinned_keys=np.zeros(0, dtype=np.int64))


def _records(artist_ids, per_artist, first_item=0):
    item = first_item
    out = []
    for a in artist_ids:
        for _ in range(per_artist):
            out.append((item, _labels({f"artist {a}"})))
            item += 1
    return out


def _membership(index):
    val_artist = {name: bool(v) for name, v in zip(index.artist_names, index.artist_is_val)}
    split = {int(i): int(s) for i, s in zip(index.item_ids, index.item_split)}
    return val_artist, split


def test_split_membership_survives_additions_and_removals():
    base = _records(range(300), 3)
    before_val, before_split = _membership(_build(base, _params()))
    assert 0.2 < np.mean(list(before_val.values())) < 0.4

    kept = [r for r in base if r[0] % 5 != 0 and not r[1].artists & {"artist 3", "artist 17"}]
    added = _records(range(300, 600), 3, first_item=10_000) + _records(range(10), 2, first_item=20_000)
    after_val, after_split = _membership(_build(kept + added, _params()))

    for name, is_val in after_val.items():
        if name in before_val:
            assert is_val == before_val[name], name
    for item, split in after_split.items():
        if item in before_split:
            assert split == before_split[item], item
    assert any(s == VAL_ARTIST for s in after_split.values())
    assert any(s == TRAIN for s in after_split.values())


def test_index_is_independent_of_input_order_and_detects_data_changes():
    records = [(i, _labels({f"artist {i % 7}"}, {f"series {i % 3}"}, {f"char {i % 5}"})) for i in range(80)]
    shuffled = records[:]
    random.Random(1).shuffle(shuffled)
    a = _build(records, _params())
    b = _build(shuffled, _params())
    assert a.index_hash == b.index_hash
    for field in ("item_ids", "item_artist", "item_split", "item_copyrights", "item_characters",
                  "pair_items", "artist_items"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field
    assert _build(records[:-1], _params()).index_hash != a.index_hash
    last = records[-1][0]
    for changed in (_labels({"artist 0"}, {"series 1"}, {"char 4"}),
                    _labels({"artist 2"}, {"series 1"}, {"char 0"}),
                    _labels({"artist 2"}, {"series 0"}, {"char 4"})):
        assert _build(records[:-1] + [(last, changed)], _params()).index_hash != a.index_hash


def test_duplicate_item_id_or_key_is_refused():
    empty = np.zeros(0, dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate item_id"):
        build_style_index([(1, 10, _labels({"a"})), (1, 11, _labels({"b"}))], _params(), empty)
    with pytest.raises(ValueError, match="duplicate item_key"):
        build_style_index([(1, 10, _labels({"a"})), (2, 10, _labels({"b"}))], _params(), empty)


def test_stable_item_key():
    key = stable_item_key(3, "sub/dir/image.png")
    assert key == stable_item_key(3, "sub\\dir\\image.png")
    assert stable_item_key(1, "caf\u00e9.png") == stable_item_key(1, "cafe\u0301.png")
    assert key != stable_item_key(4, "sub/dir/image.png")
    assert stable_item_key(1, "2x.png") != stable_item_key(12, "x.png")
    assert -2 ** 63 <= key < 2 ** 63
    # Keys are persisted in split snapshots; this value must never change.
    assert key == 1071080219563900470


def test_confound_score_hand_computed():
    records = [
        (1, _labels({"a"}, {"x"})), (2, _labels({"a"}, {"x"})), (3, _labels({"a"}, {"x", "y"})),
        (4, _labels({"a"})),
        (5, _labels({"b"}, {"x"})), (6, _labels({"b"}, {"x"})), (7, _labels({"b"})), (8, _labels({"b"})),
        (9, _labels({"c"})), (10, _labels({"c"})), (11, _labels({"c"})),
        (12, _labels({"d"}, {"x"})), (13, _labels({"d"}, {"x"})), (14, _labels({"d"}, {"y"})),
        (15, _labels({"d"}, {"y"})),
        (16, _labels({"a", "b"}, {"x"})),
    ]
    index = _build(records, _params(val_artist_ratio=0.0))
    scores = dict(zip(index.artist_names, confound_scores(index)))
    # |x| = 7 (a:3, b:2, d:2; the two-artist item is not indexed), |y| = 3; d ties x/y and takes x.
    expected = {"a": Fraction(3, 4) * Fraction(3, 7), "b": Fraction(2, 4) * Fraction(2, 7),
                "c": Fraction(0), "d": Fraction(2, 4) * Fraction(2, 7)}
    for name, value in expected.items():
        assert scores[name] == pytest.approx(float(value)), name


def test_eligibility_counts_train_images_only():
    index = _build(_records(range(200), 3), _params(val_artist_ratio=0.0, val_image_ratio=0.4,
                                                               min_images_per_artist=3))
    total = np.diff(index.artist_indptr)
    has_val_image = np.bincount(index.item_artist[index.item_split == VAL_IMAGE], minlength=total.size) > 0
    assert (total == 3).all() and has_val_image.any() and (~has_val_image).any()
    assert np.array_equal(index.artist_eligible, ~has_val_image)


def _sampler_params(**overrides):
    values = dict(artists_per_batch=8, images_per_artist=4, same_copyright_batch_ratio=0.5,
                  artist_sampling_power=0.5, seed=3, sampler_version=SAMPLER_VERSION, numpy_version="2.4.2")
    values.update(overrides)
    return SamplerParams(**values)


def test_saved_params_are_validated():
    state = _sampler_params().to_state()
    with pytest.raises(ValueError, match="artists_per_batch"):
        SamplerParams.from_state({**state, "artists_per_batch": True})
    with pytest.raises(ValueError, match="artist_sampling_power"):
        SamplerParams.from_state({**state, "artist_sampling_power": False})
    with pytest.raises(ValueError, match="missing"):
        SamplerParams.from_state({k: v for k, v in state.items() if k != "sampler_version"})
    with pytest.raises(ValueError, match="val_artist_ratio"):
        SplitParams.from_state({**_params().to_state(), "val_artist_ratio": True})


def test_val_image_split_follows_item_key_not_item_id():
    params = _params(val_artist_ratio=0.0, val_image_ratio=0.5)
    labels = [(stable_item_key(1, f"images/{i}.png"), _labels({f"artist {i % 5}"})) for i in range(200)]
    empty = np.zeros(0, dtype=np.int64)
    a = build_style_index([(i, key, lab) for i, (key, lab) in enumerate(labels)], params, empty)
    b = build_style_index([(1_000 + 3 * i, key, lab) for i, (key, lab) in enumerate(labels)], params, empty)
    split_a = dict(zip(a.item_keys.tolist(), a.item_split.tolist()))
    split_b = dict(zip(b.item_keys.tolist(), b.item_split.tolist()))
    assert split_a == split_b
    assert VAL_IMAGE in split_a.values() and TRAIN in split_a.values()
