"""P x K style batch sampler (design §6; T3, resume determinism)."""

from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_dataset import TRAIN, SplitParams, build_style_index, stable_item_key
from core.style_encoder.style_labels import StyleLabels
from core.style_encoder.style_sampler import StyleBatchSampler

P, K = 3, 2


def _labels(artist, copyrights=(), characters=()):
    return StyleLabels(frozenset({artist}), frozenset(characters), frozenset(copyrights), False)


def _fixture_records():
    rng = random.Random(0)
    records, item = [], 0
    for a in range(40):
        for j in range(4 + a % 5):
            cps = {"shared"} if a < 6 and j < 3 else {f"series {rng.randrange(8)}"}
            if (a in (6, 7) and j < 3) or (a == 8 and j < 1):
                cps = {"almost"}
            records.append((item, _labels(f"artist {a}", cps, {f"char {a} {j % 3}"})))
            item += 1
    return records


def _index(records, **split):
    values = dict(val_artist_ratio=0.1, val_image_ratio=0.1, min_images_per_artist=3, seed=5)
    values.update(split)
    keyed = [(i, stable_item_key(1, f"{i}.png"), labels) for i, labels in records]
    return build_style_index(keyed, SplitParams.create(**values), pinned_keys=np.zeros(0, dtype=np.int64))


def _sampler(index, ratio, seed=11, **kwargs):
    values = dict(artists_per_batch=P, images_per_artist=K, same_copyright_batch_ratio=ratio,
                  artist_sampling_power=0.5, seed=seed)
    values.update(kwargs)
    return StyleBatchSampler(index, **values)


def _copyrights(index, row):
    lo, hi = index.item_copyright_indptr[row], index.item_copyright_indptr[row + 1]
    return {index.copyright_names[c] for c in index.item_copyrights[lo:hi]}


def _characters(index, row):
    lo, hi = index.item_character_indptr[row], index.item_character_indptr[row + 1]
    return {index.character_names[c] for c in index.item_characters[lo:hi]}


def _check_general_shape(index, batch):
    assert batch.rows.shape == (P, K)
    assert len(set(batch.artists.tolist())) == P
    for artist, rows in zip(batch.artists, batch.rows):
        assert index.artist_eligible[artist]
        assert len(set(rows.tolist())) == K
        assert all(index.item_artist[r] == artist and index.item_split[r] == TRAIN for r in rows)


def test_same_copyright_batches_share_one_copyright():
    index = _index(_fixture_records())
    sampler = _sampler(index, ratio=1.0)
    qualifying = {index.copyright_names[c] for c in sampler.qualifying_copyrights}
    assert "shared" in qualifying
    assert "almost" not in qualifying
    for step in range(sampler.epoch_length):
        batch = sampler.batch(0, step)
        assert batch.copyright is not None
        name = index.copyright_names[batch.copyright]
        _check_general_shape(index, batch)
        assert all(name in _copyrights(index, r) for r in batch.rows.ravel())


def test_same_copyright_falls_back_when_nothing_qualifies():
    records = [(i, _labels(f"artist {i % 10}", {f"solo {i % 10}"})) for i in range(60)]
    index = _index(records, val_artist_ratio=0.0)
    sampler = _sampler(index, ratio=1.0)
    assert sampler.qualifying_copyrights.size == 0
    batch = sampler.batch(0, 0)
    assert batch.copyright is None
    _check_general_shape(index, batch)


def test_general_batches_are_train_only_and_spread_characters():
    index = _index(_fixture_records())
    sampler = _sampler(index, ratio=0.0)
    assert sampler.epoch_length == sampler.train_images // (P * K)
    assert sampler.train_images == index.stats.train_items
    for epoch in range(3):
        for batch in sampler.epoch_batches(epoch, 0):
            assert batch.copyright is None
            _check_general_shape(index, batch)
            for artist, rows in zip(batch.artists, batch.rows):
                own = index.artist_items[index.artist_indptr[artist]:index.artist_indptr[artist + 1]]
                available = set().union(*(_characters(index, r) for r in own if index.item_split[r] == TRAIN))
                chars = set().union(*(_characters(index, r) for r in rows))
                assert len(chars) == min(K, len(available))


def test_resume_reproduces_uninterrupted_batches():
    records = _fixture_records()
    index = _index(records)
    uninterrupted = [(b.epoch, b.step, b.artists.tolist(), b.rows.tolist(), b.copyright)
                     for epoch in range(2) for b in _sampler(index, ratio=0.5).epoch_batches(epoch, 0)]
    shuffled = records[:]
    random.Random(3).shuffle(shuffled)
    resumed_sampler = _sampler(_index(shuffled), ratio=0.5)
    resume_at = resumed_sampler.epoch_length + 2
    resumed = [(b.epoch, b.step, b.artists.tolist(), b.rows.tolist(), b.copyright)
               for b in resumed_sampler.epoch_batches(1, 2)]
    assert resumed == uninterrupted[resume_at:]
    assert any(c is not None for *_, c in uninterrupted)
    assert any(c is None for *_, c in uninterrupted)
    other_seed = [b.rows.tolist() for b in _sampler(index, ratio=0.5, seed=12).epoch_batches(0, 0)]
    assert other_seed != [u[3] for u in uninterrupted[:len(other_seed)]]


def test_refuses_unformable_configurations():
    index = _index(_fixture_records())
    with pytest.raises(ValueError, match="min_images_per_artist"):
        _sampler(index, ratio=0.5, images_per_artist=4)
    with pytest.raises(ValueError, match="eligible train artists"):
        _sampler(index, ratio=0.5, artists_per_batch=int(index.artist_eligible.sum()) + 1)


def test_general_batch_artist_frequency_follows_power_law():
    sizes = {"small": 4, "medium": 16, "large": 64}
    records = [(i, _labels(name)) for i, name in enumerate(n for n, size in sizes.items() for _ in range(size))]
    index = _index(records, val_artist_ratio=0.0, val_image_ratio=0.0, min_images_per_artist=4)
    sampler = _sampler(index, ratio=0.0, artists_per_batch=1, images_per_artist=1)
    draws = [index.artist_names[b.artists[0]] for epoch in range(200) for b in sampler.epoch_batches(epoch, 0)]
    weights = {name: size ** 0.5 for name, size in sizes.items()}
    total = sum(weights.values())
    for name, weight in weights.items():
        assert abs(draws.count(name) / len(draws) - weight / total) < 0.015, name
