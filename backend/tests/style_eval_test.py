"""Retrieval metrics and query eligibility (design §10, T11)."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_eval import Csr, retrieve


def _angles(degrees):
    return torch.tensor([[math.cos(math.radians(d)), math.sin(math.radians(d))] for d in degrees])


def _csr(sets):
    indptr = np.cumsum([0] + [len(s) for s in sets]).astype(np.int64)
    values = np.array([v for s in sets for v in s], dtype=np.int64)
    return Csr(indptr, values)


def _self_retrieve(emb, artists, ks, chunk_size, copyrights=None, seed=0):
    ids = np.arange(100, 100 + len(artists))
    cp = _csr(copyrights) if copyrights is not None else None
    return retrieve(emb, np.array(artists), ids, emb, np.array(artists), ids, ks, chunk_size, seed, cp, cp)


def test_hand_computed_recall_and_map_at_r():
    emb = _angles([0, 10, 100, 25, 180, 270])
    result = _self_retrieve(emb, [0, 0, 0, 1, 1, 2], ks=(1, 3), chunk_size=4)
    summary = result.summarize()
    assert (summary.queries, summary.used, summary.excluded_no_positive, summary.excluded_no_negative) == (6, 5, 1, 0)
    assert result.positives.tolist() == [2, 2, 2, 1, 1, 0]
    np.testing.assert_allclose(result.ap_at_r[:5], [0.5, 0.5, 0.0, 0.0, 0.0])
    assert summary.recall == pytest.approx({1: 0.4, 3: 0.8})
    assert summary.map_at_r == pytest.approx(0.2)


def test_query_without_other_artist_in_gallery_is_excluded():
    emb = _angles([0, 30, 60])
    ids = np.array([1, 2, 3])
    result = retrieve(emb, np.array([0, 0, 0]), ids, emb, np.array([0, 0, 0]), ids, (1,), 2, 0)
    summary = result.summarize()
    assert (summary.used, summary.excluded_no_positive, summary.excluded_no_negative) == (0, 0, 3)
    assert summary.recall == {1: None} and summary.map_at_r is None


def test_same_copyright_gallery_with_one_image_per_artist_excludes_queries():
    # Items: A/X, B/X, A/Y, B/Y, then A/{X, Z}. Without the last item every artist has one image per copyright.
    emb = _angles([0, 40, 80, 120, 160])
    artists = [0, 1, 0, 1, 0]
    cps = [[0], [0], [1], [1], [0, 2]]
    plain = _self_retrieve(emb[:4], artists[:4], (1,), 3).summarize()
    assert (plain.used, plain.excluded_no_positive) == (4, 0)
    same = _self_retrieve(emb[:4], artists[:4], (1,), 3, copyrights=cps[:4]).summarize()
    assert (same.used, same.excluded_no_positive, same.excluded_no_negative) == (0, 4, 0)

    extra = _self_retrieve(emb, artists, (1,), 2, copyrights=cps)
    assert extra.positives.tolist() == [1, 0, 0, 0, 1]
    assert extra.negatives.tolist() == [1, 2, 1, 1, 1]
    assert extra.summarize().used == 2


def test_chunked_equals_unchunked_and_signature_split_keeps_gallery():
    rng = np.random.default_rng(0)
    n = 40
    emb = torch.from_numpy(rng.normal(size=(n, 6)).astype(np.float32))
    artists = rng.integers(0, 6, n)
    cps = [list(rng.choice(4, size=rng.integers(0, 3), replace=False)) for _ in range(n)]
    for copyrights in (None, cps):
        runs = [_self_retrieve(emb, artists, (1, 5), chunk, copyrights) for chunk in (1, 7, n)]
        for other in runs[1:]:
            assert np.array_equal(other.positives, runs[0].positives)
            assert np.array_equal(other.negatives, runs[0].negatives)
            assert np.array_equal(other.hits, runs[0].hits)
            np.testing.assert_array_equal(other.ap_at_r, runs[0].ap_at_r)

    ids = np.arange(100, 100 + n)
    signed = rng.random(n) < 0.4
    full = _self_retrieve(emb, artists, (1, 5), 8)
    for flag in (signed, ~signed):
        direct = retrieve(emb[torch.from_numpy(flag)], artists[flag], ids[flag], emb, artists, ids, (1, 5), 8, 0)
        assert full.summarize(flag) == direct.summarize()


def test_collapsed_embeddings_score_at_chance_independent_of_order_and_chunking():
    # 2 images of artist 0 listed first, 200 of artist 1: ordering ties by position would give
    # Recall@1 = 1 for artist 0 and 0 for artist 1; chance is 1/201 and 199/201.
    artists = np.array([0, 0] + [1] * 200)
    emb = torch.ones(len(artists), 4)
    runs = [_self_retrieve(emb, artists, (1,), chunk, seed=5) for chunk in (7, len(artists))]
    assert np.array_equal(runs[0].hits, runs[1].hits)
    np.testing.assert_array_equal(runs[0].ap_at_r, runs[1].ap_at_r)
    recall = runs[0].hits[:, 0]
    assert recall[2:].mean() == pytest.approx(199 / 201, abs=0.03)
    assert recall[:2].sum() <= 1
    assert runs[0].ap_at_r[2:].mean() == pytest.approx(199 / 201, abs=0.03)

    reversed_order = np.arange(len(artists))[::-1].copy()
    ids = np.arange(100, 100 + len(artists))
    flipped = retrieve(emb, artists, ids, emb, artists[reversed_order], ids[reversed_order], (1,), 16, 5)
    assert np.array_equal(flipped.hits, runs[0].hits)
    other_seed = _self_retrieve(emb, artists, (1,), 16, seed=6)
    assert not np.array_equal(other_seed.hits, runs[0].hits)
