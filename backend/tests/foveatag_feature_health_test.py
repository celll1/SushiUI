"""Collapse-sensitive diagnostics for FoveaTag Phase-2 features."""

import pytest
import torch

from core.tagger.foveatag.feature_health import FeatureHealthAccumulator


def test_feature_health_separates_collapsed_and_diverse_tokens():
    collapsed = FeatureHealthAccumulator(4, max_tokens=32)
    collapsed.update(torch.ones(2, 4, 4, 4))

    diverse = FeatureHealthAccumulator(4, max_tokens=32)
    tokens = torch.eye(4).repeat(8, 1).T.reshape(1, 4, 4, 8)
    diverse.update(tokens)

    collapsed_result = collapsed.finalize()
    diverse_result = diverse.finalize()
    assert collapsed_result["channel_std_mean"] == pytest.approx(0.0)
    assert collapsed_result["mean_pairwise_cosine"] == pytest.approx(1.0)
    assert collapsed_result["effective_rank"] == pytest.approx(0.0)
    assert diverse_result["channel_std_mean"] > 0
    assert diverse_result["mean_pairwise_cosine"] < 0.25
    assert diverse_result["effective_rank"] == pytest.approx(3.0)


def test_feature_health_uses_a_bounded_deterministic_token_sample():
    features = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    first = FeatureHealthAccumulator(3, max_tokens=11)
    second = FeatureHealthAccumulator(3, max_tokens=11)
    first.update(features)
    first.update(features * 2)
    second.update(features)

    assert first.count == second.count == 11
    assert first.finalize() == second.finalize()


def test_feature_health_rejects_wrong_shape():
    health = FeatureHealthAccumulator(3)
    with pytest.raises(ValueError, match="shape"):
        health.update(torch.zeros(2, 4, 3, 3))
