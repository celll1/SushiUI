"""CPU tests for the end-to-end from-scratch FoveaTag model."""

import torch

from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.model import FoveaTagger


def _model(num_tags=5):
    config = FoveaEncoderConfig(
        widths=(8, 16, 32, 48),
        depths=(1, 1, 2, 1),
        window_size=4,
        attention_heads=(2, 3),
        mlp_ratio=2.0,
        scale_dropout=0.0,
    )
    return FoveaTagger(
        num_tags,
        encoder_config=config,
        feature_dim=16,
        evidence_slots_per_view=2,
        pool_heads=2,
    )


def _geometry(batch, views=None):
    shape = (batch, 10) if views is None else (batch, views, 10)
    geometry = torch.zeros(shape)
    geometry[..., 0:4] = torch.tensor([0.5, 0.5, 1.0, 1.0])
    geometry[..., 8:10] = 1.0
    return geometry


def test_model_uses_local_stride_16_features_and_tracks_view_indices():
    model = _model()
    details = model(
        torch.randn(2, 3, 48, 40),
        _geometry(2),
        torch.randn(2, 3, 3, 32, 32),
        _geometry(2, 3),
        local_view_mask=torch.tensor([[True, False, True], [True, True, True]]),
        return_details=True,
    )

    assert details.logits.shape == (2, 5)
    assert details.head_output.local_scores.shape == (2, 6, 5)
    assert details.evidence_view_indices.tolist() == [
        [0, 0, -1, -1, 2, 2],
        [0, 0, 1, 1, 2, 2],
    ]
    assert torch.isfinite(details.logits).all()


def test_model_supports_fast_path_without_local_views():
    model = _model(num_tags=3)
    logits = model(
        torch.randn(1, 3, 32, 32),
        _geometry(1),
        torch.empty(1, 0, 3, 16, 16),
        _geometry(1, 0),
    )

    assert logits.shape == (1, 3)
    assert torch.isfinite(logits).all()


def test_vocabulary_expansion_preserves_all_existing_global_and_local_weights():
    model = _model(num_tags=2)
    old_weight = model.head.weight.detach().clone()
    old_bias = model.head.bias.detach().clone()

    model.expand_head(4)

    torch.testing.assert_close(model.head.weight[:2], old_weight)
    torch.testing.assert_close(model.head.bias[:2], old_bias)
    assert model.head.weight.shape == (4, 32)
