"""CPU tests for staged FoveaTag inference."""

import torch
from PIL import Image

from core.tagger.foveatag.artifact import FoveaArtifactConfig, InferenceMode
from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.inference import predict_image, source_field_sizes
from core.tagger.foveatag.model import FoveaTagger
from core.tagger.foveatag.router import EvidenceRouter
from core.tagger.foveatag.views import ViewConfig


def _components(local_views=2):
    encoder = FoveaEncoderConfig(
        widths=(8, 16, 32, 48),
        depths=(1, 1, 2, 1),
        window_size=4,
        attention_heads=(2, 3),
        mlp_ratio=2.0,
        scale_dropout=0.0,
    )
    config = FoveaArtifactConfig(
        num_tags=3,
        feature_dim=16,
        evidence_slots_per_view=2,
        pool_heads=2,
        encoder=encoder,
        views=ViewConfig(overview_size=32, local_size=16, local_views=2, field_levels=(0.5, 1.0)),
        inference_modes={"test": InferenceMode(32, local_views)},
    )
    model = FoveaTagger(
        3,
        encoder_config=encoder,
        feature_dim=16,
        evidence_slots_per_view=2,
        pool_heads=2,
    )
    router = EvidenceRouter(feature_dim=32, routing_dim=16, routing_slots=2)
    return config, model, router


def test_density_levels_produce_square_source_pixel_fields():
    sizes = source_field_sizes(200, 100, 16, [0.5, 2.0])
    assert sizes == ((0.04, 0.08), (0.16, 0.32))
    assert sizes[0][0] * 200 == sizes[0][1] * 100


def test_predict_image_encodes_overview_once_then_all_locals_once():
    config, model, router = _components(local_views=2)
    calls = []
    handle = model.encoder.register_forward_hook(lambda module, inputs, output: calls.append(inputs[0].shape))
    try:
        result = predict_image(Image.new("RGB", (48, 32), (10, 20, 30)), model, router, config, mode="test")
    finally:
        handle.remove()

    assert len(calls) == 2
    assert calls[0][0] == 1
    assert calls[1][0] == 2
    assert result.logits.shape == (3,)
    assert len(result.evidence_boxes) == 3
    assert len(result.selected_boxes) == 2
    assert set(result.stage_milliseconds) == {"overview", "router", "local_and_head", "total"}
    assert all(value >= 0 for value in result.stage_milliseconds.values())


def test_zero_budget_skips_local_encoder_stage_and_returns_no_evidence_boxes():
    config, model, router = _components(local_views=0)
    calls = []
    handle = model.encoder.register_forward_hook(lambda module, inputs, output: calls.append(inputs[0].shape))
    try:
        result = predict_image(Image.new("RGB", (32, 32)), model, router, config, mode="test")
    finally:
        handle.remove()

    assert len(calls) == 1
    assert result.selected_boxes == ()
    assert result.evidence_boxes == (None, None, None)


def test_predict_image_restores_training_modes():
    config, model, router = _components()
    model.train()
    router.train()
    predict_image(Image.new("RGB", (32, 32)), model, router, config, mode="test")
    assert model.training and router.training
