"""CPU contracts for routed FoveaTag benchmark prediction artifacts."""

import json

import pytest
from PIL import Image

from core.tagger.foveatag.artifact import FoveaArtifactConfig, save_inference_artifact
from core.tagger.foveatag.benchmark import load_prediction_set
from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.model import FoveaTagger
from core.tagger.foveatag.router import EvidenceRouter
from core.tagger.foveatag.routed_prediction_runner import (
    run_routed_predictions,
    validate_joint_artifact,
)
from core.tagger.foveatag.views import ViewConfig


def _artifact(tmp_path):
    encoder = FoveaEncoderConfig(
        widths=(8, 16, 32, 48), depths=(1, 1, 2, 1), window_size=4,
        attention_heads=(2, 3), mlp_ratio=2.0, scale_dropout=0.0,
    )
    config = FoveaArtifactConfig(
        num_tags=2, feature_dim=16, evidence_slots_per_view=1, pool_heads=2,
        encoder=encoder,
        views=ViewConfig(overview_size=32, local_size=16, local_views=2),
        routing_dim=16, routing_slots=2,
    )
    model = FoveaTagger(
        2, encoder_config=encoder, feature_dim=16,
        evidence_slots_per_view=1, pool_heads=2,
    )
    router = EvidenceRouter(feature_dim=32, routing_dim=16, routing_slots=2)
    return save_inference_artifact(
        tmp_path / "joint.safetensors",
        model,
        config,
        router=router,
        vocabulary={
            "tag_to_idx": {"detail": 0, "coarse": 1},
            "idx_to_tag": {"0": "detail", "1": "coarse"},
        },
        provenance={
            "initialization": "scratch", "external_weights": [],
            "stage": "joint_routed_finetune", "threshold": 0.5,
            "router_validation": {
                "mean_utility_recovery": 0.7,
                "random_mean_utility_recovery": 0.4,
            },
        },
    )


def test_joint_artifact_requires_router_to_beat_random():
    metadata = {
        "provenance": {
            "initialization": "scratch", "external_weights": [],
            "stage": "joint_routed_finetune", "threshold": 0.5,
            "router_validation": {
                "mean_utility_recovery": 0.4,
                "random_mean_utility_recovery": 0.4,
            },
        }
    }

    with pytest.raises(ValueError, match="did not beat random"):
        validate_joint_artifact(metadata)


def test_routed_prediction_runner_writes_selected_regions_and_timings(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (40, 24), (20, 40, 60)).save(image_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "name": "phase4-smoke",
        "evidence_mode": "dense_oracle",
        "items": [{
            "sample_id": "a", "image_path": str(image_path),
            "width": 40, "height": 24, "positive_tags": ["detail"],
            "fine_detail_tags": ["detail"], "coarse_tags": [], "evidence": [],
        }],
    }), encoding="utf-8")
    artifact = _artifact(tmp_path)

    output = run_routed_predictions(
        manifest_path,
        artifact,
        tmp_path / "predictions.json",
        inference_mode="fast",
        resolutions=("native",),
        device="cpu",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    predictions = load_prediction_set(output)

    assert predictions.model_name == "foveatag-routed-fast"
    assert len(predictions.rows) == 1
    assert 0 <= predictions.rows[0].scores["detail"] <= 1
    assert len(payload["predictions"][0]["selected_boxes"]) == 2
    assert payload["predictions"][0]["stage_ms"]["overview"] >= 0
    assert payload["predictions"][0]["stage_ms"]["router"] >= 0
    assert payload["predictions"][0]["stage_ms"]["local_and_head"] >= 0
    assert payload["runtime"]["device"] == "cpu"
    assert payload["runtime"]["dtype"] == "float32"
