"""CPU contracts for Phase-3 benchmark prediction artifacts."""

import json

from PIL import Image

from core.tagger.foveatag.artifact import FoveaArtifactConfig, save_inference_artifact
from core.tagger.foveatag.benchmark import load_prediction_set
from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.model import FoveaTagger
from core.tagger.foveatag.views import ViewConfig

from core.tagger.foveatag.supervised_prediction_runner import (
    _artifact_input_mode,
    _artifact_vocabulary,
    _view_seed,
    run_supervised_predictions,
)


def test_artifact_input_mode_accepts_only_registered_phase3_branches():
    assert _artifact_input_mode(
        {"provenance": {"input_mode": "source_native"}}
    ) == "source_native"

    try:
        _artifact_input_mode({"provenance": {"input_mode": "unsupported"}})
    except ValueError as error:
        assert "valid Phase-3 input mode" in str(error)
    else:
        raise AssertionError("expected semantic special-case mode to fail")


def test_prediction_view_seed_is_stable_per_sample_and_resolution():
    first = _view_seed("sample-1", "native")

    assert first == _view_seed("sample-1", "native")
    assert first != _view_seed("sample-1", "512")
    assert first != _view_seed("sample-2", "native")


def test_artifact_vocabulary_accepts_serialized_index_mapping():
    vocabulary = {
        "tag_to_idx": {"blue eyes": 0, "text": 1},
        "idx_to_tag": {"0": "blue eyes", "1": "text"},
    }

    assert _artifact_vocabulary({"vocabulary": vocabulary}, 2) is vocabulary


def test_prediction_runner_writes_loadable_resolution_ladder_artifact(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 24), (20, 40, 60)).save(image_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "name": "phase3-smoke",
        "evidence_mode": "dense_oracle",
        "items": [{
            "sample_id": "a",
            "image_path": str(image_path),
            "width": 32,
            "height": 24,
            "positive_tags": ["detail"],
            "fine_detail_tags": ["detail"],
            "coarse_tags": [],
            "evidence": [],
        }],
    }), encoding="utf-8")
    encoder = FoveaEncoderConfig(
        widths=(8, 16, 32, 48),
        depths=(1, 1, 2, 1),
        window_size=4,
        attention_heads=(2, 3),
        mlp_ratio=2.0,
        scale_dropout=0.0,
    )
    config = FoveaArtifactConfig(
        num_tags=2,
        feature_dim=16,
        evidence_slots_per_view=1,
        pool_heads=2,
        encoder=encoder,
        views=ViewConfig(overview_size=32, local_size=16, local_views=1),
    )
    model = FoveaTagger(
        2,
        encoder_config=encoder,
        feature_dim=16,
        evidence_slots_per_view=1,
        pool_heads=2,
    )
    model_path = save_inference_artifact(
        tmp_path / "model.safetensors",
        model,
        config,
        vocabulary={
            "tag_to_idx": {"detail": 0, "coarse": 1},
            "idx_to_tag": {"0": "detail", "1": "coarse"},
        },
        provenance={"initialization": "scratch", "input_mode": "source_native"},
    )

    output = run_supervised_predictions(
        manifest_path,
        model_path,
        tmp_path / "predictions.json",
        resolutions=("native",),
        device="cpu",
    )
    output_payload = json.loads(output.read_text(encoding="utf-8"))
    predictions = load_prediction_set(output)

    assert predictions.model_name == "foveatag-source_native"
    assert len(predictions.rows) == 1
    assert 0 <= predictions.rows[0].scores["detail"] <= 1
    assert predictions.rows[0].stage_ms["forward"] >= 0
    assert output_payload["runtime"]["device"] == "cpu"
    assert output_payload["runtime"]["dtype"] == "float32"
    assert output_payload["runtime"]["prediction_rows"] == 1
    assert output_payload["runtime"]["wall_seconds"] >= 0
    assert output_payload["runtime"]["peak_allocated_mib"] is None
    assert output_payload["runtime"]["peak_reserved_mib"] is None
