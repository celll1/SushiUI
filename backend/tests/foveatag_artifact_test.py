"""CPU tests for FoveaTag artifact and resume boundaries."""

import json

import torch
from safetensors.torch import load_file

from core.tagger.foveatag.artifact import (
    FoveaArtifactConfig,
    load_inference_artifact,
    load_training_checkpoint,
    save_inference_artifact,
    save_training_checkpoint,
)
from core.tagger.foveatag.distillation import LocalSigLIP2DistillationHead
from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.model import FoveaTagger


def _config():
    encoder = FoveaEncoderConfig(
        widths=(8, 16, 32, 48),
        depths=(1, 1, 2, 1),
        window_size=4,
        attention_heads=(2, 3),
        mlp_ratio=2.0,
        scale_dropout=0.0,
    )
    return FoveaArtifactConfig(
        num_tags=3,
        feature_dim=16,
        evidence_slots_per_view=2,
        pool_heads=2,
        encoder=encoder,
    )


def _model(config):
    return FoveaTagger(
        config.num_tags,
        encoder_config=config.encoder,
        feature_dim=config.feature_dim,
        evidence_slots_per_view=config.evidence_slots_per_view,
        pool_heads=config.pool_heads,
    )


def test_inference_artifact_round_trip_excludes_training_only_projection(tmp_path):
    config = _config()
    model = _model(config)
    distillation = LocalSigLIP2DistillationHead(32, 12)
    path = save_inference_artifact(
        tmp_path / "foveatag.safetensors",
        model,
        config,
        provenance={"initialization": "scratch", "diagnostic_teacher": "siglip2"},
    )

    keys = load_file(str(path)).keys()
    assert not any("distillation" in key for key in keys)
    restored = _model(config)
    metadata = load_inference_artifact(path, restored)
    assert metadata["contains_training_only_state"] is False
    assert metadata["config"]["coordinate_convention"].endswith("v1")
    assert distillation.projection.weight.shape == (12, 32)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_resume_checkpoint_includes_distillation_and_optimizer_state(tmp_path):
    config = _config()
    model = _model(config)
    distillation = LocalSigLIP2DistillationHead(32, 12)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(distillation.parameters()),
        lr=1e-3,
    )
    original_projection = distillation.projection.weight.detach().clone()
    path = save_training_checkpoint(
        tmp_path / "resume.pt",
        model,
        optimizer,
        config,
        step=17,
        distillation_head=distillation,
    )
    with torch.no_grad():
        distillation.projection.weight.zero_()

    payload = load_training_checkpoint(
        path,
        model,
        optimizer,
        distillation_head=distillation,
    )

    assert payload["step"] == 17
    torch.testing.assert_close(distillation.projection.weight, original_projection)


def test_metadata_rejects_unknown_coordinate_contract(tmp_path):
    config = _config()
    model = _model(config)
    path = save_inference_artifact(tmp_path / "foveatag.safetensors", model, config)
    metadata_path = tmp_path / "foveatag_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"]["coordinate_convention"] = "unknown"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    try:
        load_inference_artifact(path, _model(config))
    except ValueError as error:
        assert "coordinate convention" in str(error)
    else:
        raise AssertionError("expected incompatible coordinate metadata to fail")
