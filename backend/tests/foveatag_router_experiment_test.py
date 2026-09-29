"""CPU contracts for the gate-bound Phase-4 router experiment."""

import hashlib
import json

import pytest

from core.tagger.foveatag.artifact import FoveaArtifactConfig, save_inference_artifact
from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.model import FoveaTagger
from core.tagger.foveatag.phase3_gate import record_phase3_gate
from core.tagger.foveatag.router_experiment import (
    _validate_arguments,
    validate_router_teacher,
)
from core.tagger.foveatag.views import ViewConfig


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve()


def _teacher_and_gate(tmp_path):
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
        views=ViewConfig(overview_size=32, local_size=16, local_views=2),
        routing_dim=16,
        routing_slots=2,
    )
    model = FoveaTagger(
        2,
        encoder_config=encoder,
        feature_dim=16,
        evidence_slots_per_view=1,
        pool_heads=2,
    )
    artifact = save_inference_artifact(
        tmp_path / "source_native.safetensors",
        model,
        config,
        vocabulary={
            "tag_to_idx": {"detail": 0, "scene": 1},
            "idx_to_tag": {"0": "detail", "1": "scene"},
        },
        provenance={
            "initialization": "scratch",
            "external_weights": [],
            "input_mode": "source_native",
        },
    )
    metadata = artifact.with_name("source_native_metadata.json")
    manifest = _write(tmp_path / "manifest.json", {"manifest": 1})
    baseline = _write(tmp_path / "baseline.json", {"baseline": 1})
    report = _write(tmp_path / "report.json", {"report": 1})
    predictions = {}
    for mode in ("overview_only", "overview_placebo", "source_native"):
        predictions[mode] = str(_write(tmp_path / f"{mode}.json", {
            "checkpoint_sha256": _sha256(artifact) if mode == "source_native" else "a" * 64,
            "checkpoint_metadata_sha256": (
                _sha256(metadata) if mode == "source_native" else "b" * 64
            ),
        }))
    summary = _write(tmp_path / "evaluation.json", {
        "kind": "foveatag_phase3_evaluation",
        "status": "complete",
        "benchmarks": {
            "ground": {
                "manifest": str(manifest),
                "fixed_baseline_report": str(baseline),
                "comparison_report": str(report),
                "predictions": predictions,
            }
        },
    })
    gate = record_phase3_gate(
        summary,
        tmp_path / "gate.json",
        decision="pass",
        rationale="Registered evidence passed.",
        decided_by="repository owner",
    )
    return artifact, gate


def test_router_teacher_must_be_exact_gate_bound_source_native_artifact(tmp_path):
    artifact, gate = _teacher_and_gate(tmp_path)

    loaded_gate, metadata, config = validate_router_teacher(gate, artifact)

    assert loaded_gate["decision"] == "pass"
    assert metadata["provenance"]["input_mode"] == "source_native"
    assert config.num_tags == 2


def test_router_teacher_rejects_weights_changed_after_gate(tmp_path):
    artifact, gate = _teacher_and_gate(tmp_path)
    with artifact.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="does not match"):
        validate_router_teacher(gate, artifact)


def test_router_teacher_rejects_metadata_changed_after_gate(tmp_path):
    artifact, gate = _teacher_and_gate(tmp_path)
    metadata = artifact.with_name("source_native_metadata.json")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["provenance"]["input_mode"] = "overview_only"
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata does not match"):
        validate_router_teacher(gate, artifact)


def test_router_arguments_enforce_fixed_k_within_dense_candidate_budget():
    values = {
        "train_counts": {25: 10},
        "validation_counts": {25: 2},
        "domain_weights": {25: 1.0},
        "epochs": 1,
        "epoch_samples": 10,
        "batch_size": 1,
        "workers": 0,
        "candidate_count": 4,
        "oracle_chunk_size": 2,
        "selection_budget": 5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 0,
        "checkpoint_every_steps": 1,
        "log_every": 1,
        "photo_loss_weight": 0.25,
        "max_source_pixels": 40_000_000,
        "diversity_weight": 0.05,
        "stop_after_steps": None,
    }

    with pytest.raises(ValueError, match="cannot exceed"):
        _validate_arguments(**values)
