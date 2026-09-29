"""CPU tests for evidence-bound FoveaTag Phase-4 decisions."""

import copy
import hashlib
import json

import pytest

from core.tagger.foveatag.benchmark import (
    BenchmarkManifest,
    PredictionSet,
    evaluate_predictions,
)
from core.tagger.foveatag.phase4_comparison import build_phase4_comparison
from core.tagger.foveatag.phase4_gate import (
    load_passed_phase4_gate,
    record_phase4_gate,
    routed_checkpoint_hashes,
    validate_phase4_artifact,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve()


def _summary(
    tmp_path,
    *,
    routed_checkpoint_sha="c" * 64,
    routed_metadata_sha="d" * 64,
):
    manifest_payload = {
        "format_version": 1,
        "name": "ground",
        "evidence_mode": "dense_oracle",
        "resolution_long_sides": [64],
        "items": [
            {
                "sample_id": "positive",
                "image_path": "positive.png",
                "width": 128,
                "height": 96,
                "positive_tags": ["detail", "scene"],
                "fine_detail_tags": ["detail"],
                "coarse_tags": ["scene"],
                "evidence": [],
            },
            {
                "sample_id": "negative",
                "image_path": "negative.png",
                "width": 128,
                "height": 96,
                "positive_tags": [],
                "fine_detail_tags": [],
                "coarse_tags": [],
                "evidence": [],
            },
        ],
    }
    manifest = _write(tmp_path / "manifest.json", manifest_payload)
    manifest_value = BenchmarkManifest.from_dict(manifest_payload)
    source_rows = []
    routed_rows = []
    boxes = [
        {"x": 0.0, "y": 0.0, "width": 32.0, "height": 32.0},
        {"x": 32.0, "y": 16.0, "width": 32.0, "height": 32.0},
    ]
    for resolution in ("64", "native"):
        source_rows.extend([
            {
                "sample_id": "positive",
                "resolution": resolution,
                "scores": {"detail": 0.7, "scene": 0.8},
                "stage_ms": {"forward": 2.0},
            },
            {
                "sample_id": "negative",
                "resolution": resolution,
                "scores": {"detail": 0.3, "scene": 0.2},
                "stage_ms": {"forward": 2.0},
            },
        ])
        routed_rows.extend([
            {
                "sample_id": "positive",
                "resolution": resolution,
                "scores": {"detail": 0.9, "scene": 0.85},
                "stage_ms": {"overview": 1.0, "router": 0.2, "local_and_head": 1.5},
                "selected_boxes": copy.deepcopy(boxes),
            },
            {
                "sample_id": "negative",
                "resolution": resolution,
                "scores": {"detail": 0.1, "scene": 0.15},
                "stage_ms": {"overview": 1.0, "router": 0.2, "local_and_head": 1.5},
                "selected_boxes": copy.deepcopy(boxes),
            },
        ])
    runtime = {
        "device": "cpu",
        "dtype": "float32",
        "prediction_rows": len(source_rows),
        "wall_seconds": 1.0,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
    }
    source_payload = {
        "format_version": 1,
        "model_name": "foveatag-source_native",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_metadata_sha256": "b" * 64,
        "artifact_provenance": {
            "initialization": "scratch",
            "external_weights": [],
            "input_mode": "source_native",
            "threshold": 0.5,
        },
        "runtime": runtime,
        "predictions": source_rows,
    }
    routed_payload = {
        "format_version": 1,
        "model_name": "foveatag-routed-fast",
        "checkpoint_sha256": routed_checkpoint_sha,
        "checkpoint_metadata_sha256": routed_metadata_sha,
        "artifact_provenance": {
            "initialization": "scratch",
            "external_weights": [],
            "stage": "joint_routed_finetune",
            "threshold": 0.5,
            "source_native_model_sha256": "a" * 64,
            "router_validation": {
                "budget": 2,
                "mean_utility_recovery": 0.7,
                "random_mean_utility_recovery": 0.4,
            },
        },
        "inference_mode": "fast",
        "inference_contract": {"overview_size": 384, "local_views": 2},
        "runtime": dict(runtime),
        "predictions": routed_rows,
    }
    source = _write(tmp_path / "source.json", source_payload)
    routed = _write(tmp_path / "routed.json", routed_payload)
    baseline_payload = evaluate_predictions(
        manifest_value,
        PredictionSet.from_dict({
            "format_version": 1,
            "model_name": "fixed-384",
            "predictions": copy.deepcopy(source_rows),
        }),
    )
    baseline = _write(tmp_path / "baseline.json", baseline_payload)
    report = _write(
        tmp_path / "report.json",
        build_phase4_comparison(
            manifest_value, source_payload, routed_payload, baseline_payload
        ),
    )
    return _write(tmp_path / "summary.json", {
        "format_version": 1,
        "kind": "foveatag_phase4_evaluation",
        "status": "complete",
        "active_benchmark": None,
        "inference_mode": "fast",
        "completed_benchmarks": ["ground"],
        "benchmarks": {
            "ground": {
                "manifest": str(manifest),
                "fixed_baseline_report": str(baseline),
                "source_native_predictions": str(source),
                "routed_predictions": str(routed),
                "comparison_report": str(report),
            }
        },
    })


def test_passed_phase4_gate_binds_all_evidence_and_one_routed_checkpoint(tmp_path):
    output = record_phase4_gate(
        _summary(tmp_path),
        tmp_path / "gate.json",
        decision="pass",
        rationale="Routed detail gains justify measured latency.",
        decided_by="repository owner",
    )

    payload = load_passed_phase4_gate(output)

    assert payload["decision"] == "pass"
    assert len(payload["evidence"]) == 6
    assert routed_checkpoint_hashes(payload) == {
        "checkpoint_sha256": "c" * 64,
        "checkpoint_metadata_sha256": "d" * 64,
    }


def test_failed_phase4_gate_cannot_authorize_integration(tmp_path):
    output = record_phase4_gate(
        _summary(tmp_path),
        tmp_path / "gate.json",
        decision="fail",
        rationale="Latency is not justified by quality.",
        decided_by="repository owner",
    )

    with pytest.raises(ValueError, match="did not pass"):
        load_passed_phase4_gate(output)


def test_phase4_gate_rejects_evidence_modified_after_decision(tmp_path):
    summary = _summary(tmp_path)
    output = record_phase4_gate(
        summary,
        tmp_path / "gate.json",
        decision="pass",
        rationale="Registered evidence passed.",
        decided_by="repository owner",
    )
    summary.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="evidence changed"):
        load_passed_phase4_gate(output)


def test_phase4_gate_rejects_comparison_report_not_reproducible_from_inputs(tmp_path):
    summary = _summary(tmp_path)
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    report = _write(
        tmp_path / "tampered_report.json",
        {"format_version": 1, "kind": "foveatag_phase4_routed_comparison"},
    )
    summary_payload["benchmarks"]["ground"]["comparison_report"] = str(report)
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its inputs"):
        record_phase4_gate(
            summary,
            tmp_path / "gate.json",
            decision="pass",
            rationale="invalid",
            decided_by="owner",
        )


def test_phase4_gate_loader_rejects_handwritten_incomplete_evidence(tmp_path):
    summary = _summary(tmp_path)
    gate = _write(tmp_path / "handwritten_gate.json", {
        "format_version": 1,
        "kind": "foveatag_phase4_gate_decision",
        "decision": "pass",
        "rationale": "does not bind routed evidence",
        "decided_by": "owner",
        "evidence": [{
            "role": "evaluation_summary",
            "path": str(summary),
            "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        }],
    })

    with pytest.raises(ValueError, match="evidence set does not match"):
        load_passed_phase4_gate(gate)


def test_phase4_gate_validates_the_exact_final_artifact(tmp_path):
    artifact = tmp_path / "joint.safetensors"
    artifact.write_bytes(b"joint model")
    metadata = _write(tmp_path / "joint_metadata.json", {
        "provenance": {
            "initialization": "scratch",
            "external_weights": [],
            "stage": "joint_routed_finetune",
            "threshold": 0.5,
            "router_validation": {
                "budget": 2,
                "mean_utility_recovery": 0.7,
                "random_mean_utility_recovery": 0.4,
            },
        }
    })
    summary = _summary(
        tmp_path,
        routed_checkpoint_sha=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        routed_metadata_sha=hashlib.sha256(metadata.read_bytes()).hexdigest(),
    )
    gate_path = record_phase4_gate(
        summary,
        tmp_path / "gate.json",
        decision="pass",
        rationale="exact routed artifact passed",
        decided_by="owner",
    )

    gate, loaded_metadata = validate_phase4_artifact(gate_path, artifact)
    assert gate["decision"] == "pass"
    assert loaded_metadata["provenance"]["stage"] == "joint_routed_finetune"

    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact does not match"):
        validate_phase4_artifact(gate_path, artifact)
