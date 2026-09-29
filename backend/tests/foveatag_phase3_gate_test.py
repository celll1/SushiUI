"""CPU tests for explicit, evidence-bound FoveaTag Phase-3 decisions."""

import hashlib
import json
from pathlib import Path

import pytest

from core.tagger.foveatag.benchmark import (
    BenchmarkManifest,
    PredictionSet,
    evaluate_predictions,
)
from core.tagger.foveatag.phase3_comparison import build_phase3_comparison
from core.tagger.foveatag.phase3_gate import (
    load_passed_phase3_gate,
    record_phase3_gate,
    source_native_checkpoint_hashes,
    source_native_checkpoint_sha256,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve()


def _summary(tmp_path):
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
    prediction_payloads = {}
    prediction_paths = {}
    for mode, positive, negative in (
        ("overview_only", 0.6, 0.4),
        ("overview_placebo", 0.7, 0.3),
        ("source_native", 0.9, 0.1),
    ):
        rows = []
        for resolution in ("64", "native"):
            rows.extend([
                {
                    "sample_id": "positive",
                    "resolution": resolution,
                    "scores": {"detail": positive, "scene": positive},
                    "stage_ms": {"forward": 2.0},
                },
                {
                    "sample_id": "negative",
                    "resolution": resolution,
                    "scores": {"detail": negative, "scene": negative},
                    "stage_ms": {"forward": 2.0},
                },
            ])
        payload = {
            "format_version": 1,
            "model_name": f"foveatag-{mode}",
            "artifact_provenance": {
                "initialization": "scratch",
                "external_weights": [],
                "shared_initialization_sha256": "shared",
                "shared_data_contract_sha256": "d" * 64,
                "shared_training_contract_sha256": "f" * 64,
                "input_mode": mode,
                "threshold": 0.5,
            },
            "runtime": {
                "device": "cpu",
                "dtype": "float32",
                "prediction_rows": len(rows),
                "wall_seconds": 1.0,
                "peak_allocated_mib": None,
                "peak_reserved_mib": None,
            },
            "mode": mode,
            "checkpoint_sha256": "a" * 64,
            "checkpoint_metadata_sha256": "b" * 64,
            "predictions": rows,
        }
        prediction_payloads[mode] = payload
        prediction_paths[mode] = str(_write(tmp_path / f"{mode}.json", payload))
    manifest_value = BenchmarkManifest.from_dict(manifest_payload)
    baseline_payload = evaluate_predictions(
        manifest_value,
        PredictionSet.from_dict({
            "format_version": 1,
            "model_name": "fixed-384",
            "predictions": prediction_payloads["overview_only"]["predictions"],
        }),
    )
    baseline = _write(tmp_path / "baseline.json", baseline_payload)
    report = _write(
        tmp_path / "comparison.json",
        build_phase3_comparison(manifest_value, prediction_payloads, baseline_payload),
    )
    return _write(tmp_path / "summary.json", {
        "format_version": 1,
        "kind": "foveatag_phase3_evaluation",
        "status": "complete",
        "active_benchmark": None,
        "completed_benchmarks": ["ground"],
        "benchmarks": {
            "ground": {
                "manifest": str(manifest),
                "fixed_baseline_report": str(baseline),
                "comparison_report": str(report),
                "predictions": prediction_paths,
            }
        },
    })


def test_passed_gate_is_bound_to_all_evaluation_evidence(tmp_path):
    output = record_phase3_gate(
        _summary(tmp_path),
        tmp_path / "gate.json",
        decision="pass",
        rationale="Source-native improves the registered slices within budget.",
        decided_by="repository owner",
    )

    payload = load_passed_phase3_gate(output)

    assert payload["decision"] == "pass"
    assert len(payload["evidence"]) == 7
    assert {row["role"] for row in payload["evidence"]} >= {
        "evaluation_summary", "manifest", "comparison_report",
        "prediction:source_native",
    }
    assert source_native_checkpoint_sha256(payload) == "a" * 64
    assert source_native_checkpoint_hashes(payload) == {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_metadata_sha256": "b" * 64,
    }


def test_failed_gate_cannot_authorize_phase4(tmp_path):
    output = record_phase3_gate(
        _summary(tmp_path),
        tmp_path / "gate.json",
        decision="fail",
        rationale="No material gain.",
        decided_by="repository owner",
    )

    with pytest.raises(ValueError, match="did not pass"):
        load_passed_phase3_gate(output)


def test_gate_rejects_evidence_modified_after_decision(tmp_path):
    summary = _summary(tmp_path)
    output = record_phase3_gate(
        summary,
        tmp_path / "gate.json",
        decision="pass",
        rationale="Measured evidence passed.",
        decided_by="repository owner",
    )
    summary.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="evidence changed"):
        load_passed_phase3_gate(output)


def test_gate_rejects_comparison_report_not_reproducible_from_inputs(tmp_path):
    summary = _summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    report = Path(payload["benchmarks"]["ground"]["comparison_report"])
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_payload["contract"]["expected_prediction_rows"] = 999
    report.write_text(json.dumps(report_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its inputs"):
        record_phase3_gate(
            summary,
            tmp_path / "gate.json",
            decision="pass",
            rationale="invalid",
            decided_by="owner",
        )


def test_gate_loader_rejects_handwritten_incomplete_evidence(tmp_path):
    summary = _summary(tmp_path)
    gate = _write(tmp_path / "handwritten_gate.json", {
        "format_version": 1,
        "kind": "foveatag_phase3_gate_decision",
        "decision": "pass",
        "rationale": "does not bind the branch evidence",
        "decided_by": "owner",
        "evidence": [{
            "role": "evaluation_summary",
            "path": str(summary),
            "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        }],
    })

    with pytest.raises(ValueError, match="evidence set does not match"):
        load_passed_phase3_gate(gate)
