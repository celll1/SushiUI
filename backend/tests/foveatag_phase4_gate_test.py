"""CPU tests for evidence-bound FoveaTag Phase-4 decisions."""

import json

import pytest

from core.tagger.foveatag.phase4_gate import (
    load_passed_phase4_gate,
    record_phase4_gate,
    routed_checkpoint_hashes,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve()


def _summary(tmp_path):
    manifest = _write(tmp_path / "manifest.json", {"manifest": 1})
    baseline = _write(tmp_path / "baseline.json", {"baseline": 1})
    source = _write(tmp_path / "source.json", {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_metadata_sha256": "b" * 64,
    })
    routed = _write(tmp_path / "routed.json", {
        "checkpoint_sha256": "c" * 64,
        "checkpoint_metadata_sha256": "d" * 64,
    })
    report = _write(tmp_path / "report.json", {"report": 1})
    return _write(tmp_path / "summary.json", {
        "kind": "foveatag_phase4_evaluation",
        "status": "complete",
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
