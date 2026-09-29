"""CPU tests for explicit, evidence-bound FoveaTag Phase-3 decisions."""

import json

import pytest

from core.tagger.foveatag.phase3_gate import (
    load_passed_phase3_gate,
    record_phase3_gate,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve()


def _summary(tmp_path):
    manifest = _write(tmp_path / "manifest.json", {"manifest": 1})
    baseline = _write(tmp_path / "baseline.json", {"baseline": 1})
    report = _write(tmp_path / "comparison.json", {"comparison": 1})
    predictions = {
        mode: str(_write(tmp_path / f"{mode}.json", {"mode": mode}))
        for mode in ("overview_only", "overview_placebo", "source_native")
    }
    return _write(tmp_path / "summary.json", {
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
