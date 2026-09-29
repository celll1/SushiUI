"""CPU tests for matched-FPR benchmark comparison."""

import json

import pytest

from core.tagger.foveatag.benchmark import (
    BenchmarkManifest,
    PredictionSet,
    evaluate_predictions,
    main as benchmark_main,
)
from core.tagger.foveatag.matched_fpr import compare_at_matched_fpr, main as matched_main


def test_comparison_selects_candidate_threshold_at_baseline_fpr():
    manifest = BenchmarkManifest.from_dict(
        {
            "format_version": 1,
            "name": "matched-fpr",
            "evidence_mode": "dense_oracle",
            "items": [
                {
                    "sample_id": "positive",
                    "image_path": "positive.png",
                    "width": 100,
                    "height": 100,
                    "positive_tags": ["blue eyes"],
                    "fine_detail_tags": ["blue eyes"],
                    "coarse_tags": [],
                    "evidence": [],
                },
                {
                    "sample_id": "negative",
                    "image_path": "negative.png",
                    "width": 100,
                    "height": 100,
                    "positive_tags": [],
                    "fine_detail_tags": [],
                    "coarse_tags": [],
                    "evidence": [],
                },
            ],
        }
    )
    baseline = PredictionSet.from_dict(
        {
            "format_version": 1,
            "model_name": "baseline",
            "predictions": [
                {"sample_id": "positive", "resolution": "native", "scores": {"blue eyes": 0.9}},
                {"sample_id": "negative", "resolution": "native", "scores": {"blue eyes": 0.1}},
            ],
        }
    )
    candidate = PredictionSet.from_dict(
        {
            "format_version": 1,
            "model_name": "candidate",
            "predictions": [
                {"sample_id": "positive", "resolution": "native", "scores": {"blue eyes": 0.8}},
                {"sample_id": "negative", "resolution": "native", "scores": {"blue eyes": 0.6}},
            ],
        }
    )

    comparison = compare_at_matched_fpr(
        manifest,
        evaluate_predictions(manifest, baseline, threshold=0.5),
        candidate,
        thresholds=(0.5, 0.61, 0.9),
    )

    native = comparison["resolutions"]["native"]
    assert native["baseline"]["macro_fpr"] == 0
    assert native["candidate"]["macro_fpr"] == 0
    assert native["candidate"]["macro_recall"] == 1
    assert native["candidate"]["threshold"] == pytest.approx(0.61)


def test_cli_reports_chain_input_file_hashes(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    baseline_predictions = tmp_path / "baseline_predictions.json"
    candidate_predictions = tmp_path / "candidate_predictions.json"
    baseline_report = tmp_path / "baseline_report.json"
    comparison_report = tmp_path / "comparison_report.json"
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "name": "evidence-chain",
        "evidence_mode": "dense_oracle",
        "items": [
            {
                "sample_id": "positive",
                "image_path": "positive.png",
                "width": 100,
                "height": 100,
                "positive_tags": ["blue eyes"],
                "fine_detail_tags": ["blue eyes"],
                "coarse_tags": [],
                "evidence": [],
            },
            {
                "sample_id": "negative",
                "image_path": "negative.png",
                "width": 100,
                "height": 100,
                "positive_tags": [],
                "fine_detail_tags": [],
                "coarse_tags": [],
                "evidence": [],
            },
        ],
    }), encoding="utf-8")
    for path, model, positive, negative in (
        (baseline_predictions, "baseline", 0.9, 0.1),
        (candidate_predictions, "candidate", 0.8, 0.6),
    ):
        path.write_text(json.dumps({
            "format_version": 1,
            "model_name": model,
            "predictions": [
                {
                    "sample_id": "positive",
                    "resolution": "native",
                    "scores": {"blue eyes": positive},
                },
                {
                    "sample_id": "negative",
                    "resolution": "native",
                    "scores": {"blue eyes": negative},
                },
            ],
        }), encoding="utf-8")

    assert benchmark_main([
        "evaluate", str(manifest_path), str(baseline_predictions),
        "--output", str(baseline_report),
    ]) == 0
    assert matched_main([
        str(manifest_path), str(baseline_report), str(candidate_predictions),
        "--output", str(comparison_report),
    ]) == 0

    baseline_roles = {
        row["role"] for row in json.loads(baseline_report.read_text())["evidence"]
    }
    comparison_roles = {
        row["role"] for row in json.loads(comparison_report.read_text())["evidence"]
    }
    assert baseline_roles == {"manifest", "predictions"}
    assert comparison_roles == {
        "manifest", "fixed_baseline_report", "candidate_predictions"
    }
