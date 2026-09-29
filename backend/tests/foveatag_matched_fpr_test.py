"""CPU tests for matched-FPR benchmark comparison."""

import pytest

from core.tagger.foveatag.benchmark import (
    BenchmarkManifest,
    PredictionSet,
    evaluate_predictions,
)
from core.tagger.foveatag.matched_fpr import compare_at_matched_fpr


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
