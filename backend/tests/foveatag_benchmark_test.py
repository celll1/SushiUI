"""CPU contracts for the FoveaTag Phase-0 benchmark."""

import pytest

from core.tagger.foveatag.benchmark import (
    BenchmarkManifest,
    PredictionSet,
    StageTimer,
    evaluate_predictions,
)


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest.from_dict({
        "format_version": 1,
        "name": "micro-detail-test",
        "resolution_long_sides": [384, 768],
        "items": [
            {
                "sample_id": "a",
                "image_path": "images/a.png",
                "width": 1600,
                "height": 1200,
                "positive_tags": ["blue_eyes", "1girl"],
                "fine_detail_tags": ["blue_eyes"],
                "coarse_tags": ["1girl"],
                "evidence": [{"x": 700, "y": 300, "width": 40, "height": 20, "tags": ["blue_eyes"]}],
            },
            {
                "sample_id": "b",
                "image_path": "images/b.png",
                "width": 800,
                "height": 1200,
                "positive_tags": ["1girl"],
                "fine_detail_tags": [],
                "coarse_tags": ["1girl"],
                "evidence": [],
            },
        ],
    })


def test_manifest_validates_evidence_and_builds_non_upscaled_ladder():
    manifest = _manifest()
    first = manifest.items[0]

    assert manifest.area_bin(first.evidence_area("blue_eyes")) == "tiny"
    assert manifest.resolution_ladder(first) == (
        ("384", 384, 288),
        ("768", 768, 576),
        ("native", 1600, 1200),
    )


def test_manifest_rejects_fine_tag_without_evidence():
    raw = {
        "format_version": 1,
        "name": "bad",
        "items": [{
            "sample_id": "a", "image_path": "a.png", "width": 10, "height": 10,
            "positive_tags": ["blue_eyes"], "fine_detail_tags": ["blue_eyes"],
            "coarse_tags": [], "evidence": [],
        }],
    }
    with pytest.raises(ValueError, match="lack evidence"):
        BenchmarkManifest.from_dict(raw)


def test_evaluation_reports_group_area_resolution_and_stage_timing():
    predictions = PredictionSet.from_dict({
        "format_version": 1,
        "model_name": "fixture",
        "predictions": [
            {"sample_id": "a", "resolution": "384", "scores": {"blue_eyes": 0.9, "1girl": 0.8}, "stage_ms": {"overview": 2.0}},
            {"sample_id": "b", "resolution": "384", "scores": {"blue_eyes": 0.1, "1girl": 0.7}, "stage_ms": {"overview": 4.0}},
        ],
    })

    report = evaluate_predictions(_manifest(), predictions, threshold=0.5)
    at_384 = report["resolutions"]["384"]
    assert at_384["groups"]["fine_detail"]["macro_average_precision"] == 1.0
    assert at_384["groups"]["fine_detail"]["macro_f1"] == 1.0
    assert at_384["evidence_area"]["tiny"]["recall"] == 1.0
    assert report["stage_timing_ms"]["overview"] == {
        "count": 2, "mean": 3.0, "p50": 3.0, "p95": 3.9, "max": 4.0,
    }


def test_stage_timer_records_named_stages():
    timer = StageTimer()
    with timer.stage("preprocess"):
        pass
    summary = timer.summary()
    assert summary["preprocess"]["count"] == 1
    assert summary["preprocess"]["mean"] >= 0
