"""CPU tests for FoveaTag routed Phase-4 comparison evidence."""

import copy

import pytest

from core.tagger.foveatag.benchmark import BenchmarkManifest, PredictionSet, evaluate_predictions
from core.tagger.foveatag.phase4_comparison import build_phase4_comparison


def _manifest():
    return BenchmarkManifest.from_dict({
        "format_version": 1,
        "name": "phase4-gate",
        "evidence_mode": "dense_oracle",
        "resolution_long_sides": [64],
        "items": [
            {
                "sample_id": "positive", "image_path": "positive.png",
                "width": 128, "height": 96,
                "positive_tags": ["small detail", "scene"],
                "fine_detail_tags": ["small detail"], "coarse_tags": ["scene"],
                "evidence": [],
            },
            {
                "sample_id": "negative", "image_path": "negative.png",
                "width": 128, "height": 96, "positive_tags": [],
                "fine_detail_tags": [], "coarse_tags": [], "evidence": [],
            },
        ],
    })


def _runtime(rows):
    return {
        "device": "cpu", "dtype": "float32", "prediction_rows": rows,
        "wall_seconds": 1.0, "peak_allocated_mib": None,
        "peak_reserved_mib": None,
    }


def _source():
    rows = []
    for resolution in ("64", "native"):
        rows.extend([
            {
                "sample_id": "positive", "resolution": resolution,
                "scores": {"small detail": 0.7, "scene": 0.8},
                "stage_ms": {"forward": 2.0},
            },
            {
                "sample_id": "negative", "resolution": resolution,
                "scores": {"small detail": 0.3, "scene": 0.2},
                "stage_ms": {"forward": 2.0},
            },
        ])
    return {
        "format_version": 1, "model_name": "foveatag-source_native",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_metadata_sha256": "b" * 64,
        "artifact_provenance": {
            "initialization": "scratch", "external_weights": [],
            "input_mode": "source_native", "threshold": 0.5,
        },
        "runtime": _runtime(len(rows)), "predictions": rows,
    }


def _routed():
    rows = []
    boxes = [
        {"x": 0.0, "y": 0.0, "width": 32.0, "height": 32.0},
        {"x": 32.0, "y": 16.0, "width": 32.0, "height": 32.0},
    ]
    for resolution in ("64", "native"):
        rows.extend([
            {
                "sample_id": "positive", "resolution": resolution,
                "scores": {"small detail": 0.9, "scene": 0.85},
                "stage_ms": {"overview": 1.0, "router": 0.2, "local_and_head": 1.5},
                "selected_boxes": copy.deepcopy(boxes),
            },
            {
                "sample_id": "negative", "resolution": resolution,
                "scores": {"small detail": 0.1, "scene": 0.15},
                "stage_ms": {"overview": 1.0, "router": 0.2, "local_and_head": 1.5},
                "selected_boxes": copy.deepcopy(boxes),
            },
        ])
    return {
        "format_version": 1, "model_name": "foveatag-routed-fast",
        "checkpoint_sha256": "c" * 64,
        "checkpoint_metadata_sha256": "d" * 64,
        "artifact_provenance": {
            "initialization": "scratch", "external_weights": [],
            "stage": "joint_routed_finetune", "threshold": 0.5,
            "source_native_model_sha256": "a" * 64,
            "router_validation": {
                "budget": 2,
                "mean_utility_recovery": 0.7,
                "random_mean_utility_recovery": 0.4,
            },
        },
        "inference_mode": "fast",
        "inference_contract": {"overview_size": 384, "local_views": 2},
        "runtime": _runtime(len(rows)), "predictions": rows,
    }


def _fixed(manifest, source):
    return evaluate_predictions(
        manifest,
        PredictionSet.from_dict({
            "format_version": 1,
            "model_name": "fixed-384",
            "predictions": copy.deepcopy(source["predictions"]),
        }),
    )


def test_phase4_report_binds_lineage_fixed_k_quality_and_runtime():
    manifest = _manifest()
    source = _source()
    routed = _routed()

    report = build_phase4_comparison(manifest, source, routed, _fixed(manifest, source))

    assert report["contract"]["source_native_checkpoint_sha256"] == "a" * 64
    assert report["contract"]["inference_contract"]["local_views"] == 2
    assert report["contract"]["router_validation"]["mean_utility_recovery"] == 0.7
    assert report["contract"]["routed_runtime"]["wall_seconds"] == 1.0
    assert set(report["matched_fpr_comparisons"]["routed_vs_source_native"]) == {
        "fine_detail", "coarse", "all",
    }
    assert report["evaluations"]["routed"]["stage_timing_ms"]["router"]["count"] == 4


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("lineage", "does not descend"),
        ("utility", "learned-over-random"),
        ("budget", "fixed-K"),
        ("coverage", "coverage differs"),
    ],
)
def test_phase4_report_rejects_confounded_or_unaccepted_results(mutation, message):
    manifest = _manifest()
    source = _source()
    routed = _routed()
    if mutation == "lineage":
        routed["artifact_provenance"]["source_native_model_sha256"] = "e" * 64
    elif mutation == "utility":
        routed["artifact_provenance"]["router_validation"]["mean_utility_recovery"] = 0.4
    elif mutation == "budget":
        routed["predictions"][0]["selected_boxes"].pop()
    else:
        routed["predictions"].pop()

    with pytest.raises(ValueError, match=message):
        build_phase4_comparison(manifest, source, routed, _fixed(manifest, source))
