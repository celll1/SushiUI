"""CPU tests for the pre-registered FoveaTag Phase-3 comparison."""

import copy

import pytest

from core.tagger.foveatag.benchmark import BenchmarkManifest, PredictionSet, evaluate_predictions
from core.tagger.foveatag.phase3_comparison import build_phase3_comparison


def _manifest():
    return BenchmarkManifest.from_dict({
        "format_version": 1,
        "name": "phase3-gate",
        "evidence_mode": "dense_oracle",
        "resolution_long_sides": [64],
        "items": [
            {
                "sample_id": "positive",
                "image_path": "positive.png",
                "width": 128,
                "height": 96,
                "positive_tags": ["small detail", "scene"],
                "fine_detail_tags": ["small detail"],
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
    })


def _payload(mode, positive, negative):
    rows = []
    for resolution in ("64", "native"):
        rows.extend([
            {
                "sample_id": "positive",
                "resolution": resolution,
                "scores": {"small detail": positive, "scene": positive},
                "stage_ms": {"forward": 2.0},
            },
            {
                "sample_id": "negative",
                "resolution": resolution,
                "scores": {"small detail": negative, "scene": negative},
                "stage_ms": {"forward": 2.0},
            },
        ])
    return {
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
        "checkpoint_sha256": "a" * 64,
        "checkpoint_metadata_sha256": "b" * 64,
        "predictions": rows,
    }


def test_phase3_report_validates_pairing_and_collects_all_gate_evidence():
    manifest = _manifest()
    payloads = {
        "overview_only": _payload("overview_only", 0.6, 0.4),
        "overview_placebo": _payload("overview_placebo", 0.7, 0.3),
        "source_native": _payload("source_native", 0.9, 0.1),
    }
    fixed = evaluate_predictions(
        manifest,
        PredictionSet.from_dict({
            "format_version": 1,
            "model_name": "fixed-384",
            "predictions": payloads["overview_only"]["predictions"],
        }),
    )

    report = build_phase3_comparison(manifest, payloads, fixed)

    assert report["gate_decision"] == "not_automated"
    assert report["contract"]["expected_prediction_rows"] == 4
    assert report["contract"]["shared_initialization_sha256"] == "shared"
    assert report["contract"]["shared_data_contract_sha256"] == "d" * 64
    assert report["contract"]["shared_training_contract_sha256"] == "f" * 64
    assert report["contract"]["branches"]["source_native"]["runtime"]["wall_seconds"] == 1.0
    comparisons = report["matched_fpr_comparisons"]
    assert set(comparisons) == {
        "source_native_vs_overview_only",
        "source_native_vs_overview_placebo",
        "source_native_vs_fixed_baseline",
    }
    assert set(comparisons["source_native_vs_overview_placebo"]) == {
        "fine_detail", "coarse", "all",
    }
    assert report["evaluations"]["source_native"]["stage_timing_ms"]["forward"]["count"] == 4


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("initialization", "same initialization"),
        ("data", "same data contract"),
        ("training", "same training contract"),
        ("coverage", "coverage differs"),
        ("lineage", "external-weight"),
        ("runtime", "wall_seconds"),
    ],
)
def test_phase3_report_rejects_confounded_comparisons(mutation, message):
    manifest = _manifest()
    payloads = {
        "overview_only": _payload("overview_only", 0.6, 0.4),
        "overview_placebo": _payload("overview_placebo", 0.7, 0.3),
        "source_native": _payload("source_native", 0.9, 0.1),
    }
    if mutation == "initialization":
        payloads["source_native"]["artifact_provenance"]["shared_initialization_sha256"] = "other"
    elif mutation == "data":
        payloads["source_native"]["artifact_provenance"]["shared_data_contract_sha256"] = "e" * 64
    elif mutation == "training":
        payloads["source_native"]["artifact_provenance"]["shared_training_contract_sha256"] = "0" * 64
    elif mutation == "coverage":
        payloads["source_native"]["predictions"].pop()
    else:
        if mutation == "lineage":
            payloads["source_native"]["artifact_provenance"]["external_weights"] = ["teacher"]
        else:
            payloads["source_native"]["runtime"]["wall_seconds"] = -1
    fixed = evaluate_predictions(
        manifest,
        PredictionSet.from_dict({
            "format_version": 1,
            "model_name": "fixed-384",
            "predictions": copy.deepcopy(payloads["overview_only"]["predictions"]),
        }),
    )

    with pytest.raises(ValueError, match=message):
        build_phase3_comparison(manifest, payloads, fixed)
