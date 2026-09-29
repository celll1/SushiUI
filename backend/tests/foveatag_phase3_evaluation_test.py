"""CPU orchestration tests for FoveaTag Phase-3 evaluation."""

import json

import pytest

from core.tagger.foveatag.phase3_evaluation import (
    Phase3Benchmark,
    run_phase3_evaluation,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_phase3_evaluation_regenerates_every_branch_and_builds_reports(tmp_path):
    comparison_dir = tmp_path / "comparison"
    for mode in ("overview_only", "overview_placebo", "source_native"):
        artifact = comparison_dir / mode / "best_f1.safetensors"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"model")
        _write_json(artifact.with_name("best_f1_metadata.json"), {})
    _write_json(comparison_dir / "comparison_summary.json", {
        "kind": "foveatag_phase3_supervised_comparison",
        "status": "complete",
        "completed_modes": ["overview_only", "overview_placebo", "source_native"],
    })
    manifest = _write_json(tmp_path / "manifest.json", {
        "format_version": 1,
        "name": "ground",
        "evidence_mode": "dense_oracle",
        "items": [{
            "sample_id": "a",
            "image_path": "a.png",
            "width": 32,
            "height": 32,
            "positive_tags": ["detail"],
            "fine_detail_tags": ["detail"],
            "coarse_tags": [],
            "evidence": [],
        }],
    })
    baseline = _write_json(tmp_path / "fixed.json", {"benchmark": "ground"})
    calls = []

    def prediction_runner(manifest_path, artifact, output, **kwargs):
        mode = artifact.parent.name
        calls.append((mode, kwargs))
        return _write_json(output, {
            "format_version": 1,
            "model_name": f"foveatag-{mode}",
            "predictions": [{
                "sample_id": "a",
                "resolution": "native",
                "scores": {"detail": 0.5},
            }],
        })

    builder_calls = []

    def comparison_builder(loaded_manifest, payloads, fixed):
        builder_calls.append((loaded_manifest, payloads, fixed))
        return {"kind": "comparison", "benchmark": loaded_manifest.name}

    result = run_phase3_evaluation(
        comparison_dir=comparison_dir,
        benchmarks=[Phase3Benchmark("ground", manifest, baseline)],
        resolutions=("native",),
        device="cpu",
        prediction_runner=prediction_runner,
        comparison_builder=comparison_builder,
    )

    assert [mode for mode, _kwargs in calls] == [
        "overview_only", "overview_placebo", "source_native",
    ]
    assert all(kwargs == {"resolutions": ("native",), "device": "cpu"} for _, kwargs in calls)
    assert set(builder_calls[0][1]) == {
        "overview_only", "overview_placebo", "source_native",
    }
    assert result["status"] == "complete"
    assert result["completed_benchmarks"] == ["ground"]
    report_path = comparison_dir / "evaluation" / "ground" / "phase3_comparison.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["kind"] == "comparison"


def test_phase3_evaluation_preflights_all_inputs_before_prediction(tmp_path):
    manifest = _write_json(tmp_path / "manifest.json", {})
    baseline = _write_json(tmp_path / "fixed.json", {})

    with pytest.raises(FileNotFoundError, match="best_f1"):
        run_phase3_evaluation(
            comparison_dir=tmp_path / "comparison",
            benchmarks=[Phase3Benchmark("ground", manifest, baseline)],
            prediction_runner=lambda *_args, **_kwargs: pytest.fail("must not run"),
        )


def test_phase3_evaluation_refuses_incomplete_branch_training(tmp_path):
    comparison_dir = tmp_path / "comparison"
    for mode in ("overview_only", "overview_placebo", "source_native"):
        artifact = comparison_dir / mode / "best_f1.safetensors"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"model")
        _write_json(artifact.with_name("best_f1_metadata.json"), {})
    _write_json(comparison_dir / "comparison_summary.json", {
        "kind": "foveatag_phase3_supervised_comparison",
        "status": "stopped_by_request",
        "completed_modes": ["overview_only"],
    })
    manifest = _write_json(tmp_path / "manifest.json", {})
    baseline = _write_json(tmp_path / "fixed.json", {})

    with pytest.raises(ValueError, match="training is not complete"):
        run_phase3_evaluation(
            comparison_dir=comparison_dir,
            benchmarks=[Phase3Benchmark("ground", manifest, baseline)],
            prediction_runner=lambda *_args, **_kwargs: pytest.fail("must not run"),
        )


def test_phase3_benchmark_name_cannot_escape_evaluation_directory(tmp_path):
    with pytest.raises(ValueError, match="benchmark name"):
        Phase3Benchmark("../outside", tmp_path / "manifest", tmp_path / "report")
