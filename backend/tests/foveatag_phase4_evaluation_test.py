"""CPU orchestration tests for FoveaTag Phase-4 evaluation."""

import hashlib
import json

import pytest

from core.tagger.foveatag.phase4_evaluation import (
    Phase4Benchmark,
    run_phase4_evaluation,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inputs(tmp_path):
    gate = tmp_path / "gate.json"
    source = tmp_path / "source.safetensors"
    joint = tmp_path / "joint.safetensors"
    gate.write_bytes(b"gate")
    source.write_bytes(b"source")
    joint.write_bytes(b"joint")
    _write_json(source.with_name("source_metadata.json"), {})
    _write_json(joint.with_name("joint_metadata.json"), {
        "provenance": {
            "initialization": "scratch", "external_weights": [],
            "stage": "joint_routed_finetune", "threshold": 0.5,
            "phase3_gate_sha256": _sha256(gate),
            "source_native_model_sha256": _sha256(source),
            "router_validation": {
                "budget": 2,
                "mean_utility_recovery": 0.7,
                "random_mean_utility_recovery": 0.4,
            },
        },
    })
    manifest = _write_json(tmp_path / "manifest.json", {
        "format_version": 1, "name": "ground", "evidence_mode": "dense_oracle",
        "items": [{
            "sample_id": "a", "image_path": "a.png", "width": 32, "height": 32,
            "positive_tags": ["detail"], "fine_detail_tags": ["detail"],
            "coarse_tags": [], "evidence": [],
        }],
    })
    baseline = _write_json(tmp_path / "fixed.json", {"benchmark": "ground"})
    return gate, source, joint, manifest, baseline


def test_phase4_evaluation_regenerates_source_and_routed_predictions(tmp_path):
    gate, source, joint, manifest, baseline = _inputs(tmp_path)
    calls = []

    def source_runner(manifest_path, artifact, output, **kwargs):
        calls.append(("source", artifact, kwargs))
        return _write_json(output, {"kind": "source"})

    def routed_runner(manifest_path, artifact, output, **kwargs):
        calls.append(("routed", artifact, kwargs))
        return _write_json(output, {"kind": "routed"})

    def builder(loaded_manifest, source_payload, routed_payload, fixed):
        assert source_payload["kind"] == "source"
        assert routed_payload["kind"] == "routed"
        return {"kind": "comparison", "benchmark": loaded_manifest.name}

    result = run_phase4_evaluation(
        phase3_gate=gate,
        source_native_model=source,
        joint_model=joint,
        output_dir=tmp_path / "evaluation",
        benchmarks=[Phase4Benchmark("ground", manifest, baseline)],
        inference_mode="fast",
        resolutions=("native",),
        device="cpu",
        source_prediction_runner=source_runner,
        routed_prediction_runner=routed_runner,
        comparison_builder=builder,
        source_validator=lambda *_args: None,
    )

    assert [call[0] for call in calls] == ["source", "routed"]
    assert calls[0][2] == {"resolutions": ("native",), "device": "cpu"}
    assert calls[1][2] == {
        "inference_mode": "fast", "resolutions": ("native",), "device": "cpu",
    }
    assert result["status"] == "complete"
    assert result["completed_benchmarks"] == ["ground"]
    report = tmp_path / "evaluation" / "ground" / "phase4_comparison.json"
    assert json.loads(report.read_text(encoding="utf-8"))["kind"] == "comparison"


def test_phase4_evaluation_preflights_all_inputs(tmp_path):
    gate, source, joint, manifest, baseline = _inputs(tmp_path)
    joint.unlink()

    with pytest.raises(FileNotFoundError, match="joint.safetensors"):
        run_phase4_evaluation(
            phase3_gate=gate,
            source_native_model=source,
            joint_model=joint,
            output_dir=tmp_path / "evaluation",
            benchmarks=[Phase4Benchmark("ground", manifest, baseline)],
            inference_mode="fast",
            source_prediction_runner=lambda *_args, **_kwargs: pytest.fail("must not run"),
            source_validator=lambda *_args: None,
        )


def test_phase4_evaluation_rejects_joint_artifact_from_another_gate(tmp_path):
    gate, source, joint, manifest, baseline = _inputs(tmp_path)
    metadata_path = joint.with_name("joint_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["provenance"]["phase3_gate_sha256"] = "a" * 64
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="supplied Phase-3 gate"):
        run_phase4_evaluation(
            phase3_gate=gate,
            source_native_model=source,
            joint_model=joint,
            output_dir=tmp_path / "evaluation",
            benchmarks=[Phase4Benchmark("ground", manifest, baseline)],
            inference_mode="fast",
            source_validator=lambda *_args: None,
        )


def test_phase4_benchmark_name_cannot_escape_output_directory(tmp_path):
    with pytest.raises(ValueError, match="benchmark name"):
        Phase4Benchmark("../outside", tmp_path / "manifest", tmp_path / "report")
