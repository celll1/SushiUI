"""CPU orchestration tests for the Phase-1 control evaluation."""

import hashlib
import json

import pytest

from core.tagger.foveatag.control_evaluation import (
    ControlBenchmark,
    run_control_evaluation,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path):
    training = tmp_path / "training"
    weights = _write(training / "best_f1.safetensors", b"weights")
    encoder = _write(tmp_path / "encoder.safetensors", b"encoder")
    vocabulary = _write(tmp_path / "vocabulary.json", {})
    base = _write(tmp_path / "base.safetensors", b"base")
    provenance = {
        "checkpoint": str(encoder.resolve()),
        "checkpoint_sha256": _sha(encoder),
        "vocabulary": str(vocabulary.resolve()),
        "vocabulary_sha256": _sha(vocabulary),
        "vision_encoder": str(base.resolve()),
        "vision_encoder_sha256": _sha(base),
    }
    contract = {"seed": 7}
    _write(training / "best_f1_metadata.json", {
        "kind": "foveatag_siglip2_control_weights",
        "encoder_provenance": provenance,
        "run_contract": contract,
        "threshold": 0.4,
    })
    _write(training / "summary.json", {
        "kind": "foveatag_siglip2_control_run",
        "encoder_provenance": provenance,
        "run_contract": contract,
        "epochs": 1,
        "history": [{}],
    })
    manifest = _write(tmp_path / "manifest.json", {
        "format_version": 1,
        "name": "ground",
        "evidence_mode": "dense_oracle",
        "items": [{
            "sample_id": "a",
            "image_path": "a.png",
            "width": 32,
            "height": 32,
            "positive_tags": ["detail", "person"],
            "fine_detail_tags": ["detail"],
            "coarse_tags": ["person"],
            "evidence": [],
        }],
    })
    baseline = _write(tmp_path / "baseline.json", {
        "benchmark": "ground",
        "model_name": "fixed",
        "resolutions": {"native": {"per_tag": {
            "detail": {"fpr": 0, "precision": 1, "recall": 1, "average_precision": 1, "f1": 1},
            "person": {"fpr": 0, "precision": 1, "recall": 1, "average_precision": 1, "f1": 1},
        }}},
    })
    return training, encoder, vocabulary, weights, manifest, baseline


def test_control_evaluation_regenerates_predictions_and_both_groups(tmp_path):
    training, encoder, vocabulary, weights, manifest, baseline = _inputs(tmp_path)
    calls = []

    def prediction_runner(manifest_path, checkpoint, vocab, control, output, **kwargs):
        calls.append((manifest_path, checkpoint, vocab, control, kwargs))
        return _write(output, {
            "format_version": 1,
            "model_name": "control",
            "checkpoint_sha256": _sha(weights),
            "predictions": [{
                "sample_id": "a",
                "resolution": "native",
                "scores": {"detail": 0.8, "person": 0.9},
            }],
        })

    groups = []

    def comparison_builder(_manifest, _baseline, _predictions, *, group):
        groups.append(group)
        return {"group": group}

    output = tmp_path / "evaluation"
    result = run_control_evaluation(
        training_dir=training,
        encoder_checkpoint=encoder,
        vocabulary=vocabulary,
        output_dir=output,
        benchmarks=[ControlBenchmark("ground", manifest, baseline)],
        resolutions=("native",),
        prediction_runner=prediction_runner,
        comparison_builder=comparison_builder,
    )

    assert len(calls) == 1
    assert calls[0][-1] == {"local_views": 4, "resolutions": ("native",)}
    assert groups == ["fine_detail", "coarse"]
    assert result["status"] == "complete"
    assert result["completed_benchmarks"] == ["ground"]
    report = json.loads(
        (output / "ground" / "control_report.json").read_text(encoding="utf-8")
    )
    assert {row["role"] for row in report["evidence"]} == {
        "manifest", "control_predictions"
    }


def test_control_evaluation_preflights_training_completion(tmp_path):
    training, encoder, vocabulary, _weights, manifest, baseline = _inputs(tmp_path)
    summary = json.loads((training / "summary.json").read_text(encoding="utf-8"))
    summary["history"] = []
    _write(training / "summary.json", summary)

    with pytest.raises(ValueError, match="training is not complete"):
        run_control_evaluation(
            training_dir=training,
            encoder_checkpoint=encoder,
            vocabulary=vocabulary,
            output_dir=tmp_path / "evaluation",
            benchmarks=[ControlBenchmark("ground", manifest, baseline)],
            prediction_runner=lambda *_args, **_kwargs: pytest.fail("must not run"),
        )


def test_control_evaluation_preflights_every_benchmark_before_prediction(tmp_path):
    training, encoder, vocabulary, _weights, manifest, baseline = _inputs(tmp_path)
    wrong_baseline = _write(tmp_path / "wrong_baseline.json", {
        "benchmark": "different-benchmark",
    })

    with pytest.raises(ValueError, match="fixed baseline benchmark does not match"):
        run_control_evaluation(
            training_dir=training,
            encoder_checkpoint=encoder,
            vocabulary=vocabulary,
            output_dir=tmp_path / "evaluation",
            benchmarks=[
                ControlBenchmark("ground", manifest, baseline),
                ControlBenchmark("wrong", manifest, wrong_baseline),
            ],
            prediction_runner=lambda *_args, **_kwargs: pytest.fail("must not run"),
        )


def test_control_evaluation_rejects_changed_external_encoder(tmp_path):
    training, encoder, vocabulary, _weights, manifest, baseline = _inputs(tmp_path)
    encoder.write_bytes(b"changed")

    with pytest.raises(ValueError, match="checkpoint hash does not match"):
        run_control_evaluation(
            training_dir=training,
            encoder_checkpoint=encoder,
            vocabulary=vocabulary,
            output_dir=tmp_path / "evaluation",
            benchmarks=[ControlBenchmark("ground", manifest, baseline)],
            prediction_runner=lambda *_args, **_kwargs: pytest.fail("must not run"),
        )


def test_control_benchmark_name_cannot_escape_output_directory(tmp_path):
    with pytest.raises(ValueError, match="benchmark name"):
        ControlBenchmark("../outside", tmp_path / "manifest", tmp_path / "report")
