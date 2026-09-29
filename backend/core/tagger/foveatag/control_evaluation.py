"""Run the registered Phase-1 control benchmarks as one evidence-bound unit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from .benchmark import (
    PredictionSet,
    evaluate_predictions,
    file_evidence,
    load_benchmark_manifest,
)
from .control_prediction_runner import run_control_predictions
from .control_run import _checkpoint_vocabulary, _file_sha256
from .matched_fpr import compare_at_matched_fpr


@dataclass(frozen=True)
class ControlBenchmark:
    name: str
    manifest: Path
    fixed_baseline_report: Path

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError("benchmark name must contain only letters, digits, '.', '_', or '-'")


def run_control_evaluation(
    *,
    training_dir: Path,
    encoder_checkpoint: Path,
    vocabulary: Path,
    output_dir: Path,
    benchmarks: Sequence[ControlBenchmark],
    local_views: int = 4,
    resolutions: Optional[Sequence[str]] = None,
    prediction_runner: Callable[..., Path] = run_control_predictions,
    comparison_builder: Callable[..., dict[str, Any]] = compare_at_matched_fpr,
) -> dict[str, Any]:
    if not benchmarks:
        raise ValueError("at least one benchmark is required")
    if local_views < 0:
        raise ValueError("local_views must be non-negative")
    names = [benchmark.name for benchmark in benchmarks]
    if len(names) != len(set(names)):
        raise ValueError("benchmark names must be unique")

    weights = training_dir / "best_f1.safetensors"
    metadata_path = weights.with_name(f"{weights.stem}_metadata.json")
    summary_path = training_dir / "summary.json"
    vocabulary = _checkpoint_vocabulary(encoder_checkpoint, vocabulary)
    required = [weights, metadata_path, summary_path, encoder_checkpoint, vocabulary]
    for benchmark in benchmarks:
        required.extend((benchmark.manifest, benchmark.fixed_baseline_report))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase-1 evaluation inputs are missing: {missing}")

    summary = _load_json(summary_path)
    metadata = _load_json(metadata_path)
    _validate_completed_training(summary, metadata)
    _validate_external_lineage(metadata, encoder_checkpoint, vocabulary)
    lineage_paths = {
        "control_weights": weights,
        "control_metadata": metadata_path,
        "training_summary": summary_path,
        "encoder_checkpoint": encoder_checkpoint,
        "vocabulary": vocabulary,
    }
    base_encoder = metadata["encoder_provenance"].get("vision_encoder")
    if base_encoder:
        lineage_paths["base_vision_encoder"] = Path(str(base_encoder))
    lineage_evidence = file_evidence(lineage_paths)

    prepared = {}
    for benchmark in benchmarks:
        manifest = load_benchmark_manifest(benchmark.manifest)
        baseline = _load_json(benchmark.fixed_baseline_report)
        if baseline.get("benchmark") != manifest.name:
            raise ValueError(
                f"fixed baseline benchmark does not match {benchmark.name} manifest"
            )
        benchmark_paths = {
            "manifest": benchmark.manifest,
            "fixed_baseline_report": benchmark.fixed_baseline_report,
        }
        prepared[benchmark.name] = (
            manifest,
            baseline,
            benchmark_paths,
            file_evidence(benchmark_paths),
        )

    reports: dict[str, Any] = {}
    for benchmark in benchmarks:
        benchmark_dir = output_dir / benchmark.name
        _write_json_atomic(
            output_dir / "summary.json",
            _summary("running", benchmark.name, reports),
        )
        manifest, baseline, benchmark_paths, benchmark_evidence = prepared[benchmark.name]
        prediction_path = benchmark_dir / "control_predictions.json"
        prediction_runner(
            benchmark.manifest,
            encoder_checkpoint,
            vocabulary,
            weights,
            prediction_path,
            local_views=local_views,
            resolutions=resolutions,
        )
        prediction_payload = _load_json(prediction_path)
        if file_evidence(lineage_paths) != lineage_evidence:
            raise RuntimeError("Phase-1 lineage changed during evaluation")
        if file_evidence(benchmark_paths) != benchmark_evidence:
            raise RuntimeError("Phase-1 benchmark inputs changed during prediction")
        if prediction_payload.get("checkpoint_sha256") != _file_sha256(weights):
            raise ValueError("control predictions do not reference the supplied weights")
        predictions = PredictionSet.from_dict(prediction_payload)
        threshold = float(metadata["threshold"])
        evaluation = evaluate_predictions(manifest, predictions, threshold=threshold)
        evaluation_paths = {
            "manifest": benchmark.manifest,
            "control_predictions": prediction_path,
        }
        evaluation_evidence = file_evidence(evaluation_paths)
        if file_evidence(evaluation_paths) != evaluation_evidence:
            raise RuntimeError("Phase-1 evaluation inputs changed")
        evaluation["evidence"] = evaluation_evidence
        evaluation_path = benchmark_dir / "control_report.json"
        _write_json_atomic(evaluation_path, evaluation)

        matched_paths = {}
        for group in ("fine_detail", "coarse"):
            matched_inputs = {
                "manifest": benchmark.manifest,
                "fixed_baseline_report": benchmark.fixed_baseline_report,
                "candidate_predictions": prediction_path,
            }
            matched_evidence = file_evidence(matched_inputs)
            matched = comparison_builder(manifest, baseline, predictions, group=group)
            if file_evidence(matched_inputs) != matched_evidence:
                raise RuntimeError("Phase-1 matched-FPR inputs changed")
            matched["evidence"] = matched_evidence
            matched_path = benchmark_dir / f"matched_fpr_{group}.json"
            _write_json_atomic(matched_path, matched)
            matched_paths[group] = str(matched_path.resolve())
        reports[benchmark.name] = {
            "manifest": str(benchmark.manifest.resolve()),
            "fixed_baseline_report": str(benchmark.fixed_baseline_report.resolve()),
            "control_predictions": str(prediction_path.resolve()),
            "control_report": str(evaluation_path.resolve()),
            "matched_fpr": matched_paths,
        }
    result = _summary("complete", None, reports)
    _write_json_atomic(output_dir / "summary.json", result)
    return result


def _validate_completed_training(
    summary: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    if summary.get("kind") != "foveatag_siglip2_control_run":
        raise ValueError("Phase-1 training summary has the wrong kind")
    epochs = int(summary.get("epochs", -1))
    history = summary.get("history")
    if epochs <= 0 or not isinstance(history, list) or len(history) != epochs:
        raise ValueError("Phase-1 control training is not complete")
    if metadata.get("kind") != "foveatag_siglip2_control_weights":
        raise ValueError("Phase-1 best weights metadata has the wrong kind")
    for field in ("encoder_provenance", "run_contract"):
        if not isinstance(metadata.get(field), Mapping):
            raise ValueError(f"Phase-1 best weights lack {field}")
        if dict(metadata[field]) != dict(summary.get(field) or {}):
            raise ValueError(f"Phase-1 summary and best weights disagree on {field}")


def _validate_external_lineage(
    metadata: Mapping[str, Any], encoder_checkpoint: Path, vocabulary: Path
) -> None:
    provenance = metadata["encoder_provenance"]
    expected = (
        ("checkpoint", "checkpoint_sha256", encoder_checkpoint),
        ("vocabulary", "vocabulary_sha256", vocabulary),
    )
    for path_field, hash_field, path in expected:
        if Path(str(provenance.get(path_field, ""))).resolve() != path.resolve():
            raise ValueError(f"Phase-1 {path_field} path does not match")
        if provenance.get(hash_field) != _file_sha256(path):
            raise ValueError(f"Phase-1 {path_field} hash does not match")
    base = provenance.get("vision_encoder")
    base_hash = provenance.get("vision_encoder_sha256")
    if base:
        base_path = Path(str(base))
        if not base_path.is_file() or base_hash != _file_sha256(base_path):
            raise ValueError("Phase-1 base vision encoder hash does not match")


def _summary(status: str, active_benchmark: Optional[str], reports: Mapping[str, Any]):
    return {
        "format_version": 1,
        "kind": "foveatag_phase1_control_evaluation",
        "status": status,
        "active_benchmark": active_benchmark,
        "completed_benchmarks": list(reports),
        "benchmarks": dict(reports),
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_benchmark(values: Sequence[str]) -> ControlBenchmark:
    if len(values) != 3:
        raise ValueError("--benchmark requires NAME MANIFEST FIXED_BASELINE_REPORT")
    return ControlBenchmark(values[0], Path(values[1]), Path(values[2]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark", nargs=3, action="append", required=True,
        metavar=("NAME", "MANIFEST", "FIXED_BASELINE_REPORT"),
    )
    parser.add_argument("--local-views", type=int, default=4)
    parser.add_argument("--resolutions", nargs="+")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = run_control_evaluation(
        training_dir=args.training_dir,
        encoder_checkpoint=args.encoder_checkpoint,
        vocabulary=args.vocabulary,
        output_dir=args.output_dir,
        benchmarks=[_parse_benchmark(value) for value in args.benchmark],
        local_views=args.local_views,
        resolutions=args.resolutions,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
