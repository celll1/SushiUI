"""Run registered Phase-4 source-versus-routed benchmark evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from .benchmark import load_benchmark_manifest
from .phase4_comparison import build_phase4_comparison
from .routed_prediction_runner import run_routed_predictions, validate_joint_artifact
from .router_experiment import validate_router_teacher
from .supervised_prediction_runner import _file_sha256, run_supervised_predictions


@dataclass(frozen=True)
class Phase4Benchmark:
    name: str
    manifest: Path
    fixed_baseline_report: Path

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError("benchmark name must contain only letters, digits, '.', '_', or '-'")


def run_phase4_evaluation(
    *,
    phase3_gate: Path,
    source_native_model: Path,
    joint_model: Path,
    output_dir: Path,
    benchmarks: Sequence[Phase4Benchmark],
    inference_mode: str,
    resolutions: Optional[Sequence[str]] = None,
    device: str = "cuda",
    source_prediction_runner: Callable[..., Path] = run_supervised_predictions,
    routed_prediction_runner: Callable[..., Path] = run_routed_predictions,
    comparison_builder: Callable[..., dict[str, Any]] = build_phase4_comparison,
    source_validator: Callable[..., Any] = validate_router_teacher,
) -> dict[str, Any]:
    if not benchmarks:
        raise ValueError("at least one benchmark is required")
    names = [benchmark.name for benchmark in benchmarks]
    if len(names) != len(set(names)):
        raise ValueError("benchmark names must be unique")
    required = [
        phase3_gate,
        source_native_model,
        _sidecar_path(source_native_model),
        joint_model,
        _sidecar_path(joint_model),
    ]
    for benchmark in benchmarks:
        required.extend((benchmark.manifest, benchmark.fixed_baseline_report))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase-4 evaluation inputs are missing: {missing}")
    source_validator(phase3_gate, source_native_model)
    joint_metadata = _load_json(_sidecar_path(joint_model))
    provenance = validate_joint_artifact(joint_metadata)
    if provenance.get("phase3_gate_sha256") != _file_sha256(phase3_gate):
        raise ValueError("joint artifact does not reference the supplied Phase-3 gate")
    if provenance.get("source_native_model_sha256") != _file_sha256(source_native_model):
        raise ValueError("joint artifact does not descend from the supplied source-native model")

    reports: dict[str, Any] = {}
    for benchmark in benchmarks:
        benchmark_dir = output_dir / benchmark.name
        _write_json_atomic(
            output_dir / "summary.json",
            _summary("running", benchmark.name, inference_mode, reports),
        )
        source_path = benchmark_dir / "source_native_predictions.json"
        routed_path = benchmark_dir / "routed_predictions.json"
        source_prediction_runner(
            benchmark.manifest,
            source_native_model,
            source_path,
            resolutions=resolutions,
            device=device,
        )
        routed_prediction_runner(
            benchmark.manifest,
            joint_model,
            routed_path,
            inference_mode=inference_mode,
            resolutions=resolutions,
            device=device,
        )
        report = comparison_builder(
            load_benchmark_manifest(benchmark.manifest),
            _load_json(source_path),
            _load_json(routed_path),
            _load_json(benchmark.fixed_baseline_report),
        )
        report_path = benchmark_dir / "phase4_comparison.json"
        _write_json_atomic(report_path, report)
        reports[benchmark.name] = {
            "manifest": str(benchmark.manifest.resolve()),
            "fixed_baseline_report": str(benchmark.fixed_baseline_report.resolve()),
            "source_native_predictions": str(source_path.resolve()),
            "routed_predictions": str(routed_path.resolve()),
            "comparison_report": str(report_path.resolve()),
        }
    result = _summary("complete", None, inference_mode, reports)
    _write_json_atomic(output_dir / "summary.json", result)
    return result


def _summary(status, active_benchmark, inference_mode, reports):
    return {
        "format_version": 1,
        "kind": "foveatag_phase4_evaluation",
        "status": status,
        "active_benchmark": active_benchmark,
        "inference_mode": inference_mode,
        "completed_benchmarks": list(reports),
        "benchmarks": dict(reports),
    }


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metadata.json")


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


def _parse_benchmark(values: Sequence[str]) -> Phase4Benchmark:
    if len(values) != 3:
        raise ValueError("--benchmark requires NAME MANIFEST FIXED_BASELINE_REPORT")
    return Phase4Benchmark(values[0], Path(values[1]), Path(values[2]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-gate", type=Path, required=True)
    parser.add_argument("--source-native-model", type=Path, required=True)
    parser.add_argument("--joint-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-mode", required=True)
    parser.add_argument(
        "--benchmark", nargs=3, action="append", required=True,
        metavar=("NAME", "MANIFEST", "FIXED_BASELINE_REPORT"),
    )
    parser.add_argument("--resolutions", nargs="+")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = run_phase4_evaluation(
        phase3_gate=args.phase3_gate,
        source_native_model=args.source_native_model,
        joint_model=args.joint_model,
        output_dir=args.output_dir,
        benchmarks=[_parse_benchmark(value) for value in args.benchmark],
        inference_mode=args.inference_mode,
        resolutions=args.resolutions,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
