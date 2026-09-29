"""Run Phase-3 A/B/C predictions and paired reports after training completes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from .benchmark import load_benchmark_manifest
from .phase3_comparison import build_phase3_comparison
from .supervised_data import PHASE3_INPUT_MODES
from .supervised_prediction_runner import run_supervised_predictions


@dataclass(frozen=True)
class Phase3Benchmark:
    name: str
    manifest: Path
    fixed_baseline_report: Path

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError("benchmark name must contain only letters, digits, '.', '_', or '-'")


def run_phase3_evaluation(
    *,
    comparison_dir: Path,
    benchmarks: Sequence[Phase3Benchmark],
    resolutions: Optional[Sequence[str]] = None,
    device: str = "cuda",
    prediction_runner: Callable[..., Path] = run_supervised_predictions,
    comparison_builder: Callable[..., dict[str, Any]] = build_phase3_comparison,
) -> dict[str, Any]:
    """Regenerate every branch prediction before building paired reports."""
    if not benchmarks:
        raise ValueError("at least one benchmark is required")
    names = [benchmark.name for benchmark in benchmarks]
    if len(names) != len(set(names)):
        raise ValueError("benchmark names must be unique")
    artifacts = {
        mode: comparison_dir / mode / "best_f1.safetensors"
        for mode in PHASE3_INPUT_MODES
    }
    training_summary_path = comparison_dir / "comparison_summary.json"
    missing = [
        str(path)
        for path in artifacts.values()
        for path in (path, path.with_name(f"{path.stem}_metadata.json"))
        if not path.is_file()
    ]
    if not training_summary_path.is_file():
        missing.append(str(training_summary_path))
    for benchmark in benchmarks:
        if not benchmark.manifest.is_file():
            missing.append(str(benchmark.manifest))
        if not benchmark.fixed_baseline_report.is_file():
            missing.append(str(benchmark.fixed_baseline_report))
    if missing:
        raise FileNotFoundError(f"Phase-3 evaluation inputs are missing: {missing}")
    training_summary = _load_json(training_summary_path)
    if training_summary.get("kind") != "foveatag_phase3_supervised_comparison":
        raise ValueError("Phase-3 training summary has the wrong kind")
    if training_summary.get("status") != "complete":
        raise ValueError("Phase-3 A/B/C training is not complete")
    if training_summary.get("completed_modes") != list(PHASE3_INPUT_MODES):
        raise ValueError("Phase-3 training summary does not contain every mode")

    evaluation_root = comparison_dir / "evaluation"
    reports: dict[str, Any] = {}
    for benchmark in benchmarks:
        benchmark_dir = evaluation_root / benchmark.name
        _write_json_atomic(
            evaluation_root / "summary.json",
            _evaluation_summary(
                status="running",
                active_benchmark=benchmark.name,
                reports=reports,
            ),
        )
        payloads = {}
        prediction_paths = {}
        for mode in PHASE3_INPUT_MODES:
            output = benchmark_dir / f"{mode}_predictions.json"
            prediction_runner(
                benchmark.manifest,
                artifacts[mode],
                output,
                resolutions=resolutions,
                device=device,
            )
            payloads[mode] = _load_json(output)
            prediction_paths[mode] = str(output.resolve())
        report = comparison_builder(
            load_benchmark_manifest(benchmark.manifest),
            payloads,
            _load_json(benchmark.fixed_baseline_report),
        )
        report_path = benchmark_dir / "phase3_comparison.json"
        _write_json_atomic(report_path, report)
        reports[benchmark.name] = {
            "manifest": str(benchmark.manifest.resolve()),
            "fixed_baseline_report": str(benchmark.fixed_baseline_report.resolve()),
            "predictions": prediction_paths,
            "comparison_report": str(report_path.resolve()),
        }
    result = _evaluation_summary(
        status="complete",
        active_benchmark=None,
        reports=reports,
    )
    _write_json_atomic(evaluation_root / "summary.json", result)
    return result


def _evaluation_summary(
    *, status: str, active_benchmark: Optional[str], reports: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "foveatag_phase3_evaluation",
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
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_benchmark(values: Sequence[str]) -> Phase3Benchmark:
    if len(values) != 3:
        raise ValueError("--benchmark requires NAME MANIFEST FIXED_BASELINE_REPORT")
    return Phase3Benchmark(values[0], Path(values[1]), Path(values[2]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "MANIFEST", "FIXED_BASELINE_REPORT"),
    )
    parser.add_argument("--resolutions", nargs="+")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = run_phase3_evaluation(
        comparison_dir=args.comparison_dir,
        benchmarks=[_parse_benchmark(value) for value in args.benchmark],
        resolutions=args.resolutions,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
