"""Validate and execute the registered Phase-1 residual diagnostic plan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .control_evaluation import ControlBenchmark, run_control_evaluation
from .control_run import _file_sha256, run_control_training


@dataclass(frozen=True)
class ResolvedControlPlan:
    raw: Mapping[str, Any]
    checkpoint: Path
    vocabulary: Path
    datasets_db: Path
    holdout_manifests: tuple[Path, ...]
    training_dir: Path
    evaluation_dir: Path
    benchmarks: tuple[ControlBenchmark, ...]


def load_control_plan(path: Path | str, *, repo_root: Path) -> ResolvedControlPlan:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Phase-1 plan must contain a JSON object")
    if value.get("format_version") != 1:
        raise ValueError("unsupported Phase-1 plan format")
    if value.get("kind") != "foveatag_phase1_residual_diagnostic_plan":
        raise ValueError("artifact is not a Phase-1 residual diagnostic plan")
    model = _mapping(value, "model")
    checkpoint = _verify_external(_mapping(model, "checkpoint"), "checkpoint")
    vocabulary = _verify_external(_mapping(model, "vocabulary"), "vocabulary")
    _verify_external(_mapping(model, "base_vision_encoder"), "base vision encoder")

    data = _mapping(value, "data")
    datasets_db = _verify_database(_mapping(data, "datasets_db"))
    holdouts = tuple(
        _verify_repo_file(row, repo_root, "holdout manifest")
        for row in _sequence(data, "holdout_manifests")
    )
    training = _mapping(value, "training")
    epochs = _positive_int(training, "epochs")
    epoch_samples = _positive_int(training, "epoch_samples")
    batch_size = _positive_int(training, "batch_size")
    optimizer_steps = _positive_int(training, "optimizer_steps")
    if optimizer_steps != epochs * math.ceil(epoch_samples / batch_size):
        raise ValueError("Phase-1 plan optimizer_steps does not match its sample budget")
    if _mapping(training, "loss") != {
        "kind": "asymmetric_loss",
        "gamma_negative": 4.0,
        "gamma_positive": 1.0,
        "negative_clip": 0.05,
        "epsilon": 1e-6,
        "detach_focal_weight": False,
        "reduction": "mean",
    }:
        raise ValueError("Phase-1 plan ASL contract does not match the implemented loss")

    evaluation = _mapping(value, "evaluation")
    if list(_sequence(evaluation, "groups")) != ["fine_detail", "coarse"]:
        raise ValueError("Phase-1 plan must evaluate fine_detail and coarse groups")
    benchmarks = []
    for row in _sequence(evaluation, "benchmarks"):
        row = _as_mapping(row, "evaluation benchmark")
        manifest = _repo_path(row, "manifest_repo_path", repo_root)
        baseline = _repo_path(row, "fixed_baseline_report_repo_path", repo_root)
        if not manifest.is_file() or not baseline.is_file():
            raise FileNotFoundError("Phase-1 evaluation benchmark input is missing")
        if _file_sha256(baseline) != str(row.get("fixed_baseline_report_sha256", "")):
            raise ValueError(f"Phase-1 fixed baseline hash changed: {baseline}")
        benchmarks.append(ControlBenchmark(str(row["name"]), manifest, baseline))
    outputs = _mapping(value, "outputs")
    return ResolvedControlPlan(
        raw=dict(value),
        checkpoint=checkpoint,
        vocabulary=vocabulary,
        datasets_db=datasets_db,
        holdout_manifests=holdouts,
        training_dir=Path(str(outputs["training_dir"])),
        evaluation_dir=Path(str(outputs["evaluation_dir"])),
        benchmarks=tuple(benchmarks),
    )


def execute_control_training_plan(
    plan: ResolvedControlPlan,
    *,
    resume: Optional[Path] = None,
    runner: Callable[..., Mapping[str, Any]] = run_control_training,
) -> Mapping[str, Any]:
    latest = plan.training_dir / "latest.pt"
    if resume is None and latest.exists():
        raise FileExistsError("Phase-1 output already has latest.pt; pass an explicit resume")
    data = _mapping(plan.raw, "data")
    training = _mapping(plan.raw, "training")
    return runner(
        checkpoint=plan.checkpoint,
        vocabulary_path=plan.vocabulary,
        datasets_db=plan.datasets_db,
        output_dir=plan.training_dir,
        train_counts=_int_mapping(data, "train_counts"),
        validation_counts=_int_mapping(data, "validation_counts"),
        domain_weights=_float_mapping(data, "domain_weights"),
        holdout_manifests=plan.holdout_manifests,
        photo_dataset_id=int(data["photo_dataset_id"]),
        photo_loss_weight=float(data["photo_loss_weight"]),
        epochs=int(training["epochs"]),
        epoch_samples=int(training["epoch_samples"]),
        batch_size=int(training["batch_size"]),
        local_views=int(training["local_views"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_steps=int(training["warmup_steps"]),
        log_every=100,
        seed=int(training["seed"]),
        max_source_pixels=int(training["max_source_pixels"]),
        resume=resume,
    )


def execute_control_evaluation_plan(
    plan: ResolvedControlPlan,
    *,
    runner: Callable[..., Mapping[str, Any]] = run_control_evaluation,
) -> Mapping[str, Any]:
    training = _mapping(plan.raw, "training")
    return runner(
        training_dir=plan.training_dir,
        encoder_checkpoint=plan.checkpoint,
        vocabulary=plan.vocabulary,
        output_dir=plan.evaluation_dir,
        benchmarks=plan.benchmarks,
        local_views=int(training["local_views"]),
    )


def _verify_external(row: Mapping[str, Any], name: str) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"Phase-1 {name} is missing: {path}")
    if path.stat().st_size != int(row.get("bytes", -1)):
        raise ValueError(f"Phase-1 {name} size changed: {path}")
    if _file_sha256(path) != str(row.get("sha256", "")):
        raise ValueError(f"Phase-1 {name} hash changed: {path}")
    return path


def _verify_database(row: Mapping[str, Any]) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"Phase-1 datasets database is missing: {path}")
    stat = path.stat()
    if stat.st_size != int(row.get("bytes", -1)) or stat.st_mtime_ns != int(
        row.get("mtime_ns", -1)
    ):
        raise ValueError("Phase-1 datasets database identity changed")
    return path


def _verify_repo_file(row: Any, repo_root: Path, name: str) -> Path:
    row = _as_mapping(row, name)
    path = _repo_path(row, "repo_path", repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"Phase-1 {name} is missing: {path}")
    if path.stat().st_size != int(row.get("bytes", -1)):
        raise ValueError(f"Phase-1 {name} size changed: {path}")
    if _file_sha256(path) != str(row.get("sha256", "")):
        raise ValueError(f"Phase-1 {name} hash changed: {path}")
    return path


def _repo_path(row: Mapping[str, Any], field: str, repo_root: Path) -> Path:
    relative = Path(str(row.get(field, "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Phase-1 {field} must stay within the repository")
    return (repo_root / relative).resolve()


def _mapping(parent: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    return _as_mapping(parent.get(field), field)


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase-1 {name} must be an object")
    return value


def _sequence(parent: Mapping[str, Any], field: str) -> Sequence[Any]:
    value = parent.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Phase-1 {field} must be a non-empty list")
    return value


def _positive_int(parent: Mapping[str, Any], field: str) -> int:
    value = int(parent.get(field, 0))
    if value <= 0:
        raise ValueError(f"Phase-1 {field} must be positive")
    return value


def _int_mapping(parent: Mapping[str, Any], field: str) -> dict[int, int]:
    return {int(key): int(value) for key, value in _mapping(parent, field).items()}


def _float_mapping(parent: Mapping[str, Any], field: str) -> dict[int, float]:
    return {int(key): float(value) for key, value in _mapping(parent, field).items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[4])
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    plan = load_control_plan(args.plan, repo_root=args.repo_root)
    if args.action == "train":
        result = execute_control_training_plan(plan, resume=args.resume)
    else:
        if args.resume is not None:
            raise ValueError("--resume is only valid for train")
        result = execute_control_evaluation_plan(plan)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
