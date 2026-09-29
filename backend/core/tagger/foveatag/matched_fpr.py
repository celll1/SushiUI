"""Compare benchmark predictions at the fixed baseline's false-positive rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .benchmark import (
    BenchmarkManifest,
    PredictionSet,
    evaluate_predictions,
    load_benchmark_manifest,
    load_prediction_set,
)


DEFAULT_THRESHOLDS = (
    0.001,
    0.005,
    *(value / 100 for value in range(1, 100)),
    0.995,
    0.999,
)


def compare_at_matched_fpr(
    manifest: BenchmarkManifest,
    baseline_report: Mapping[str, Any],
    candidate: PredictionSet,
    *,
    group: str = "fine_detail",
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Select one candidate threshold per resolution without exceeding baseline macro FPR."""
    if group not in ("fine_detail", "coarse", "all"):
        raise ValueError("group must be fine_detail, coarse, or all")
    if not thresholds or any(not 0 < threshold < 1 for threshold in thresholds):
        raise ValueError("thresholds must lie within (0, 1)")
    tags = _group_tags(manifest, group)
    reports = [
        (float(threshold), evaluate_predictions(manifest, candidate, threshold=float(threshold)))
        for threshold in thresholds
    ]
    comparisons = {}
    baseline_resolutions = baseline_report.get("resolutions", {})
    candidate_resolutions = set(reports[0][1]["resolutions"])
    for resolution in sorted(candidate_resolutions):
        if resolution not in baseline_resolutions:
            raise ValueError(f"baseline report lacks candidate resolution {resolution}")
        baseline_values = _operating_point(baseline_resolutions[resolution], tags)
        target_fpr = baseline_values["macro_fpr"]
        candidates = []
        for threshold, report in reports:
            values = _operating_point(report["resolutions"][resolution], tags)
            candidates.append((threshold, values, report["resolutions"][resolution]))
        feasible = [entry for entry in candidates if entry[1]["macro_fpr"] <= target_fpr]
        if feasible:
            selected = max(
                feasible,
                key=lambda entry: (entry[1]["macro_fpr"], entry[1]["macro_recall"], -entry[0]),
            )
        else:
            selected = min(candidates, key=lambda entry: entry[1]["macro_fpr"])
        threshold, candidate_values, candidate_resolution = selected
        comparisons[resolution] = {
            "baseline": baseline_values,
            "candidate": {
                **candidate_values,
                "macro_average_precision": candidate_resolution["groups"][group][
                    "macro_average_precision"
                ],
                "macro_f1": candidate_resolution["groups"][group]["macro_f1"],
                "threshold": threshold,
            },
            "delta": {
                "macro_average_precision": (
                    candidate_resolution["groups"][group]["macro_average_precision"]
                    - baseline_resolutions[resolution]["groups"][group]["macro_average_precision"]
                ),
                "macro_recall": candidate_values["macro_recall"] - baseline_values["macro_recall"],
                "macro_fpr": candidate_values["macro_fpr"] - target_fpr,
            },
        }
    return {
        "format_version": 1,
        "benchmark": manifest.name,
        "baseline_model": baseline_report.get("model_name"),
        "candidate_model": candidate.model_name,
        "group": group,
        "threshold_search": list(thresholds),
        "resolutions": comparisons,
    }


def _group_tags(manifest: BenchmarkManifest, group: str) -> tuple[str, ...]:
    fine = {tag for item in manifest.items for tag in item.fine_detail_tags}
    coarse = {tag for item in manifest.items for tag in item.coarse_tags}
    if group == "fine_detail":
        return tuple(sorted(fine))
    if group == "coarse":
        return tuple(sorted(coarse))
    return tuple(sorted(fine | coarse))


def _operating_point(resolution_report: Mapping[str, Any], tags: Sequence[str]) -> dict[str, float]:
    per_tag = resolution_report["per_tag"]
    rows = [per_tag[tag] for tag in tags if tag in per_tag]
    if not rows:
        raise ValueError("resolution report has no requested tags")
    return {
        "macro_fpr": _mean(row["fpr"] for row in rows),
        "macro_precision": _mean(row["precision"] for row in rows),
        "macro_recall": _mean(row["recall"] for row in rows),
        "macro_average_precision": _mean(row["average_precision"] for row in rows),
        "macro_f1": _mean(row["f1"] for row in rows),
    }


def _mean(values) -> float:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("candidate_predictions", type=Path)
    parser.add_argument("--group", choices=("fine_detail", "coarse", "all"), default="fine_detail")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_at_matched_fpr(
        load_benchmark_manifest(args.manifest),
        json.loads(args.baseline_report.read_text(encoding="utf-8")),
        load_prediction_set(args.candidate_predictions),
        group=args.group,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
