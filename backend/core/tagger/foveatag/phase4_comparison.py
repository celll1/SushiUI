"""Compare a routed joint FoveaTag artifact with its Phase-3 source branch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .benchmark import (
    BenchmarkManifest,
    PredictionSet,
    evaluate_predictions,
    load_benchmark_manifest,
)
from .matched_fpr import compare_at_matched_fpr
from .phase3_comparison import _present_groups, _validated_runtime


def build_phase4_comparison(
    manifest: BenchmarkManifest,
    source_payload: Mapping[str, Any],
    routed_payload: Mapping[str, Any],
    fixed_baseline_report: Mapping[str, Any],
) -> dict[str, Any]:
    source = PredictionSet.from_dict(source_payload)
    routed = PredictionSet.from_dict(routed_payload)
    expected_keys = {
        (item.sample_id, resolution)
        for item in manifest.items
        for resolution, _width, _height in manifest.resolution_ladder(item)
    }
    benchmark_tags = {tag for item in manifest.items for tag in item.positive_tags}
    _validate_prediction_coverage(source, expected_keys, benchmark_tags, "source-native")
    _validate_prediction_coverage(routed, expected_keys, benchmark_tags, "routed")
    if source.model_name != "foveatag-source_native":
        raise ValueError("Phase-4 comparison has an unexpected source-native model")
    source_provenance = _mapping(source_payload.get("artifact_provenance"), "source provenance")
    routed_provenance = _mapping(routed_payload.get("artifact_provenance"), "routed provenance")
    if source_provenance.get("initialization") != "scratch" or source_provenance.get("external_weights") != []:
        raise ValueError("source-native predictions do not have pure-scratch lineage")
    if source_provenance.get("input_mode") != "source_native":
        raise ValueError("Phase-4 comparison source is not the source-native branch")
    if routed_provenance.get("initialization") != "scratch" or routed_provenance.get("external_weights") != []:
        raise ValueError("routed predictions do not have pure-scratch lineage")
    if routed_provenance.get("stage") != "joint_routed_finetune":
        raise ValueError("routed predictions are not from joint fine-tuning")
    source_hash = _digest(source_payload, "checkpoint_sha256", "source-native")
    _digest(source_payload, "checkpoint_metadata_sha256", "source-native")
    _digest(routed_payload, "checkpoint_sha256", "routed")
    _digest(routed_payload, "checkpoint_metadata_sha256", "routed")
    if routed_provenance.get("source_native_model_sha256") != source_hash:
        raise ValueError("routed artifact does not descend from the compared source-native model")
    utility = _mapping(routed_provenance.get("router_validation"), "router validation")
    learned = _finite_number(utility.get("mean_utility_recovery"), "learned utility")
    random_baseline = _finite_number(
        utility.get("random_mean_utility_recovery"), "random utility"
    )
    if learned <= random_baseline:
        raise ValueError("routed artifact does not retain learned-over-random utility")
    inference_contract = _mapping(
        routed_payload.get("inference_contract"), "routed inference contract"
    )
    local_views = int(inference_contract.get("local_views", -1))
    overview_size = int(inference_contract.get("overview_size", -1))
    if local_views <= 0 or overview_size <= 0:
        raise ValueError("routed inference contract has no positive local-view budget")
    if routed.model_name != f"foveatag-routed-{routed_payload.get('inference_mode')}":
        raise ValueError("routed predictions have an unexpected model_name")
    if int(utility.get("budget", -1)) != local_views:
        raise ValueError("routed inference K does not match router validation K")
    raw_rows = routed_payload.get("predictions")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("routed predictions must be a sequence")
    for row in raw_rows:
        boxes = _mapping_or_sequence(row, "selected_boxes")
        if len(boxes) != local_views:
            raise ValueError("routed prediction does not use its registered fixed-K budget")
        for box in boxes:
            box = _mapping(box, "selected box")
            values = {
                name: _finite_number(box.get(name), f"selected box {name}")
                for name in ("x", "y", "width", "height")
            }
            if values["width"] <= 0 or values["height"] <= 0:
                raise ValueError("selected box dimensions must be positive")
    if fixed_baseline_report.get("benchmark") != manifest.name:
        raise ValueError("fixed baseline report belongs to a different benchmark")
    source_threshold = _threshold(source_provenance, "source-native")
    routed_threshold = _threshold(routed_provenance, "routed")
    source_evaluation = evaluate_predictions(manifest, source, threshold=source_threshold)
    routed_evaluation = evaluate_predictions(manifest, routed, threshold=routed_threshold)
    groups = _present_groups(manifest)
    comparisons = {
        "routed_vs_source_native": {
            group: compare_at_matched_fpr(
                manifest,
                source_evaluation,
                routed,
                group=group,
            )
            for group in groups
        },
        "routed_vs_fixed_baseline": {
            group: compare_at_matched_fpr(
                manifest,
                fixed_baseline_report,
                routed,
                group=group,
            )
            for group in groups
        },
    }
    return {
        "format_version": 1,
        "kind": "foveatag_phase4_routed_comparison",
        "benchmark": manifest.name,
        "contract": {
            "source_native_checkpoint_sha256": source_hash,
            "routed_source_native_checkpoint_sha256": routed_provenance[
                "source_native_model_sha256"
            ],
            "inference_mode": routed_payload.get("inference_mode"),
            "inference_contract": dict(inference_contract),
            "prediction_rows": len(expected_keys),
            "source_runtime": _validated_runtime(
                source_payload.get("runtime"), mode="source-native", expected_rows=len(expected_keys)
            ),
            "routed_runtime": _validated_runtime(
                routed_payload.get("runtime"), mode="routed", expected_rows=len(expected_keys)
            ),
            "router_validation": dict(utility),
        },
        "evaluations": {
            "source_native": source_evaluation,
            "routed": routed_evaluation,
        },
        "matched_fpr_comparisons": comparisons,
    }


def _validate_prediction_coverage(predictions, expected_keys, tags, name):
    keys = {(row.sample_id, row.resolution) for row in predictions.rows}
    if keys != expected_keys or len(predictions.rows) != len(expected_keys):
        raise ValueError(f"{name} prediction coverage differs from the manifest")
    for row in predictions.rows:
        missing = tags - set(row.scores)
        if missing:
            raise ValueError(f"{name} prediction lacks benchmark tags {sorted(missing)[:3]}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _mapping_or_sequence(row: Any, name: str) -> Sequence[Any]:
    if not isinstance(row, Mapping):
        raise ValueError("routed prediction row must be an object")
    value = row.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"routed prediction {name} must be a sequence")
    return value


def _digest(payload: Mapping[str, Any], key: str, name: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} predictions lack a valid {key}")
    return value


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be finite")
    return result


def _threshold(provenance: Mapping[str, Any], name: str) -> float:
    value = _finite_number(provenance.get("threshold"), f"{name} threshold")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} threshold is outside [0, 1]")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-native", type=Path, required=True)
    parser.add_argument("--routed", type=Path, required=True)
    parser.add_argument("--fixed-baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = build_phase4_comparison(
        load_benchmark_manifest(args.manifest),
        _load_json(args.source_native),
        _load_json(args.routed),
        _load_json(args.fixed_baseline_report),
    )
    _write_json_atomic(args.output, report)
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
