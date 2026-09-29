"""Build the pre-registered Phase-3 A/B/C benchmark comparison report."""

from __future__ import annotations

import argparse
import json
import math
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
from .supervised_data import PHASE3_INPUT_MODES


def build_phase3_comparison(
    manifest: BenchmarkManifest,
    prediction_payloads: Mapping[str, Mapping[str, Any]],
    fixed_baseline_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the paired experiment and collect gate evidence without deciding it."""
    if set(prediction_payloads) != set(PHASE3_INPUT_MODES):
        raise ValueError(f"predictions must contain exactly {PHASE3_INPUT_MODES}")
    predictions = {
        mode: PredictionSet.from_dict(prediction_payloads[mode])
        for mode in PHASE3_INPUT_MODES
    }
    expected_keys = {
        (item.sample_id, resolution)
        for item in manifest.items
        for resolution, _width, _height in manifest.resolution_ladder(item)
    }
    benchmark_tags = {
        tag for item in manifest.items for tag in item.positive_tags
    }
    shared_hashes = set()
    contracts = {}
    for mode, prediction_set in predictions.items():
        payload = prediction_payloads[mode]
        provenance = payload.get("artifact_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{mode} predictions lack artifact provenance")
        if provenance.get("initialization") != "scratch":
            raise ValueError(f"{mode} is not a scratch-initialized branch")
        if provenance.get("external_weights") != []:
            raise ValueError(f"{mode} contains external-weight provenance")
        if provenance.get("input_mode") != mode:
            raise ValueError(f"{mode} artifact provenance has the wrong input mode")
        shared_hash = provenance.get("shared_initialization_sha256")
        if not isinstance(shared_hash, str) or not shared_hash:
            raise ValueError(f"{mode} lacks the shared initialization hash")
        shared_hashes.add(shared_hash)
        if prediction_set.model_name != f"foveatag-{mode}":
            raise ValueError(f"{mode} has an unexpected model_name")
        for name in ("checkpoint_sha256", "checkpoint_metadata_sha256"):
            digest = payload.get(name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{mode} predictions lack a valid {name}")
        keys = {(row.sample_id, row.resolution) for row in prediction_set.rows}
        if keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise ValueError(
                f"{mode} prediction coverage differs from the manifest "
                f"(missing={missing[:3]}, extra={extra[:3]})"
            )
        for row in prediction_set.rows:
            missing_tags = benchmark_tags - set(row.scores)
            if missing_tags:
                raise ValueError(
                    f"{mode} prediction {row.sample_id}/{row.resolution} "
                    f"lacks tags {sorted(missing_tags)[:3]}"
                )
        threshold = float(provenance.get("threshold", 0.5))
        if not 0 <= threshold <= 1:
            raise ValueError(f"{mode} validation threshold is outside [0, 1]")
        runtime = _validated_runtime(
            payload.get("runtime"), mode=mode, expected_rows=len(prediction_set.rows)
        )
        contracts[mode] = {
            "model_name": prediction_set.model_name,
            "prediction_rows": len(prediction_set.rows),
            "validation_threshold": threshold,
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "checkpoint_metadata_sha256": payload["checkpoint_metadata_sha256"],
            "runtime": runtime,
        }
    if len(shared_hashes) != 1:
        raise ValueError("Phase-3 branches do not share the same initialization")
    if fixed_baseline_report.get("benchmark") != manifest.name:
        raise ValueError("fixed baseline report belongs to a different benchmark")

    evaluations = {
        mode: evaluate_predictions(
            manifest,
            predictions[mode],
            threshold=contracts[mode]["validation_threshold"],
        )
        for mode in PHASE3_INPUT_MODES
    }
    groups = _present_groups(manifest)
    source_native = predictions["source_native"]
    comparisons = {}
    for baseline_mode in ("overview_only", "overview_placebo"):
        comparisons[f"source_native_vs_{baseline_mode}"] = {
            group: compare_at_matched_fpr(
                manifest,
                evaluations[baseline_mode],
                source_native,
                group=group,
            )
            for group in groups
        }
    comparisons["source_native_vs_fixed_baseline"] = {
        group: compare_at_matched_fpr(
            manifest,
            fixed_baseline_report,
            source_native,
            group=group,
        )
        for group in groups
    }
    return {
        "format_version": 1,
        "kind": "foveatag_phase3_abc_comparison",
        "benchmark": manifest.name,
        "gate_decision": "not_automated",
        "gate_decision_reason": (
            "The pre-registered design has no numeric materiality or latency "
            "threshold; this report preserves evidence for the owner's decision."
        ),
        "contract": {
            "initialization": "scratch",
            "external_weights": [],
            "shared_initialization_sha256": next(iter(shared_hashes)),
            "expected_prediction_rows": len(expected_keys),
            "branches": contracts,
        },
        "evaluations": evaluations,
        "matched_fpr_comparisons": comparisons,
    }


def _present_groups(manifest: BenchmarkManifest) -> tuple[str, ...]:
    fine = {tag for item in manifest.items for tag in item.fine_detail_tags}
    coarse = {tag for item in manifest.items for tag in item.coarse_tags}
    groups = []
    if fine:
        groups.append("fine_detail")
    if coarse:
        groups.append("coarse")
    if fine or coarse:
        groups.append("all")
    if not groups:
        raise ValueError("benchmark has no fine-detail or coarse tag groups")
    return tuple(groups)


def _validated_runtime(
    raw: Any, *, mode: str, expected_rows: int
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{mode} predictions lack runtime measurements")
    device = raw.get("device")
    dtype = raw.get("dtype")
    if not isinstance(device, str) or not device or not isinstance(dtype, str) or not dtype:
        raise ValueError(f"{mode} runtime device and dtype must be non-empty strings")
    try:
        row_count = int(raw.get("prediction_rows", -1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{mode} runtime prediction_rows must be an integer") from error
    if row_count != expected_rows:
        raise ValueError(f"{mode} runtime row count does not match predictions")
    try:
        wall_seconds = float(raw.get("wall_seconds", -1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{mode} runtime wall_seconds must be numeric") from error
    if not math.isfinite(wall_seconds) or wall_seconds < 0:
        raise ValueError(f"{mode} runtime wall_seconds must be finite and non-negative")
    memory = {}
    for name in ("peak_allocated_mib", "peak_reserved_mib"):
        value = raw.get(name)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{mode} runtime {name} must be numeric") from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{mode} runtime {name} must be finite and non-negative")
        memory[name] = value
    return {
        "device": device,
        "dtype": dtype,
        "prediction_rows": expected_rows,
        "wall_seconds": wall_seconds,
        **memory,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overview-only", type=Path, required=True)
    parser.add_argument("--overview-placebo", type=Path, required=True)
    parser.add_argument("--source-native", type=Path, required=True)
    parser.add_argument("--fixed-baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = build_phase3_comparison(
        load_benchmark_manifest(args.manifest),
        {
            "overview_only": _load_json(args.overview_only),
            "overview_placebo": _load_json(args.overview_placebo),
            "source_native": _load_json(args.source_native),
        },
        _load_json(args.fixed_baseline_report),
    )
    _write_json_atomic(args.output, report)
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
