"""Record and verify the explicit owner decision for FoveaTag Phase 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .benchmark import load_benchmark_manifest
from .phase4_comparison import build_phase4_comparison


PHASE4_GATE_FORMAT_VERSION = 1


def record_phase4_gate(
    evaluation_summary_path: Path | str,
    output_path: Path | str,
    *,
    decision: str,
    rationale: str,
    decided_by: str,
) -> Path:
    if decision not in {"pass", "fail"}:
        raise ValueError("decision must be pass or fail")
    if not rationale.strip() or not decided_by.strip():
        raise ValueError("rationale and decided_by must be non-empty")
    summary_path = Path(evaluation_summary_path).resolve()
    summary = _load_json(summary_path)
    evidence = _validated_evaluation_evidence(summary_path, summary)
    rows = [
        {"role": role, "path": str(path), "sha256": _file_sha256(path)}
        for path, role in sorted(evidence.items(), key=lambda item: str(item[0]))
    ]
    payload = {
        "format_version": PHASE4_GATE_FORMAT_VERSION,
        "kind": "foveatag_phase4_gate_decision",
        "decision": decision,
        "rationale": rationale.strip(),
        "decided_by": decided_by.strip(),
        "evidence": rows,
    }
    output = Path(output_path)
    _write_json_atomic(output, payload)
    return output


def _validated_evaluation_evidence(
    summary_path: Path, summary: Mapping[str, Any]
) -> dict[Path, str]:
    if summary.get("format_version") != 1:
        raise ValueError("unsupported Phase-4 evaluation format")
    if summary.get("kind") != "foveatag_phase4_evaluation":
        raise ValueError("artifact is not a Phase-4 evaluation summary")
    if summary.get("status") != "complete":
        raise ValueError("Phase-4 evaluation is not complete")
    if summary.get("active_benchmark") is not None:
        raise ValueError("completed Phase-4 evaluation still has an active benchmark")
    inference_mode = summary.get("inference_mode")
    if not isinstance(inference_mode, str) or not inference_mode:
        raise ValueError("Phase-4 evaluation lacks its inference mode")
    benchmarks = summary.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or not benchmarks:
        raise ValueError("Phase-4 evaluation summary has no benchmarks")
    completed = summary.get("completed_benchmarks")
    if not isinstance(completed, list) or completed != list(benchmarks):
        raise ValueError("Phase-4 completed benchmark list does not match its reports")
    evidence = {summary_path: "evaluation_summary"}
    source_hashes = set()
    routed_hashes = set()
    for name, raw in benchmarks.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"benchmark {name} summary is invalid")
        paths = {
            field: _required_file(raw.get(field), f"benchmark {name} {field}")
            for field in (
                "manifest",
                "fixed_baseline_report",
                "source_native_predictions",
                "routed_predictions",
                "comparison_report",
            )
        }
        source = _load_json(paths["source_native_predictions"])
        routed = _load_json(paths["routed_predictions"])
        rebuilt = build_phase4_comparison(
            load_benchmark_manifest(paths["manifest"]),
            source,
            routed,
            _load_json(paths["fixed_baseline_report"]),
        )
        if rebuilt.get("benchmark") != name:
            raise ValueError(f"benchmark {name} summary name does not match its manifest")
        if rebuilt.get("contract", {}).get("inference_mode") != inference_mode:
            raise ValueError(f"benchmark {name} inference mode does not match the summary")
        if _load_json(paths["comparison_report"]) != rebuilt:
            raise ValueError(f"benchmark {name} comparison report does not match its inputs")
        source_hashes.add((
            source.get("checkpoint_sha256"),
            source.get("checkpoint_metadata_sha256"),
        ))
        routed_hashes.add((
            routed.get("checkpoint_sha256"),
            routed.get("checkpoint_metadata_sha256"),
        ))
        for field, path in paths.items():
            evidence[path] = field
    if len(source_hashes) != 1:
        raise ValueError("Phase-4 benchmarks do not use one source-native checkpoint")
    if len(routed_hashes) != 1:
        raise ValueError("Phase-4 benchmarks do not use one routed checkpoint")
    return evidence


def load_passed_phase4_gate(path: Path | str) -> dict[str, Any]:
    payload = _load_json(Path(path))
    if payload.get("format_version") != PHASE4_GATE_FORMAT_VERSION:
        raise ValueError("unsupported Phase-4 gate artifact")
    if payload.get("kind") != "foveatag_phase4_gate_decision":
        raise ValueError("artifact is not a Phase-4 gate decision")
    if payload.get("decision") != "pass":
        raise ValueError("Phase-4 gate did not pass")
    if not str(payload.get("rationale", "")).strip() or not str(
        payload.get("decided_by", "")
    ).strip():
        raise ValueError("Phase-4 gate lacks decision provenance")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Phase-4 gate has no evidence hashes")
    resolved_evidence: dict[Path, str] = {}
    summary_paths = []
    for row in evidence:
        if not isinstance(row, Mapping):
            raise ValueError("Phase-4 gate evidence row is invalid")
        evidence_path = _required_file(row.get("path"), "gate evidence")
        role = row.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("Phase-4 gate evidence role is invalid")
        if evidence_path in resolved_evidence:
            raise ValueError(f"Phase-4 gate contains duplicate evidence: {evidence_path}")
        expected = row.get("sha256")
        if not isinstance(expected, str) or _file_sha256(evidence_path) != expected:
            raise ValueError(f"Phase-4 gate evidence changed: {evidence_path}")
        resolved_evidence[evidence_path] = role
        if role == "evaluation_summary":
            summary_paths.append(evidence_path)
    if len(summary_paths) != 1:
        raise ValueError("Phase-4 gate must bind one evaluation summary")
    summary_path = summary_paths[0]
    expected_evidence = _validated_evaluation_evidence(
        summary_path, _load_json(summary_path)
    )
    if resolved_evidence != expected_evidence:
        raise ValueError("Phase-4 gate evidence set does not match the evaluation")
    return dict(payload)


def routed_checkpoint_hashes(gate: Mapping[str, Any]) -> dict[str, str]:
    pairs = set()
    for row in gate.get("evidence", ()):
        if isinstance(row, Mapping) and row.get("role") == "routed_predictions":
            prediction = _load_json(_required_file(row.get("path"), "routed prediction"))
            pair = tuple(
                prediction.get(name)
                for name in ("checkpoint_sha256", "checkpoint_metadata_sha256")
            )
            if any(not isinstance(digest, str) or len(digest) != 64 for digest in pair):
                raise ValueError("routed prediction lacks checkpoint hashes")
            pairs.add(pair)
    if len(pairs) != 1:
        raise ValueError("Phase-4 gate does not bind one routed checkpoint")
    model_hash, metadata_hash = next(iter(pairs))
    return {
        "checkpoint_sha256": model_hash,
        "checkpoint_metadata_sha256": metadata_hash,
    }


def _required_file(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} path is missing")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", choices=("pass", "fail"), required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--decided-by", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output = record_phase4_gate(
        args.evaluation_summary,
        args.output,
        decision=args.decision,
        rationale=args.rationale,
        decided_by=args.decided_by,
    )
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
