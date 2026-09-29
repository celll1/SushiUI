"""Record and verify the explicit human decision for the Phase-3 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


PHASE3_GATE_FORMAT_VERSION = 1


def record_phase3_gate(
    evaluation_summary_path: Path | str,
    output_path: Path | str,
    *,
    decision: str,
    rationale: str,
    decided_by: str,
) -> Path:
    """Bind a manual pass/fail decision to every evaluated input and report."""
    if decision not in {"pass", "fail"}:
        raise ValueError("decision must be pass or fail")
    if not rationale.strip() or not decided_by.strip():
        raise ValueError("rationale and decided_by must be non-empty")
    summary_path = Path(evaluation_summary_path).resolve()
    summary = _load_json(summary_path)
    if summary.get("kind") != "foveatag_phase3_evaluation":
        raise ValueError("artifact is not a Phase-3 evaluation summary")
    if summary.get("status") != "complete":
        raise ValueError("Phase-3 evaluation is not complete")
    benchmarks = summary.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or not benchmarks:
        raise ValueError("Phase-3 evaluation summary has no benchmarks")
    evidence = {summary_path: "evaluation_summary"}
    for name, raw in benchmarks.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"benchmark {name} summary is invalid")
        for field in ("manifest", "fixed_baseline_report", "comparison_report"):
            evidence[_required_file(raw.get(field), f"benchmark {name} {field}")] = field
        predictions = raw.get("predictions")
        if not isinstance(predictions, Mapping) or not predictions:
            raise ValueError(f"benchmark {name} has no branch predictions")
        for mode, path in predictions.items():
            evidence[_required_file(path, f"benchmark {name} prediction {mode}")] = (
                f"prediction:{mode}"
            )
    rows = [
        {
            "role": role,
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for path, role in sorted(evidence.items(), key=lambda item: str(item[0]))
    ]
    payload = {
        "format_version": PHASE3_GATE_FORMAT_VERSION,
        "kind": "foveatag_phase3_gate_decision",
        "decision": decision,
        "rationale": rationale.strip(),
        "decided_by": decided_by.strip(),
        "evidence": rows,
    }
    output = Path(output_path)
    _write_json_atomic(output, payload)
    return output


def load_passed_phase3_gate(path: Path | str) -> dict[str, Any]:
    """Require a pass decision and reject evidence changed after that decision."""
    payload = _load_json(Path(path))
    if payload.get("format_version") != PHASE3_GATE_FORMAT_VERSION:
        raise ValueError("unsupported Phase-3 gate artifact")
    if payload.get("kind") != "foveatag_phase3_gate_decision":
        raise ValueError("artifact is not a Phase-3 gate decision")
    if payload.get("decision") != "pass":
        raise ValueError("Phase-3 gate did not pass")
    if not str(payload.get("rationale", "")).strip() or not str(
        payload.get("decided_by", "")
    ).strip():
        raise ValueError("Phase-3 gate lacks decision provenance")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Phase-3 gate has no evidence hashes")
    for row in evidence:
        if not isinstance(row, Mapping):
            raise ValueError("Phase-3 gate evidence row is invalid")
        evidence_path = _required_file(row.get("path"), "gate evidence")
        expected = row.get("sha256")
        if not isinstance(expected, str) or _file_sha256(evidence_path) != expected:
            raise ValueError(f"Phase-3 gate evidence changed: {evidence_path}")
    return dict(payload)


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
    output = record_phase3_gate(
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
