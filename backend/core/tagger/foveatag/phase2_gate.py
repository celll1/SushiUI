"""Bind the Phase-2 encoder decision to collapse and lineage evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


PHASE2_GATE_FORMAT_VERSION = 1


def record_phase2_gate(
    evaluation_path: Path | str,
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
    evaluation_path = Path(evaluation_path).resolve()
    evaluation = _load_json(evaluation_path)
    if evaluation.get("format_version") != 1:
        raise ValueError("unsupported Phase-2 evaluation format")
    if evaluation.get("kind") != "foveatag_phase2_encoder_evaluation":
        raise ValueError("artifact is not a Phase-2 encoder evaluation")
    _validate_health(evaluation.get("encoder_validation"), "trained encoder")
    _validate_health(
        evaluation.get("scratch_initialization_validation"),
        "scratch initialization",
    )
    evidence = {evaluation_path: "encoder_evaluation"}
    encoder = _verified_evaluation_file(evaluation, "encoder", "encoder_sha256")
    metadata = _verified_evaluation_file(
        evaluation, "encoder_metadata", "encoder_metadata_sha256"
    )
    run_summary = _verified_evaluation_file(
        evaluation, "run_summary", "run_summary_sha256"
    )
    metadata_payload = _load_json(metadata)
    summary_payload = _load_json(run_summary)
    _validate_evaluation_lineage(
        evaluation,
        encoder=encoder,
        metadata=metadata,
        metadata_payload=metadata_payload,
        run_summary=run_summary,
        summary_payload=summary_payload,
    )
    evidence[encoder] = "encoder"
    evidence[metadata] = "encoder_metadata"
    evidence[run_summary] = "run_summary"
    manifest_rows = evaluation.get("validation_manifests")
    if not isinstance(manifest_rows, list) or not manifest_rows:
        raise ValueError("Phase-2 evaluation has no validation manifest evidence")
    expected_manifests = _expected_validation_manifests(
        encoder.parent / "manifests", metadata_payload
    )
    seen_manifests: set[tuple[int, str]] = set()
    for row in manifest_rows:
        if not isinstance(row, Mapping):
            raise ValueError("validation manifest evidence row is invalid")
        try:
            key = (int(row.get("domain")), str(row.get("role")))
        except (TypeError, ValueError) as error:
            raise ValueError("validation manifest evidence identity is invalid") from error
        if key in seen_manifests or key not in expected_manifests:
            raise ValueError(f"unexpected or duplicate validation manifest evidence: {key}")
        path = _required_file(row.get("path"), "validation manifest evidence")
        expected = row.get("sha256")
        if not isinstance(expected, str) or _file_sha256(path) != expected:
            raise ValueError(f"validation manifest evidence changed: {path}")
        expected_path, contract_hash, expected_bytes = expected_manifests[key]
        if path != expected_path:
            raise ValueError(f"validation manifest path does not match run contract: {path}")
        if contract_hash is not None and expected != contract_hash:
            raise ValueError(f"validation manifest hash does not match run contract: {path}")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise ValueError(f"validation manifest index size does not match run contract: {path}")
        seen_manifests.add(key)
        evidence[path] = f"validation_{row.get('role', 'unknown')}:{row.get('domain')}"
    if seen_manifests != set(expected_manifests):
        raise ValueError("Phase-2 evaluation does not cover every registered validation manifest")
    payload = {
        "format_version": PHASE2_GATE_FORMAT_VERSION,
        "kind": "foveatag_phase2_gate_decision",
        "decision": decision,
        "rationale": rationale.strip(),
        "decided_by": decided_by.strip(),
        "evidence": [
            {"role": role, "path": str(path), "sha256": _file_sha256(path)}
            for path, role in sorted(evidence.items(), key=lambda item: str(item[0]))
        ],
    }
    output = Path(output_path)
    _write_json_atomic(output, payload)
    return output


def load_passed_phase2_gate(path: Path | str) -> dict[str, Any]:
    payload = _load_json(Path(path))
    if payload.get("format_version") != PHASE2_GATE_FORMAT_VERSION:
        raise ValueError("unsupported Phase-2 gate artifact")
    if payload.get("kind") != "foveatag_phase2_gate_decision":
        raise ValueError("artifact is not a Phase-2 gate decision")
    if payload.get("decision") != "pass":
        raise ValueError("Phase-2 gate did not pass")
    if not str(payload.get("rationale", "")).strip() or not str(
        payload.get("decided_by", "")
    ).strip():
        raise ValueError("Phase-2 gate lacks decision provenance")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Phase-2 gate has no evidence hashes")
    for row in evidence:
        if not isinstance(row, Mapping):
            raise ValueError("Phase-2 gate evidence row is invalid")
        evidence_path = _required_file(row.get("path"), "gate evidence")
        expected = row.get("sha256")
        if not isinstance(expected, str) or _file_sha256(evidence_path) != expected:
            raise ValueError(f"Phase-2 gate evidence changed: {evidence_path}")
    return dict(payload)


def require_phase2_encoder(gate: Mapping[str, Any], encoder_path: Path | str) -> None:
    encoder_path = Path(encoder_path).resolve()
    metadata_path = encoder_path.with_name(f"{encoder_path.stem}_metadata.json")
    expected = {
        "encoder": (encoder_path, _file_sha256(encoder_path)),
        "encoder_metadata": (metadata_path, _file_sha256(metadata_path)),
    }
    for role, (path, digest) in expected.items():
        rows = [row for row in gate.get("evidence", ()) if row.get("role") == role]
        if len(rows) != 1:
            raise ValueError(f"Phase-2 gate does not bind one {role}")
        if Path(rows[0]["path"]).resolve() != path or rows[0].get("sha256") != digest:
            raise ValueError(f"Phase-2 gate does not bind the supplied {role}")


def _validate_evaluation_lineage(
    evaluation: Mapping[str, Any],
    *,
    encoder: Path,
    metadata: Path,
    metadata_payload: Mapping[str, Any],
    run_summary: Path,
    summary_payload: Mapping[str, Any],
) -> None:
    expected_metadata = encoder.with_name(f"{encoder.stem}_metadata.json")
    if metadata != expected_metadata:
        raise ValueError("Phase-2 evaluation metadata is not paired with its encoder")
    if run_summary != encoder.parent / "summary.json":
        raise ValueError("Phase-2 evaluation summary is not paired with its encoder")
    if metadata_payload.get("format_version") != 1:
        raise ValueError("unsupported Phase-2 encoder metadata format")
    if metadata_payload.get("kind") != "foveatag_scratch_encoder":
        raise ValueError("Phase-2 encoder metadata has the wrong kind")
    if (
        metadata_payload.get("initialization") != "scratch"
        or metadata_payload.get("external_weights") != []
    ):
        raise ValueError("Phase-2 encoder metadata does not have pure-scratch lineage")
    contract = metadata_payload.get("run_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Phase-2 encoder metadata lacks a run contract")
    if evaluation.get("run_contract_sha256") != _json_sha256(contract):
        raise ValueError("Phase-2 evaluation run contract hash does not match metadata")
    if summary_payload.get("kind") != "foveatag_teacher_free_pretraining_run":
        raise ValueError("Phase-2 run summary has the wrong kind")
    if summary_payload.get("status") != "complete":
        raise ValueError("Phase-2 run summary is not complete")
    if summary_payload.get("run_contract") != dict(contract):
        raise ValueError("Phase-2 run summary contract does not match encoder metadata")
    if int(summary_payload.get("epoch", -1)) != int(contract.get("epochs", -2)):
        raise ValueError("Phase-2 run summary did not complete every registered epoch")
    for field in ("epoch", "global_step"):
        if int(evaluation.get(field, -1)) != int(metadata_payload.get(field, -2)):
            raise ValueError(f"Phase-2 evaluation {field} does not match encoder metadata")
    training_validation = evaluation.get("training_validation")
    if training_validation != metadata_payload.get("validation"):
        raise ValueError("Phase-2 training validation does not match encoder metadata")
    _validate_training_validation(training_validation)
    weighted = training_validation["weighted"]
    best_loss = summary_payload.get("best_validation_loss")
    if not _finite(best_loss) or not math.isclose(
        float(weighted["loss"]), float(best_loss), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("Phase-2 encoder is not the completed run's best checkpoint")


def _validate_training_validation(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Phase-2 evaluation lacks held-out training validation")
    weighted = value.get("weighted")
    domains = value.get("domains")
    if not isinstance(weighted, Mapping) or not isinstance(domains, Mapping) or not domains:
        raise ValueError("Phase-2 held-out training validation is incomplete")
    for metric in ("loss", "correspondence", "reconstruction", "chroma"):
        if not _finite(weighted.get(metric)):
            raise ValueError(f"Phase-2 held-out validation lacks weighted {metric}")
    for domain, row in domains.items():
        if not isinstance(row, Mapping) or int(row.get("samples", 0)) <= 0:
            raise ValueError(f"Phase-2 held-out validation domain {domain} has no samples")
        for metric in ("loss", "correspondence", "reconstruction", "chroma"):
            if not _finite(row.get(metric)):
                raise ValueError(f"Phase-2 held-out validation domain {domain} lacks {metric}")


def _expected_validation_manifests(
    manifest_dir: Path, metadata: Mapping[str, Any]
) -> dict[tuple[int, str], tuple[Path, Optional[str], Optional[int]]]:
    contract = metadata.get("run_contract")
    domains = contract.get("manifest_domains") if isinstance(contract, Mapping) else None
    if not isinstance(domains, Mapping) or not domains:
        raise ValueError("Phase-2 run contract lacks validation manifests")
    expected = {}
    for domain, details in domains.items():
        if not isinstance(details, Mapping) or not isinstance(details.get("validation"), Mapping):
            raise ValueError(f"Phase-2 validation manifest contract is invalid: {domain}")
        validation = details["validation"]
        count = int(validation.get("count", -1))
        digest = validation.get("sha256")
        if count <= 0 or not isinstance(digest, str) or not digest:
            raise ValueError(f"Phase-2 validation manifest contract is incomplete: {domain}")
        domain_id = int(domain)
        expected[(domain_id, "paths")] = (
            (manifest_dir / str(validation.get("path", ""))).resolve(),
            digest,
            None,
        )
        expected[(domain_id, "index")] = (
            (manifest_dir / str(validation.get("index_path", ""))).resolve(),
            None,
            count * 8,
        )
    return expected


def _validate_health(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase-2 evaluation lacks {name} health")
    domains = value.get("domains")
    if not isinstance(domains, Mapping) or not domains:
        raise ValueError(f"Phase-2 evaluation lacks {name} domains")
    for domain, row in domains.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"{name} domain {domain} is invalid")
        if not _finite(row.get("correspondence")):
            raise ValueError(f"{name} domain {domain} lacks correspondence")
        health = row.get("feature_health")
        if not isinstance(health, Mapping):
            raise ValueError(f"{name} domain {domain} lacks feature health")
        for view in ("masked_local", "overview"):
            metrics = health.get(view)
            if not isinstance(metrics, Mapping) or int(metrics.get("tokens", 0)) <= 1:
                raise ValueError(f"{name} domain {domain} lacks {view} tokens")
            for metric in (
                "mean_token_norm",
                "channel_std_mean",
                "mean_pairwise_cosine",
                "effective_rank",
                "effective_rank_fraction",
                "participation_rank",
            ):
                if not _finite(metrics.get(metric)):
                    raise ValueError(f"{name} domain {domain} has invalid {view} {metric}")


def _verified_evaluation_file(
    evaluation: Mapping[str, Any], path_field: str, hash_field: str
) -> Path:
    path = _required_file(evaluation.get(path_field), path_field)
    expected = evaluation.get(hash_field)
    if not isinstance(expected, str) or _file_sha256(path) != expected:
        raise ValueError(f"Phase-2 {path_field} changed after evaluation")
    return path


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


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


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", choices=("pass", "fail"), required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--decided-by", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output = record_phase2_gate(
        args.evaluation,
        args.output,
        decision=args.decision,
        rationale=args.rationale,
        decided_by=args.decided_by,
    )
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
