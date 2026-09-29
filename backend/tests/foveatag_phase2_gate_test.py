"""Phase-2 encoder acceptance is explicit and hash-bound."""

import hashlib
import json

import pytest

from core.tagger.foveatag.phase2_gate import (
    load_passed_phase2_gate,
    record_phase2_gate,
    require_phase2_encoder,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _health():
    metrics = {
        "tokens": 32,
        "mean_token_norm": 1.0,
        "channel_std_mean": 0.2,
        "mean_pairwise_cosine": 0.3,
        "effective_rank": 8.0,
        "effective_rank_fraction": 0.25,
        "participation_rank": 6.0,
    }
    return {"weighted_correspondence": 0.1, "domains": {"25": {
        "samples": 2,
        "skipped": 0,
        "correspondence": 0.1,
        "feature_health": {"masked_local": metrics, "overview": metrics},
    }}}


def _evaluation(tmp_path):
    encoder = tmp_path / "best_encoder.safetensors"
    metadata = tmp_path / "best_encoder_metadata.json"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = manifest_dir / "validation.jsonl"
    index = manifest_dir / "validation.jsonl.idx"
    run_summary = tmp_path / "summary.json"
    encoder.write_bytes(b"encoder")
    manifest.write_text('"image.png"\n', encoding="utf-8")
    index.write_bytes(bytes(8))
    contract = {
        "epochs": 1,
        "manifest_domains": {
            "25": {
                "validation": {
                    "path": manifest.name,
                    "index_path": index.name,
                    "count": 1,
                    "sha256": _sha(manifest),
                }
            }
        },
    }
    validation = {
        "weighted": {
            "loss": 0.25,
            "correspondence": 0.1,
            "reconstruction": 0.12,
            "chroma": 0.06,
        },
        "domains": {
            "25": {
                "samples": 1,
                "skipped": 0,
                "loss": 0.25,
                "correspondence": 0.1,
                "reconstruction": 0.12,
                "chroma": 0.06,
            }
        },
    }
    metadata.write_text(json.dumps({
        "format_version": 1,
        "kind": "foveatag_scratch_encoder",
        "initialization": "scratch",
        "external_weights": [],
        "run_contract": contract,
        "epoch": 0,
        "global_step": 10,
        "validation": validation,
    }), encoding="utf-8")
    run_summary.write_text(json.dumps({
        "kind": "foveatag_teacher_free_pretraining_run",
        "status": "complete",
        "epoch": 1,
        "run_contract": contract,
        "best_validation_loss": 0.25,
    }), encoding="utf-8")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "format_version": 1,
        "kind": "foveatag_phase2_encoder_evaluation",
        "encoder": str(encoder),
        "encoder_sha256": _sha(encoder),
        "encoder_metadata": str(metadata),
        "encoder_metadata_sha256": _sha(metadata),
        "run_summary": str(run_summary),
        "run_summary_sha256": _sha(run_summary),
        "epoch": 0,
        "global_step": 10,
        "run_contract_sha256": _json_sha(contract),
        "training_validation": validation,
        "encoder_validation": _health(),
        "scratch_initialization_validation": _health(),
        "validation_manifests": [
            {"domain": 25, "role": "paths", "path": str(manifest), "sha256": _sha(manifest)},
            {"domain": 25, "role": "index", "path": str(index), "sha256": _sha(index)},
        ],
    }), encoding="utf-8")
    return evaluation, encoder, manifest


def test_passed_phase2_gate_binds_encoder_and_all_evidence(tmp_path):
    evaluation, encoder, manifest = _evaluation(tmp_path)
    gate_path = record_phase2_gate(
        evaluation,
        tmp_path / "gate.json",
        decision="pass",
        rationale="loss fell without rank collapse",
        decided_by="owner",
    )

    gate = load_passed_phase2_gate(gate_path)
    require_phase2_encoder(gate, encoder)
    manifest.write_text('"changed.png"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        load_passed_phase2_gate(gate_path)


def test_failed_phase2_gate_cannot_start_phase3(tmp_path):
    evaluation, _encoder, _manifest = _evaluation(tmp_path)
    gate_path = record_phase2_gate(
        evaluation,
        tmp_path / "gate.json",
        decision="fail",
        rationale="rank collapsed",
        decided_by="owner",
    )

    with pytest.raises(ValueError, match="did not pass"):
        load_passed_phase2_gate(gate_path)


def test_phase2_gate_requires_scratch_reference(tmp_path):
    evaluation, _encoder, _manifest = _evaluation(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    payload["scratch_initialization_validation"] = None
    evaluation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="scratch initialization"):
        record_phase2_gate(
            evaluation,
            tmp_path / "gate.json",
            decision="pass",
            rationale="invalid",
            decided_by="owner",
        )


def test_phase2_gate_rejects_evaluation_not_matching_encoder_metadata(tmp_path):
    evaluation, _encoder, _manifest = _evaluation(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    payload["training_validation"]["weighted"]["chroma"] = 0.01
    evaluation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match encoder metadata"):
        record_phase2_gate(
            evaluation,
            tmp_path / "gate.json",
            decision="pass",
            rationale="invalid",
            decided_by="owner",
        )


def test_phase2_gate_rejects_unregistered_validation_manifest(tmp_path):
    evaluation, _encoder, _manifest = _evaluation(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text('"other.png"\n', encoding="utf-8")
    payload["validation_manifests"][0].update({
        "path": str(replacement),
        "sha256": _sha(replacement),
    })
    evaluation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="path does not match run contract"):
        record_phase2_gate(
            evaluation,
            tmp_path / "gate.json",
            decision="pass",
            rationale="invalid",
            decided_by="owner",
        )


def test_phase2_gate_loader_rejects_handwritten_incomplete_evidence(tmp_path):
    evaluation, _encoder, _manifest = _evaluation(tmp_path)
    gate = tmp_path / "handwritten_gate.json"
    gate.write_text(json.dumps({
        "format_version": 1,
        "kind": "foveatag_phase2_gate_decision",
        "decision": "pass",
        "rationale": "claims to pass without binding the encoder",
        "decided_by": "owner",
        "evidence": [{
            "role": "encoder_evaluation",
            "path": str(evaluation),
            "sha256": _sha(evaluation),
        }],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence set does not match"):
        load_passed_phase2_gate(gate)


def test_phase2_gate_loader_revalidates_evaluation_roles(tmp_path):
    gate = record_phase2_gate(
        _evaluation(tmp_path)[0],
        tmp_path / "gate.json",
        decision="pass",
        rationale="valid before role tampering",
        decided_by="owner",
    )
    payload = json.loads(gate.read_text(encoding="utf-8"))
    encoder_row = next(row for row in payload["evidence"] if row["role"] == "encoder")
    encoder_row["role"] = "untrusted_encoder"
    gate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence set does not match"):
        load_passed_phase2_gate(gate)
