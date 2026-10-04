"""Contracts for post-run Phase-2 checkpoint evaluation."""

import hashlib
import json

import pytest
import torch

from core.tagger.foveatag.encoder import FoveaEncoder, FoveaEncoderConfig
from core.tagger.foveatag.pretraining import FoveaPretrainer
from core.tagger.foveatag.pretraining_evaluation import (
    _completed_run_summary,
    _model_from_payload,
    _new_model,
    _pyramid_cache_root,
    _module_state_sha256,
    _verify_validation_manifests,
)


def _tiny_config():
    return FoveaEncoderConfig(
        widths=(8, 12, 16, 32),
        depths=(1, 1, 1, 1),
        attention_heads=(4, 4),
        window_size=4,
    )


def test_checkpoint_model_loader_requires_scratch_lineage():
    config = _tiny_config()
    source = FoveaPretrainer(config)
    contract = {
        "initialization": "scratch",
        "external_weights": [],
        "encoder": {
            "widths": config.widths,
            "depths": config.depths,
            "window_size": config.window_size,
            "attention_heads": config.attention_heads,
            "mlp_ratio": config.mlp_ratio,
            "geometry_dim": config.geometry_dim,
            "scale_dropout": config.scale_dropout,
        },
        "mask_ratio": 0.4,
        "objective": {
            "correspondence_weight": 1.0,
            "reconstruction_weight": 1.0,
            "chroma_weight": 0.5,
            "variance_weight": 1.0,
            "covariance_weight": 0.01,
            "centering_weight": 0.1,
            "target_standard_deviation": 1.0,
            "distribution_epsilon": 1e-4,
            "max_coordinate_distance": 0.02,
        },
    }
    payload = {
        "format_version": 1,
        "kind": "foveatag_scratch_pretraining",
        "run_contract": contract,
        "model": source.state_dict(),
    }

    restored, restored_contract = _model_from_payload(payload)
    assert restored_contract == contract
    torch.testing.assert_close(restored.encoder.stem_conv.weight, source.encoder.stem_conv.weight)
    payload["run_contract"] = {**contract, "external_weights": ["teacher"]}
    with pytest.raises(ValueError, match="pure-scratch"):
        _model_from_payload(payload)


def test_validation_manifest_verifier_checks_content_and_index(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_bytes(b'"a.png"\n"b.png"\n')
    index = tmp_path / "validation.jsonl.idx"
    index.write_bytes(bytes(16))
    metadata = {"domains": {"25": {"validation": {
        "path": path.name,
        "index_path": index.name,
        "count": 2,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }}}}

    _verify_validation_manifests(tmp_path, metadata)
    path.write_bytes(b'"changed.png"\n')
    with pytest.raises(ValueError, match="hash changed"):
        _verify_validation_manifests(tmp_path, metadata)


def test_scratch_baseline_can_reproduce_seeded_initialization():
    config = _tiny_config()
    contract = {
        "encoder": {
            "widths": config.widths,
            "depths": config.depths,
            "window_size": config.window_size,
            "attention_heads": config.attention_heads,
            "mlp_ratio": config.mlp_ratio,
            "geometry_dim": config.geometry_dim,
            "scale_dropout": config.scale_dropout,
        },
        "mask_ratio": 0.4,
        "objective": {
            "correspondence_weight": 1.0,
            "reconstruction_weight": 1.0,
            "chroma_weight": 0.5,
            "variance_weight": 1.0,
            "covariance_weight": 0.01,
            "centering_weight": 0.1,
            "target_standard_deviation": 1.0,
            "distribution_epsilon": 1e-4,
            "max_coordinate_distance": 0.02,
        },
    }
    torch.manual_seed(17)
    first = _new_model(contract)
    torch.manual_seed(17)
    second = _new_model(contract)

    torch.testing.assert_close(first.encoder.stem_conv.weight, second.encoder.stem_conv.weight)


def test_scratch_encoder_fingerprint_is_canonical_and_weight_sensitive():
    config = _tiny_config()
    torch.manual_seed(17)
    first = FoveaPretrainer(config).encoder
    torch.manual_seed(17)
    reference = FoveaEncoder(config)

    assert _module_state_sha256(first) == _module_state_sha256(reference)
    with torch.no_grad():
        reference.stem_conv.weight.view(-1)[0] += 1
    assert _module_state_sha256(first) != _module_state_sha256(reference)


def test_evaluation_resolves_cache_root_from_manifest_contract():
    assert _pyramid_cache_root({}) is None
    assert _pyramid_cache_root({"manifest_request": {}}) is None
    assert _pyramid_cache_root({
        "manifest_request": {"pyramid_cache": {"root": "cache-root"}}
    }) == "cache-root"


def test_encoder_evaluation_requires_completed_run_and_best_artifact(tmp_path):
    contract = {"epochs": 2, "token": "same"}
    metadata = {"validation": {"weighted": {"loss": 0.25}}}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "kind": "foveatag_teacher_free_pretraining_run",
        "status": "complete",
        "epoch": 2,
        "run_contract": contract,
        "best_validation_loss": 0.25,
    }), encoding="utf-8")

    result = _completed_run_summary(summary, contract, metadata)
    assert result["epoch"] == 2
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["status"] = "stopped_by_request"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        _completed_run_summary(summary, contract, metadata)
