"""Contracts for post-run Phase-2 checkpoint evaluation."""

import hashlib

import pytest
import torch

from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.pretraining import FoveaPretrainer
from core.tagger.foveatag.pretraining_evaluation import (
    _model_from_payload,
    _new_model,
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
            "max_coordinate_distance": 0.02,
        },
    }
    torch.manual_seed(17)
    first = _new_model(contract)
    torch.manual_seed(17)
    second = _new_model(contract)

    torch.testing.assert_close(first.encoder.stem_conv.weight, second.encoder.stem_conv.weight)
