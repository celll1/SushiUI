"""CPU tests for compact FoveaTag router checkpoints."""

import torch

from core.tagger.foveatag.router import EvidenceRouter
from core.tagger.foveatag.router_checkpoint import (
    load_router_checkpoint,
    load_router_weights,
    save_router_checkpoint,
    save_router_weights,
)


def _router():
    return EvidenceRouter(feature_dim=8, routing_dim=12, routing_slots=2)


def test_router_checkpoint_roundtrip_excludes_frozen_tagger(tmp_path):
    router = _router()
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
    sum(parameter.sum() for parameter in router.parameters()).backward()
    optimizer.step()
    contract = {"tagger_sha256": "abc", "candidate_count": 32}
    path = save_router_checkpoint(
        tmp_path / "latest.pt",
        router,
        optimizer,
        scheduler=None,
        epoch=1,
        sample_offset=4,
        global_step=7,
        best_utility_recovery=0.6,
        run_contract=contract,
        history=[{"epoch": 0}],
        epoch_accumulator={"samples": 4},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" not in payload and "tagger" not in payload

    restored = _router()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    loaded = load_router_checkpoint(
        path,
        restored,
        restored_optimizer,
        scheduler=None,
        expected_run_contract=contract,
    )

    assert loaded["global_step"] == 7
    assert loaded["epoch_accumulator"] == {"samples": 4}
    for name, value in router.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    assert restored_optimizer.state_dict()["state"]


def test_router_checkpoint_rejects_changed_tagger_lineage(tmp_path):
    router = _router()
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
    path = save_router_checkpoint(
        tmp_path / "latest.pt",
        router,
        optimizer,
        scheduler=None,
        epoch=0,
        sample_offset=0,
        global_step=0,
        best_utility_recovery=0.0,
        run_contract={"tagger_sha256": "first"},
        history=[],
    )

    try:
        load_router_checkpoint(
            path,
            router,
            optimizer,
            scheduler=None,
            expected_run_contract={"tagger_sha256": "second"},
        )
    except ValueError as error:
        assert "contract" in str(error)
    else:
        raise AssertionError("expected changed tagger lineage to fail")


def test_router_inference_weights_roundtrip_with_validation_metadata(tmp_path):
    router = _router()
    contract = {"tagger_sha256": "abc", "routing_slots": 2}
    path = save_router_weights(
        tmp_path / "best_router.safetensors",
        router,
        run_contract=contract,
        epoch=2,
        global_step=9,
        validation={"mean_utility_recovery": 0.75},
    )
    restored = _router()
    metadata = load_router_weights(
        path,
        restored,
        expected_run_contract=contract,
    )

    assert metadata["validation"]["mean_utility_recovery"] == 0.75
    for name, value in router.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
