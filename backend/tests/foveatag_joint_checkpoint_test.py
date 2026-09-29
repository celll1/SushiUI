"""CPU tests for exact-resume FoveaTag joint checkpoints."""

import random

import pytest
import torch
from torch import nn

from core.tagger.foveatag.joint_checkpoint import (
    load_joint_checkpoint,
    save_joint_checkpoint,
)


def _state():
    model = nn.Linear(3, 2)
    router = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *router.parameters()],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    loss = router(model(torch.randn(4, 3))).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, router, optimizer, scheduler


def test_joint_checkpoint_roundtrip_restores_both_modules_and_progress(tmp_path):
    random.seed(17)
    torch.manual_seed(19)
    model, router, optimizer, scheduler = _state()
    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_router = {name: value.detach().clone() for name, value in router.state_dict().items()}
    contract = {"source_model_sha256": "a" * 64, "router_sha256": "b" * 64}
    path = save_joint_checkpoint(
        tmp_path / "latest.pt",
        model,
        router,
        optimizer,
        scheduler=scheduler,
        epoch=2,
        sample_offset=13,
        global_step=29,
        best_validation_f1=0.4,
        best_threshold=0.35,
        run_contract=contract,
        history=[{"epoch": 1}],
        epoch_accumulator={"samples": 13},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for parameter in router.parameters():
            parameter.zero_()

    payload = load_joint_checkpoint(
        path,
        model,
        router,
        optimizer,
        scheduler=scheduler,
        expected_run_contract=contract,
        restore_rng=False,
    )

    assert (payload["epoch"], payload["sample_offset"], payload["global_step"]) == (2, 13, 29)
    assert payload["epoch_accumulator"] == {"samples": 13}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name])
    for name, value in router.state_dict().items():
        torch.testing.assert_close(value, expected_router[name])


def test_joint_checkpoint_rejects_changed_source_lineage(tmp_path):
    model, router, optimizer, scheduler = _state()
    path = save_joint_checkpoint(
        tmp_path / "latest.pt",
        model,
        router,
        optimizer,
        scheduler=scheduler,
        epoch=0,
        sample_offset=0,
        global_step=1,
        best_validation_f1=0.0,
        best_threshold=0.5,
        run_contract={"source_model_sha256": "first"},
        history=[],
    )

    with pytest.raises(ValueError, match="run contract does not match"):
        load_joint_checkpoint(
            path,
            model,
            router,
            optimizer,
            scheduler=scheduler,
            expected_run_contract={"source_model_sha256": "second"},
        )


def test_joint_checkpoint_requires_matching_scheduler_presence(tmp_path):
    model, router, optimizer, scheduler = _state()
    path = save_joint_checkpoint(
        tmp_path / "latest.pt",
        model,
        router,
        optimizer,
        scheduler=scheduler,
        epoch=0,
        sample_offset=0,
        global_step=1,
        best_validation_f1=0.0,
        best_threshold=0.5,
        run_contract={"lineage": "fixed"},
        history=[],
    )

    with pytest.raises(ValueError, match="scheduler state"):
        load_joint_checkpoint(
            path,
            model,
            router,
            optimizer,
            scheduler=None,
            expected_run_contract={"lineage": "fixed"},
        )
