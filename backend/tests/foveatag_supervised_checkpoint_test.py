"""CPU tests for Phase-3 shared initialization and resume artifacts."""

import torch
from torch import nn

from core.tagger.foveatag.supervised_checkpoint import (
    load_supervised_checkpoint,
    load_supervised_initialization,
    save_supervised_checkpoint,
    save_supervised_initialization,
)


def _model():
    return nn.Sequential(nn.Linear(2, 3), nn.SiLU(), nn.Linear(3, 2))


def test_shared_initialization_gives_branches_identical_tensors(tmp_path):
    source = _model()
    contract = {"encoder_sha256": "abc", "vocabulary_sha256": "def", "seed": 4}
    path = save_supervised_initialization(
        tmp_path / "shared.safetensors",
        source,
        initialization_contract=contract,
    )
    first = _model()
    second = _model()

    load_supervised_initialization(
        path, first, expected_initialization_contract=contract
    )
    load_supervised_initialization(
        path, second, expected_initialization_contract=contract
    )

    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name])
        torch.testing.assert_close(value, second.state_dict()[name])


def test_shared_initialization_rejects_changed_contract(tmp_path):
    path = save_supervised_initialization(
        tmp_path / "shared.safetensors",
        _model(),
        initialization_contract={"seed": 1},
    )

    try:
        load_supervised_initialization(
            path, _model(), expected_initialization_contract={"seed": 2}
        )
    except ValueError as error:
        assert "contract" in str(error)
    else:
        raise AssertionError("expected changed initialization contract to fail")


def test_supervised_checkpoint_round_trip_and_contract(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    contract = {"input_mode": "source_native", "seed": 7}
    path = save_supervised_checkpoint(
        tmp_path / "latest.pt",
        model,
        optimizer,
        scheduler=scheduler,
        epoch=2,
        sample_offset=48,
        global_step=13,
        best_validation_f1=0.4,
        best_threshold=0.35,
        run_contract=contract,
        history=[{"epoch": 1}],
        epoch_accumulator={"samples": 48, "loss_sum": 12.0},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    payload = load_supervised_checkpoint(
        path,
        model,
        optimizer,
        scheduler=scheduler,
        expected_run_contract=contract,
        restore_rng=False,
    )

    assert payload["epoch"] == 2
    assert payload["sample_offset"] == 48
    assert payload["global_step"] == 13
    assert payload["epoch_accumulator"]["samples"] == 48
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
