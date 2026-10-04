"""CPU tests for FoveaTag Phase-2 resumable artifacts."""

import json

import torch
from safetensors.torch import load_file
from torch import nn

from core.tagger.foveatag.pretraining_checkpoint import (
    load_pretrained_encoder,
    load_pretraining_checkpoint,
    save_pretrained_encoder,
    save_pretraining_checkpoint,
)


class _Pretrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)


def test_pretraining_checkpoint_round_trip_and_contract(tmp_path):
    model = _Pretrainer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = model.decoder(model.encoder(torch.ones(1, 2))).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    contract = {"seed": 4, "datasets": [3, 7]}
    path = save_pretraining_checkpoint(
        tmp_path / "latest.pt",
        model,
        optimizer,
        scheduler=scheduler,
        epoch=2,
        sample_offset=48,
        global_step=13,
        best_validation_loss=0.7,
        run_contract=contract,
        history=[{"epoch": 1}],
        epoch_accumulator={"samples": 48, "sums": {"loss": 12.0}},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    payload = load_pretraining_checkpoint(
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


def test_pretraining_checkpoint_rejects_contract_mismatch(tmp_path):
    model = _Pretrainer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_pretraining_checkpoint(
        tmp_path / "latest.pt",
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        sample_offset=0,
        global_step=0,
        best_validation_loss=1.0,
        run_contract={"seed": 1},
        history=[],
    )

    try:
        load_pretraining_checkpoint(
            path,
            model,
            optimizer,
            scheduler=None,
            expected_run_contract={"seed": 2},
        )
    except ValueError as error:
        assert "contract" in str(error)
    else:
        raise AssertionError("expected incompatible run contract to fail")


def test_encoder_artifact_excludes_decoder_and_records_scratch_lineage(tmp_path):
    model = _Pretrainer()
    path = save_pretrained_encoder(
        tmp_path / "encoder.safetensors",
        model.encoder,
        run_contract={"objective": "teacher-free"},
        epoch=3,
        global_step=20,
        validation={"loss": 0.5},
    )

    assert set(load_file(str(path))) == {"weight", "bias"}
    metadata = json.loads((tmp_path / "encoder_metadata.json").read_text())
    assert metadata["initialization"] == "scratch"
    assert metadata["external_weights"] == []

    restored = nn.Linear(2, 2)
    loaded_metadata = load_pretrained_encoder(
        path,
        restored,
        expected_run_contract={"objective": "teacher-free"},
    )
    assert loaded_metadata == metadata
    for name, value in model.encoder.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_encoder_loader_rejects_external_weight_lineage(tmp_path):
    model = _Pretrainer()
    path = save_pretrained_encoder(
        tmp_path / "encoder.safetensors",
        model.encoder,
        run_contract={"objective": "teacher-free"},
        epoch=1,
        global_step=2,
        validation={"loss": 0.5},
    )
    metadata_path = tmp_path / "encoder_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["external_weights"] = ["teacher"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    try:
        load_pretrained_encoder(path, nn.Linear(2, 2))
    except ValueError as error:
        assert "scratch lineage" in str(error)
    else:
        raise AssertionError("expected external-weight lineage to fail")
