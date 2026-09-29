"""CPU contracts for Phase-1 control checkpoint resume."""

import pytest
import torch
from torch import nn

from core.tagger.foveatag.control_checkpoint import (
    load_control_checkpoint,
    load_control_weights,
    save_control_checkpoint,
    save_control_weights,
)


class _Control(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Linear(2, 2)
        self.vision_encoder.requires_grad_(False)
        self.baseline_head = nn.Linear(2, 2)
        self.baseline_head.requires_grad_(False)
        self.head = nn.Linear(2, 2)


def test_control_checkpoint_round_trip_excludes_external_encoder(tmp_path):
    model = _Control()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    loss = model.head(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    original_head = model.head.weight.detach().clone()
    original_encoder = model.vision_encoder.weight.detach().clone()
    original_baseline = model.baseline_head.weight.detach().clone()
    provenance = {"checkpoint": "fixed.safetensors", "vocabulary": "fixed.json"}

    path = save_control_checkpoint(
        tmp_path / "control.pt",
        model,
        optimizer,
        next_epoch=3,
        global_step=19,
        encoder_provenance=provenance,
        best_validation_f1=0.42,
        best_threshold=0.35,
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert all(not key.startswith("vision_encoder.") for key in saved["control_model"])
    assert all(not key.startswith("baseline_head.") for key in saved["control_model"])

    with torch.no_grad():
        model.head.weight.zero_()
        model.vision_encoder.weight.add_(1)
        model.baseline_head.weight.add_(1)
    payload = load_control_checkpoint(
        path,
        model,
        optimizer,
        expected_encoder_provenance=provenance,
    )

    assert payload["next_epoch"] == 3
    assert payload["global_step"] == 19
    torch.testing.assert_close(model.head.weight, original_head)
    torch.testing.assert_close(model.vision_encoder.weight, original_encoder + 1)
    torch.testing.assert_close(model.baseline_head.weight, original_baseline + 1)


def test_control_checkpoint_rejects_encoder_provenance_mismatch(tmp_path):
    model = _Control()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    path = save_control_checkpoint(
        tmp_path / "control.pt",
        model,
        optimizer,
        next_epoch=1,
        global_step=0,
        encoder_provenance={"checkpoint": "fixed.safetensors"},
    )

    with pytest.raises(ValueError, match="provenance does not match"):
        load_control_checkpoint(
            path,
            model,
            optimizer,
            expected_encoder_provenance={"checkpoint": "other.safetensors"},
        )


def test_control_weights_round_trip_uses_compact_inference_artifact(tmp_path):
    model = _Control()
    provenance = {"checkpoint": "fixed.safetensors"}
    original_head = model.head.weight.detach().clone()
    path = save_control_weights(
        tmp_path / "best_f1.safetensors",
        model,
        encoder_provenance=provenance,
        epoch=2,
        global_step=31,
        validation_f1=0.4,
        threshold=0.25,
    )
    with torch.no_grad():
        model.head.weight.zero_()

    metadata = load_control_weights(
        path,
        model,
        expected_encoder_provenance=provenance,
    )

    assert metadata["epoch"] == 2
    assert metadata["global_step"] == 31
    torch.testing.assert_close(model.head.weight, original_head)
