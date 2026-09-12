"""Validation collection remains the single source for tagger evaluation."""

import torch
from torch import nn

from core.tagger.tagger_trainer import TaggerTrainer, _compute_all_metrics, _find_best_threshold


class _LogitModel(nn.Module):
    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes):
        return pixel_values


def _batch(logits: torch.Tensor, labels: torch.Tensor):
    batch_size = logits.shape[0]
    return logits, torch.zeros(batch_size, 1), torch.zeros(batch_size, 1), labels, None


def test_validate_uses_collected_predictions_without_changing_metrics():
    trainer = TaggerTrainer.__new__(TaggerTrainer)
    logits = torch.tensor([[2.0, -2.0], [-1.0, 3.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loader = [_batch(logits, labels)]

    predictions, collected_labels = trainer._collect_val_preds(
        _LogitModel(), loader, torch.device("cpu"), None,
    )
    threshold, f1 = _find_best_threshold(predictions, collected_labels)
    expected = _compute_all_metrics(predictions, collected_labels, threshold)

    actual = trainer._validate(_LogitModel(), loader, torch.device("cpu"), None)

    assert actual == {
        "f1": f1,
        "threshold": threshold,
        "precision": expected["precision"],
        "recall": expected["recall"],
    }


def test_collect_validation_predictions_pads_stale_vocabulary_labels():
    trainer = TaggerTrainer.__new__(TaggerTrainer)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    labels = torch.tensor([[1.0, 0.0]])

    _, collected_labels = trainer._collect_val_preds(
        _LogitModel(), [_batch(logits, labels)], torch.device("cpu"), None,
    )

    assert collected_labels.shape == (1, 3)
    assert collected_labels[0, 2] == 0


def test_collect_validation_predictions_rejects_an_empty_loader():
    trainer = TaggerTrainer.__new__(TaggerTrainer)
    try:
        trainer._collect_val_preds(_LogitModel(), [None], torch.device("cpu"), None)
    except RuntimeError as error:
        assert "no readable batches" in str(error)
        return
    raise AssertionError("empty validation loader was accepted")
