import torch

from core.tagger.foveatag.training_metrics import TrainingMetricWindow, confusion_counts


def test_counts_ignore_unknowns_and_do_not_weight_confusion_by_loss_weight():
    counts = confusion_counts(torch.tensor([[4., 4., -4., 4., -4.]]),
        torch.tensor([[1., 0., 1., 0., 0.]]), torch.tensor([[.25, .25, 1., 0., 1.]]), .35)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "known_negative": 2}
    window = TrainingMetricWindow()
    window.add(counts, 1, 201, .35)
    window.add(counts, 1, 202, .35)
    report = window.report()
    assert report["start_step"] == 201 and report["end_step"] == 202
    assert report["samples"] == report["batches"] == 2
    assert report["precision"] == report["recall"] == report["fp_rate"] == .5
    assert report["fp_per_image"] == 1


def test_undefined_rates_are_null_and_threshold_change_starts_new_window():
    window = TrainingMetricWindow()
    assert window.report()["precision"] is None
    counts = {"tp": 0, "fp": 0, "fn": 0, "known_negative": 0}
    window.add(counts, 8, 10, .35)
    window.add(counts, 8, 11, .5)
    report = window.report()
    assert report["start_step"] == report["end_step"] == 11
    assert report["samples"] == 8 and report["precision"] is None and report["fp_rate"] is None
