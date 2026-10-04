"""Annotation-based, fixed-threshold training-window metrics (not validation)."""
from __future__ import annotations

import torch


def confusion_counts(logits, labels, loss_mask, threshold):
    with torch.no_grad():
        predicted = logits.detach().float().sigmoid() >= threshold
        positive, known = labels >= .5, loss_mask > 0
        counts = torch.stack(((predicted & positive & known).sum(),
            (predicted & ~positive & known).sum(), (~predicted & positive & known).sum(),
            (~positive & known).sum())).cpu().tolist()
    return dict(zip(("tp", "fp", "fn", "known_negative"), counts))


class TrainingMetricWindow:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {key: 0 for key in ("tp", "fp", "fn", "known_negative")}
        self.samples = self.batches = 0
        self.start_step = self.end_step = self.threshold = None

    def add(self, counts, samples, step, threshold):
        if self.threshold is not None and self.threshold != threshold:
            self.reset()
        if self.start_step is None:
            self.start_step, self.threshold = step, threshold
        self.end_step = step
        self.samples += samples
        self.batches += 1
        for key in self.counts:
            self.counts[key] += counts[key]

    def report(self):
        tp, fp, fn, negatives = (self.counts[key] for key in ("tp", "fp", "fn", "known_negative"))
        return {"start_step": self.start_step, "end_step": self.end_step,
            "samples": self.samples, "batches": self.batches, "threshold": self.threshold,
            **self.counts, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fp_rate": fp / negatives if negatives else None,
            "fp_per_image": fp / self.samples if self.samples else None,
            "scope": "training_annotations_pre_update"}
