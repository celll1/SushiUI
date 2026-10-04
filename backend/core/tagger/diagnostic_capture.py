"""Picklable five-field samples/batches with actual-forward source provenance."""
from dataclasses import dataclass

import torch


@dataclass
class DiagnosticBatch:
    values: tuple
    diagnostic_sources: tuple

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def pin_memory(self):
        return type(self)(tuple(value.pin_memory() for value in self.values), self.diagnostic_sources)


def capture_forward(batch, logits, labels, masks, *, step, epoch, threshold, vocabulary,
                    tag_metrics=None, focus=None, prediction_origin="training_forward_before_optimizer_update"):
    sources = getattr(batch, "diagnostic_sources", ())
    index = next((i for i, source in enumerate(sources) if source and (source.get("source_path") or source.get("source_bytes"))), None)
    if index is None:
        return {"step": step, "skip_reason": "no_source_metadata_in_step_batch"}
    snapshot = {"step": step, "epoch": epoch, "source": sources[index],
        "probabilities": logits[index].detach().float().sigmoid().cpu(),
        "positive": (labels[index] >= .5).detach().cpu(), "known": (masks[index] > 0).detach().cpu(),
        "threshold": float(threshold), "vocabulary": vocabulary.to_dict(),
        "prediction_origin": prediction_origin, "boxes": torch.empty(0, 4),
        "random_slots": torch.empty(0, dtype=torch.bool), "local_size": None,
        "focus_kind": "none"}
    if tag_metrics is not None:
        snapshot["tag_metrics"] = {key: torch.as_tensor(tag_metrics[key]).clone()
            for key in ("best_thr", "n_pos")}
    if focus is not None:
        snapshot.update(focus(index))
    return snapshot


def training_forward(model, inputs, attention_mask, spatial_shapes, labels, masks, criterion, step):
    if getattr(model, "routing", None):
        logits, auxiliary, focus = model.forward_routed(inputs, labels, masks, criterion, step)
    else:
        logits, auxiliary, focus = model(inputs, attention_mask, spatial_shapes), None, None
    classification = criterion(logits, labels, masks)
    return logits, classification, classification if auxiliary is None else classification + auxiliary, focus
