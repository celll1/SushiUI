"""Frozen tagger used to remove content information from style embeddings (design §8.2)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HSICTagger:
    model: torch.nn.Module
    processor: object
    is_naflex: bool
    category_indices: dict[str, torch.Tensor]
    checkpoint_name: str
    checkpoint_sha256: str
    max_num_patches: int = 256  # Matches the tagger inference default.


def category_columns(vocabulary: dict, weights: dict[str, float]) -> dict[str, torch.Tensor]:
    tag_to_idx = vocabulary.get("tag_to_idx") or {}
    tag_to_category = vocabulary.get("tag_to_category") or {}
    if any(str(category).casefold() == "artist" for category in tag_to_category.values()):
        raise ValueError("the HSIC tagger vocabulary contains Artist tags; use an artist-excluded checkpoint")
    out = {}
    for category, weight in weights.items():
        if weight <= 0:
            continue
        indices = sorted(int(idx) for tag, idx in tag_to_idx.items()
                         if str(tag_to_category.get(tag, "")).casefold() == category.casefold())
        if not indices:
            raise ValueError(f"the HSIC tagger vocabulary has no {category} columns with weight {weight}")
        out[category] = torch.tensor(indices, dtype=torch.int64)
    return out


def load_hsic_tagger(path: str, weights: dict[str, float], device: torch.device) -> HSICTagger:
    if not path.lower().endswith(".safetensors") or not os.path.isfile(path):
        raise ValueError(f"HSIC tagger checkpoint must be an existing .safetensors file: {path}")
    stem = os.path.splitext(os.path.basename(path))[0]
    directory = os.path.dirname(os.path.abspath(path))
    snapshot_vocab = os.path.join(directory, f"{stem}_vocabulary.json")
    if not os.path.isfile(snapshot_vocab):
        raise ValueError(f"HSIC needs the checkpoint's vocabulary snapshot: {snapshot_vocab}")

    from core.tagger.siglip2_inference_manager import SigLIP2InferenceManager

    manager = SigLIP2InferenceManager()
    manager.device = str(device)
    manager.load_model(path, snapshot_vocab)
    if manager.model is None or manager.vocabulary is None:
        raise ValueError("HSIC tagger did not load a PyTorch model and vocabulary")
    columns = category_columns(manager.vocabulary, weights)
    num_tags = manager.model.head.out_features
    if any((idx >= num_tags).any().item() for idx in columns.values()):
        raise ValueError("HSIC tagger vocabulary indices exceed its classifier head")
    if manager._cls_emb_hook_handle is not None:
        manager._cls_emb_hook_handle.remove()
        manager._cls_emb_hook_handle = None
    manager.model.requires_grad_(False).eval()
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return HSICTagger(manager.model, manager.processor, manager.is_naflex, columns,
                      os.path.basename(path), digest.hexdigest())
