"""Deterministic concept selection independent of display thresholds."""

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Optional, Sequence

import torch

from core.tagger.tag_vocabulary import CATEGORY_ORDER, normalize_tag

CONCEPT_CATEGORIES = tuple(CATEGORY_ORDER) + ("Unknown",)


@dataclass(frozen=True)
class ConceptBatch:
    ids: torch.Tensor
    scores: torch.Tensor
    categories: torch.Tensor
    sources: torch.Tensor
    mask: torch.Tensor
    warnings: tuple[tuple[str, ...], ...]


class ConceptSelector:
    VERSION = 1

    def __init__(self, names: Sequence[str], categories: Sequence[str], *, max_concepts: int,
                 allowed_categories: Sequence[str], alias_resolver: Optional[Callable[[str], str]] = None):
        self.names = tuple(normalize_tag(name) for name in names)
        self.categories = tuple(categories)
        if not self.names or len(self.names) != len(categories) or len(set(self.names)) != len(self.names):
            raise ValueError("Concept vocabulary must be nonempty, unique, and aligned")
        if any(not name for name in self.names) or max_concepts < 1:
            raise ValueError("Concept names and budget must be positive")
        if any(category not in CONCEPT_CATEGORIES for category in (*categories, *allowed_categories)):
            raise ValueError("Unsupported concept category")
        self.max_concepts = max_concepts
        self.alias_resolver = alias_resolver
        self.lookup = {name: index for index, name in enumerate(self.names)}
        self.allowed = {i for i, cat in enumerate(categories) if cat in allowed_categories}
        self.tie_order = sorted(self.allowed, key=lambda index: self.names[index])
        payload = json.dumps(list(zip(self.names, categories)), ensure_ascii=False, separators=(",", ":"))
        self.vocabulary_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @torch.no_grad()
    def select(self, logits: torch.Tensor, *, mode: str, provided=None) -> ConceptBatch:
        if logits.ndim != 2 or logits.shape[1] != len(self.names) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite logits aligned with the concept vocabulary")
        if mode not in {"predicted", "provided", "merged"}:
            raise ValueError("Unknown tag_source")
        batch = logits.shape[0]
        if mode != "predicted" and (provided is None or len(provided) != batch):
            raise ValueError("Provided tags must contain one list per image")
        if mode == "predicted" and provided is not None:
            raise ValueError("predicted mode does not accept provided tags")
        ids = torch.zeros(batch, self.max_concepts, device=logits.device, dtype=torch.long)
        scores = logits.new_zeros(batch, self.max_concepts)
        categories = torch.zeros_like(ids)
        sources = torch.zeros_like(ids)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        warnings = []
        for row in range(batch):
            selected, notes = [], []
            if mode != "predicted":
                if isinstance(provided[row], str):
                    raise ValueError("Provided tags must be lists")
                for raw in provided[row]:
                    name = normalize_tag(raw)
                    if self.alias_resolver is not None:
                        name = normalize_tag(self.alias_resolver(name))
                    index = self.lookup.get(name)
                    if index is None:
                        notes.append(f"unknown_tag:{name}")
                    elif index not in self.allowed:
                        notes.append(f"excluded_tag:{name}")
                    elif index not in selected:
                        selected.append(index)
                if len(selected) > self.max_concepts:
                    notes.append("provided_tags_truncated")
                selected = selected[:self.max_concepts]
            provided_count = len(selected)
            if mode != "provided" and self.tie_order:
                ordered = torch.tensor(self.tie_order, device=logits.device)
                rank = torch.argsort(logits[row, ordered], descending=True, stable=True)
                for index in ordered[rank[:self.max_concepts + provided_count]].tolist():
                    if len(selected) >= self.max_concepts:
                        break
                    if index not in selected:
                        selected.append(index)
            for column, index in enumerate(selected):
                ids[row, column] = index
                categories[row, column] = CONCEPT_CATEGORIES.index(self.categories[index])
                sources[row, column] = int(column < provided_count)
                if column >= provided_count:
                    scores[row, column] = logits[row, index].sigmoid()
                mask[row, column] = True
            warnings.append(tuple(notes))
        return ConceptBatch(ids, scores, categories, sources, mask, tuple(warnings))
