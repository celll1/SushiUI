"""
TEComponentRegistry — arch-independent text-encoder component layer.  [SKELETON — P2]

Generalizes ``models/sdxl_te_registry.py`` (TE_REGISTRY, ``is_custom_te``,
``load_sdxl_te``, ``encode_text``, positional-embedding extension) to be
arch-independent, driven by a ``ComponentWiringSpec`` (components/wiring.py).

Planned P2 surface:
    load_te(spec, path) -> encoder body
    encode_text(te, tokenizer, prompts, ...) -> (hidden, pooled)

P0/P1: skeleton only. Do NOT move sdxl code yet — this exists so downstream
imports resolve and the package shape is fixed.
"""

from __future__ import annotations


def load_te(spec, path):  # pragma: no cover - P2
    raise NotImplementedError("te_registry.load_te is implemented in phase P2")


def encode_text(te, tokenizer, prompts, **kwargs):  # pragma: no cover - P2
    raise NotImplementedError("te_registry.encode_text is implemented in phase P2")
