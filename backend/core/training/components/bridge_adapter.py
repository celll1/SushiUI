"""
BridgeAdapter — dim-bridge between a TE and a backbone.  [SKELETON — P2]

Generalizes ``SDXLTEAdapters`` (the dim-bridge loaded from ``sushi.te_adapter.*``
keys): maps an arbitrary TE's ``(hidden, pooled)`` into the backbone's
``(te_out_dim, te_pooled_dim)`` contract. This is the trainable edge that lets a
foreign TE (e.g. Qwen3) drive a native backbone (e.g. SDXL) without touching the
backbone's train_step (plan A.4 design test).

Planned P2 surface:
    BridgeAdapter(in_dim, out_dim, in_pooled, out_pooled)
        .forward(hidden, pooled) -> (hidden', pooled')

P0/P1: skeleton only. Do NOT move SDXLTEAdapters code yet.
"""

from __future__ import annotations


class BridgeAdapter:  # pragma: no cover - P2
    def __init__(self, in_dim, out_dim, in_pooled=None, out_pooled=None):
        raise NotImplementedError("BridgeAdapter is implemented in phase P2")
