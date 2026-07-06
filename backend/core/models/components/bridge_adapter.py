"""Dim-bridge adapters between a TE and a backbone (plan A.4).

Generalizes ``core/models/sdxl_te_adapter.py``: it hosts the canonical
implementation of ``SDXLTEAdapters`` (the dim-bridge loaded from
``sushi.te_adapter.*`` keys) plus the arch-independent ``BridgeAdapter``. The old
``core.models.sdxl_te_adapter`` module is a thin re-export shim of this one, so
existing imports keep resolving to the SAME objects (identity preserved).

CHECKPOINT FORMAT FROZEN (plan F): ``SDXLTEAdapters`` is moved byte-identically —
its submodule names (``hidden.net.*`` / ``pooled.net.*``) are what produce the
``sushi.te_adapter.*`` checkpoint keys, so the on-disk format is unchanged.

When a backbone's text encoder is swapped, the new encoder's hidden states (D_te)
and a pooled vector (D_te) do not match the backbone's fixed interface. These small
trainable adapters bridge the gap; the backbone body is untouched. Unlike the REPA
projector (training-only), these adapters are part of the model at inference: they
are saved into the single-file (``sushi.te_adapter.*``) and reloaded.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None):
        super().__init__()
        width = hidden or max(out_dim, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.SiLU(),
            nn.Linear(width, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SDXLTEAdapters(nn.Module):
    """Bundle: per-token hidden adapter (D_te -> 2048) + pooled adapter (D_te -> 1280).

    forward(hidden_states[B,L,D_te], pooled[B,D_te]) -> (enc[B,L,2048], pooled[B,1280]).
    """

    def __init__(self, in_dim: int, hidden_out: int = 2048, pooled_out: int = 1280):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_out = int(hidden_out)
        self.pooled_out = int(pooled_out)
        self.hidden = _MLP(in_dim, hidden_out)
        self.pooled = _MLP(in_dim, pooled_out)

    def forward(self, hidden_states: torch.Tensor, pooled: torch.Tensor):
        return self.hidden(hidden_states), self.pooled(pooled)

    def forward_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.hidden(hidden_states)

    def forward_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.pooled(pooled)


class BridgeAdapter(nn.Module):
    """Arch-independent dim-bridge (ADDITIVE generalization of ``SDXLTEAdapters``).

    Maps an arbitrary TE's ``(hidden, pooled)`` into a backbone's
    ``(out_dim, out_pooled)`` contract. ``out_pooled=None`` builds a hidden-only
    bridge (no pooled/added-cond path, e.g. DiT backbones). ``in_pooled`` defaults
    to ``in_dim`` when a pooled path is requested but the pooled input width is not
    given separately.

    The submodule names (``hidden`` / ``pooled`` of ``_MLP``) match
    ``SDXLTEAdapters`` so an SDXL-shaped bridge produces the same ``sushi.te_adapter.*``
    key layout — but this class is new and no frozen checkpoint depends on it; SDXL
    save/load continues to use ``SDXLTEAdapters`` directly.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 in_pooled: Optional[int] = None, out_pooled: Optional[int] = None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.in_pooled = int(in_pooled) if in_pooled is not None else int(in_dim)
        self.out_pooled = int(out_pooled) if out_pooled is not None else None
        self.hidden = _MLP(self.in_dim, self.out_dim)
        self.pooled = _MLP(self.in_pooled, self.out_pooled) if self.out_pooled else None

    def forward(self, hidden_states: torch.Tensor, pooled: Optional[torch.Tensor] = None):
        h = self.hidden(hidden_states)
        if self.pooled is not None and pooled is not None:
            return h, self.pooled(pooled)
        return h, pooled
