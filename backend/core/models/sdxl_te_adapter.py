"""Trainable text-encoder adapters for custom-TE SDXL.

When the SDXL text encoder is swapped (SigLIP2 text / FLAN-T5 / Qwen3), the new
encoder's hidden states (D_te) and a pooled vector (D_te) do not match the U-Net's
fixed interface (cross_attention_dim=2048, add_embedding pooled=1280). These small
trainable adapters bridge the gap; the U-Net body is untouched.

Unlike the REPA projector (training-only), these adapters are part of the model at
inference: they are saved into the single-file (sushi.te_adapter.*) and reloaded.
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
