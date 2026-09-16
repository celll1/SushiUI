"""Fixed-length conditioning bridge from SenseNova prefix state to SDXL I/F."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


def selected_layer_indices(num_layers: int) -> tuple[int, ...]:
    """Checkpoint-relative 25/50/75/100% layer selection without duplicates."""
    if int(num_layers) <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    last = int(num_layers) - 1
    values = (round(0.25 * last), round(0.50 * last), round(0.75 * last), last)
    return tuple(dict.fromkeys(int(value) for value in values))


@dataclass(frozen=True)
class ChimeraBridgeConfig:
    hidden_size: int
    kv_width: int
    selected_layers: tuple[int, ...]
    context_tokens: int = 77
    context_dim: int = 2048
    pooled_dim: int = 1280
    bridge_dim: int = 1024
    num_heads: int = 8

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "kv_width": self.kv_width,
            "context_tokens": self.context_tokens,
            "context_dim": self.context_dim,
            "pooled_dim": self.pooled_dim,
            "bridge_dim": self.bridge_dim,
            "num_heads": self.num_heads,
        }
        bad = {name: value for name, value in positive.items() if int(value) <= 0}
        if bad:
            raise ValueError(f"bridge dimensions must be positive: {bad}")
        if not self.selected_layers or len(set(self.selected_layers)) != len(self.selected_layers):
            raise ValueError("selected_layers must be a non-empty unique sequence")
        if self.bridge_dim % self.num_heads:
            raise ValueError(
                f"bridge_dim {self.bridge_dim} must be divisible by num_heads {self.num_heads}"
            )


@dataclass
class ChimeraBridgeOutput:
    encoder_hidden_states: torch.Tensor
    pooled_text_embeds: torch.Tensor
    context_positions: torch.Tensor
    resampler_weights: torch.Tensor
    position_variance: torch.Tensor


class ConditioningBridge(nn.Module):
    """Project layer-specific SenseNova K/V into fixed SDXL conditioning.

    K/V arrive as ``[B,H_kv,L,D_head]``; ``kv_width`` is ``H_kv*D_head``.
    Each layer owns its projection because its attention basis is independent.
    """

    def __init__(self, config: ChimeraBridgeConfig):
        super().__init__()
        self.config = config
        self.hidden_norm = nn.LayerNorm(config.hidden_size)
        self.hidden_projection = nn.Linear(config.hidden_size, config.bridge_dim)
        self.kv_projections = nn.ModuleDict({
            str(layer): nn.Linear(2 * config.kv_width, config.bridge_dim)
            for layer in config.selected_layers
        })
        self.kv_gates = nn.ParameterDict({
            str(layer): nn.Parameter(torch.zeros(())) for layer in config.selected_layers
        })
        self.layer_embeddings = nn.ParameterDict({
            str(layer): nn.Parameter(torch.zeros(config.bridge_dim))
            for layer in config.selected_layers
        })
        self.queries = nn.Parameter(torch.empty(config.context_tokens, config.bridge_dim))
        nn.init.normal_(self.queries, std=config.bridge_dim ** -0.5)
        self.resampler = nn.MultiheadAttention(
            config.bridge_dim, config.num_heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(config.bridge_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.bridge_dim, 4 * config.bridge_dim),
            nn.GELU(),
            nn.Linear(4 * config.bridge_dim, config.bridge_dim),
        )
        self.context_projection = nn.Linear(config.bridge_dim, config.context_dim)
        self.pooled_projection = nn.Linear(config.bridge_dim, config.pooled_dim)

    @staticmethod
    def _flatten_kv(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError(
                f"K/V must have equal [B,H,L,D] shapes, got {tuple(key.shape)} and {tuple(value.shape)}"
            )
        key = key.transpose(1, 2).flatten(2)
        value = value.transpose(1, 2).flatten(2)
        return torch.cat((key, value), dim=-1)

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        selected_kv: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> ChimeraBridgeOutput:
        if last_hidden_state.ndim != 3:
            raise ValueError(
                f"last_hidden_state must be [B,L,D], got {tuple(last_hidden_state.shape)}"
            )
        batch, sequence, hidden = last_hidden_state.shape
        if hidden != self.config.hidden_size:
            raise ValueError(f"hidden width {hidden} != configured {self.config.hidden_size}")
        if attention_mask.shape != (batch, sequence):
            raise ValueError(
                f"attention_mask must be {(batch, sequence)}, got {tuple(attention_mask.shape)}"
            )
        if positions.shape != (batch, sequence, 3):
            raise ValueError(
                f"positions must be {(batch, sequence, 3)}, got {tuple(positions.shape)}"
            )
        mask = attention_mask.to(device=last_hidden_state.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every bridge item must contain at least one unmasked prefix token")

        memory = self.hidden_projection(self.hidden_norm(last_hidden_state))
        memory_positions = positions.to(device=last_hidden_state.device, dtype=torch.float32)
        expected = set(self.config.selected_layers)
        if set(selected_kv) != expected:
            raise ValueError(
                f"selected K/V layers {sorted(selected_kv)} differ from configured {sorted(expected)}"
            )
        for layer in self.config.selected_layers:
            key, value = selected_kv[layer]
            flat = self._flatten_kv(key, value)
            if flat.shape[:2] != (batch, sequence) or flat.shape[-1] != 2 * self.config.kv_width:
                raise ValueError(
                    f"layer {layer} flattened K/V shape {tuple(flat.shape)} is incompatible with "
                    f"batch={batch}, sequence={sequence}, kv_width={self.config.kv_width}"
                )
            name = str(layer)
            projected = self.kv_projections[name](flat)
            residual = projected + self.layer_embeddings[name]
            memory = memory + torch.tanh(self.kv_gates[name]) * residual

        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        attended, per_head_weights = self.resampler(
            queries,
            memory,
            memory,
            key_padding_mask=~mask,
            need_weights=True,
            average_attn_weights=False,
        )
        refined = self.output_norm(attended + self.feed_forward(attended))
        context = self.context_projection(refined)
        pooled = self.pooled_projection(refined.mean(dim=1))

        weights = per_head_weights.float().mean(dim=1)
        weights = weights * mask[:, None, :].float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        context_positions = torch.matmul(weights, memory_positions.float())
        delta = memory_positions[:, None, :, :] - context_positions[:, :, None, :]
        position_variance = (weights[..., None] * delta.square()).sum(dim=2)
        return ChimeraBridgeOutput(
            encoder_hidden_states=context,
            pooled_text_embeds=pooled,
            context_positions=context_positions,
            resampler_weights=weights,
            position_variance=position_variance,
        )
