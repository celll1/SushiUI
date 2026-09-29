"""Bounded representation-health diagnostics for Phase-2 validation."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


class FeatureHealthAccumulator:
    """Estimate collapse-sensitive statistics from a fixed token budget."""

    def __init__(
        self,
        channels: int,
        *,
        max_tokens: int = 4096,
        expected_updates: int | None = None,
    ) -> None:
        if channels <= 0 or max_tokens <= 1:
            raise ValueError("channels must be positive and max_tokens must exceed one")
        if expected_updates is not None and expected_updates <= 0:
            raise ValueError("expected_updates must be positive when provided")
        self.channels = int(channels)
        self.max_tokens = int(max_tokens)
        self.expected_updates = (
            None if expected_updates is None else int(expected_updates)
        )
        self.count = 0
        self._last_update_index = -1
        self._sum = torch.zeros(channels, dtype=torch.float64)
        self._gram = torch.zeros((channels, channels), dtype=torch.float64)
        self._unit_sum = torch.zeros(channels, dtype=torch.float64)
        self._norm_sum = 0.0

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        *,
        update_index: int | None = None,
    ) -> None:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"features must have shape [batch, {self.channels}, height, width]"
            )
        remaining = self._allocation(update_index)
        if remaining <= 0:
            return
        tokens = features.detach().permute(0, 2, 3, 1).reshape(-1, self.channels)
        if tokens.shape[0] > remaining:
            indices = torch.linspace(
                0,
                tokens.shape[0] - 1,
                remaining,
                device=tokens.device,
            ).round().to(torch.long)
            tokens = tokens.index_select(0, indices)
        tokens = tokens.to(dtype=torch.float32)
        unit = F.normalize(tokens, dim=1, eps=1e-12)
        self._sum += tokens.sum(dim=0).to(device="cpu", dtype=torch.float64)
        self._gram += (tokens.T @ tokens).to(device="cpu", dtype=torch.float64)
        self._unit_sum += unit.sum(dim=0).to(device="cpu", dtype=torch.float64)
        self._norm_sum += float(torch.linalg.vector_norm(tokens, dim=1).sum())
        self.count += int(tokens.shape[0])

    def _allocation(self, update_index: int | None) -> int:
        if self.expected_updates is None:
            if update_index is not None:
                raise ValueError("update_index requires expected_updates")
            return self.max_tokens - self.count
        if update_index is None:
            raise ValueError("update_index is required when expected_updates is set")
        index = int(update_index)
        if index <= self._last_update_index or index >= self.expected_updates:
            raise ValueError("update_index must increase within expected_updates")
        self._last_update_index = index
        target_count = (index + 1) * self.max_tokens // self.expected_updates
        return target_count - self.count

    def finalize(self) -> dict[str, Any]:
        if self.count <= 1:
            return {
                "tokens": self.count,
                "mean_token_norm": math.nan,
                "channel_std_mean": math.nan,
                "mean_pairwise_cosine": math.nan,
                "effective_rank": math.nan,
                "effective_rank_fraction": math.nan,
                "participation_rank": math.nan,
            }
        mean = self._sum / self.count
        covariance = self._gram / self.count - torch.outer(mean, mean)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        variance = float(eigenvalues.sum())
        if variance > 0.0:
            probabilities = eigenvalues / variance
            nonzero = probabilities > 0
            entropy = float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
            effective_rank = math.exp(entropy)
            participation_rank = variance**2 / float(eigenvalues.square().sum())
        else:
            effective_rank = 0.0
            participation_rank = 0.0
        pair_count = self.count * (self.count - 1)
        pairwise_cosine = (float(self._unit_sum.square().sum()) - self.count) / pair_count
        return {
            "tokens": self.count,
            "mean_token_norm": self._norm_sum / self.count,
            "channel_std_mean": float(covariance.diagonal().clamp_min(0.0).sqrt().mean()),
            "mean_pairwise_cosine": max(-1.0, min(1.0, pairwise_cosine)),
            "effective_rank": effective_rank,
            "effective_rank_fraction": effective_rank / self.channels,
            "participation_rank": participation_rank,
        }
