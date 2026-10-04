"""Evidence-slot aggregation and expandable classification head for FoveaTag."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EvidenceHeadOutput:
    logits: torch.Tensor
    local_scores: torch.Tensor
    best_evidence_indices: torch.Tensor
    mixed_evidence: torch.Tensor


class EvidenceSlotMixer(nn.Module):
    """Condition each local slot on the overview without changing slot count."""

    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        hidden_dim = hidden_dim or feature_dim * 2
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.feature_dim = feature_dim
        self.local_norm = nn.LayerNorm(feature_dim)
        self.global_norm = nn.LayerNorm(feature_dim)
        self.update = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )

    def forward(self, global_summary: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        _validate_features(global_summary, evidence, self.feature_dim)
        if evidence.shape[1] == 0:
            return evidence
        local = self.local_norm(evidence)
        overview = self.global_norm(global_summary).unsqueeze(1).expand_as(local)
        update, gate = self.update(torch.cat((local, overview), dim=-1)).chunk(2, dim=-1)
        return evidence + torch.sigmoid(gate) * update


class EvidenceTagHead(nn.Module):
    """Combine global context with a per-tag smooth maximum over evidence slots."""

    def __init__(
        self,
        feature_dim: int,
        num_tags: int,
        *,
        temperature: float = 0.1,
        mixer_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_tags <= 0:
            raise ValueError("feature_dim and num_tags must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.feature_dim = feature_dim
        self.num_tags = num_tags
        self.temperature = float(temperature)
        self.mixer = EvidenceSlotMixer(feature_dim, mixer_hidden_dim)
        # A single row-expandable layer keeps compatibility with the existing
        # tagger vocabulary/optimizer migration contract.
        self.head = nn.Linear(feature_dim * 2, num_tags)

    @property
    def global_weight(self) -> torch.Tensor:
        return self.head.weight[:, : self.feature_dim]

    @property
    def local_weight(self) -> torch.Tensor:
        return self.head.weight[:, self.feature_dim :]

    def forward(
        self,
        global_summary: torch.Tensor,
        evidence: torch.Tensor,
        evidence_mask: Optional[torch.Tensor] = None,
    ) -> EvidenceHeadOutput:
        _validate_features(global_summary, evidence, self.feature_dim)
        batch_size, slot_count, _ = evidence.shape
        mask = _validate_mask(evidence_mask, batch_size, slot_count, evidence.device)
        mixed = self.mixer(global_summary, evidence)

        global_logits = F.linear(global_summary, self.global_weight, self.head.bias)
        local_scores = F.linear(mixed, self.local_weight)
        if slot_count == 0:
            local_term = global_logits.new_zeros(global_logits.shape)
            best_indices = torch.full(
                global_logits.shape,
                -1,
                dtype=torch.long,
                device=global_logits.device,
            )
        else:
            masked_scores = local_scores.masked_fill(~mask.unsqueeze(-1), -torch.inf)
            valid_samples = mask.any(dim=1)
            local_term = self.temperature * torch.logsumexp(
                masked_scores / self.temperature,
                dim=1,
            )
            valid_counts = mask.sum(dim=1).clamp_min(1).to(local_term.dtype)
            local_term = local_term - self.temperature * valid_counts.log().unsqueeze(1)
            local_term = torch.where(valid_samples.unsqueeze(1), local_term, 0.0)
            best_indices = masked_scores.argmax(dim=1)
            best_indices = torch.where(
                valid_samples.unsqueeze(1),
                best_indices,
                torch.full_like(best_indices, -1),
            )

        return EvidenceHeadOutput(
            logits=global_logits + local_term,
            local_scores=local_scores,
            best_evidence_indices=best_indices,
            mixed_evidence=mixed,
        )

    def expand_head(self, new_num_tags: int) -> Tuple[nn.Parameter, Optional[nn.Parameter]]:
        """Append randomly initialized vocabulary rows while preserving old rows."""
        if new_num_tags < self.num_tags:
            raise ValueError("EvidenceTagHead cannot shrink its vocabulary")
        if new_num_tags == self.num_tags:
            return self.head.weight, self.head.bias
        old_head = self.head
        expanded = nn.Linear(
            old_head.in_features,
            new_num_tags,
            bias=old_head.bias is not None,
            device=old_head.weight.device,
            dtype=old_head.weight.dtype,
        )
        with torch.no_grad():
            expanded.weight[: self.num_tags].copy_(old_head.weight)
            if old_head.bias is not None and expanded.bias is not None:
                expanded.bias[: self.num_tags].copy_(old_head.bias)
        self.head = expanded
        self.num_tags = new_num_tags
        return expanded.weight, expanded.bias


class ResidualEvidenceTagHead(nn.Module):
    """Add zero-initialized local evidence to fixed baseline logits."""

    def __init__(
        self,
        feature_dim: int,
        num_tags: int,
        *,
        temperature: float = 0.1,
        mixer_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_tags <= 0:
            raise ValueError("feature_dim and num_tags must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.feature_dim = feature_dim
        self.num_tags = num_tags
        self.temperature = float(temperature)
        self.mixer = EvidenceSlotMixer(feature_dim, mixer_hidden_dim)
        self.head = nn.Linear(feature_dim, num_tags, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(
        self,
        baseline_logits: torch.Tensor,
        global_summary: torch.Tensor,
        evidence: torch.Tensor,
        evidence_mask: Optional[torch.Tensor] = None,
    ) -> EvidenceHeadOutput:
        _validate_features(global_summary, evidence, self.feature_dim)
        batch_size, slot_count, _ = evidence.shape
        if baseline_logits.shape != (batch_size, self.num_tags):
            raise ValueError(
                f"baseline_logits must have shape [{batch_size}, {self.num_tags}]"
            )
        mask = _validate_mask(evidence_mask, batch_size, slot_count, evidence.device)
        mixed = self.mixer(global_summary, evidence)
        local_scores = self.head(mixed)
        if slot_count == 0:
            local_term = baseline_logits.new_zeros(baseline_logits.shape)
            best_indices = torch.full(
                baseline_logits.shape,
                -1,
                dtype=torch.long,
                device=baseline_logits.device,
            )
        else:
            masked_scores = local_scores.masked_fill(~mask.unsqueeze(-1), -torch.inf)
            valid_samples = mask.any(dim=1)
            local_term = self.temperature * torch.logsumexp(
                masked_scores / self.temperature,
                dim=1,
            )
            valid_counts = mask.sum(dim=1).clamp_min(1).to(local_term.dtype)
            local_term = local_term - self.temperature * valid_counts.log().unsqueeze(1)
            local_term = torch.where(valid_samples.unsqueeze(-1), local_term, 0.0)
            best_indices = masked_scores.argmax(dim=1)
            best_indices = torch.where(
                valid_samples.unsqueeze(-1),
                best_indices,
                torch.full_like(best_indices, -1),
            )
        return EvidenceHeadOutput(
            logits=baseline_logits + local_term,
            local_scores=local_scores,
            best_evidence_indices=best_indices,
            mixed_evidence=mixed,
        )

    def expand_head(self, new_num_tags: int) -> Tuple[nn.Parameter, None]:
        if new_num_tags < self.num_tags:
            raise ValueError("ResidualEvidenceTagHead cannot shrink its vocabulary")
        if new_num_tags == self.num_tags:
            return self.head.weight, None
        old_head = self.head
        expanded = nn.Linear(
            old_head.in_features,
            new_num_tags,
            bias=False,
            device=old_head.weight.device,
            dtype=old_head.weight.dtype,
        )
        nn.init.zeros_(expanded.weight)
        with torch.no_grad():
            expanded.weight[: self.num_tags].copy_(old_head.weight)
        self.head = expanded
        self.num_tags = new_num_tags
        return expanded.weight, None


def _validate_features(
    global_summary: torch.Tensor,
    evidence: torch.Tensor,
    feature_dim: int,
) -> None:
    if global_summary.ndim != 2 or global_summary.shape[-1] != feature_dim:
        raise ValueError(f"global_summary must have shape [batch, {feature_dim}]")
    if evidence.ndim != 3 or evidence.shape[-1] != feature_dim:
        raise ValueError(f"evidence must have shape [batch, slots, {feature_dim}]")
    if evidence.shape[0] != global_summary.shape[0]:
        raise ValueError("global_summary and evidence batch sizes must match")
    if evidence.device != global_summary.device:
        raise ValueError("global_summary and evidence must be on the same device")


def _validate_mask(
    mask: Optional[torch.Tensor],
    batch_size: int,
    slot_count: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch_size, slot_count), dtype=torch.bool, device=device)
    if mask.shape != (batch_size, slot_count):
        raise ValueError(f"evidence_mask must have shape [{batch_size}, {slot_count}]")
    return mask.to(device=device, dtype=torch.bool)
