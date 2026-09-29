"""SigLIP2 encoder-swap control for FoveaTag Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from .head import EvidenceHeadOutput, EvidenceTagHead


@dataclass(frozen=True)
class SigLIP2ViewInputs:
    pixel_values: torch.Tensor
    pixel_attention_mask: Optional[torch.Tensor] = None
    spatial_shapes: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SigLIP2ControlOutput:
    logits: torch.Tensor
    head_output: EvidenceHeadOutput
    evidence_view_indices: torch.Tensor


class MaskedSlotPooler(nn.Module):
    """Pool a masked token sequence into a fixed number of learned slots."""

    def __init__(self, hidden_size: int, slot_count: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_size <= 0 or slot_count <= 0:
            raise ValueError("hidden_size and slot_count must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.slot_count = slot_count
        self.query = nn.Parameter(torch.empty(1, slot_count, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        nn.init.normal_(self.query, std=hidden_size ** -0.5)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"tokens must have shape [batch, sequence, {self.hidden_size}]"
            )
        if tokens.shape[1] == 0:
            raise ValueError("token sequence cannot be empty")
        valid = _token_mask(token_mask, tokens)
        # MultiheadAttention returns NaN when every key is masked. Invalid views
        # are discarded later, so give their first token a harmless key here.
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        query = self.query.expand(tokens.shape[0], -1, -1)
        normalized = self.norm(tokens)
        pooled, _ = self.attention(
            query,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        return pooled


class SigLIP2ControlTagger(nn.Module):
    """Apply FoveaTag views and evidence aggregation to an existing SigLIP2 encoder."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        *,
        hidden_size: int,
        feature_dim: int,
        num_tags: int,
        evidence_slots_per_view: int = 2,
        pool_heads: int = 4,
        is_naflex: bool = True,
        freeze_encoder: bool = False,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.vision_encoder = vision_encoder
        self.is_naflex = is_naflex
        self.evidence_slots_per_view = evidence_slots_per_view
        self.global_pooler = MaskedSlotPooler(hidden_size, 1, pool_heads)
        self.local_pooler = MaskedSlotPooler(hidden_size, evidence_slots_per_view, pool_heads)
        self.feature_projection = nn.Linear(hidden_size, feature_dim)
        self.evidence_head = EvidenceTagHead(
            feature_dim,
            num_tags,
            temperature=temperature,
        )
        if freeze_encoder:
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad = False

    @property
    def head(self) -> nn.Linear:
        return self.evidence_head.head

    def expand_head(self, new_num_tags: int) -> Tuple[nn.Parameter, Optional[nn.Parameter]]:
        return self.evidence_head.expand_head(new_num_tags)

    def forward(
        self,
        overview: SigLIP2ViewInputs,
        local: SigLIP2ViewInputs,
        *,
        local_view_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, SigLIP2ControlOutput]:
        overview_tokens = self._encode(overview)
        overview_mask = _optional_flat_mask(overview.pixel_attention_mask, overview_tokens)
        global_summary = self.feature_projection(
            self.global_pooler(overview_tokens, overview_mask).squeeze(1)
        )

        local_values, batch_size, view_count = _flatten_local_tensor(local.pixel_values)
        if batch_size != overview_tokens.shape[0]:
            raise ValueError("overview and local view batch sizes must match")
        view_mask = _view_mask(local_view_mask, batch_size, view_count, global_summary.device)
        if view_count == 0:
            evidence = global_summary.new_empty((batch_size, 0, global_summary.shape[-1]))
        else:
            flat_local = SigLIP2ViewInputs(
                pixel_values=local_values,
                pixel_attention_mask=_flatten_optional(local.pixel_attention_mask, batch_size, view_count),
                spatial_shapes=_flatten_optional(local.spatial_shapes, batch_size, view_count),
            )
            local_tokens = self._encode(flat_local)
            local_token_mask = _optional_flat_mask(flat_local.pixel_attention_mask, local_tokens)
            pooled = self.local_pooler(local_tokens, local_token_mask)
            projected = self.feature_projection(pooled)
            evidence = projected.reshape(
                batch_size,
                view_count * self.evidence_slots_per_view,
                projected.shape[-1],
            )

        evidence_mask = view_mask.repeat_interleave(self.evidence_slots_per_view, dim=1)
        head_output = self.evidence_head(global_summary, evidence, evidence_mask)
        if not return_details:
            return head_output.logits
        indices = torch.arange(view_count, device=global_summary.device, dtype=torch.long)
        indices = indices.repeat_interleave(self.evidence_slots_per_view)
        indices = indices.unsqueeze(0).expand(batch_size, -1).clone()
        indices.masked_fill_(~evidence_mask, -1)
        return SigLIP2ControlOutput(
            logits=head_output.logits,
            head_output=head_output,
            evidence_view_indices=indices,
        )

    def _encode(self, inputs: SigLIP2ViewInputs) -> torch.Tensor:
        if self.is_naflex:
            output = self.vision_encoder(
                pixel_values=inputs.pixel_values,
                pixel_attention_mask=inputs.pixel_attention_mask,
                spatial_shapes=inputs.spatial_shapes,
            )
        else:
            output = self.vision_encoder(pixel_values=inputs.pixel_values)
        tokens = getattr(output, "last_hidden_state", None)
        if tokens is None:
            raise ValueError("vision encoder output must expose last_hidden_state")
        return tokens


def _token_mask(mask: Optional[torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"token mask must have shape {tuple(tokens.shape[:2])}")
    return mask.to(device=tokens.device, dtype=torch.bool)


def _optional_flat_mask(mask: Optional[torch.Tensor], tokens: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"pixel_attention_mask must have shape {tuple(tokens.shape[:2])}")
    return mask


def _flatten_local_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if tensor.ndim < 3:
        raise ValueError("local pixel_values must have shape [batch, views, ...]")
    batch_size, view_count = tensor.shape[:2]
    return tensor.reshape(batch_size * view_count, *tensor.shape[2:]), batch_size, view_count


def _flatten_optional(
    tensor: Optional[torch.Tensor],
    batch_size: int,
    view_count: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.shape[:2] != (batch_size, view_count):
        raise ValueError("local auxiliary tensors must start with [batch, views]")
    return tensor.reshape(batch_size * view_count, *tensor.shape[2:])


def _view_mask(
    mask: Optional[torch.Tensor],
    batch_size: int,
    view_count: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch_size, view_count), dtype=torch.bool, device=device)
    if mask.shape != (batch_size, view_count):
        raise ValueError(f"local_view_mask must have shape [{batch_size}, {view_count}]")
    return mask.to(device=device, dtype=torch.bool)
