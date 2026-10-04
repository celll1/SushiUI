"""End-to-end from-scratch FoveaTag model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from .encoder import FoveaEncoder, FoveaEncoderConfig, FoveaEncoderOutput
from .head import EvidenceHeadOutput, EvidenceTagHead
from .siglip2_control import MaskedSlotPooler


@dataclass(frozen=True)
class FoveaTaggerOutput:
    logits: torch.Tensor
    global_features: torch.Tensor
    head_output: EvidenceHeadOutput
    evidence_view_indices: torch.Tensor


class FoveaTagger(nn.Module):
    """Shared multi-view encoder plus per-tag multiple-instance head."""

    def __init__(
        self,
        num_tags: int,
        *,
        encoder_config: FoveaEncoderConfig = FoveaEncoderConfig(),
        feature_dim: int = 512,
        evidence_slots_per_view: int = 2,
        pool_heads: int = 8,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if evidence_slots_per_view <= 0:
            raise ValueError("evidence_slots_per_view must be positive")
        local_dim = encoder_config.widths[2]
        global_dim = encoder_config.widths[3]
        self.encoder = FoveaEncoder(encoder_config)
        self.evidence_slots_per_view = evidence_slots_per_view
        self.local_pooler = MaskedSlotPooler(local_dim, evidence_slots_per_view, pool_heads)
        self.local_projection = nn.Linear(local_dim, feature_dim)
        self.global_projection = nn.Linear(global_dim, feature_dim)
        self.evidence_head = EvidenceTagHead(
            feature_dim,
            num_tags,
            temperature=temperature,
        )

    @property
    def head(self) -> nn.Linear:
        return self.evidence_head.head

    def expand_head(self, new_num_tags: int) -> Tuple[nn.Parameter, Optional[nn.Parameter]]:
        return self.evidence_head.expand_head(new_num_tags)

    def forward(
        self,
        overview_images: torch.Tensor,
        overview_geometry: torch.Tensor,
        local_images: torch.Tensor,
        local_geometry: torch.Tensor,
        *,
        local_view_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, FoveaTaggerOutput]:
        if local_images.ndim != 5 or local_images.shape[2] != 3:
            raise ValueError("local_images must have shape [batch, views, 3, height, width]")
        batch_size, view_count = local_images.shape[:2]
        if overview_images.shape[0] != batch_size:
            raise ValueError("overview and local batch sizes must match")
        if local_geometry.shape != (batch_size, view_count, self.encoder.config.geometry_dim):
            raise ValueError(
                "local_geometry must have shape [batch, views, geometry_dim]"
            )
        overview = self.encoder(overview_images, overview_geometry)
        return self.classify_encoded_overview(
            overview,
            local_images,
            local_geometry,
            local_view_mask=local_view_mask,
            return_details=return_details,
        )

    def classify_encoded_overview(
        self,
        overview: FoveaEncoderOutput,
        local_images: torch.Tensor,
        local_geometry: torch.Tensor,
        *,
        local_view_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, FoveaTaggerOutput]:
        """Classify local views without recomputing a cached overview encoding."""
        if local_images.ndim != 5 or local_images.shape[2] != 3:
            raise ValueError("local_images must have shape [batch, views, 3, height, width]")
        batch_size, view_count = local_images.shape[:2]
        if overview.global_summary.shape[0] != batch_size:
            raise ValueError("encoded overview and local batch sizes must match")
        if local_geometry.shape != (batch_size, view_count, self.encoder.config.geometry_dim):
            raise ValueError("local_geometry must have shape [batch, views, geometry_dim]")
        view_mask = _view_mask(
            local_view_mask,
            batch_size,
            view_count,
            overview.global_summary.device,
        )
        global_summary = self.global_projection(overview.global_summary)

        if view_count:
            flat_images = local_images.reshape(
                batch_size * view_count,
                *local_images.shape[2:],
            )
            flat_geometry = local_geometry.reshape(batch_size * view_count, -1)
            local = self.encoder(flat_images, flat_geometry)
            local_tokens = local.local_map.flatten(2).transpose(1, 2)
            pooled = self.local_pooler(local_tokens)
            evidence = self.local_projection(pooled).reshape(
                batch_size,
                view_count * self.evidence_slots_per_view,
                -1,
            )
        else:
            evidence = global_summary.new_empty((batch_size, 0, global_summary.shape[-1]))

        evidence_mask = view_mask.repeat_interleave(self.evidence_slots_per_view, dim=1)
        head_output = self.evidence_head(global_summary, evidence, evidence_mask)
        if not return_details:
            return head_output.logits
        indices = torch.arange(view_count, device=global_summary.device, dtype=torch.long)
        indices = indices.repeat_interleave(self.evidence_slots_per_view)
        indices = indices.unsqueeze(0).expand(batch_size, -1).clone()
        indices.masked_fill_(~evidence_mask.to(indices.device), -1)
        return FoveaTaggerOutput(
            logits=head_output.logits,
            global_features=global_summary,
            head_output=head_output,
            evidence_view_indices=indices,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


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
