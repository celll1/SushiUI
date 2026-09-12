"""MiniMax-H3 target-video window connectivity for FlexAttention."""

from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from core.attention import AttentionMechanism


@dataclass
class H3VideoWindowPlan:
    position_ids: torch.Tensor
    target_video_rows: torch.Tensor
    temporal_radius: float
    spatial_radius: float
    block_size: int = 128
    mechanism: AttentionMechanism = field(
        default=AttentionMechanism.H3_VIDEO_WINDOW, init=False
    )
    _block_mask: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.position_ids.ndim != 2 or self.position_ids.shape[1] != 3:
            raise ValueError("H3 sparse position_ids must have shape [sequence, 3]")
        if self.target_video_rows.shape != (self.position_ids.shape[0],):
            raise ValueError("H3 sparse target-video mask must match the packed sequence")
        if self.target_video_rows.dtype != torch.bool:
            raise ValueError("H3 sparse target-video mask must be boolean")
        if self.target_video_rows.device != self.position_ids.device:
            raise ValueError("H3 sparse metadata must be on one device")
        if self.temporal_radius < 0 or self.spatial_radius < 0:
            raise ValueError("H3 sparse radii must be non-negative")
        if self.block_size <= 0:
            raise ValueError("H3 sparse block_size must be positive")

        length = self.position_ids.shape[0]
        padded = ((length + self.block_size - 1) // self.block_size) * self.block_size
        target = torch.zeros(padded, dtype=torch.bool, device=self.target_video_rows.device)
        target[:length] = self.target_video_rows
        self._pure_target_blocks = target.view(-1, self.block_size).all(dim=1)

    @classmethod
    def from_layout(
        cls,
        layout: Dict[str, Any],
        *,
        temporal_radius: float,
        spatial_radius: float,
        block_size: int = 128,
    ) -> "H3VideoWindowPlan":
        position_ids = layout["position_ids"]
        video_indices = layout["video_indices"]
        condition_rows = int(layout.get("num_condition_video_rows", 0) or 0)
        target_video_rows = torch.zeros(
            position_ids.shape[0], dtype=torch.bool, device=position_ids.device
        )
        target_video_rows[video_indices[condition_rows:]] = True
        return cls(
            position_ids=position_ids,
            target_video_rows=target_video_rows,
            temporal_radius=float(temporal_radius),
            spatial_radius=float(spatial_radius),
            block_size=int(block_size),
        )

    def allowed(self, query_index: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
        query_block = query_index // self.block_size
        key_block = key_index // self.block_size
        sparse_pair = self._pure_target_blocks[query_block] & self._pure_target_blocks[key_block]
        query_position = self.position_ids[query_index]
        key_position = self.position_ids[key_index]
        delta = (query_position - key_position).abs()
        local = (
            (delta[..., 0] <= self.temporal_radius)
            & (delta[..., 1] <= self.spatial_radius)
            & (delta[..., 2] <= self.spatial_radius)
        )
        return ~sparse_pair | local

    def mask_mod(self, batch, head, query_index, key_index):
        del batch, head
        return self.allowed(query_index, key_index)

    def block_mask(self):
        if self._block_mask is None:
            from torch.nn.attention.flex_attention import create_block_mask

            length = self.position_ids.shape[0]
            self._block_mask = create_block_mask(
                self.mask_mod,
                B=None,
                H=None,
                Q_LEN=length,
                KV_LEN=length,
                device=self.position_ids.device,
                BLOCK_SIZE=self.block_size,
                _compile=True,
            )
        return self._block_mask


def build_h3_attention_plan(
    method: str,
    layout: Dict[str, Any],
    *,
    temporal_radius: float,
    spatial_radius: float,
    block_size: int = 128,
) -> H3VideoWindowPlan | None:
    if method == AttentionMechanism.DENSE.value:
        return None
    if method != AttentionMechanism.H3_VIDEO_WINDOW.value:
        raise ValueError(f"MiniMax-H3 does not implement attention mechanism {method!r}")
    return H3VideoWindowPlan.from_layout(
        layout,
        temporal_radius=temporal_radius,
        spatial_radius=spatial_radius,
        block_size=block_size,
    )
