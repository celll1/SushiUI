"""SenseNova-positioned attention for the Chimera SDXL U-Net."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

import torch
from diffusers.models.attention_processor import Attention

from core.attention import AttentionMode, dispatch_attention, dispatch_attention_varlen

from .positional import apply_sensenova_rope, apply_sensenova_rope_qk, spatial_query_positions


@dataclass(frozen=True)
class ChimeraAttentionContext:
    context_positions: torch.Tensor
    target_height: int
    target_width: int
    crop_top: int = 0
    crop_left: int = 0
    prefix_terminal_t: float = 0.0
    cache_key: Hashable | None = None


def _spatial_shape(sequence: int, target_height: int, target_width: int) -> tuple[int, int]:
    """Recover an attention site's grid from its sequence and canvas ratio."""
    candidates = [
        (height, sequence // height)
        for height in range(1, int(sequence**0.5) + 1)
        if sequence % height == 0
    ]
    candidates += [(width, height) for height, width in candidates if height != width]
    target_ratio = float(target_height) / float(target_width)
    return min(candidates, key=lambda hw: abs(hw[0] / hw[1] - target_ratio))


class ChimeraAttnProcessor:
    """Parameter-free SDXL attention with SenseNova 3-D RoPE.

    Cross-attention K/V are generation-local. The cache holds post-RoPE K and
    projected V, and is never serialized with the U-Net.
    """

    def __init__(
        self,
        backend: str = "normal",
        *,
        mode: AttentionMode = AttentionMode.INFERENCE,
        rope_theta: float = 1_000_000.0,
        rope_theta_hw: float = 10_000.0,
    ) -> None:
        self.backend = backend
        self.mode = mode
        self.rope_theta = float(rope_theta)
        self.rope_theta_hw = float(rope_theta_hw)
        self.context: ChimeraAttentionContext | None = None
        self._cross_kv_cache: dict[Hashable, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_context(self, context: ChimeraAttentionContext | None) -> None:
        self.context = context

    def clear_cache(self) -> None:
        self._cross_kv_cache.clear()

    @staticmethod
    def _valid_key_rows(
        attention_mask: torch.Tensor | None,
        *,
        batch: int,
        length: int,
    ) -> torch.Tensor | None:
        """Recover a right-padded validity mask from diffusers' score bias."""
        if attention_mask is None:
            return None
        mask = attention_mask
        while mask.ndim > 2 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        if mask.shape != (batch, length):
            raise ValueError(
                f"Chimera context mask must reduce to {(batch, length)}, got "
                f"{tuple(attention_mask.shape)}"
            )
        valid = mask if mask.dtype == torch.bool else mask > -1.0
        valid = valid.to(dtype=torch.bool)
        if bool(valid.all()):
            return None
        lengths = valid.sum(dim=1)
        if bool((lengths == 0).any()):
            raise ValueError("Chimera context mask cannot contain an empty prefix")
        expected = torch.arange(length, device=valid.device)[None, :] < lengths[:, None]
        if not torch.equal(valid, expected):
            raise ValueError("Chimera context mask must be contiguous right padding")
        return valid

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        context = self.context
        if context is None:
            raise RuntimeError("Chimera attention context is not armed")
        residual = hidden_states
        is_self_attention = encoder_hidden_states is None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        inner_dim = query.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        site_height, site_width = _spatial_shape(
            sequence_length, context.target_height, context.target_width
        )
        query_positions = spatial_query_positions(
            site_height,
            site_width,
            target_height=context.target_height,
            target_width=context.target_width,
            crop_top=context.crop_top,
            crop_left=context.crop_left,
            prefix_terminal_t=context.prefix_terminal_t,
            device=query.device,
        ).expand(batch_size, -1, -1)

        cache_id = None
        if not is_self_attention and context.cache_key is not None:
            cache_id = (
                context.cache_key,
                str(query.device),
                query.dtype,
                batch_size,
                encoder_hidden_states.shape[1],
            )
        cached = self._cross_kv_cache.get(cache_id) if cache_id is not None else None
        if cached is None:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if is_self_attention:
                query, key = apply_sensenova_rope_qk(
                    query,
                    key,
                    query_positions,
                    query_positions,
                    rope_theta=self.rope_theta,
                    rope_theta_hw=self.rope_theta_hw,
                )
            else:
                key_positions = context.context_positions.to(device=query.device)
                if key_positions.shape[0] == 1 and batch_size != 1:
                    key_positions = key_positions.expand(batch_size, -1, -1)
                if key_positions.shape != (batch_size, key.shape[2], 3):
                    raise ValueError(
                        "Chimera context positions must match cross-attention keys: "
                        f"positions={tuple(key_positions.shape)}, "
                        f"keys={(batch_size, key.shape[2], 3)}"
                    )
                query, key = apply_sensenova_rope_qk(
                    query,
                    key,
                    query_positions,
                    key_positions,
                    rope_theta=self.rope_theta,
                    rope_theta_hw=self.rope_theta_hw,
                )
                if cache_id is not None:
                    self._cross_kv_cache[cache_id] = (key, value)
        else:
            key, value = cached
            query = apply_sensenova_rope(
                query,
                query_positions,
                rope_theta=self.rope_theta,
                rope_theta_hw=self.rope_theta_hw,
            )

        valid_rows = self._valid_key_rows(
            attention_mask,
            batch=batch_size,
            length=key.shape[2],
        )
        if valid_rows is None:
            hidden_states = dispatch_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self.backend,
                mode=self.mode,
                layout="BHSD",
            )
        else:
            # Packed varlen preserves FlashAttention for ragged caption batches;
            # passing a dense padding mask would force its mask-less kernel to
            # the native backend for every U-Net cross-attention site.
            q_bshd = query.transpose(1, 2).contiguous()
            k_bshd = key.transpose(1, 2).contiguous()
            v_bshd = value.transpose(1, 2).contiguous()
            q_length = q_bshd.shape[1]
            key_lengths = valid_rows.sum(dim=1, dtype=torch.int32)
            packed_q = q_bshd.reshape(-1, attn.heads, head_dim)
            packed_k = k_bshd[valid_rows]
            packed_v = v_bshd[valid_rows]
            cu_q = torch.arange(
                0,
                (batch_size + 1) * q_length,
                q_length,
                device=query.device,
                dtype=torch.int32,
            )
            cu_k = torch.cat((
                torch.zeros(1, device=query.device, dtype=torch.int32),
                key_lengths.cumsum(0),
            ))
            packed_out = dispatch_attention_varlen(
                packed_q,
                packed_k,
                packed_v,
                cu_q,
                cu_k,
                q_length,
                int(key_lengths.max().item()),
                dropout_p=0.0,
                is_causal=False,
                backend=self.backend,
                mode=self.mode,
            )
            hidden_states = packed_out.reshape(
                batch_size, q_length, attn.heads, head_dim
            ).transpose(1, 2).contiguous()
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states.to(query.dtype)))
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install_chimera_attention_processors(
    unet: torch.nn.Module,
    *,
    backend: str = "normal",
    mode: AttentionMode = AttentionMode.INFERENCE,
    rope_theta: float = 1_000_000.0,
    rope_theta_hw: float = 10_000.0,
) -> None:
    processors = {
        name: ChimeraAttnProcessor(
            backend,
            mode=mode,
            rope_theta=rope_theta,
            rope_theta_hw=rope_theta_hw,
        )
        for name in unet.attn_processors
    }
    unet.set_attn_processor(processors)


def set_chimera_attention_context(
    unet: torch.nn.Module, context: ChimeraAttentionContext | None
) -> None:
    for processor in unet.attn_processors.values():
        if isinstance(processor, ChimeraAttnProcessor):
            processor.set_context(context)


def clear_chimera_attention_caches(unet: torch.nn.Module) -> None:
    for processor in unet.attn_processors.values():
        if isinstance(processor, ChimeraAttnProcessor):
            processor.set_context(None)
            processor.clear_cache()
