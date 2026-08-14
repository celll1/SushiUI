# SushiUI vendored copy — MODIFIED (see the "SushiUI modifications" note below).
#
# Source: https://github.com/huggingface/diffusers, PR #14456 "Add MiniMax Music 3",
#         commit dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d, retrieved 2026-08-14,
#         file `src/diffusers/models/transformers/transformer_minimax_music3.py`.
# Upstream file license: Apache-2.0 (see the copyright header below).
#
# The MiniMax Music 3 *weights* this code consumes are covered by the MiniMax
# Music 3 Community License Agreement; see docs/guides/MINIMAX_MUSIC3_DESIGN.md.
#
# SushiUI modifications:
#   * relative diffusers imports rewritten to absolute `diffusers.*` paths
#     (this file lives outside the `diffusers` package tree).
#   * attention runs through SushiUI's unified conduit
#     (`core.attention.dispatch_attention`) instead of diffusers'
#     `dispatch_attention_fn`, so the repo-wide `attention_type` vocabulary
#     (native / flash / sage / tq) selects the kernel here as it does on every
#     other architecture. The per-module `_attn_backend` / `_attn_mode` are
#     stamped once per forward by `_stamp_attention_backend`, mirroring
#     `minimax_h3/vendor/transformer_minimax_h3.py`. This is full
#     self-attention over one packed 1D sequence: no mask, no causal, equal
#     q/kv heads (head_dim 64 here, inside every backend's allowed set).
#   * everything else — the Fourier time embedding, the partial rotary
#     embedding, the block structure, the pre/post 1x1-conv residual wrapper
#     around `proj_in`/`proj_out`, and the prepended-timestep-token trick — is
#     unchanged from upstream; this is a checkpoint contract.
#
# Why vendored: PR #14456 targets diffusers `0.40.0.dev0` and is unmerged;
# this repo runs diffusers 0.38.0 for nine other architectures and must not be
# pinned to an unreleased dev branch for one architecture. See
# docs/guides/MINIMAX_MUSIC3_DESIGN.md, "Dependency gate".

# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.torch_utils import lru_cache_unless_export
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from core.attention import AttentionMode, dispatch_attention


class MiniMaxMusic3FourierEmbedding(nn.Module):
    """Random Fourier features over the flow-matching time in `[0, 1]`. The projection is a trained checkpoint weight."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_dim // 2, 1))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * timestep.unsqueeze(-1) @ self.weight.T
        return torch.cat((angles.cos(), angles.sin()), dim=-1)


class MiniMaxMusic3RotaryEmbedding(nn.Module):
    """Partial rotary embedding: only the first `rotary_dim` dimensions of each head rotate."""

    def __init__(self, rotary_dim: int, theta: float = 10000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.theta = theta

    @lru_cache_unless_export(maxsize=32)
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, device=device).float() / self.rotary_dim))
        steps = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(steps, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos().contiguous(), freqs.sin().contiguous()


def _apply_partial_rotary_emb(
    hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    # hidden_states: [batch, seq, heads, head_dim]; only the leading rotary dims rotate.
    cos, sin = rotary_emb
    rotary_dim = cos.shape[-1]
    cos = cos[:, None, :].to(hidden_states.dtype)
    sin = sin[:, None, :].to(hidden_states.dtype)
    rotated = hidden_states[..., :rotary_dim]
    half_first, half_second = rotated.chunk(2, dim=-1)
    rotate_half = torch.cat((-half_second, half_first), dim=-1)
    rotated = rotated * cos + rotate_half * sin
    return torch.cat((rotated, hidden_states[..., rotary_dim:]), dim=-1)


class MiniMaxMusic3AttnProcessor:
    r"""
    Full self-attention over one packed 1D latent sequence. There is no cross-attention and no mask anywhere in this
    transformer.
    """

    # SushiUI: the conduit's backend/mode are stamped onto the ATTENTION MODULE
    # (`attn._attn_backend` / `attn._attn_mode`) once per forward by
    # `MiniMaxMusic3Transformer1DModel._stamp_attention_backend`, not onto the
    # processor -- one processor instance is shared per attention module. These
    # two class attributes are the diffusers processor contract and are no
    # longer read.
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "MiniMaxMusic3Attention",
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.view(batch_size, seq_len, attn.heads, attn.head_dim)
        key = key.view(batch_size, seq_len, attn.heads, attn.head_dim)
        value = value.view(batch_size, seq_len, attn.heads, attn.head_dim)

        query = _apply_partial_rotary_emb(query, rotary_emb)
        key = _apply_partial_rotary_emb(key, rotary_emb)

        # SushiUI: routed through the unified conduit. query/key/value are already BSHD
        # ([B, S, H, D] -- `view` above splits the head axis out of the last dim, it does not transpose), which is
        # the conduit's canonical layout. head_dim is 64 on the released checkpoint, inside every backend's allowed
        # head-dim set, and q/kv share `attn.heads` so no GQA term is needed. No mask, no causal: full self-attention
        # over one document.
        hidden_states = dispatch_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=getattr(attn, "_attn_backend", "native"),
            mode=getattr(attn, "_attn_mode", AttentionMode.INFERENCE),
            layout="BSHD",
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class MiniMaxMusic3Attention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MiniMaxMusic3AttnProcessor
    _available_processors = [MiniMaxMusic3AttnProcessor]

    def __init__(self, dim: int, heads: int, head_dim: int, processor: Optional[MiniMaxMusic3AttnProcessor] = None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])
        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb=rotary_emb)


class MiniMaxMusic3TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MiniMaxMusic3Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2)
        self.ff_out = nn.Linear(ff_inner_dim, dim)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary_emb)
        gate_states, gate = self.ff_in(self.norm2(hidden_states)).chunk(2, dim=-1)
        hidden_states = hidden_states + self.ff_out(gate_states * torch.nn.functional.silu(gate))
        return hidden_states


class MiniMaxMusic3Transformer1DModel(ModelMixin, ConfigMixin):
    r"""
    The flow-matching diffusion transformer of MiniMax Music 3. It denoises Flow-VAE audio latents conditioned on
    per-frame hidden states produced by the autoregressive language-model stage.

    Inputs are 1D latent sequences of shape `(batch, in_channels, length)`. The conditioning signal
    (`encoder_hidden_states`, shape `(batch, length, condition_dim)`) must already be aligned to the latent timeline —
    see `MiniMaxMusic3ConditionEncoder`. The flow-matching `timestep` runs from 0 (noise) to 1 (data).
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxMusic3TransformerBlock"]
    _repeated_blocks = ["MiniMaxMusic3TransformerBlock"]
    _skip_layerwise_casting_patterns = ["time_proj", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        condition_dim: int = 2048,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        ff_inner_dim: int = 8192,
        rotary_dim: int = 32,
        fourier_embedding_dim: int = 256,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        # The transformer input concatenates [latent, zeros(in_channels), condition] along channels.
        concat_channels = 2 * in_channels + condition_dim

        self.time_proj = MiniMaxMusic3FourierEmbedding(fourier_embedding_dim)
        self.time_embed = TimestepEmbedding(fourier_embedding_dim, inner_dim)

        self.preprocess_conv = nn.Conv1d(concat_channels, concat_channels, 1, bias=False)
        self.proj_in = nn.Linear(concat_channels, inner_dim, bias=False)
        self.rotary_emb = MiniMaxMusic3RotaryEmbedding(rotary_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxMusic3TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, ff_inner_dim)
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)

        self.gradient_checkpointing = False

    # SushiUI addition: attention-backend selection for the unified conduit.
    def _stamp_attention_backend(self) -> None:
        """Propagate this model's ``_attn_backend`` to every attention module.

        The MODE is derived from the autograd state rather than configured:
        inference under ``no_grad`` may use inference-only backends (sage),
        while a training forward with grad enabled must not -- the conduit
        refuses them for ``AttentionMode.TRAINING``. Mirrors
        ``minimax_h3/vendor/transformer_minimax_h3.py``.
        """
        backend = getattr(self, "_attn_backend", "native")
        mode = AttentionMode.TRAINING if torch.is_grad_enabled() else AttentionMode.INFERENCE
        for m in self.modules():
            if isinstance(m, MiniMaxMusic3Attention):
                m._attn_backend = backend
                m._attn_mode = mode

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor] | Transformer2DModelOutput:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, in_channels, length)`):
                Noisy Flow-VAE latents.
            timestep (`torch.Tensor` of shape `(batch,)`):
                Flow-matching time in `[0, 1]`, where 0 is pure noise and 1 is data.
            encoder_hidden_states (`torch.Tensor` of shape `(batch, length, condition_dim)`):
                Frame-aligned conditioning from `MiniMaxMusic3ConditionEncoder`. Pass zeros for the unconditional
                branch of classifier-free guidance.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            The predicted flow-matching velocity with the same shape as `hidden_states`.
        """
        self._stamp_attention_backend()  # SushiUI: unified attention conduit
        zeros = torch.zeros_like(hidden_states)
        hidden_states = torch.cat((hidden_states, zeros, encoder_hidden_states.transpose(1, 2)), dim=1)
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        hidden_states = hidden_states.transpose(1, 2)

        temb = self.time_embed(self.time_proj(timestep))

        hidden_states = self.proj_in(hidden_states)
        # The timestep embedding is prepended as one extra token and removed after the blocks.
        hidden_states = torch.cat((temb.unsqueeze(1), hidden_states), dim=1)
        rotary_emb = self.rotary_emb(hidden_states.shape[1], hidden_states.device)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, rotary_emb)
            else:
                hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.proj_out(hidden_states[:, 1:])
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)
