# SushiUI vendored copy — MODIFIED (see the "SushiUI modifications" note below).
#
# Source: https://github.com/huggingface/diffusers, PR #14456 "Add MiniMax Music 3",
#         commit dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d, retrieved 2026-08-14,
#         file `src/diffusers/models/transformers/minimax_music3_rvq_depth_decoder.py`.
# Upstream file license: Apache-2.0 (see the copyright header below).
#
# The MiniMax Music 3 *weights* this code consumes are covered by the MiniMax
# Music 3 Community License Agreement; see docs/guides/MINIMAX_MUSIC3_DESIGN.md.
#
# SushiUI modifications:
#   * relative diffusers imports rewritten to absolute `diffusers.*` paths.
#   * attention runs through SushiUI's unified conduit
#     (`core.attention.dispatch_attention`) instead of diffusers'
#     `dispatch_attention_fn`. Unlike the flow-matching transformer, this
#     attention IS causal, over a tiny sequence (<= `max_position_embeddings`
#     == 16: the projected global hidden state, the embedded semantic code,
#     and up to 6 already-sampled residual codes). `is_causal=True` is passed
#     straight through to the conduit; every backend supports causal masking,
#     so causality itself never triggers a capability-guard downgrade here.
#     head_dim DOES: `hidden_size / num_attention_heads` == 4096 / 16 == 256,
#     which is above sage's `max_head_dim` (128) and outside tq's
#     `allowed_head_dims` ({64, 128}) -- both are downgraded to native on
#     this model on head_dim alone (see `core/attention/registry.py`'s
#     BACKENDS table). flash's cap is 256, so flash is NOT downgraded here.
#
# Why vendored: see transformer_minimax_music3.py's header in this same
# directory and docs/guides/MINIMAX_MUSIC3_DESIGN.md, "Dependency gate".

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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm

from core.attention import AttentionMode, dispatch_attention


class MiniMaxMusic3DepthAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: "MiniMaxMusic3DepthAttention", hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.view(batch_size, seq_len, attn.heads, attn.head_dim)
        key = key.view(batch_size, seq_len, attn.heads, attn.head_dim)
        value = value.view(batch_size, seq_len, attn.heads, attn.head_dim)

        # SushiUI: routed through the unified conduit. Already BSHD; causal, no mask, equal q/kv heads.
        hidden_states = dispatch_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            backend=getattr(attn, "_attn_backend", "native"),
            mode=getattr(attn, "_attn_mode", AttentionMode.INFERENCE),
            layout="BSHD",
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return attn.to_out(hidden_states)


class MiniMaxMusic3DepthAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MiniMaxMusic3DepthAttnProcessor
    _available_processors = [MiniMaxMusic3DepthAttnProcessor]

    def __init__(self, dim: int, heads: int, processor: Optional[MiniMaxMusic3DepthAttnProcessor] = None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)


class MiniMaxMusic3DepthDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = MiniMaxMusic3DepthAttention(dim, heads)
        self.post_attention_layernorm = RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.input_layernorm(hidden_states))
        norm_states = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.down_proj(F.silu(self.gate_proj(norm_states)) * self.up_proj(norm_states))


# SushiUI addition (not upstream): the per-module attention-backend stamp, mirroring every other vendored
# transformer's `_stamp_attention_backend`. The upstream file has no such helper because upstream routes through
# diffusers' global `dispatch_attention_fn` backend selection instead of a per-model `_attn_backend` attribute.
def _stamp_depth_decoder_attention_backend(model: "MiniMaxMusic3RVQDepthDecoder") -> None:
    backend = getattr(model, "_attn_backend", "native")
    mode = AttentionMode.TRAINING if torch.is_grad_enabled() else AttentionMode.INFERENCE
    for m in model.modules():
        if isinstance(m, MiniMaxMusic3DepthAttention):
            m._attn_backend = backend
            m._attn_mode = mode


class MiniMaxMusic3RVQDepthDecoder(ModelMixin, ConfigMixin):
    r"""
    The local language model of MiniMax Music 3. Within each audio frame it autoregressively predicts the seven
    residual RVQ codebooks (c1..c7) from the global language model's hidden state and the frame's semantic code, and
    exposes the per-step hidden states that condition the flow-matching transformer.

    It also owns the embedding table for the residual codebooks, which the pipeline uses to embed complete frames for
    the global language model's feedback loop.
    """

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        intermediate_size: int = 6144,
        audio_vocab_size: int = 1024,
        num_codebooks: int = 8,
        max_position_embeddings: int = 16,
    ):
        super().__init__()
        self.audio_embeddings = nn.Embedding(audio_vocab_size * (num_codebooks - 1), hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList(
            [
                MiniMaxMusic3DepthDecoderBlock(hidden_size, num_attention_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.audio_heads = nn.ModuleList(
            [nn.Linear(hidden_size, audio_vocab_size, bias=False) for _ in range(num_codebooks - 1)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            inputs_embeds (`torch.Tensor` of shape `(batch, steps, hidden_size)`):
                Projected depth-sequence embeddings: the global hidden state followed by the embedded codes sampled so
                far, each passed through `projection`.

        Returns:
            `torch.Tensor` of shape `(batch, steps, hidden_size)`: normalized hidden states; the last step feeds the
            next codebook head.
        """
        _stamp_depth_decoder_attention_backend(self)  # SushiUI: unified attention conduit
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.pos_embedding(positions).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)
