# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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
#
# ---------------------------------------------------------------------------
# Vendored into SushiUI from diffusers-main
# (src/diffusers/models/transformers/transformer_krea2.py), adapted to the local
# diffusers 0.38.0 API surface:
#   * All attention is routed through SushiUI's unified conduit
#     (core.attention.dispatch_attention) instead of diffusers'
#     ``dispatch_attention_fn``; the per-module ``_attn_backend`` / ``_attn_mode``
#     attributes select the kernel (native SDPA / FlashAttention / SageAttention)
#     exactly like minit2i vendor/mmjit.py. GQA (48 query / 12 kv heads) is handled
#     by the conduit: on the native path it pre-expands K/V (repeat_interleave)
#     instead of using SDPA's own ``enable_gqa`` broadcast, which is far slower;
#     sage downgrades on unequal heads; flash broadcasts GQA natively.
#   * ``maybe_adjust_dtype_for_device`` (absent in 0.38) is inlined.
#   * The PEFT / AttentionMixin machinery (LoRA) is dropped for the Phase A
#     inference build; module and parameter NAMES are preserved verbatim so the
#     published diffusers-format checkpoints load unchanged.
# ---------------------------------------------------------------------------

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from core.attention import AttentionMode, dispatch_attention


def _rope_freqs_dtype(device: torch.device) -> torch.dtype:
    """float64 RoPE frequencies everywhere except MPS (no float64 support there).

    Replaces diffusers-main ``maybe_adjust_dtype_for_device`` which is absent in
    diffusers 0.38.0.
    """
    if device is not None and device.type == "mps":
        return torch.float32
    return torch.float64


class Krea2RMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale: the effective multiplier is ``1 + weight``,
    matching the Krea 2 checkpoint format. Activations are upcast to float32 for the
    normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = F.rms_norm(hidden_states.float(), (self.dim,), weight=self.weight + 1.0, eps=self.eps)
        return hidden_states.to(dtype)


class Krea2Attention(nn.Module):
    """Self-attention with grouped-query projections, q/k RMSNorm, rotary embeddings
    and a sigmoid output gate. Attention runs through SushiUI's unified conduit.

    ``_style_ctx`` / ``block_idx`` support training-free reference-style KV
    injection (see ``core.inference.reference_style``). Both default to
    ``None`` at the class level: attention modules that are never stamped
    (the text-fusion attention blocks, or any main block when no style
    transfer is requested) take the byte-identical original code path.
    """

    _style_ctx = None
    block_idx = None

    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: Optional[int] = None, eps: float = 1e-5
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads

        self.to_q = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.to_k = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_v = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False), nn.Dropout(0.0)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple] = None,
    ) -> torch.Tensor:
        # BSHD projections: [B, S, H, D].
        query = self.to_q(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        gate = self.to_gate(hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # --- Reference-style KV injection (training-free) ---
        # Must run strictly AFTER qk-RMSNorm and AFTER RoPE: scaling/AdaIN-ing K
        # before RMSNorm would be silently erased (RMSNorm is scale-invariant),
        # and RoPE must already be baked into K before it is stashed/injected
        # (the rotary phase carries the token position; injecting pre-RoPE K
        # would desync the reference's positions from the target's).
        ctx = self._style_ctx
        if ctx is not None and self.block_idx is not None and ctx.active_for_block(self.block_idx):
            img_start, img_end = ctx.img_start, ctx.img_end
            if ctx.mode == "capture":
                # Stash post-norm/post-RoPE image-token Q/K/V. The reference
                # QUERY is captured (not just K/V) because target-Q's AdaIN
                # alignment is stylized by the REFERENCE Query, not the
                # reference Key (verbatim ComfyUI-Krea2-StyleTransfer
                # ``_cross_batch_adain_qk``).
                ctx.store[self.block_idx] = (
                    query[:, img_start:img_end].detach().clone(),
                    key[:, img_start:img_end].detach().clone(),
                    value[:, img_start:img_end].detach().clone(),
                )
            elif ctx.mode == "inject" and ctx.refs is not None:
                # Multi-reference ("stack" / "common_concept"): centralizes the
                # per-ref active/freq/make_ref_value logic in
                # StyleContext.collect_block_refs so this hook stays thin. The
                # single-ref branch below (``ctx.mode == "inject"`` reached only
                # when ``ctx.refs is None``) is completely untouched -- this
                # branch is only ever reached for 2+ refs (see
                # ``krea2_pipeline_ops._run_loop``'s multi-ref capture and the
                # ``style_refs``/``StyleContext(refs=...)`` wiring in
                # ``pipeline_backends.krea2``).
                from core.inference.reference_style import inject_kv_multi

                target_v_img = value[:, img_start:img_end]
                block_refs = ctx.collect_block_refs(self.block_idx, target_v_img, key.device, key.dtype)
                if block_refs:
                    ref_len_before = key.shape[1]
                    key, value, query = inject_kv_multi(
                        key, value, query, img_start, img_end, block_refs, ctx.combine_mode
                    )
                    if attention_mask is not None and key.shape[1] != ref_len_before:
                        pad_len = key.shape[1] - ref_len_before
                        pad = attention_mask.new_ones(attention_mask.shape[0], 1, 1, pad_len)
                        attention_mask = torch.cat([attention_mask, pad], dim=-1)
            elif ctx.mode == "inject":
                ref_qkv = ctx.store.get(self.block_idx)
                if ref_qkv is not None:
                    from core.inference.reference_style import inject_kv, make_ref_value

                    ref_q, ref_k, ref_v = ref_qkv
                    cfg = ctx.config
                    if cfg.ref_k_strength != 0.0 or cfg.adain_strength > 0.0:
                        freq_vec = cfg.get_freq_scale_vector(self.head_dim, ctx.progress, key.device, key.dtype)
                        target_v_img = value[:, img_start:img_end]
                        ref_v_final = make_ref_value(
                            target_v_img, ref_v, cfg.value_mode, cfg.value_adain_strength, cfg.ref_value_mix
                        )
                        ref_len_before = key.shape[1]
                        key, value, query = inject_kv(
                            key, value, ref_k, ref_v_final, img_start, img_end,
                            cfg.ref_k_strength, freq_vec, cfg.adain_strength, q=query, ref_q=ref_q,
                        )
                        if attention_mask is not None and key.shape[1] != ref_len_before:
                            ref_len = ref_k.shape[1]
                            pad = attention_mask.new_ones(attention_mask.shape[0], 1, 1, ref_len)
                            attention_mask = torch.cat([attention_mask, pad], dim=-1)

        hidden_states = dispatch_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=getattr(self, "_attn_backend", "native"),
            mode=getattr(self, "_attn_mode", AttentionMode.INFERENCE),
            layout="BSHD",
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return self.to_out[0](hidden_states)


class Krea2SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class Krea2TextFusionBlock(nn.Module):
    """Pre-norm transformer block (no rotary embeddings, no time modulation) used by
    the text fusion stage."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, intermediate_size: int, eps: float) -> None:
        super().__init__()
        self.norm1 = Krea2RMSNorm(dim, eps=eps)
        self.norm2 = Krea2RMSNorm(dim, eps=eps)
        self.attn = Krea2Attention(dim, num_heads, num_kv_heads, eps=eps)
        self.ff = Krea2SwiGLU(dim, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class Krea2TextFusion(nn.Module):
    """Fuses the stack of tapped text-encoder hidden states into a single sequence of
    text features."""

    def __init__(
        self,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_layerwise_blocks)
            ]
        )
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_refiner_blocks)
            ]
        )

    def forward(self, encoder_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape

        hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states.contiguous())

        hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim).permute(0, 1, 3, 2)
        hidden_states = self.projector(hidden_states).squeeze(-1)

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        return hidden_states


class Krea2TransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, norm_eps: float
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(hidden_size, num_heads, num_kv_heads, eps=norm_eps)
        self.ff = Krea2SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


class Krea2TimestepEmbedding(nn.Module):
    """Sinusoidal flow-time embedding (cos-first, input scaled by 1000) followed by a
    two-layer MLP."""

    def __init__(self, embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half)
        args = (timestep.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))


class Krea2TextProjection(nn.Module):
    """Projects the fused text features into the transformer width."""

    def __init__(self, text_dim: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(F.gelu(hidden_states, approximate="tanh"))


class Krea2FinalLayer(nn.Module):
    """Final adaptive RMSNorm and output projection."""

    def __init__(self, hidden_size: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        modulation = temb + self.scale_shift_table
        scale, shift = modulation.chunk(2, dim=1)
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


class Krea2RotaryPosEmbed(nn.Module):
    """3-axis (t, h, w) rotary position embedding (FluxPosEmbed-style)."""

    def __init__(self, theta: int, axes_dim: list):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = _rope_freqs_dtype(ids.device)
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class Krea2Transformer2DModel(ModelMixin, ConfigMixin):
    r"""The single-stream MMDiT flow-matching backbone used by the Krea 2 pipeline."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Krea2TransformerBlock", "Krea2TextFusionBlock", "Krea2FinalLayer"]
    _keep_in_fp32_modules = ["norm", "norm1", "norm2", "norm_q", "norm_k"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 12,
        intermediate_size: int = 16384,
        timestep_embed_dim: int = 256,
        text_hidden_dim: int = 2560,
        num_text_layers: int = 12,
        text_num_attention_heads: int = 20,
        text_num_key_value_heads: int = 20,
        text_intermediate_size: int = 6912,
        num_layerwise_text_blocks: int = 2,
        num_refiner_text_blocks: int = 2,
        axes_dims_rope: tuple = (32, 48, 48),
        rope_theta: float = 1000.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        hidden_size = attention_head_dim * num_attention_heads
        if sum(axes_dims_rope) != attention_head_dim:
            raise ValueError(
                f"sum(axes_dims_rope)={sum(axes_dims_rope)} must equal attention_head_dim={attention_head_dim}"
            )

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.gradient_checkpointing = False

        self.img_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_embed = Krea2TimestepEmbedding(timestep_embed_dim, hidden_size)
        self.time_mod_proj = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.text_fusion = Krea2TextFusion(
            num_text_layers=num_text_layers,
            dim=text_hidden_dim,
            num_heads=text_num_attention_heads,
            num_kv_heads=text_num_key_value_heads,
            intermediate_size=text_intermediate_size,
            num_layerwise_blocks=num_layerwise_text_blocks,
            num_refiner_blocks=num_refiner_text_blocks,
            eps=norm_eps,
        )
        self.txt_in = Krea2TextProjection(text_hidden_dim, hidden_size, eps=norm_eps)
        self.rotary_emb = Krea2RotaryPosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        self.transformer_blocks = nn.ModuleList(
            [
                Krea2TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer = Krea2FinalLayer(hidden_size, out_channels=in_channels, eps=norm_eps)

    def enable_gradient_checkpointing(self, *args, **kwargs) -> None:
        """Enable activation checkpointing over the 28 main transformer blocks
        (mirrors the minit2i vendor pattern; forward reads ``self.gradient_checkpointing``
        and ``self._gradient_checkpointing_func``)."""
        self.gradient_checkpointing = True

        def _ckpt(module, *inputs):
            return torch.utils.checkpoint.checkpoint(module.__call__, *inputs, use_reentrant=False)

        self._gradient_checkpointing_func = _ckpt

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _stamp_style_context(self, text_seq_len: int, image_seq_len: int) -> None:
        """Propagate ``self._style_ctx`` (set externally by the pipeline_ops
        denoise loop, ``None`` by default) to every main ``transformer_blocks``
        attention module and its static ``block_idx``, and record this
        forward's image-token range on the context. Does NOT touch the
        text-fusion attention blocks (style transfer only targets the main
        DiT self-attention). When ``self._style_ctx`` is absent/None this is a
        cheap no-op assignment loop -- attention forward remains byte-identical.
        """
        ctx = getattr(self, "_style_ctx", None)
        if ctx is not None:
            ctx.img_start = text_seq_len
            ctx.img_end = text_seq_len + image_seq_len
            ctx.config.resolve_default_block_range(len(self.transformer_blocks))
        for idx, block in enumerate(self.transformer_blocks):
            block.attn.block_idx = idx
            block.attn._style_ctx = ctx

    def _stamp_attention_backend(self) -> None:
        """Propagate this model's ``_attn_backend`` to every attention module and derive
        the mode from the autograd state (inference under no_grad -> sage allowed;
        training with grad -> conduit refuses sage). Mirrors minit2i vendor/mmjit.py."""
        backend = getattr(self, "_attn_backend", "native")
        mode = AttentionMode.TRAINING if torch.is_grad_enabled() else AttentionMode.INFERENCE
        for m in self.modules():
            if isinstance(m, Krea2Attention):
                m._attn_backend = backend
                m._attn_mode = mode

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict] = None,
        return_dict: bool = True,
    ):
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must have shape (sequence_length, 3), got {tuple(position_ids.shape)}.")

        self._stamp_attention_backend()

        batch_size, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]
        self._stamp_style_context(text_seq_len, image_seq_len)

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            text_attention_mask = encoder_attention_mask[:, None, None, :]
            image_mask = encoder_attention_mask.new_ones((batch_size, image_seq_len))
            attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[:, None, None, :]

        encoder_hidden_states = self.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        image_rotary_emb = self.rotary_emb(position_ids)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, temb_mod, image_rotary_emb, attention_mask
                )
            else:
                hidden_states = block(hidden_states, temb_mod, image_rotary_emb, attention_mask)

        hidden_states = hidden_states[:, text_seq_len:]
        output = self.final_layer(hidden_states, temb)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
