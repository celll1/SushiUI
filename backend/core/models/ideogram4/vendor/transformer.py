# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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
# Vendored into SushiUI from the Apache-2.0 Ideogram 4 reference implementation
# (huggingface/diffusers ``transformer_ideogram4.py``). Import paths are made
# absolute so the module works against the installed diffusers (0.38) without
# requiring the unreleased diffusers Ideogram4 classes.
# ---------------------------------------------------------------------------

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import apply_lora_scale, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Per-token role indicators used to label entries of the packed text+image sequence.
SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3

# Image grid coordinates start at this offset so they never collide with text token indices.
IMAGE_POSITION_OFFSET = 65536


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Ideogram4MRoPE(nn.Module):
    """Multi-axis (t, h, w) interleaved rotary position embedding."""

    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        base: int,
        mrope_section: tuple,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim

    def forward(self, position_ids: torch.Tensor):
        # position_ids: (B, L, 3) of int (axes are t, h, w).
        if position_ids.ndim != 3 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must have shape (B, L, 3), got {tuple(position_ids.shape)}.")
        batch_size, seq_len, _ = position_ids.shape

        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, batch_size, -1, 1)
        freqs = inv_freq @ pos.unsqueeze(2)
        freqs = freqs.transpose(2, 3)  # (3, B, L, inv_freq_size)

        # Interleaved mrope: pull H freqs into idx 1 mod 3, W freqs into idx 2 mod 3.
        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs_t.device)
            freqs_t[..., idx] = freqs[axis][..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos(), emb.sin()


# ---------------------------------------------------------------------------
# Unified attention dispatch for Ideogram 4 (attention conduit integration).
#
# The vendored processors previously called diffusers ``dispatch_attention_fn``
# with ``backend=None`` (always the native/default kernel). Inference plumbing
# (``pipeline_backends/ideogram4.set_ideogram4_attention_backend``) now stamps
# ``_attention_backend`` with the diffusers AttentionBackendName string (via
# ``to_diffusers_backend``): ``"native"`` | ``"flash"`` | ``"sage"``.
#
# Ideogram 4's self-attention runs a single packed sequence with a per-sample
# block-diagonal segment mask ``(B, 1, L, L)``. FlashAttention's dense kernel
# cannot consume that mask, so a plain ``backend="flash"`` call would degrade to
# a native no-op. Per the design (D2/R2 REVISION v2) we instead convert the
# block-diagonal segments to ``cu_seqlens`` and run ``flash_attn_varlen_func`` so
# FlashAttention actually engages. head_dim=256 requires Ampere+/Hopper (compute
# capability >= 8.0); on older GPUs (or when the varlen kernel is unavailable) we
# fall back cleanly to the exact native dense-mask path. SageAttention is
# native-only for Ideogram 4 (head_dim=256 exceeds sage's 128 max), mirroring the
# conduit head_dim guard.
# ---------------------------------------------------------------------------

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _ideogram4_fa_varlen_capable(device) -> bool:
    """True when FlashAttention head_dim=256 is supported on this device.

    FA-2 supports head_dim up to 256 only on Ampere+ / Hopper (compute
    capability >= 8.0). CPU / older GPUs -> native fallback.
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return (major, minor) >= (8, 0)


def _ideogram4_segment_cu_seqlens(segment_ids: torch.Tensor):
    """Build ``(cu_seqlens, max_seqlen)`` from the packed segment ids.

    ``segment_ids`` is ``(B, L)`` (long). Within each sample, tokens that share a
    segment id form a CONTIGUOUS run (the packed ``[left-pad][text+image]``
    layout uses id ``-1`` for the padding prefix and id ``1`` for the real
    suffix). Runs from different samples are ALWAYS kept separate so no token
    attends across the batch -- exactly reproducing the per-sample
    ``(B, 1, L, L)`` block-diagonal mask.

    Returns an int32 ``cu_seqlens`` (cumulative segment boundaries over the
    flattened ``B*L`` token axis) on ``segment_ids``' device and the python-int
    max segment length, as required by ``flash_attn_varlen_func``.
    """
    batch, seq_len = segment_ids.shape
    seg_cpu = segment_ids.detach().to("cpu")
    cu = [0]
    running = 0
    max_seqlen = 0
    for b in range(batch):
        row = seg_cpu[b]
        if seq_len == 0:
            continue
        change = (row[1:] != row[:-1]).nonzero(as_tuple=True)[0]
        boundaries = (change + 1).tolist()
        starts = [0] + boundaries
        ends = boundaries + [seq_len]
        for s, e in zip(starts, ends):
            seg_len = e - s
            running += seg_len
            cu.append(running)
            if seg_len > max_seqlen:
                max_seqlen = seg_len
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=segment_ids.device)
    return cu_seqlens, max_seqlen


def _ideogram4_flash_varlen(query, key, value, segment_ids):
    """Run ``flash_attn_varlen_func`` over the block-diagonal segments.

    ``query``/``key``/``value`` are ``(B, L, num_heads, head_dim)``. Returns the
    output in the SAME ``(B, L, num_heads, head_dim)`` layout, or ``None`` if the
    varlen kernel is unavailable / fails (caller falls back to native).
    """
    try:
        from flash_attn import flash_attn_varlen_func
    except Exception:
        return None
    try:
        batch, seq_len, num_heads, head_dim = query.shape
        cu_seqlens, max_seqlen = _ideogram4_segment_cu_seqlens(segment_ids)

        orig_dtype = query.dtype
        needs_cast = orig_dtype not in _HALF_DTYPES
        q = query.to(torch.bfloat16) if needs_cast else query
        k = key.to(torch.bfloat16) if needs_cast else key
        v = value.to(torch.bfloat16) if needs_cast else value

        # (B, L, H, D) -> (B*L, H, D) flattened token axis for varlen.
        q = q.reshape(batch * seq_len, num_heads, head_dim).contiguous()
        k = k.reshape(batch * seq_len, num_heads, head_dim).contiguous()
        v = v.reshape(batch * seq_len, num_heads, head_dim).contiguous()

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=None,  # None -> 1/sqrt(head_dim), matches SDPA default
            causal=False,
        )
        out = out.reshape(batch, seq_len, num_heads, head_dim)
        if needs_cast:
            out = out.to(orig_dtype)
        return out
    except Exception as e:  # noqa: BLE001 - never raise into the model
        print(f"[Ideogram4] flash_attn_varlen failed ({e}); falling back to native")
        return None


def ideogram4_dispatch_attention(
    query,
    key,
    value,
    attention_mask,
    backend,
    parallel_config=None,
    segment_ids=None,
):
    """Backend-aware self-attention dispatch shared by all Ideogram 4 processors.

    ``backend`` is the diffusers AttentionBackendName string stamped by
    ``set_ideogram4_attention_backend`` (``None`` -> diffusers default native).

    * ``flash`` + block-diagonal mask: convert segments to ``cu_seqlens`` and run
      ``flash_attn_varlen_func`` (D2). Falls back to the exact native dense-mask
      path when the varlen kernel is unavailable or the GPU is pre-Ampere.
    * ``sage`` with head_dim > 128 (Ideogram 4 is 256): downgraded to native,
      mirroring the conduit head_dim guard.
    * everything else (native, or flash without a mask): diffusers
      ``dispatch_attention_fn`` unchanged -- byte-identical to the legacy path
      when ``backend`` is ``None``/``"native"``.
    """
    head_dim = query.shape[-1]

    if backend == "flash" and attention_mask is not None:
        if segment_ids is not None and _ideogram4_fa_varlen_capable(query.device):
            out = _ideogram4_flash_varlen(query, key, value, segment_ids)
            if out is not None:
                return out
        # varlen unavailable / pre-Ampere / kernel failure -> exact native fallback.
        return dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend="native",
            parallel_config=parallel_config,
        )

    if backend == "sage" and head_dim > 128:
        backend = "native"

    return dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        backend=backend,
        parallel_config=parallel_config,
    )


class Ideogram4AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "Ideogram4Attention",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb,
        segment_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # MRoPE applied in (B, L, num_heads, head_dim) layout; cos/sin broadcast over the head axis.
        cos, sin = image_rotary_emb
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        hidden_states = ideogram4_dispatch_attention(
            query,
            key,
            value,
            attention_mask,
            self._attention_backend,
            self._parallel_config,
            segment_ids,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.to_out[0](hidden_states)


class Ideogram4Attention(nn.Module, AttentionModuleMixin):
    """Self-attention with split Q/K/V, q/k RMSNorm, MRoPE and a block-diagonal segment mask."""

    _default_processor_cls = Ideogram4AttnProcessor
    _available_processors = [Ideogram4AttnProcessor]

    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_bias = False

        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.norm_k = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False), nn.Dropout(0.0)])

        self.set_processor(self._default_processor_cls())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        image_rotary_emb=None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k in kwargs if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb, **kwargs)


class Ideogram4MLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


@maybe_allow_in_graph
class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adaln_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

        self.attention_norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)
        self.attention_norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)

        self.adaln_modulation = nn.Linear(adaln_dim, 4 * hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb,
        adaln_input: torch.Tensor,
        segment_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(hidden_states) * scale_msa,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            segment_ids=segment_ids,
        )
        hidden_states = hidden_states + gate_msa * self.attention_norm2(attn_out)
        hidden_states = hidden_states + gate_mlp * self.ffn_norm2(
            self.feed_forward(self.ffn_norm1(hidden_states) * scale_mlp)
        )
        return hidden_states


def _sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    """Sinusoidal scalar embedding followed by a small MLP."""

    def __init__(self, dim: int, input_range: tuple) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        if self.range_max <= self.range_min:
            raise ValueError("input_range[1] must be greater than input_range[0]")
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(in_dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adaln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adaln_dim, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(conditioning))
        return self.linear(self.norm_final(hidden_states) * scale)


class Ideogram4Transformer2DModel(ModelMixin, ConfigMixin, AttentionMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    The flow-matching transformer backbone used by the Ideogram 4 pipeline.

    The transformer operates on a single packed sequence containing both
    text-conditioning tokens (produced by a multimodal text encoder) and the
    patchified image latents. Per-token indicators distinguish the two roles, and a
    block-diagonal attention mask derived from `segment_ids` restricts each sample
    to attend only to itself within a packed batch.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Ideogram4TransformerBlock"]
    _repeated_blocks = ["Ideogram4TransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedding", "adaln_proj", "embed_image_indicator"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        num_layers: int = 34,
        attention_head_dim: int = 256,
        num_attention_heads: int = 18,
        intermediate_size: int = 12288,
        adaln_dim: int = 512,
        llm_features_dim: int = 53248,
        rope_theta: int = 5_000_000,
        mrope_section: tuple = (24, 20, 20),
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        hidden_size = attention_head_dim * num_attention_heads
        head_dim = attention_head_dim

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.gradient_checkpointing = False

        self.input_proj = nn.Linear(in_channels, hidden_size, bias=True)
        self.llm_cond_norm = RMSNorm(llm_features_dim, eps=1e-6, elementwise_affine=True)
        self.llm_cond_proj = nn.Linear(llm_features_dim, hidden_size, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(hidden_size, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(hidden_size, adaln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, hidden_size)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=rope_theta,
            mrope_section=mrope_section,
        )

        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    norm_eps=norm_eps,
                    adaln_dim=adaln_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer = Ideogram4FinalLayer(
            hidden_size=hidden_size,
            out_channels=in_channels,
            adaln_dim=adaln_dim,
        )

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
        attention_kwargs: dict = None,
        return_dict: bool = True,
    ):
        r"""Predict the flow-matching velocity for the image-token positions of the packed sequence."""
        batch_size, seq_len, in_channels = hidden_states.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected last dim {self.in_channels}, got {in_channels}.")

        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(hidden_states.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(hidden_states.dtype).unsqueeze(-1)

        encoder_hidden_states = encoder_hidden_states * llm_token_mask
        hidden_states = hidden_states * output_image_mask
        hidden_states = self.input_proj(hidden_states) * output_image_mask

        # Keep shape (B, 1, ...) when t is per-sample so downstream adaln projections do not pay for L identical copies.
        t_cond = self.t_embedding(timestep)
        if timestep.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        encoder_hidden_states = self.llm_cond_norm(encoder_hidden_states)
        encoder_hidden_states = self.llm_cond_proj(encoder_hidden_states) * llm_token_mask

        hidden_states = hidden_states + encoder_hidden_states

        image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
        hidden_states = hidden_states + image_indicator_embedding

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)
        image_rotary_emb = (cos, sin)

        # Block-diagonal mask from segment ids: tokens only attend within their segment. Shared by every block.
        attention_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

        # Optional block-swap offloader (set by pipeline.py for VRAM optimization):
        # streams each block's weights between CPU and GPU around its forward.
        offloader = getattr(self, "_block_offloader", None)

        # First Block Cache (FBCache): OFF by default (_fbcache is None -> byte-identical,
        # including the block-swap wait/submit and gradient-checkpointing paths below). When a
        # FirstBlockCache is attached by the Ideogram4 denoise loop, run only layers[0], take its
        # residual on the single evolving image/hidden tensor as the indicator, and either reuse
        # the cached full residual (skip layers[1:]) or run them and refresh the cache. Only
        # hidden_states evolves through the layer list (attention_mask / image_rotary_emb /
        # adaln_input are read-only per-step context), so one image tensor is both indicator and
        # cache. Mutually exclusive with Spectrum and Block Swap (guarded in the pipeline), so this
        # branch never runs alongside _block_offloader.
        fbcache = getattr(self, "_fbcache", None)
        if fbcache is not None:
            fbcache_step = getattr(self, "_fbcache_step", 0)
            original = hidden_states
            first_out = self.layers[0](hidden_states, attention_mask, image_rotary_emb, adaln_input, segment_ids)
            indicator_residual = first_out - original
            if fbcache.use_cache(indicator_residual, fbcache_step):
                # Cache hit: reuse the full-transformer residual, skip layers[1:].
                hidden_states = original + fbcache.get()
            else:
                # Cache miss: run remaining layers from first_out, refresh the cached residual.
                hidden_states = first_out
                for block in self.layers[1:]:
                    hidden_states = block(hidden_states, attention_mask, image_rotary_emb, adaln_input, segment_ids)
                fbcache.store(hidden_states - original)
        else:
            for block_idx, block in enumerate(self.layers):
                if offloader is not None:
                    offloader.wait_for_block(block_idx)
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        block, hidden_states, attention_mask, image_rotary_emb, adaln_input, segment_ids
                    )
                else:
                    hidden_states = block(hidden_states, attention_mask, image_rotary_emb, adaln_input, segment_ids)
                if offloader is not None:
                    offloader.submit_move_blocks_forward(block_idx)

        output = self.final_layer(hidden_states, conditioning=adaln_input)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
