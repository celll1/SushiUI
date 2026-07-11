# Anima Model Architecture (inference subset)
#
# Vendored from kohya-ss/sd-scripts library/anima_models.py (Apache-2.0).
# Original code: NVIDIA CORPORATION & AFFILIATES, licensed under Apache-2.0.
# Adapted for SushiUI inference:
#   - library.attention -> .anima_attention (PyTorch SDPA only)
#   - removed Unsloth offloaded gradient checkpointing
#   - removed block swap / ModelOffloader (handled by SushiUI VRAM mgmt)
#   - kept the public interface (Anima, LLMAdapter) intact

import math
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F

from . import anima_attention as attention

import logging
logger = logging.getLogger(__name__)


# ----- RoPE utilities -----

def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_base(
    t: torch.Tensor,
    freqs: torch.Tensor,
    start_positions: torch.Tensor = None,
    tensor_format: str = "sbhd",
    interleaved: bool = False,
) -> torch.Tensor:
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    if start_positions is not None:
        max_offset = torch.max(start_positions)
        assert max_offset + cur_seq_len <= max_seq_len
        freqs = torch.concatenate([freqs[i : i + cur_seq_len] for i in start_positions], dim=1)

    assert cur_seq_len <= max_seq_len
    freqs = freqs[:cur_seq_len]

    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * cos_) + (_rotate_half(t, interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
    interleaved: bool = False,
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
    cp_size: int = 1,
) -> torch.Tensor:
    assert not fused
    return _apply_rotary_pos_emb_base(t, freqs, start_positions, tensor_format, interleaved=interleaved)


# ----- Basic building blocks -----

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)
        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


# ----- DiT Attention -----

class Attention(nn.Module):
    """Multi-head attention with QK-norm and optional RoPE (self-attention only).

    ``_style_ctx`` / ``block_idx`` support training-free reference-style KV
    injection (see ``core.inference.reference_style``). Both default to
    ``None`` at the class level: attention modules that are never stamped
    (cross-attention, or any self-attention block when no style transfer is
    requested) take the byte-identical original code path. Only stamped on
    ``self_attn`` modules (see ``Anima._stamp_style_context``) -- Anima's
    cross-attention reads text tokens (``crossattn_emb``), not the image
    K/V, so style transfer never targets it.
    """

    _style_ctx = None
    block_idx = None

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)
        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(self, x, context=None, rope_emb=None):
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format)
            k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format)
        return q, k, v

    def forward(self, x, attn_params, context=None, rope_emb=None):
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        if q.dtype != v.dtype and torch.is_autocast_enabled():
            q = q.to(v.dtype)
            k = k.to(v.dtype)

        # --- Reference-style KV injection (training-free, self-attention only) ---
        # Must run strictly AFTER qk-RMSNorm and AFTER RoPE (both already applied
        # inside compute_qkv for self-attention): scaling/AdaIN-ing K before
        # RMSNorm would be silently erased (RMSNorm is scale-invariant), and RoPE
        # must already be baked into K before it is stashed/injected (the rotary
        # phase carries the token position; injecting pre-RoPE K would desync the
        # reference's positions from the target's).
        #
        # Anima's self-attention sequence is IMAGE-ONLY (no text concatenation --
        # ``is_selfattn`` implies ``context is None``, and the caller always
        # passes the flattened image stream ``b (t h w) d``), so the injected
        # image-token range is the WHOLE sequence: no text/image split is needed,
        # unlike Krea2's combined text+image self-attention stream.
        ctx = self._style_ctx
        if self.is_selfattn and ctx is not None and self.block_idx is not None and ctx.active_for_block(self.block_idx):
            img_start, img_end = 0, q.shape[1]
            if ctx.mode == "capture":
                # Stash post-norm/post-RoPE image-token Q/K/V. The reference
                # QUERY is captured (not just K/V) because target-Q's AdaIN
                # alignment is stylized by the REFERENCE Query, not the
                # reference Key (verbatim ComfyUI-Krea2-StyleTransfer
                # ``_cross_batch_adain_qk``).
                ctx.store[self.block_idx] = (
                    q[:, img_start:img_end].detach().clone(),
                    k[:, img_start:img_end].detach().clone(),
                    v[:, img_start:img_end].detach().clone(),
                )
            elif ctx.mode == "inject":
                ref_qkv = ctx.store.get(self.block_idx)
                if ref_qkv is not None:
                    from core.inference.reference_style import inject_kv, make_ref_value

                    ref_q, ref_k, ref_v = ref_qkv
                    cfg = ctx.config
                    if cfg.ref_k_strength != 0.0 or cfg.adain_strength > 0.0:
                        # Anima's 3D video RoPE uses the "rotate-half" convention
                        # (interleaved=False in apply_rotary_pos_emb), NOT the
                        # per-axis interleave-real layout that
                        # ``frequency_scale_vector`` assumes (Krea2/FLUX-style).
                        # Deriving a correct per-axis frequency-suppression curve
                        # for this layout is a separate RoPE-layout adaptation;
                        # until that is done, use an all-ones vector (no
                        # frequency-content suppression on the reference Key --
                        # a quality knob, not a correctness requirement: the
                        # ref_k_strength scale + AdaIN alignment below still
                        # apply in full).
                        freq_vec = torch.ones(self.head_dim, device=k.device, dtype=k.dtype)
                        target_v_img = v[:, img_start:img_end]
                        ref_v_final = make_ref_value(
                            target_v_img, ref_v, cfg.value_mode, cfg.value_adain_strength, cfg.ref_value_mix
                        )
                        k, v, q = inject_kv(
                            k, v, ref_k, ref_v_final, img_start, img_end,
                            cfg.ref_k_strength, freq_vec, cfg.adain_strength, q=q, ref_q=ref_q,
                        )

        result = attention.attention([q, k, v], attn_params=attn_params)
        return self.output_dropout(self.output_proj(result))


# ----- Positional Embeddings -----

class VideoPositionEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def seq_dim(self) -> int:
        return 1

    def forward(self, x_B_T_H_W_C, fps=None):
        return self.generate_embeddings(x_B_T_H_W_C.shape, fps=fps)

    def generate_embeddings(self, B_T_H_W_C, fps):
        raise NotImplementedError


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self, *, head_dim, len_h, len_w, len_t,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t
        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t
        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        self.dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t

    def generate_embeddings(self, B_T_H_W_C, fps=None, h_ntk_factor=None, w_ntk_factor=None, t_ntk_factor=None):
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        assert H <= self.max_h and W <= self.max_w
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert uniform_fps or B == 1 or T == 1
            if fps is None:
                assert T == 1
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ] * 2,
            dim=-1,
        )
        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self) -> int:
        return 0


class LearnablePosEmbAxis(VideoPositionEmb):
    def __init__(self, *, interpolation, model_channels, len_h, len_w, len_t, **kwargs):
        del kwargs
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"]
        self.model_channels = model_channels
        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.model_channels)
        torch.nn.init.trunc_normal_(self.pos_emb_h, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_w, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_t, std=std, a=-3 * std, b=3 * std)

    def generate_embeddings(self, B_T_H_W_C, fps=None):
        B, T, H, W, _ = B_T_H_W_C
        emb_h_H = self.pos_emb_h[:H]
        emb_w_W = self.pos_emb_w[:W]
        emb_t_T = self.pos_emb_t[:T]
        emb = (
            repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
            + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
            + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
        )
        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(1e-6, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return emb / norm.to(emb.dtype)


# ----- Timestep Embedding -----

class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return rearrange(emb.to(dtype=in_dtype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        if self.use_adaln_lora:
            return sample, emb
        return emb, None


# ----- Patch Embedding -----

class PatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size, temporal_patch_size, in_channels=3, out_channels=768):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size, m=spatial_patch_size, n=spatial_patch_size,
            ),
            nn.Linear(in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        return self.proj(x)


# ----- Final Layer -----

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, spatial_patch_size, temporal_patch_size, out_channels,
                 use_adaln_lora=False, adaln_lora_dim=256):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False,
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False),
            )
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)
        self.layer_norm.reset_parameters()

    def forward(self, x_B_T_H_W_D, emb_B_T_D, adaln_lora_B_T_3D=None, use_fp32=False):
        with torch.autocast(device_type=x_B_T_H_W_D.device.type, dtype=torch.float32, enabled=use_fp32):
            if self.use_adaln_lora:
                assert adaln_lora_B_T_3D is not None
                shift_B_T_D, scale_B_T_D = (
                    self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
                ).chunk(2, dim=-1)
            else:
                shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)
        shift_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
        scale_B_T_1_1_D = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")
        x_B_T_H_W_D = self.layer_norm(x_B_T_H_W_D) * (1 + scale_B_T_1_1_D) + shift_B_T_1_1_D
        return self.linear(x_B_T_H_W_D)


# ----- DiT Block -----

class Block(nn.Module):
    """DiT block with self-attn + cross-attn + MLP, each modulated by AdaLN."""

    def __init__(self, x_dim, context_dim, num_heads, mlp_ratio=4.0,
                 use_adaln_lora=False, adaln_lora_dim=256):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(x_dim, None, num_heads, x_dim // num_heads, qkv_format="bshd")
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd")
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        # Gradient checkpointing mode (set by Anima.enable_gradient_checkpointing).
        # One of:
        #   "none"             — no checkpointing (default)
        #   "standard"         — torch.utils.checkpoint (activations on GPU)
        #   "cpu_offload"      — torch.utils.checkpoint + blocking CPU offload
        #   "async_cpu_offload"— custom autograd.Function with non_blocking
        #                        CPU offload (see core.training.async_checkpoint)
        self.gradient_checkpoint_mode: str = "none"

    # Backwards-compat shim: code that flips a boolean continues to work.
    @property
    def gradient_checkpointing(self) -> bool:
        return self.gradient_checkpoint_mode != "none"

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        if isinstance(value, str):
            self.gradient_checkpoint_mode = value
        else:
            self.gradient_checkpoint_mode = "standard" if value else "none"

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params,
                use_fp32=False, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None,
                extra_per_block_pos_emb=None):
        # Optional gradient checkpointing for training-time VRAM reduction.
        mode = self.gradient_checkpoint_mode if self.training else "none"

        if mode == "none":
            return self._forward_impl(
                x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params,
                use_fp32=use_fp32,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_per_block_pos_emb,
            )

        def _custom(x_in, emb_in, ctx_in, rope_in, adaln_in, extra_in):
            return self._forward_impl(
                x_in, emb_in, ctx_in, attn_params,
                use_fp32=use_fp32,
                rope_emb_L_1_1_D=rope_in,
                adaln_lora_B_T_3D=adaln_in,
                extra_per_block_pos_emb=extra_in,
            )

        if mode == "standard":
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                _custom, x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
                rope_emb_L_1_1_D, adaln_lora_B_T_3D, extra_per_block_pos_emb,
                use_reentrant=False,
            )

        if mode == "cpu_offload":
            # Blocking CPU offload of the saved activations. torch.utils.checkpoint
            # only stores the function INPUTS in its ctx (with use_reentrant=False),
            # so we pre-stage the inputs as CPU tensors before handing them to
            # checkpoint(). The wrapper then moves them back to the block's
            # compute device for the actual forward / backward recompute. The
            # block's output stays on GPU so the next block / the final layer
            # see a normal CUDA tensor.
            from torch.utils.checkpoint import checkpoint
            compute_device = next(self.parameters()).device

            def _wrap_offload(*inputs):
                dev_inputs = tuple(
                    t.to(compute_device) if isinstance(t, torch.Tensor) else t
                    for t in inputs
                )
                return _custom(*dev_inputs)

            def _to_cpu_if_tensor(t):
                return t.cpu() if isinstance(t, torch.Tensor) else t

            cpu_args = (
                _to_cpu_if_tensor(x_B_T_H_W_D),
                _to_cpu_if_tensor(emb_B_T_D),
                _to_cpu_if_tensor(crossattn_emb),
                _to_cpu_if_tensor(rope_emb_L_1_1_D),
                _to_cpu_if_tensor(adaln_lora_B_T_3D),
                _to_cpu_if_tensor(extra_per_block_pos_emb),
            )
            return checkpoint(_wrap_offload, *cpu_args, use_reentrant=False)

        if mode == "async_cpu_offload":
            # Non-blocking CPU offload variant: same structure as cpu_offload
            # but the input-staging transfers use non_blocking=True so the
            # copies can overlap with compute. We reuse torch.utils.checkpoint
            # rather than a custom autograd.Function so the requires_grad-
            # propagation edge cases (e.g. tests with non-grad inputs and
            # gradient coming from parameters inside the wrap) keep working.
            from torch.utils.checkpoint import checkpoint
            compute_device = next(self.parameters()).device

            def _wrap_async(*inputs):
                dev_inputs = tuple(
                    t.to(compute_device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                    for t in inputs
                )
                return _custom(*dev_inputs)

            def _to_cpu_async(t):
                # pin_memory enables true non-blocking H2D copy on recompute.
                if not isinstance(t, torch.Tensor):
                    return t
                if t.is_cuda:
                    cpu_copy = torch.empty(t.shape, dtype=t.dtype, device="cpu",
                                            pin_memory=torch.cuda.is_available())
                    cpu_copy.copy_(t, non_blocking=True)
                    return cpu_copy
                return t

            cpu_args = (
                _to_cpu_async(x_B_T_H_W_D),
                _to_cpu_async(emb_B_T_D),
                _to_cpu_async(crossattn_emb),
                _to_cpu_async(rope_emb_L_1_1_D),
                _to_cpu_async(adaln_lora_B_T_3D),
                _to_cpu_async(extra_per_block_pos_emb),
            )
            return checkpoint(_wrap_async, *cpu_args, use_reentrant=False)

        # Unknown mode — fall back to a no-checkpoint forward and warn once.
        if not getattr(self, "_warned_unknown_mode", False):
            print(f"[Anima Block] WARNING: unknown gradient_checkpoint_mode "
                  f"'{mode}', running without checkpointing")
            self._warned_unknown_mode = True
        return self._forward_impl(
            x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params,
            use_fp32=use_fp32,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            extra_per_block_pos_emb=extra_per_block_pos_emb,
        )

    def _forward_impl(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params,
                       use_fp32=False, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None,
                       extra_per_block_pos_emb=None):
        if use_fp32:
            x_B_T_H_W_D = x_B_T_H_W_D.float()
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        with torch.autocast(device_type=x_B_T_H_W_D.device.type, dtype=torch.float32, enabled=use_fp32):
            if self.use_adaln_lora:
                shift_sa, scale_sa, gate_sa = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_ca, scale_ca, gate_ca = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
            else:
                shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
                shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape for broadcasting: (B, T, D) -> (B, T, 1, 1, D)
        def _r(t): return rearrange(t, "b t d -> b t 1 1 d")
        shift_sa_, scale_sa_, gate_sa_ = _r(shift_sa), _r(scale_sa), _r(gate_sa)
        shift_ca_, scale_ca_, gate_ca_ = _r(shift_ca), _r(scale_ca), _r(gate_ca)
        shift_mlp_, scale_mlp_, gate_mlp_ = _r(shift_mlp), _r(scale_mlp), _r(gate_mlp)

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _adaln(_x, _norm, _scale, _shift):
            return _norm(_x) * (1 + _scale) + _shift

        # 1. Self-attention
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_self_attn, scale_sa_, shift_sa_)
        result = rearrange(
            self.self_attn(rearrange(normed, "b t h w d -> b (t h w) d"), attn_params, None, rope_emb=rope_emb_L_1_1_D),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa_ * result

        # 2. Cross-attention
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_ca_, shift_ca_)
        result = rearrange(
            self.cross_attn(rearrange(normed, "b t h w d -> b (t h w) d"), attn_params, crossattn_emb, rope_emb=rope_emb_L_1_1_D),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = result * gate_ca_ + x_B_T_H_W_D

        # 3. MLP
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp_, shift_mlp_)
        result = self.mlp(normed)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_ * result
        return x_B_T_H_W_D


# ----- LLM Adapter -----

class LLMAdapterRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def _adapter_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _adapter_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_adapter_rotate_half(x) * sin)


class AdapterRotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLMAdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = LLMAdapterRMSNorm(self.head_dim)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = LLMAdapterRMSNorm(self.head_dim)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        v = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            q = _adapter_apply_rotary_pos_emb(q, cos, sin)
            cos, sin = position_embeddings_context
            k = _adapter_apply_rotary_pos_emb(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(out)


class LLMAdapterTransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, self_attn=False, layer_norm=False):
        super().__init__()
        self.has_self_attn = self_attn
        if self.has_self_attn:
            self.norm_self_attn = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
            self.self_attn = LLMAdapterAttention(model_dim, model_dim, num_heads, model_dim // num_heads)
        self.norm_cross_attn = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        self.cross_attn = LLMAdapterAttention(model_dim, source_dim, num_heads, model_dim // num_heads)
        self.norm_mlp = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None,
                position_embeddings=None, position_embeddings_context=None):
        if self.has_self_attn:
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(
                normed, mask=target_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )
            x = x + attn_out
        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(
            normed, mask=source_attention_mask, context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        x = x + attn_out
        return x + self.mlp(self.norm_mlp(x))

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)


class LLMAdapter(nn.Module):
    """Bridge module: Qwen3 embeddings (source) -> T5-compatible space (target)."""

    def __init__(self, source_dim, target_dim, model_dim, num_layers=6, num_heads=16,
                 embed=None, self_attn=False, layer_norm=False):
        super().__init__()
        if embed is not None:
            self.embed = nn.Embedding.from_pretrained(embed.weight)
        else:
            self.embed = nn.Embedding(32128, target_dim)
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = AdapterRotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([
            LLMAdapterTransformerBlock(source_dim, model_dim, num_heads=num_heads,
                                       self_attn=self_attn, layer_norm=layer_norm)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = LLMAdapterRMSNorm(target_dim)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)
        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x, context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


# ----- Main DiT Model -----

class Anima(nn.Module):
    """Cosmos-Predict2 DiT model for image generation.

    28 transformer blocks with AdaLN-LoRA modulation, 3D RoPE, optional LLM Adapter.
    """

    LATENT_CHANNELS = 16

    def __init__(
        self,
        max_img_h: int = 512,
        max_img_w: int = 512,
        max_frames: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        concat_padding_mask: bool = True,
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "rope3d",
        pos_emb_learnable: bool = True,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 4.0,
        rope_w_extrapolation_ratio: float = 4.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = False,
        use_llm_adapter: bool = True,
        attn_mode: str = "torch",
        split_attn: bool = False,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.use_llm_adapter = use_llm_adapter
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        self.build_patch_embed()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        if self.use_llm_adapter:
            self.llm_adapter = LLMAdapter(
                source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, self_attn=True,
            )

        self.blocks = nn.ModuleList([
            Block(
                x_dim=model_channels, context_dim=crossattn_emb_channels, num_heads=num_heads,
                mlp_ratio=mlp_ratio, use_adaln_lora=use_adaln_lora, adaln_lora_dim=adaln_lora_dim,
            )
            for _ in range(num_blocks)
        ])

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels, spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal, out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora, adaln_lora_dim=self.adaln_lora_dim,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)
        self.init_weights()

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()
        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()
        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False,
                                       async_offload: bool = False) -> None:
        """Enable activation checkpointing on every Block.

        Modes (precedence, highest first):
          async_offload=True  -> non-blocking CPU offload of activations
                                 (custom autograd.Function; overlaps copy
                                 with compute)
          cpu_offload=True    -> blocking CPU offload of activations via
                                 torch.utils.checkpoint
          both False          -> standard torch.utils.checkpoint (activations
                                 stay on GPU)
        """
        if async_offload:
            mode = "async_cpu_offload"
        elif cpu_offload:
            mode = "cpu_offload"
        else:
            mode = "standard"
        for block in self.blocks:
            block.gradient_checkpoint_mode = mode

    def disable_gradient_checkpointing(self) -> None:
        for block in self.blocks:
            block.gradient_checkpoint_mode = "none"

    def _stamp_style_context(self) -> None:
        """Propagate ``self._style_ctx`` (set externally by the inference
        pipeline's denoise loop, ``None`` by default) to every block's
        SELF-attention module and its static ``block_idx``. Does NOT touch
        ``block.cross_attn`` (style transfer only targets the image
        self-attention stream). Training-only features (TREAD, BlockSkip,
        stochastic depth) are gated on ``self.training`` and style transfer is
        inference-only, so the two never overlap on the same forward -- but
        gate here defensively too (``self.training`` -> force ``None``) so a
        stray ``_style_ctx`` left set on the module can never leak into a
        training forward. When ``self._style_ctx`` is absent/None (the default)
        this is a cheap no-op assignment loop -- attention forward remains
        byte-identical.
        """
        ctx = None if self.training else getattr(self, "_style_ctx", None)
        if ctx is not None:
            ctx.config.resolve_default_block_range(len(self.blocks))
        for idx, block in enumerate(self.blocks):
            block.self_attn.block_idx = idx
            block.self_attn._style_ctx = ctx

    def build_patch_embed(self) -> None:
        in_channels = self.in_channels + 1 if self.concat_padding_mask else self.in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial, temporal_patch_size=self.patch_temporal,
            in_channels=in_channels, out_channels=self.model_channels,
        )

    def build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps, min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(**kwargs)

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)

    def prepare_embedded_sequence(self, x_B_C_T_H_W, fps=None, padding_mask=None):
        from torchvision import transforms
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1,
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)
        return x_B_T_H_W_D, None, extra_pos_emb

    def unpatchify(self, x_B_T_H_W_M):
        return rearrange(
            x_B_T_H_W_M, "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial, p2=self.patch_spatial, t=self.patch_temporal,
        )

    def _blockskip_forward(self, cfg, x0, t_emb, crossattn_emb, attn_params,
                           use_fp32, rope, adaln, extra_pos):
        """DiT-BlockSkip two-pass forward over the image stream (arXiv 2603.20755).

        Pass 1 (no_grad, full network): capture residual features
          Delta_front = f_n - f_0        (over skipped front blocks [0, n))
          Delta_back  = f_L - f_{L-m}     (over skipped back blocks [L-m, L))
        where f_i is the input to block i and f_L the input to the final layer.

        Pass 2 (gradient, middle blocks [n, L-m) only):
          x = f_0 + Delta_front           (== f_n; front is frozen so this is exact)
          x = middle_blocks(x)            (LoRA-trained, grad flows here)
          x = x + Delta_back              (== f_L when the middle is unchanged)

        Returns the reconstructed stream (input to the final layer). Deltas are
        detached constants (no grad), so backprop is confined to the middle blocks.
        """
        n = int(cfg["front"])
        m = int(cfg["back"])
        L = len(self.blocks)
        lo = n
        hi = L - m

        def _run(x, blk):
            return blk(
                x, t_emb, crossattn_emb, attn_params, use_fp32,
                rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln,
                extra_per_block_pos_emb=extra_pos,
            )

        # Pass 1: frozen full forward, capturing span-boundary features.
        with torch.no_grad():
            xt = x0
            f_lo = None
            f_hi = None
            for i, blk in enumerate(self.blocks):
                if i == lo:
                    f_lo = xt
                if i == hi:
                    f_hi = xt
                xt = _run(xt, blk)
            f_L = xt
            if f_lo is None:      # n == 0 (no front skip)
                f_lo = x0
            if f_hi is None:      # m == 0 (no back skip): hi == L, never entered
                f_hi = f_L
            delta_front = f_lo - x0
            delta_back = f_L - f_hi

        # Persist + reload the residuals (paper stores one set per iteration; also
        # the seam for a future separate precompute phase). Lossless round-trip.
        writer = cfg.get("on_residual")
        if writer is not None:
            delta_front, delta_back = writer(delta_front, delta_back)

        # Pass 2: gradient forward over the middle blocks only.
        x = x0 + delta_front
        for blk in self.blocks[lo:hi]:
            x = _run(x, blk)
        x = x + delta_back
        return x

    def forward_mini_train_dit(self, x_B_C_T_H_W, timesteps_B_T, crossattn_emb,
                                fps=None, padding_mask=None, source_attention_mask=None,
                                t5_input_ids=None, t5_attn_mask=None):
        if t5_input_ids is not None and self.use_llm_adapter and hasattr(self, "llm_adapter"):
            crossattn_emb = self.llm_adapter(
                source_hidden_states=crossattn_emb,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=source_attention_mask,
            )
            if t5_attn_mask is not None:
                crossattn_emb[~t5_attn_mask.bool()] = 0

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask,
        )
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        attn_params = attention.AttentionParams.create_attention_params(self.attn_mode, self.split_attn)
        use_fp32 = x_B_T_H_W_D.dtype == torch.float16

        # Training-free reference-style transfer: stamp block_idx + the
        # (possibly None) style context onto every self-attention module
        # before any block runs. Cheap no-op when style transfer is inactive.
        self._stamp_style_context()

        # Optional block-swap offloader (set by the pipeline backend for VRAM
        # optimization): streams each block's weights between CPU and GPU around
        # its forward. Gated on the attribute so the default path is unchanged.
        offloader = getattr(self, "_block_offloader", None)

        # First Block Cache (FBCache): OFF by default (_fbcache is None -> byte-identical,
        # including the block-swap wait/submit path below). When a FirstBlockCache is attached
        # by the Anima denoising loop, run only blocks[0], take its residual on the image stream
        # x_B_T_H_W_D as the indicator, and either reuse the cached full residual (skip blocks[1:])
        # or run them and refresh the cache. Only x_B_T_H_W_D evolves through the block list
        # (crossattn_emb is read-only context), so a single image tensor is both indicator and
        # cache. Mutually exclusive with Spectrum and Block Swap (guarded in the pipeline), so
        # this branch never runs alongside _block_offloader.
        fbcache = getattr(self, "_fbcache", None)

        # DiT-BlockSkip (arXiv 2603.20755) — training-only MEMORY-REDUCTION. When
        # the Anima LoRA trainer attaches a config for a training forward, skip the
        # first `front` and last `back` blocks: a no_grad full pass captures each
        # skipped span's residual feature Delta (span input->output), and the
        # gradient pass runs ONLY the middle blocks, re-adding Delta at the span
        # boundaries. Backprop flows only through the middle blocks (LoRA lives
        # there), so the skipped blocks retain no backward activations. Gated on
        # self.training (sampling/validation always run the full network) and never
        # composes with FBCache (inference-only) or block-swap (guarded off).
        blockskip = getattr(self, "_blockskip_config", None) if self.training else None
        if blockskip is not None:
            x_B_T_H_W_D = self._blockskip_forward(
                blockskip, x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb,
                attn_params, use_fp32, rope_emb_L_1_1_D, adaln_lora_B_T_3D,
                extra_pos_emb,
            )
        elif fbcache is not None:
            fbcache_step = getattr(self, "_fbcache_step", 0)
            original = x_B_T_H_W_D
            first_out = self.blocks[0](
                x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, attn_params, use_fp32,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb,
            )
            indicator = first_out - original
            if fbcache.use_cache(indicator, fbcache_step):
                # Cache hit: reuse the full-transformer residual, skip blocks[1:].
                x_B_T_H_W_D = original + fbcache.get()
            else:
                # Cache miss: run remaining blocks from first_out, refresh the cached residual.
                x_B_T_H_W_D = first_out
                for block in self.blocks[1:]:
                    x_B_T_H_W_D = block(
                        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, attn_params, use_fp32,
                        rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                        extra_per_block_pos_emb=extra_pos_emb,
                    )
                fbcache.store(x_B_T_H_W_D - original)
        else:
            # TREAD token routing (arXiv 2501.04765): OFF by default (_tread_config
            # is None). When the Anima trainer attaches a route config for a training
            # step, a random subset of (1 - drop_ratio) tokens is gathered at
            # block[start], run through blocks[start:end] ONLY, then scattered back
            # into the full stream at block[end] (bypassed tokens keep their pre-span
            # values — the paper's identity/residual transport). Training-only:
            # gated on self.training AND cleared for sampling/validation, and never
            # runs alongside FBCache (inference-only, handled above). Block-swap
            # stays compatible: every block __call__ still fires in index order, so
            # the offloader wait/submit coupling is preserved — routing only changes
            # WHICH tokens (and RoPE / extra-pos rows) each block sees.
            tread = getattr(self, "_tread_config", None) if self.training else None
            B, T, H, W, D = x_B_T_H_W_D.shape
            num_tokens = T * H * W

            route_active = False
            if tread is not None:
                start_b = int(tread.get("start_block", 0))
                end_b = int(tread.get("end_block", 0))
                drop_ratio = float(tread.get("drop_ratio", 0.0))
                # Route only when the span is valid, drops >=1 token, and the
                # sequence is a plain image grid (T == 1). adaLN modulation is
                # per-timestep (broadcast over spatial tokens); with T == 1 the
                # gathered subset — represented as a [B, 1, 1, keep, D] grid —
                # shares one modulation, so routing is exact. T > 1 (video) would
                # need per-token modulation, so it is skipped with a one-time warn.
                if not (0 <= start_b < end_b <= len(self.blocks) and 0.0 < drop_ratio < 1.0):
                    if not getattr(self, "_warned_tread_span", False):
                        print(f"[Anima TREAD] WARNING: invalid route "
                              f"(start={start_b}, end={end_b}, drop={drop_ratio}, "
                              f"blocks={len(self.blocks)}); routing disabled")
                        self._warned_tread_span = True
                elif T != 1:
                    if not getattr(self, "_warned_tread_video", False):
                        print(f"[Anima TREAD] WARNING: token routing requires T==1 "
                              f"(got T={T}); routing disabled for this run")
                        self._warned_tread_video = True
                elif num_tokens > 1:
                    route_active = True

            kept_idx = None
            rope_span = None
            extra_span = None
            x_flat_full = None
            if route_active:
                from core.training.token_routing import (
                    select_kept_indices, gather_tokens, scatter_tokens,
                )
                kept_idx = select_kept_indices(num_tokens, drop_ratio, x_B_T_H_W_D.device)

            # Low-rate stochastic depth (per-batch block dropout): OFF by default
            # (_block_skip_config is None). Training-only, gated on self.training and
            # cleared for sampling/validation by the trainer. Each eligible block
            # (front/back, outside the protected middle span) is independently
            # dropped this step with prob skip_rate; executed eligible blocks have
            # their residual DELTA scaled by 1/(1-skip_rate) so the expected
            # contribution matches the full eval network (torchvision-style inverted
            # scaling — eval runs every block, unscaled). Block-swap stays correct:
            # a dropped block still fires the offloader wait/submit (only the compute
            # is skipped), so the conductor never desyncs. When TREAD routing is
            # active, in-span blocks are EXCLUDED from dropout (they see the gathered
            # token subset — mixing the two transforms would be ill-defined), so the
            # two techniques compose on disjoint block ranges.
            bskip = getattr(self, "_block_skip_config", None) if self.training else None
            block_skip_active = False
            skip_mask = None
            eligible_set = None
            inv_keep = 1.0
            if bskip is not None:
                skip_rate = float(bskip.get("skip_rate", 0.0))
                protect_start = int(bskip.get("protect_start", 0))
                protect_end = int(bskip.get("protect_end", 0))
                if skip_rate > 0.0:
                    from core.training.block_dropout import compute_skip_mask
                    exclude = set(range(start_b, end_b)) if route_active else None
                    skip_mask, elig = compute_skip_mask(
                        len(self.blocks), skip_rate, protect_start, protect_end,
                        x_B_T_H_W_D.device, exclude=exclude,
                    )
                    eligible_set = set(elig)
                    inv_keep = 1.0 / (1.0 - skip_rate)
                    block_skip_active = True

            for block_idx, block in enumerate(self.blocks):
                if offloader is not None:
                    offloader.wait_for_block(block_idx)

                # Dropped block: identity (skip compute only). Offloader bookkeeping
                # below still fires so block-swap stays in lock-step.
                if block_skip_active and skip_mask[block_idx]:
                    if offloader is not None:
                        offloader.submit_move_blocks_forward(block_idx)
                    continue

                if route_active and block_idx == start_b:
                    # Enter route: snapshot the full stream, gather the kept subset
                    # into a [B, 1, 1, keep, D] pseudo-grid so blocks run unchanged.
                    x_flat_full = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
                    x_kept = gather_tokens(x_flat_full, kept_idx)
                    x_B_T_H_W_D = x_kept[:, None, None, :, :]
                    if rope_emb_L_1_1_D is not None:
                        rope_span = rope_emb_L_1_1_D.index_select(0, kept_idx)
                    if extra_pos_emb is not None:
                        extra_flat = rearrange(extra_pos_emb, "b t h w d -> b (t h w) d")
                        extra_span = gather_tokens(extra_flat, kept_idx)[:, None, None, :, :]

                in_span = route_active and (start_b <= block_idx < end_b)
                cur_rope = rope_span if in_span else rope_emb_L_1_1_D
                cur_extra = extra_span if in_span else extra_pos_emb

                # Executed eligible block: rescale its residual delta by 1/(1-p) so
                # E[residual] matches eval. Non-eligible (protected / excluded)
                # blocks were never subject to dropout, so run them unscaled.
                scale_delta = block_skip_active and (block_idx in eligible_set)
                x_before = x_B_T_H_W_D if scale_delta else None

                x_B_T_H_W_D = block(
                    x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, attn_params, use_fp32,
                    rope_emb_L_1_1_D=cur_rope,
                    adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                    extra_per_block_pos_emb=cur_extra,
                )

                if scale_delta:
                    x_B_T_H_W_D = x_before + (x_B_T_H_W_D - x_before) * inv_keep

                if offloader is not None:
                    offloader.submit_move_blocks_forward(block_idx)

                if route_active and block_idx == end_b - 1:
                    # Exit route: scatter processed tokens back into the full stream
                    # (bypassed tokens keep their pre-span values), restore grid shape.
                    x_kept_out = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
                    x_B_T_H_W_D = scatter_tokens(x_flat_full, x_kept_out, kept_idx)
                    x_B_T_H_W_D = rearrange(
                        x_B_T_H_W_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W,
                    )

        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D, t_embedding_B_T_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D, use_fp32=use_fp32,
        )
        return self.unpatchify(x_B_T_H_W_O)

    def forward(self, x, timesteps, context=None, fps=None, padding_mask=None,
                target_input_ids=None, target_attention_mask=None, source_attention_mask=None,
                **kwargs):
        context = self._preprocess_text_embeds(context, target_input_ids, target_attention_mask, source_attention_mask)
        return self.forward_mini_train_dit(x, timesteps, context, fps=fps, padding_mask=padding_mask, **kwargs)

    def _preprocess_text_embeds(self, source_hidden_states, target_input_ids,
                                target_attention_mask=None, source_attention_mask=None):
        if target_input_ids is not None:
            context = self.llm_adapter(
                source_hidden_states, target_input_ids,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
            )
            context[~target_attention_mask.bool()] = 0
            return context
        return source_hidden_states


# Default Anima DiT config (anima-base-v1.0). Match sd-scripts anima_utils.py.
ANIMA_DIT_CONFIG = dict(
    max_img_h=512,
    max_img_w=512,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    concat_padding_mask=True,
    crossattn_emb_channels=1024,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    min_fps=1,
    max_fps=30,
    use_adaln_lora=True,
    adaln_lora_dim=256,
    num_blocks=28,
    num_heads=16,
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=4.0,
    rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
    extra_h_extrapolation_ratio=1.0,
    extra_w_extrapolation_ratio=1.0,
    extra_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False,
    use_llm_adapter=True,
)
