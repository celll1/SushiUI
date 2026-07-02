"""MiniT2I MM-JiT architecture (pixel-space MM-DiT, flow matching, x0 prediction).

Vendored into SushiUI from the MIT-licensed MiniT2I reference (mmdit.py /
pipeline.py, github.com/Hope7Happiness/minit2i-torch). Module names and parameter
structure are preserved exactly so the published diffusers checkpoints load
unchanged; the forward path is generalized from square-only to arbitrary
(grid_h, grid_w) so non-square / non-512 resolutions work.

Pixel space: in_channels = 3 (RGB), no VAE. Images in [-1, 1]; flow
x_t = images*t + noise*(1-t) with noise_scale 2; model predicts x0.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from core.attention import AttentionMode, dispatch_attention


def mem_efficient_sdpa(q, k, v, backend="native", mode=AttentionMode.INFERENCE):
    """scaled_dot_product_attention with a head_dim that is always SDPA-fast.

    q/k/v: [B, H, N, D] (BHSD). The flash and memory-efficient SDPA backends
    require the head dim to be a multiple of 8; otherwise SDPA silently falls
    back to the math backend, which materialises the [B, H, N, N] score matrix
    (O(N^2) VRAM). The l16 variant uses head_dim=52, so at high token counts this
    alone costs tens of GB. Zero-padding D up to the next multiple of 8 leaves
    QK^T and the value mix unchanged (the padded lanes contribute 0), so passing
    the ORIGINAL-D scale makes the result numerically identical while keeping the
    O(N) fast path. b16 (D=64) needs no padding and hits the fast path directly.

    ``backend``/``mode`` route the (padded) attention through the unified conduit
    (native SDPA / FlashAttention / SageAttention) via ``dispatch_attention``.
    The default ``backend='native'`` reproduces the previous SDPA behaviour
    exactly. The explicit ``scale = orig_D ** -0.5`` MUST be passed: a kernel's
    default scale would use the PADDED head dim, changing the softmax temperature.
    l16 pads D=52->56 (sage excludes 52 -> conduit downgrades to native; flash
    accepts the padded 56); b16 D=64 is accepted by both flash and sage. No mask
    is ever forwarded (the joint sequence is dense).
    """
    d = q.shape[-1]
    scale = d ** -0.5
    pad = (-d) % 8
    if pad:
        q = F.pad(q, (0, pad))
        k = F.pad(k, (0, pad))
        v = F.pad(v, (0, pad))
    out = dispatch_attention(
        q, k, v,
        scale=scale,
        backend=backend,
        mode=mode,
        layout="BHSD",
    )
    if pad:
        return out[..., :d]
    return out


def rotate_half(x):
    x1, x2 = x.reshape(*x.shape[:-1], 2, -1).unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


class BottleneckPatchEmbed(nn.Module):
    """Pixel -> token: Conv(3 -> pca) patchify, then 1x1 Conv(pca -> hidden)."""

    def __init__(self, img_size=512, patch_size=16, in_channels=3, pca_channels=128, hidden_size=1248):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj1 = nn.Conv2d(in_channels, pca_channels, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_channels, hidden_size, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.proj2(self.proj1(x))  # [b, hidden, gh, gw]
        gh, gw = x.shape[-2], x.shape[-1]
        return x.flatten(2).transpose(1, 2), gh, gw  # [b, gh*gw, hidden], gh, gw


class SwiGLUMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        hidden_dim = (hidden_features + 7) // 8 * 8
        self.w1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.w3 = nn.Linear(in_features, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, in_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TextRotaryEmbedding1D(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, x):
        b, length, h, d = x.shape
        inv = 1.0 / (self.theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        pos = torch.arange(length, device=x.device, dtype=torch.float32)
        angles = torch.einsum("l,f->lf", pos, inv)
        angles = torch.cat([angles, angles], dim=-1)
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)
        return x * cos[None, :, None, :] + rotate_half(x) * sin[None, :, None, :]


class VisionRotaryEmbeddingFast(nn.Module):
    """2D vision RoPE generalized to arbitrary (grid_h, grid_w).

    For grid_h == grid_w this is numerically identical to the square-only
    reference (token order is row-major: index = h*grid_w + w).
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = head_dim // 2
        self.theta = theta

    def forward(self, x, grid_h: int, grid_w: int):
        length = x.shape[1]
        if grid_h * grid_w != length:
            raise ValueError(f"image token length {length} != grid_h*grid_w {grid_h*grid_w}")
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)[: self.dim // 2] / self.dim)
        )
        th = torch.arange(grid_h, device=x.device, dtype=torch.float32)
        tw = torch.arange(grid_w, device=x.device, dtype=torch.float32)
        base_h = torch.einsum("l,f->lf", th, freqs)  # [gh, dim//4]
        base_w = torch.einsum("l,f->lf", tw, freqs)  # [gw, dim//4]
        f_h = base_h[:, None, :].expand(grid_h, grid_w, -1)
        f_w = base_w[None, :, :].expand(grid_h, grid_w, -1)
        angles = torch.cat([f_h, f_w], dim=-1)
        angles = torch.cat([angles, angles], dim=-1).reshape(length, -1)
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)
        return x * cos[None, :, None, :] + rotate_half(x) * sin[None, :, None, :]


class MultiModalRotaryEmbeddingFast(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.text_rope = TextRotaryEmbedding1D(head_dim)
        self.vision_rope = VisionRotaryEmbeddingFast(head_dim)

    def forward(self, x, txt_len: int, grid_h: int, grid_w: int):
        txt = self.text_rope(x[:, :txt_len])
        img = self.vision_rope(x[:, txt_len:], grid_h, grid_w)
        return torch.cat([txt, img], dim=1)


class PlainTextTransformerBlock(nn.Module):
    def __init__(self, hidden_size=1248, num_heads=24, head_dim=52, mlp_ratio=2.7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, inner_dim * 3)
        self.attn_proj = nn.Linear(inner_dim, hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = TextRotaryEmbedding1D(head_dim)

    def forward(self, txt):
        b, length, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        # Memory-efficient attention (flash / mem-efficient SDPA) instead of an
        # explicit [B,heads,L,L] score matrix — O(L) memory, needed for high-res
        # sequences. head_dim is padded to a multiple of 8 so SDPA never falls
        # back to the O(L^2) math backend (mathematically equivalent).
        out = mem_efficient_sdpa(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            backend=getattr(self, "_attn_backend", "native"),
            mode=getattr(self, "_attn_mode", AttentionMode.INFERENCE),
        ).transpose(1, 2).reshape(b, length, -1)
        txt = txt + self.attn_proj(out)
        txt = txt + self.mlp(self.norm2(txt))
        return txt


class DoubleStreamDiTBlock(nn.Module):
    def __init__(self, hidden_size=1248, txt_hidden_size=1248, num_heads=24, head_dim=52, mlp_ratio=2.7):
        super().__init__()
        self.hidden_size = hidden_size
        self.txt_hidden_size = txt_hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.img_norm1 = RMSNorm(hidden_size)
        self.img_norm2 = RMSNorm(hidden_size)
        self.txt_norm1 = RMSNorm(txt_hidden_size)
        self.txt_norm2 = RMSNorm(txt_hidden_size)
        self.img_qkv = nn.Linear(hidden_size, inner_dim * 3)
        self.txt_qkv = nn.Linear(txt_hidden_size, inner_dim * 3)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = MultiModalRotaryEmbeddingFast(head_dim)
        self.img_attn_proj = nn.Linear(inner_dim, hidden_size)
        self.txt_attn_proj = nn.Linear(inner_dim, txt_hidden_size)
        self.img_mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = SwiGLUMlp(txt_hidden_size, int(txt_hidden_size * mlp_ratio))

    def forward(self, x, txt, vec, grid_h: int, grid_w: int):
        b, li, _ = x.shape
        lt = txt.shape[1]
        x_norm = self.img_norm1(x)
        txt_norm = self.txt_norm1(txt)
        qkv_i = self.img_qkv(x_norm).reshape(b, li, 3, self.num_heads, self.head_dim)
        qkv_t = self.txt_qkv(txt_norm).reshape(b, lt, 3, self.num_heads, self.head_dim)
        q_i, k_i, v_i = qkv_i[:, :, 0], qkv_i[:, :, 1], qkv_i[:, :, 2]
        q_t, k_t, v_t = qkv_t[:, :, 0], qkv_t[:, :, 1], qkv_t[:, :, 2]
        q_i, k_i = self.q_norm(q_i), self.k_norm(k_i)
        q_t, k_t = self.q_norm(q_t), self.k_norm(k_t)
        q = self.rope(torch.cat([q_t, q_i], dim=1), txt_len=lt, grid_h=grid_h, grid_w=grid_w)
        k = self.rope(torch.cat([k_t, k_i], dim=1), txt_len=lt, grid_h=grid_h, grid_w=grid_w)
        v = torch.cat([v_t, v_i], dim=1)
        # Memory-efficient attention (flash / mem-efficient SDPA) over the joint
        # [text + image] sequence — avoids the O(L²) score matrix that OOMs at
        # high res. head_dim is padded to a multiple of 8 so SDPA never falls back
        # to the O(L²) math backend (mathematically equivalent).
        out = mem_efficient_sdpa(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            backend=getattr(self, "_attn_backend", "native"),
            mode=getattr(self, "_attn_mode", AttentionMode.INFERENCE),
        ).transpose(1, 2).contiguous()  # [b, seq, heads, hd]
        x = x + self.img_attn_proj(out[:, lt:].reshape(b, li, -1))
        txt = txt + self.txt_attn_proj(out[:, :lt].reshape(b, lt, -1))
        x = x + self.img_mlp(self.img_norm2(x))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return x, txt


class FinalLayer(nn.Module):
    def __init__(self, hidden_size=1248, patch_size=16, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x, vec=None):
        return self.linear(self.norm_final(x))


def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, device, dtype):
    """2D sincos pos-embed for a (grid_h, grid_w) row-major token grid.

    Token order matches BottleneckPatchEmbed.flatten(2): index = h*grid_w + w.
    For grid_h == grid_w this equals the square-only reference (which encodes
    [enc(w), enc(h)] in that token order).
    """
    h_idx = torch.arange(grid_h, device=device, dtype=torch.float32)[:, None].expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w, device=device, dtype=torch.float32)[None, :].expand(grid_h, grid_w).reshape(-1)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, w_idx)  # reference emb_h encodes the W coordinate
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, h_idx)  # reference emb_w encodes the H coordinate
    return torch.cat([emb_w, emb_h], dim=1).to(dtype=dtype)


@dataclass
class MMJiTConfig:
    image_size: int = 512
    patch_size: int = 16
    in_channels: int = 3
    txt_input_size: int = 1024
    hidden_size: int = 768
    txt_hidden_size: int = 768
    cond_vec_size: int = 768
    depth_double: int = 17
    txt_preamble_depth: int = 2
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 2.6666666666666665
    pca_channels: int = 128
    prompt_length: int = 256
    n_T: int = 100
    prediction: str = "x"
    sampler: str = "euler"
    cfg_channels: int = 3
    cfg_interval: tuple = (0.0, 1.0)
    llm: str = "google/flan-t5-large"
    # Data space: "none" = pixel-space RGB (in_channels=3, patch_size=16); "sdxl" /
    # "flux1" = VAE-latent space (in_channels = VAE latent channels, patch_size=2).
    # Only affects the I/O layers (proj1 / FinalLayer); the transformer body is
    # channel-agnostic. noise_scale defaults to 2.0 for pixel, 1.0 for latent.
    vae_type: str = "none"
    noise_scale: float = 2.0


class MMJiT(nn.Module):
    def __init__(self, cfg: MMJiTConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_img_size = cfg.image_size // cfg.patch_size
        self.img_embedder = BottleneckPatchEmbed(
            cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.pca_channels, cfg.hidden_size
        )
        self.txt_embedder = nn.Linear(cfg.txt_input_size, cfg.txt_hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.txt_input_size))
        self.t_embedder = TimestepEmbedder(cfg.cond_vec_size)
        self.pooled_embedder = nn.Linear(cfg.txt_input_size, cfg.cond_vec_size, bias=False)
        self.txt_preamble_blocks = nn.ModuleList(
            [PlainTextTransformerBlock(cfg.txt_hidden_size, cfg.num_heads, cfg.head_dim, cfg.mlp_ratio)
             for _ in range(cfg.txt_preamble_depth)]
        )
        self.double_blocks = nn.ModuleList(
            [DoubleStreamDiTBlock(cfg.hidden_size, cfg.txt_hidden_size, cfg.num_heads, cfg.head_dim, cfg.mlp_ratio)
             for _ in range(cfg.depth_double)]
        )
        self.final_layer = FinalLayer(cfg.hidden_size, cfg.patch_size, cfg.in_channels)
        self.gradient_checkpointing = False
        # REPA tap: when _repa_tap_depth is set (0-based block index), forward()
        # stashes the grad-connected image hidden state after that double block into
        # _repa_tap_out for representation-alignment loss. None = disabled (no-op).
        # Capturing the loop variable (the checkpoint output) is gradient-checkpoint
        # safe, unlike a forward hook which would see the no-grad recompute tensor.
        self._repa_tap_depth = None
        self._repa_tap_out = None

    def unpatchify(self, x, grid_h: int, grid_w: int):
        b = x.shape[0]
        p = self.cfg.patch_size
        c = self.cfg.in_channels
        x = x.reshape(b, grid_h, grid_w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(b, c, grid_h * p, grid_w * p)

    def forward(self, img, t, context, attn_mask):
        if img.ndim == 4 and img.shape[1] != self.cfg.in_channels:
            img = img.permute(0, 3, 1, 2)
        attn_mask = attn_mask.to(device=context.device)
        context = torch.where(attn_mask[:, :, None] > 0.5, context, self.mask_token.to(dtype=context.dtype))
        x, gh, gw = self.img_embedder(img)
        pos = get_2d_sincos_pos_embed(self.cfg.hidden_size, gh, gw, x.device, x.dtype)
        x = x + pos[None]
        t_vec = self.t_embedder(t)
        # Use a wrapping-agnostic dtype (txt_embedder / pooled_embedder may be
        # wrapped by a LoRALinearLayer during training, which has no `.weight`).
        txt_dtype = next(self.txt_embedder.parameters()).dtype
        pooled_dtype = next(self.pooled_embedder.parameters()).dtype
        txt = self.txt_embedder(context.to(dtype=txt_dtype))
        pooled_text = context.mean(dim=1)
        vec = t_vec + self.pooled_embedder(pooled_text.to(dtype=pooled_dtype))
        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        # Attention backend/mode propagation: the vendored (and NAG/NegPip
        # monkey-patched) block forwards route their mem_efficient_sdpa call
        # through the unified conduit, reading a per-block ``_attn_backend`` /
        # ``_attn_mode``. Stamp both onto every attention-bearing block from this
        # net's ``_attn_backend`` (set by the inference plumbing in
        # pipeline_backends/minit2i.py and by the training hook). Mode is derived
        # from the autograd state: inference denoise loops run under
        # ``torch.no_grad`` (INFERENCE, sage allowed) while training runs with grad
        # enabled (TRAINING, sage refused by the conduit). Default 'native' keeps
        # the prior SDPA path byte-identical.
        _attn_backend = getattr(self, "_attn_backend", "native")
        _attn_mode = AttentionMode.TRAINING if torch.is_grad_enabled() else AttentionMode.INFERENCE
        for _blk in self.txt_preamble_blocks:
            _blk._attn_backend = _attn_backend
            _blk._attn_mode = _attn_mode
        for _blk in self.double_blocks:
            _blk._attn_backend = _attn_backend
            _blk._attn_mode = _attn_mode
        for block in self.txt_preamble_blocks:
            if use_ckpt:
                txt = torch.utils.checkpoint.checkpoint(block, txt, use_reentrant=False)
            else:
                txt = block(txt)
        self._repa_tap_out = None
        # First Block Cache (FBCache): OFF by default (_fbcache is None -> byte-identical,
        # including the block-swap wait/submit path below). When a FirstBlockCache is attached
        # by the MiniT2I denoising loop, run only double_blocks[0], take its residual on the
        # image stream `x` as the indicator, and either reuse the cached full residual (skip
        # double_blocks[1:]) or run them and refresh the cache. Both `x` (image) and `txt`
        # (text) evolve through the double-block list and both feed `combined` afterwards, so
        # the cache stores a tuple of BOTH residuals (x_residual, txt_residual); the indicator
        # is the image residual only (the cheap change signal). Mutually exclusive with Spectrum
        # and Block Swap (guarded in the pipeline), so this branch never runs alongside
        # _block_offloader. txt_preamble_blocks above are untouched (small, kept resident).
        #
        # REPA tap: the tap captures `x` at _repa_tap_depth during real compute (training-time
        # auxiliary). On a MISS the full loop runs so the tap fires exactly as before; on a HIT
        # the skipped blocks are not executed so the tap is not set for a skipped depth (FBCache
        # is inference-only and REPA is not read at inference, so this is safe).
        fbcache = getattr(self, "_fbcache", None)
        if fbcache is not None:
            fbcache_step = getattr(self, "_fbcache_step", 0)
            orig_x, orig_txt = x, txt
            first = self.double_blocks[0]
            if use_ckpt:
                x, txt = torch.utils.checkpoint.checkpoint(first, x, txt, vec, gh, gw, use_reentrant=False)
            else:
                x, txt = first(x, txt, vec, gh, gw)
            if self._repa_tap_depth is not None and 0 == self._repa_tap_depth:
                self._repa_tap_out = x
            indicator = x - orig_x
            if fbcache.use_cache(indicator, fbcache_step):
                # Cache hit: reuse the full-transformer residuals, skip double_blocks[1:].
                x_res, txt_res = fbcache.get()
                x = orig_x + x_res
                txt = orig_txt + txt_res
            else:
                # Cache miss: run remaining blocks, refresh the cached (x, txt) residuals.
                for _depth, block in enumerate(self.double_blocks[1:], start=1):
                    if use_ckpt:
                        x, txt = torch.utils.checkpoint.checkpoint(block, x, txt, vec, gh, gw, use_reentrant=False)
                    else:
                        x, txt = block(x, txt, vec, gh, gw)
                    if self._repa_tap_depth is not None and _depth == self._repa_tap_depth:
                        self._repa_tap_out = x
                fbcache.store((x - orig_x, txt - orig_txt))
        else:
            # Block swap (inference only): when a TransformerBlockOffloader is attached, only
            # the heavy `double_blocks` are streamed CPU<->GPU (txt_preamble_blocks and all
            # other modules stay GPU-resident). Gated strictly on `_block_offloader`; when it
            # is None the default path below is byte-for-byte unchanged.
            _offloader = getattr(self, "_block_offloader", None)
            for _depth, block in enumerate(self.double_blocks):
                if _offloader is not None:
                    _offloader.wait_for_block(_depth)
                if use_ckpt:
                    x, txt = torch.utils.checkpoint.checkpoint(block, x, txt, vec, gh, gw, use_reentrant=False)
                else:
                    x, txt = block(x, txt, vec, gh, gw)
                if _offloader is not None:
                    _offloader.submit_move_blocks_forward(_depth)
                if self._repa_tap_depth is not None and _depth == self._repa_tap_depth:
                    self._repa_tap_out = x
        combined = torch.cat([txt, x], dim=1)
        out = self.final_layer(combined, vec)
        img_out = out[:, txt.shape[1]:, :]
        return self.unpatchify(img_out, gh, gw)


class DiffusionModel(nn.Module):
    def __init__(self, cfg: Optional[MMJiTConfig] = None):
        super().__init__()
        self.cfg = cfg or MMJiTConfig()
        self.net = MMJiT(self.cfg)

    def pred_velocity(self, x, t, text, mask):
        x0 = self.net(x, t, text, mask)
        return (x0 - x) / torch.clamp(1 - t[:, None, None, None], min=0.05)
