"""
DEUS U-Net Architecture (Dual-Embeddings U-Net Structure)

Based on SDXL U-Net with modifications:
- RoPE 2D positional encoding (from Z-Image)
- Sparse skip connections (every N blocks instead of every block)
- Improved Conv blocks
- 4-channel latent input (SDXL VAE)
- Multi-modal conditioning (SigLIP-2 text + optional images)

Architecture:
- Input: 4-channel latents
- Conditioning: SigLIP-2 embeddings (1152-dim, variable length)
- Down blocks: 4 stages with sparse skip connections (SDXL style: 320, 640, 1280, 1280)
- Mid block: Attention + FFN
- Up blocks: 4 stages with sparse skip connections from down blocks
- Output: 4-channel latents

Model sizes:
- Small: ~1.5B params
- Medium: ~2.8B params (default)
- Large: ~4.0B params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class UNetConfig:
    """U-Net configuration."""

    # Latent specs
    in_channels: int = 4  # SDXL VAE latent channels
    out_channels: int = 4

    # Model size
    model_channels: int = 320  # Base channel count
    channel_mult: Tuple[int, ...] = (1, 2, 4)  # Channel multipliers per stage
    num_res_blocks: int = 2  # Residual blocks per DownBlock
    num_res_blocks_per_up_block: int = 2  # Residual blocks per UpBlock (can differ from DownBlock)

    # Attention (SDXL style: block-specific settings)
    # For SDXL compatibility: head_dim=64 fixed, heads vary by block
    attention_head_dim: int = 64  # SDXL uses fixed head_dim=64
    num_attention_heads: Tuple[int, ...] = (5, 10, 20)  # Per block: Down0, Down1, Down2 / Up2, Up1, Up0
    transformer_layers_per_block: Tuple[int, ...] = (0, 2, 2)  # SDXL: Down0=0, Down1=2, Down2=2
    transformer_layers_per_up_block: Tuple[int, ...] = (0, 0, 0)  # SDXL: Up0=0, Up1=0, Up2=0
    transformer_layers_per_mid_block: int = 10  # SDXL: Mid block has 10 layers
    context_dim: int = 1152  # SigLIP-2 hidden size

    # Skip connections
    skip_connection_interval: int = 2  # Sparse skip: connect every N blocks

    # Time embedding
    time_embed_dim: int = 1280  # Time embedding dimension

    # Dropout
    dropout: float = 0.0

    # Model variant
    variant: str = "medium"  # "small", "medium", "large"

    @property
    def latent_channels(self) -> int:
        """Latent channels (for compatibility with checkpoint_utils)."""
        return self.in_channels

    @classmethod
    def from_variant(cls, variant: str = "medium"):
        """Create config from variant name.

        Based on SDXL U-Net architecture:
        - SDXL Base: 2.6B params (block_out_channels: 320, 640, 1280)
        - Our variants scale this for target sizes
        """
        configs = {
            "small": {
                # Target: ~1.5B params (smaller than SDXL)
                "model_channels": 256,
                "channel_mult": (1, 2, 4),  # 256, 512, 1024
                "num_res_blocks": 2,
                "num_attention_heads": 16,  # 256 / 16 = 16 (head_dim)
                "transformer_depth": 6,
            },
            "medium": {
                # Target: ~2.6B params (SDXL Base equivalent)
                # DEUS: Completely matches SDXL structure
                # Differences: RoPE (conditioning), no time_ids, sparse skip connections
                "model_channels": 320,
                "channel_mult": (1, 2, 4),  # 320, 640, 1280 (3 blocks, same as SDXL)
                "num_res_blocks": 2,  # Down blocks: 2 resnets each (SDXL)
                "num_res_blocks_per_up_block": 3,  # Up blocks: 3 resnets each (SDXL)
                "attention_head_dim": 64,  # SDXL fixed head_dim
                "num_attention_heads": (5, 10, 20),  # SDXL: Down0=5, Down1=10, Down2=20
                "transformer_layers_per_block": (0, 2, 2),  # SDXL: Down0=0, Down1=2, Down2=2
                "transformer_layers_per_up_block": (0, 0, 0),  # SDXL: Up0=0, Up1=0, Up2=0
                "transformer_layers_per_mid_block": 10,  # SDXL: Mid block has 10 layers
            },
            "large": {
                # Target: ~3.2B params (slightly larger than SDXL)
                "model_channels": 384,
                "channel_mult": (1, 2, 4),  # 384, 768, 1536
                "num_res_blocks": 2,
                "num_attention_heads": 24,  # 384 / 24 = 16 (head_dim)
                "transformer_depth": 10,
            },
        }

        if variant not in configs:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

        config_dict = configs[variant]
        config_dict["variant"] = variant
        return cls(**config_dict)


class RoPE2D(nn.Module):
    """
    Resolution-Adaptive 2D Rotary Position Embedding.

    Applies resolution-adaptive rotary embeddings to spatial dimensions (H, W) of latents.
    Position indices are normalized by resolution ratio to maintain consistent positional
    information across different resolutions.

    Key improvement: Allows generation at resolutions different from training while
    maintaining consistent behavior and quality.

    References:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - Position Interpolation: https://arxiv.org/abs/2306.15595
    """

    def __init__(
        self,
        dim: int,
        train_resolution: int = 128,
        max_resolution: int = 512,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.train_resolution = train_resolution  # Training resolution in latent space (1024px / 8 = 128)
        self.max_resolution = max_resolution      # Max expected resolution (4096px / 8 = 512)
        self.base = base

        # Precompute frequency bands: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for RoPE embeddings: {(H, W): cached_embedding}
        # Cached on CPU to save VRAM, moved to device when needed
        self._cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply resolution-adaptive 2D RoPE to input tensor.
        
        Uses caching to avoid recomputing RoPE embeddings for the same resolution.

        Args:
            x: Input [batch, channels, height, width]

        Returns:
            x with RoPE applied [batch, channels, height, width]
        """
        B, C, H, W = x.shape
        
        # Check cache first (keyed by resolution)
        cache_key = (H, W)
        
        if cache_key in self._cache:
            # Use cached embedding (move to device and dtype)
            emb_2d = self._cache[cache_key].to(device=x.device, dtype=x.dtype)
        else:
            # Compute RoPE embedding (first time for this resolution)
            emb_2d = self._compute_rope_embedding(H, W, x.device, x.dtype, C)
            # Cache on CPU to save VRAM
            self._cache[cache_key] = emb_2d.cpu()
        
        return x + emb_2d
    
    def _compute_rope_embedding(
        self, 
        H: int, 
        W: int, 
        device: torch.device, 
        dtype: torch.dtype,
        C: int
    ) -> torch.Tensor:
        """
        Compute RoPE embedding for given resolution.
        
        Args:
            H: Height
            W: Width
            device: Target device
            dtype: Target dtype
            C: Channel count (for matching)
        
        Returns:
            RoPE embedding [1, C, H, W]
        """
        # Resolution scaling factors (normalize by training resolution)
        scale_h = H / self.train_resolution
        scale_w = W / self.train_resolution

        # Normalized position indices (relative to training resolution)
        # At 2x training resolution: [0, 0.5, 1.0, 1.5, ..., 127.5]
        # At 1x training resolution: [0, 1, 2, ..., 127]
        # At 0.5x training resolution: [0, 2, 4, ..., 126]
        pos_h = torch.arange(H, device=device, dtype=torch.float32) / scale_h
        pos_w = torch.arange(W, device=device, dtype=torch.float32) / scale_w

        # Compute sinusoidal embeddings
        # For height dimension
        freqs_h = torch.einsum("i,j->ij", pos_h, self.inv_freq.to(device))  # [H, dim//2]
        emb_h = torch.cat([freqs_h.sin(), freqs_h.cos()], dim=-1)  # [H, dim]

        # For width dimension
        freqs_w = torch.einsum("i,j->ij", pos_w, self.inv_freq.to(device))  # [W, dim//2]
        emb_w = torch.cat([freqs_w.sin(), freqs_w.cos()], dim=-1)  # [W, dim]

        # Expand to 2D grid: [H, W, dim]
        emb_h = emb_h.unsqueeze(1).expand(-1, W, -1)  # [H, W, dim]
        emb_w = emb_w.unsqueeze(0).expand(H, -1, -1)  # [H, W, dim]

        # Combine (addition maintains independence of H and W components)
        emb_2d = emb_h + emb_w  # [H, W, dim]

        # Match channel count (repeat or slice)
        if self.dim < C:
            emb_2d = emb_2d.repeat(1, 1, (C // self.dim) + 1)[:, :, :C]
        else:
            emb_2d = emb_2d[:, :, :C]

        # Reshape to [1, C, H, W] and convert to target dtype
        emb_2d = emb_2d.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)  # [1, C, H, W]

        return emb_2d
    
    def clear_cache(self):
        """Clear the RoPE cache (useful for memory management)."""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached RoPE embeddings."""
        return len(self._cache)


class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation (SDXL style).
    
    GEGLU(x) = Linear(x) -> chunk(2) -> hidden * GELU(gate)
    This matches SDXL's FeedForward implementation.
    """
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class ImprovedConvBlock(nn.Module):
    """
    Improved Convolutional Block with DiC enhancements.

    Enhancements:
    - GroupNorm instead of BatchNorm
    - GELU activation (DiC style)
    - Residual connection
    - Mid-block condition injection (DiC)
    - Conditional gating (DiC)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        dropout: float = 0.0,
        condition_dim: Optional[int] = None  # DiC: condition dimension for gating
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.GELU()  # DiC: GELU activation

        # DiC: Mid-block condition injection and conditional gating
        self.condition_dim = condition_dim
        if condition_dim is not None:
            # Condition projection for gating
            self.condition_proj = nn.Linear(condition_dim, out_channels * 2)  # scale and shift
        else:
            self.condition_proj = None

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input [batch, in_channels, height, width]
            condition: Optional condition [batch, condition_dim] for DiC conditional gating
        """
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)

        # DiC: Mid-block condition injection and conditional gating
        if condition is not None and self.condition_proj is not None:
            # Project condition to scale and shift
            cond_params = self.condition_proj(condition)  # [B, out_channels * 2]
            scale, shift = cond_params.chunk(2, dim=-1)  # [B, out_channels] each
            # Apply conditional gating: x = x * (1 + scale) + shift
            scale = scale[:, :, None, None]  # [B, C, 1, 1]
            shift = shift[:, :, None, None]  # [B, C, 1, 1]
            x = x * (1 + scale) + shift

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.act(x)

        return x


class CrossAttentionBlock(nn.Module):
    """
    Memory-efficient Cross-Attention block using Flash Attention.

    Uses scaled_dot_product_attention instead of nn.MultiheadAttention
    to avoid storing massive attention weight matrices (25.8GB per layer).

    Attends to SigLIP-2 text/image embeddings.
    """

    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()

        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        
        # SDXL style: Use fixed head_dim=64 if provided, otherwise calculate from channels/heads
        if head_dim is not None:
            self.head_dim = head_dim
            # Verify that channels == num_heads * head_dim
            assert channels == num_heads * head_dim, \
                f"channels ({channels}) must equal num_heads ({num_heads}) * head_dim ({head_dim})"
        else:
            self.head_dim = channels // num_heads
            assert channels % num_heads == 0, \
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        # Layer norms
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)

        # Self-attention projections
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)

        # Cross-attention projections
        self.to_q_cross = nn.Linear(channels, channels, bias=False)
        self.to_k_cross = nn.Linear(context_dim, channels, bias=False)
        self.to_v_cross = nn.Linear(context_dim, channels, bias=False)
        self.to_out_cross = nn.Linear(channels, channels)

        # FFN with GEGLU (SDXL style)
        # GEGLU: Linear(channels -> channels * 8) -> chunk(2) -> hidden * GELU(gate) -> Linear(channels * 4 -> channels)
        # This matches SDXL's FeedForward implementation
        self.ffn = nn.ModuleList([
            GEGLU(channels, channels * 8, bias=True),  # GEGLU: Linear(channels -> channels*8) -> chunk -> hidden * GELU(gate)
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels, bias=True),  # channels*8 / 2 = channels*4 after chunk
            nn.Dropout(dropout)
        ])

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Latent features [batch, seq_len, channels] (already flattened, like SDXL)
            context: Conditioning [batch, seq_len, context_dim]

        Returns:
            Updated features [batch, seq_len, channels] (same shape as input)
        """
        # Input is already [B, H*W, C] (SDXL style)
        # No shape conversion needed - this is the key optimization!

        # Self-attention with Flash Attention
        x_norm = self.norm1(x)
        
        # Compute QKV projections
        q = self.to_q(x_norm)  # [B, H*W, C]
        k = self.to_k(x_norm)  # [B, H*W, C]
        v = self.to_v(x_norm)  # [B, H*W, C]
        
        # Reshape for attention: [B, H*W, C] -> [B, H*W, num_heads, head_dim] -> [B, num_heads, H*W, head_dim]
        B, seq_len, C = q.shape
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash Attention v2 (no attention weights stored)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0
        )

        # Reshape back: [B, num_heads, H*W, head_dim] -> [B, H*W, C]
        attn_output = attn_output.transpose(1, 2).reshape(B, seq_len, C)
        attn_output = self.to_out(attn_output)
        x = x + attn_output

        # Cross-attention with Flash Attention
        x_norm = self.norm2(x)
        seq_len_ctx = context.size(1)

        # Compute QKV projections
        q_cross = self.to_q_cross(x_norm)  # [B, H*W, C]
        k_cross = self.to_k_cross(context)  # [B, seq_len_ctx, C]
        v_cross = self.to_v_cross(context)  # [B, seq_len_ctx, C]
        
        # Reshape for attention
        q_cross = q_cross.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_cross = k_cross.view(B, seq_len_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        v_cross = v_cross.view(B, seq_len_ctx, self.num_heads, self.head_dim).transpose(1, 2)

        cross_output = torch.nn.functional.scaled_dot_product_attention(
            q_cross, k_cross, v_cross,
            dropout_p=self.dropout if self.training else 0.0
        )

        # Reshape back: [B, num_heads, H*W, head_dim] -> [B, H*W, C]
        cross_output = cross_output.transpose(1, 2).reshape(B, seq_len, C)
        cross_output = self.to_out_cross(cross_output)
        x = x + cross_output

        # FFN with GEGLU (SDXL style)
        x_norm = self.norm3(x)
        # GEGLU: proj -> chunk(2) -> hidden * GELU(gate)
        x_ffn = self.ffn[0](x_norm)  # GEGLU: outputs channels * 4
        x_ffn = self.ffn[1](x_ffn)  # Dropout
        x_ffn = self.ffn[2](x_ffn)  # Linear(channels * 4 -> channels)
        x_ffn = self.ffn[3](x_ffn)  # Dropout
        x = x + x_ffn

        # Return [B, H*W, C] (no shape conversion - handled by DownBlock/UpBlock)
        return x


class ResnetBlock(nn.Module):
    """
    Residual block with time embedding and improved conv.
    DiC enhancements:
    - Mid-block condition injection
    - Conditional gating
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None  # DiC: context dimension for conditional gating
    ):
        super().__init__()

        # DiC: Pass context_dim to ImprovedConvBlock for conditional gating
        self.conv_block = ImprovedConvBlock(
            in_channels, 
            out_channels, 
            dropout=dropout,
            condition_dim=context_dim
        )

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input [batch, in_channels, height, width]
            time_emb: Time embedding [batch, time_embed_dim]
            context: Optional context [batch, context_dim] for DiC conditional gating

        Returns:
            Output [batch, out_channels, height, width]
        """
        # DiC: Apply conv block with mid-block condition injection
        # If context is provided, use pooled context for conditional gating
        condition = None
        if context is not None:
            # Pool context: [B, L, C] -> [B, C] (mean pooling)
            condition = context.mean(dim=1)  # [B, context_dim]
        
        x = self.conv_block(x, condition=condition)

        # Add time embedding
        time_proj = self.time_mlp(time_emb)[:, :, None, None]  # [B, C, 1, 1]
        x = x + time_proj

        return x


class DownBlock(nn.Module):
    """Down-sampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        context_dim: int,
        num_res_blocks: int = 2,
        num_attention_heads: int = 8,
        transformer_depth: int = 2,
        dropout: float = 0.0,
        downsample: bool = True
    ):
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.transformer_depth = transformer_depth
        self.downsample = downsample

        # Resnet blocks (DiC: with context_dim for conditional gating)
        self.resnets = nn.ModuleList([
            ResnetBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim,
                dropout,
                context_dim=context_dim  # DiC: pass context_dim for conditional gating
            )
            for i in range(num_res_blocks)
        ])

        # SDXL-style: Transformer2DModel wrapper (proj_in, norm, proj_out)
        # Only add if transformer_depth > 0
        if transformer_depth > 0:
            # SDXL: norm (GroupNorm) before proj_in
            self.norm = nn.GroupNorm(32, out_channels, eps=1e-6)
            # SDXL: proj_in (Linear projection)
            self.proj_in = nn.Linear(out_channels, out_channels, bias=True)
            # SDXL: proj_out (Linear projection)
            self.proj_out = nn.Linear(out_channels, out_channels, bias=True)
            
            # Attention blocks (only if transformer_depth > 0)
            # head_dim is passed explicitly for SDXL compatibility (fixed 64)
            head_dim = 64  # SDXL fixed head_dim
            self.attentions = nn.ModuleList([
                CrossAttentionBlock(
                    out_channels,
                    context_dim,
                    num_attention_heads,
                    head_dim=head_dim,
                    dropout=dropout
                )
                for _ in range(transformer_depth)
            ])
        else:
            # SDXL style: Some blocks have no attention (e.g., DownBlock0)
            self.norm = None
            self.proj_in = None
            self.proj_out = None
            self.attentions = nn.ModuleList([])

        # Downsample
        if downsample:
            self.downsample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample_conv = None

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (output, skip_connection)
        """
        import time
        is_step1 = hasattr(self, '_parent_is_step1') and self._parent_is_step1
        
        if is_step1:
            db_internal_start = time.time()
            torch.cuda.synchronize()
            resnet_times = []
            attn_times = []
        
        # Resnet blocks (process in [B, C, H, W] format)
        for i, resnet in enumerate(self.resnets):
            if is_step1:
                resnet_start = time.time()
                torch.cuda.synchronize()
            
            x = resnet(x, time_emb, context)  # DiC: pass context for conditional gating
            
            if is_step1:
                torch.cuda.synchronize()
                resnet_time = (time.time() - resnet_start) * 1000
                resnet_times.append(resnet_time)

        # SDXL-style: Transformer2DModel wrapper (proj_in, norm, proj_out)
        # Only process if attention exists
        if len(self.attentions) > 0:
            B, C, H, W = x.shape
            spatial_size = H * W
            residual = x  # Save for residual connection
            
            # SDXL: Apply norm (GroupNorm) before proj_in
            x_norm = self.norm(x)  # [B, C, H, W]
            
            # Convert to [B, H*W, C] (like SDXL's Transformer2DModel)
            x_flat = x_norm.view(B, C, spatial_size).transpose(1, 2)  # [B, H*W, C]
            
            # SDXL: Apply proj_in
            x_flat = self.proj_in(x_flat)  # [B, H*W, C]

            # Attention blocks (process in [B, H*W, C] format - SDXL style)
            for i, attn in enumerate(self.attentions):
                if is_step1:
                    attn_start = time.time()
                    torch.cuda.synchronize()
                
                x_flat = attn(x_flat, context)
                
                if is_step1:
                    torch.cuda.synchronize()
                    attn_time = (time.time() - attn_start) * 1000
                    attn_times.append(attn_time)

            # SDXL: Apply proj_out
            x_flat = self.proj_out(x_flat)  # [B, H*W, C]

            # Convert back to [B, C, H, W] (like SDXL's Transformer2DModel output)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)
            x = x + residual  # Residual connection (like SDXL)
        else:
            # No attention blocks (SDXL style: DownBlock0, UpBlock0)
            if is_step1:
                attn_times = [0.0]  # Empty list for logging

        # Save skip connection before downsampling
        skip = x

        # Downsample
        if is_step1:
            downsample_start = time.time()
            torch.cuda.synchronize()
        
        if self.downsample_conv is not None:
            x = self.downsample_conv(x)
        
        if is_step1:
            torch.cuda.synchronize()
            downsample_time = (time.time() - downsample_start) * 1000 if self.downsample_conv is not None else 0
            total_db_time = (time.time() - db_internal_start) * 1000
            print(f"[DEUS] [DownBlock Internal] Resnets: {sum(resnet_times):.2f}ms ({', '.join([f'{t:.2f}ms' for t in resnet_times])})")
            print(f"[DEUS] [DownBlock Internal] Attentions: {sum(attn_times):.2f}ms ({', '.join([f'{t:.2f}ms' for t in attn_times])})")
            if downsample_time > 0:
                print(f"[DEUS] [DownBlock Internal] Downsample: {downsample_time:.2f}ms")
            print(f"[DEUS] [DownBlock Internal] Total: {total_db_time:.2f}ms")

        return x, skip


class UpBlock(nn.Module):
    """Up-sampling block with sparse skip connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_embed_dim: int,
        context_dim: int,
        num_res_blocks: int = 2,
        num_attention_heads: int = 8,
        transformer_depth: int = 2,
        dropout: float = 0.0,
        upsample: bool = True
    ):
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.transformer_depth = transformer_depth
        self.upsample = upsample

        # Upsample first
        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample_conv = None

        # Combine with skip connection
        combined_channels = in_channels + skip_channels

        # Resnet blocks
        self.resnets = nn.ModuleList([
            ResnetBlock(
                combined_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim,
                dropout
            )
            for i in range(num_res_blocks)
        ])

        # SDXL-style: Transformer2DModel wrapper (proj_in, norm, proj_out)
        # Only add if transformer_depth > 0
        if transformer_depth > 0:
            # SDXL: norm (GroupNorm) before proj_in
            self.norm = nn.GroupNorm(32, out_channels, eps=1e-6)
            # SDXL: proj_in (Linear projection)
            self.proj_in = nn.Linear(out_channels, out_channels, bias=True)
            # SDXL: proj_out (Linear projection)
            self.proj_out = nn.Linear(out_channels, out_channels, bias=True)
            
            # Attention blocks (only if transformer_depth > 0)
            # head_dim is passed explicitly for SDXL compatibility (fixed 64)
            head_dim = 64  # SDXL fixed head_dim
            self.attentions = nn.ModuleList([
                CrossAttentionBlock(
                    out_channels,
                    context_dim,
                    num_attention_heads,
                    head_dim=head_dim,
                    dropout=dropout
                )
                for _ in range(transformer_depth)
            ])
        else:
            # SDXL style: Some blocks have no attention (e.g., UpBlock0)
            self.norm = None
            self.proj_in = None
            self.proj_out = None
            self.attentions = nn.ModuleList([])

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor],
        time_emb: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        import time
        is_step1 = hasattr(self, '_parent_is_step1') and self._parent_is_step1
        
        if is_step1:
            ub_internal_start = time.time()
            torch.cuda.synchronize()
        
        # Upsample
        if is_step1:
            upsample_start = time.time()
            torch.cuda.synchronize()
        
        if self.upsample_conv is not None:
            x = self.upsample_conv(x)
        
        if is_step1:
            torch.cuda.synchronize()
            upsample_time = (time.time() - upsample_start) * 1000 if self.upsample_conv is not None else 0

        # Concatenate with skip connection (if provided)
        if is_step1:
            concat_start = time.time()
            torch.cuda.synchronize()
        
        if skip is not None:
            # Interpolate skip to match x's spatial size (if they don't match)
            # This handles cases where downsampling/upsampling has rounding errors
            if skip.shape[2:] != x.shape[2:]:
                skip = torch.nn.functional.interpolate(
                    skip,
                    size=x.shape[2:],  # Match (H, W) of x
                    mode='nearest'
                )

            x = torch.cat([x, skip], dim=1)
        
        if is_step1:
            torch.cuda.synchronize()
            concat_time = (time.time() - concat_start) * 1000 if skip is not None else 0
            resnet_times = []
            attn_times = []

        # Resnet blocks (process in [B, C, H, W] format)
        for i, resnet in enumerate(self.resnets):
            if is_step1:
                resnet_start = time.time()
                torch.cuda.synchronize()
            
            x = resnet(x, time_emb, context)  # DiC: pass context for conditional gating
            
            if is_step1:
                torch.cuda.synchronize()
                resnet_time = (time.time() - resnet_start) * 1000
                resnet_times.append(resnet_time)

        # SDXL-style: Transformer2DModel wrapper (proj_in, norm, proj_out)
        # Only process if attention exists
        if len(self.attentions) > 0:
            B, C, H, W = x.shape
            spatial_size = H * W
            residual = x  # Save for residual connection
            
            # SDXL: Apply norm (GroupNorm) before proj_in
            x_norm = self.norm(x)  # [B, C, H, W]
            
            # Convert to [B, H*W, C] (like SDXL's Transformer2DModel)
            x_flat = x_norm.view(B, C, spatial_size).transpose(1, 2)  # [B, H*W, C]
            
            # SDXL: Apply proj_in
            x_flat = self.proj_in(x_flat)  # [B, H*W, C]

            # Attention blocks (process in [B, H*W, C] format - SDXL style)
            for i, attn in enumerate(self.attentions):
                if is_step1:
                    attn_start = time.time()
                    torch.cuda.synchronize()
                
                x_flat = attn(x_flat, context)
                
                if is_step1:
                    torch.cuda.synchronize()
                    attn_time = (time.time() - attn_start) * 1000
                    attn_times.append(attn_time)

            # SDXL: Apply proj_out
            x_flat = self.proj_out(x_flat)  # [B, H*W, C]

            # Convert back to [B, C, H, W] (like SDXL's Transformer2DModel output)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)
            x = x + residual  # Residual connection (like SDXL)
        else:
            # No attention blocks (SDXL style: DownBlock0, UpBlock0)
            if is_step1:
                attn_times = [0.0]  # Empty list for logging
        
        if is_step1:
            torch.cuda.synchronize()
            total_ub_time = (time.time() - ub_internal_start) * 1000
            if upsample_time > 0:
                print(f"[DEUS] [UpBlock Internal] Upsample: {upsample_time:.2f}ms")
            if concat_time > 0:
                print(f"[DEUS] [UpBlock Internal] Concat skip: {concat_time:.2f}ms")
            print(f"[DEUS] [UpBlock Internal] Resnets: {sum(resnet_times):.2f}ms ({', '.join([f'{t:.2f}ms' for t in resnet_times])})")
            print(f"[DEUS] [UpBlock Internal] Attentions: {sum(attn_times):.2f}ms ({', '.join([f'{t:.2f}ms' for t in attn_times])})")
            print(f"[DEUS] [UpBlock Internal] Total: {total_ub_time:.2f}ms")

        return x


class DeusUNet(nn.Module):
    """
    DEUS U-Net architecture with multi-modal conditioning.
    (Dual-Embeddings U-Net Structure)

    Features:
    - 4-channel latent input/output (SDXL VAE)
    - SigLIP-2 multi-modal conditioning (text + optional images)
    - RoPE 2D positional encoding
    - Sparse skip connections
    - Improved conv blocks
    """

    def __init__(self, config: Optional[UNetConfig] = None):
        super().__init__()

        if config is None:
            config = UNetConfig.from_variant("medium")

        self.config = config

        # Time embedding
        time_embed_dim = config.time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(config.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # RoPE 2D (Resolution-Adaptive)
        self.rope_2d = RoPE2D(
            dim=config.model_channels,
            train_resolution=128,  # 1024px / 8 (VAE downscaling factor)
            max_resolution=512,    # 4096px / 8 (maximum resolution support)
            base=10000.0
        )

        # Input projection
        self.conv_in = nn.Conv2d(config.in_channels, config.model_channels, kernel_size=3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList()
        in_ch = config.model_channels
        for i, mult in enumerate(config.channel_mult):
            out_ch = config.model_channels * mult
            downsample = i < len(config.channel_mult) - 1
            
            # SDXL style: Different attention heads per block
            num_heads = config.num_attention_heads[i]
            transformer_depth = config.transformer_layers_per_block[i]

            self.down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    time_embed_dim,
                    config.context_dim,
                    config.num_res_blocks,
                    num_heads,
                    transformer_depth,
                    config.dropout,
                    downsample
                )
            )
            in_ch = out_ch

        # Mid block
        mid_ch = config.model_channels * config.channel_mult[-1]
        # SDXL style: Mid block uses 20 heads (last in num_attention_heads) and 10 transformer layers
        mid_num_heads = config.num_attention_heads[-1]  # 20 heads for 1280 channels
        mid_transformer_depth = config.transformer_layers_per_mid_block  # 10 layers
        
        # SDXL-style: Transformer2DModel wrapper for Mid block
        mid_norm = nn.GroupNorm(32, mid_ch, eps=1e-6)
        mid_proj_in = nn.Linear(mid_ch, mid_ch, bias=True)
        mid_proj_out = nn.Linear(mid_ch, mid_ch, bias=True)
        
        mid_attentions = nn.ModuleList([
            CrossAttentionBlock(
                mid_ch,
                config.context_dim,
                mid_num_heads,
                head_dim=config.attention_head_dim,
                dropout=config.dropout
            )
            for _ in range(mid_transformer_depth)
        ])
        
        self.mid_block = nn.ModuleList([
            ResnetBlock(mid_ch, mid_ch, time_embed_dim, config.dropout, context_dim=config.context_dim),  # DiC
            mid_norm,  # SDXL: norm
            mid_proj_in,  # SDXL: proj_in
            mid_attentions,  # Attention blocks
            mid_proj_out,  # SDXL: proj_out
            ResnetBlock(mid_ch, mid_ch, time_embed_dim, config.dropout, context_dim=config.context_dim)  # DiC
        ])

        # Up blocks
        self.up_blocks = nn.ModuleList()
        in_ch = mid_ch
        for i, mult in enumerate(reversed(config.channel_mult)):
            out_ch = config.model_channels * mult
            upsample = i < len(config.channel_mult) - 1

            # Determine skip channels (sparse skip connections)
            down_idx = len(config.channel_mult) - 1 - i
            if down_idx % config.skip_connection_interval == 0:
                skip_ch = config.model_channels * config.channel_mult[down_idx]
            else:
                skip_ch = 0  # No skip connection

            # SDXL style: Different attention heads per block (reversed order)
            # Up blocks: Up0=20heads, Up1=10heads, Up2=5heads (matches Down2, Down1, Down0)
            # But SDXL Up blocks have 0 transformer layers
            up_idx = len(config.channel_mult) - 1 - i
            num_heads = config.num_attention_heads[up_idx]
            transformer_depth = config.transformer_layers_per_up_block[up_idx]  # Use Up-specific setting

            self.up_blocks.append(
                UpBlock(
                    in_ch,
                    out_ch,
                    skip_ch,
                    time_embed_dim,
                    config.context_dim,
                    config.num_res_blocks_per_up_block,  # Use separate setting for UpBlock
                    num_heads,
                    transformer_depth,
                    config.dropout,
                    upsample
                )
            )
            in_ch = out_ch

        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, config.model_channels),
            nn.SiLU(),
            nn.Conv2d(config.model_channels, config.out_channels, kernel_size=3, padding=1)
        )

        print(f"[UNet] DEUS U-Net initialized:")
        print(f"  Variant: {config.variant}")
        print(f"  Model channels: {config.model_channels}")
        print(f"  Channel mult: {config.channel_mult}")
        print(f"  Skip connection interval: {config.skip_connection_interval}")
        print(f"  Attention head dim: {config.attention_head_dim} (SDXL style)")
        print(f"  Attention heads per block: {config.num_attention_heads}")
        print(f"  Transformer layers per block: {config.transformer_layers_per_block}")
        print(f"  Transformer layers (mid block): {config.transformer_layers_per_mid_block}")
        print(f"  Latent channels: {config.in_channels} -> {config.out_channels}")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sample: Noisy latents [batch, 4, height, width]
            timestep: Timesteps [batch] or [1]
            encoder_hidden_states: SigLIP-2 embeddings [batch, seq_len, 1152]

        Returns:
            Predicted noise [batch, 4, height, width]
        """
        import time
        
        # Check if this is Step1 (first forward pass) for detailed profiling
        # Use a simple counter to detect first call
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0
            print(f"[DEUS] [U-Net Internal] Initializing forward count (first call)")
        
        self._forward_count += 1
        is_step1 = (self._forward_count == 1)
        
        # Debug: Always log forward count to verify it's working
        print(f"[DEUS] [U-Net Internal] Forward count: {self._forward_count}, is_step1: {is_step1}")
        
        if is_step1:
            unet_start_time = time.time()
            torch.cuda.synchronize()
            print(f"[DEUS] [U-Net Internal] ========== DETAILED U-NET TIMING ==========")
            print(f"[DEUS] [U-Net Internal] Input shape: {sample.shape}, dtype: {sample.dtype}")
            print(f"[DEUS] [U-Net Internal] Encoder hidden states shape: {encoder_hidden_states.shape}")
        
        # Save input spatial size for final interpolation
        input_size = sample.shape[2:]  # (H, W)

        # Time embedding
        if is_step1:
            time_emb_start = time.time()
            torch.cuda.synchronize()
        
        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)

        t_emb = self.get_timestep_embedding(timestep, self.config.model_channels)
        # Convert to sample dtype (get_timestep_embedding always returns float32)
        t_emb = t_emb.to(dtype=sample.dtype)
        t_emb = self.time_embed(t_emb)
        
        if is_step1:
            torch.cuda.synchronize()
            time_emb_time = (time.time() - time_emb_start) * 1000
            print(f"[DEUS] [U-Net Internal] Time embedding: {time_emb_time:.2f}ms")

        # Input projection
        if is_step1:
            conv_in_start = time.time()
            torch.cuda.synchronize()
        
        x = self.conv_in(sample)
        
        if is_step1:
            torch.cuda.synchronize()
            conv_in_time = (time.time() - conv_in_start) * 1000
            print(f"[DEUS] [U-Net Internal] Input projection (conv_in): {conv_in_time:.2f}ms")

        # Apply RoPE
        if is_step1:
            rope_start = time.time()
            torch.cuda.synchronize()
        
        x = self.rope_2d(x)
        
        if is_step1:
            torch.cuda.synchronize()
            rope_time = (time.time() - rope_start) * 1000
            print(f"[DEUS] [U-Net Internal] RoPE 2D: {rope_time:.2f}ms")

        # Down blocks (with sparse skip connections)
        # Pre-allocate list to avoid dynamic growth
        num_down_blocks = len(self.down_blocks)
        skip_connections = [None] * num_down_blocks
        
        if is_step1:
            down_blocks_start = time.time()
            torch.cuda.synchronize()
            down_block_times = []

        for i, down_block in enumerate(self.down_blocks):
            if is_step1:
                db_start = time.time()
                torch.cuda.synchronize()
                # Pass step1 flag to down_block
                down_block._parent_is_step1 = True
            
            x, skip = down_block(x, t_emb, encoder_hidden_states)

            # Save skip only if interval matches
            if i % self.config.skip_connection_interval == 0:
                skip_connections[i] = skip
            # else: skip_connections[i] remains None
            
            if is_step1:
                torch.cuda.synchronize()
                db_time = (time.time() - db_start) * 1000
                down_block_times.append(db_time)
                print(f"[DEUS] [U-Net Internal] Down block {i}: {db_time:.2f}ms")
                # Clear flag
                if hasattr(down_block, '_parent_is_step1'):
                    delattr(down_block, '_parent_is_step1')
        
        if is_step1:
            torch.cuda.synchronize()
            down_blocks_time = (time.time() - down_blocks_start) * 1000
            print(f"[DEUS] [U-Net Internal] Down blocks total: {down_blocks_time:.2f}ms")

        # Mid block (SDXL style: Resnet -> Attention -> Resnet)
        if is_step1:
            mid_start = time.time()
            torch.cuda.synchronize()
        
        # First ResnetBlock (process in [B, C, H, W] format)
        x = self.mid_block[0](x, t_emb, encoder_hidden_states)  # DiC: pass context
        
        # SDXL-style: Transformer2DModel wrapper (norm, proj_in, proj_out)
        B, C, H, W = x.shape
        spatial_size = H * W
        residual = x  # Save for residual connection
        
        # SDXL: Apply norm (GroupNorm) before proj_in
        mid_norm = self.mid_block[1]
        x_norm = mid_norm(x)  # [B, C, H, W]
        
        # Convert to [B, H*W, C] (like SDXL's Transformer2DModel)
        x_flat = x_norm.view(B, C, spatial_size).transpose(1, 2)  # [B, H*W, C]
        
        # SDXL: Apply proj_in
        mid_proj_in = self.mid_block[2]
        x_flat = mid_proj_in(x_flat)  # [B, H*W, C]
        
        # CrossAttentionBlocks (process in [B, H*W, C] format - SDXL style with multiple layers)
        mid_attentions = self.mid_block[3]  # This is now a ModuleList
        for attn in mid_attentions:
            x_flat = attn(x_flat, encoder_hidden_states)
        
        # SDXL: Apply proj_out
        mid_proj_out = self.mid_block[4]
        x_flat = mid_proj_out(x_flat)  # [B, H*W, C]
        
        # Convert back to [B, C, H, W]
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        x = x + residual  # Residual connection
        
        # Second ResnetBlock (process in [B, C, H, W] format)
        x = self.mid_block[5](x, t_emb, encoder_hidden_states)  # DiC: pass context
        
        if is_step1:
            torch.cuda.synchronize()
            mid_time = (time.time() - mid_start) * 1000
            print(f"[DEUS] [U-Net Internal] Mid block: {mid_time:.2f}ms")

        # Up blocks (with sparse skip connections)
        # Use reversed() iterator to avoid creating new list
        if is_step1:
            up_blocks_start = time.time()
            torch.cuda.synchronize()
            up_block_times = []
        
        for i, (up_block, skip) in enumerate(zip(self.up_blocks, reversed(skip_connections))):
            if is_step1:
                ub_start = time.time()
                torch.cuda.synchronize()
                # Pass step1 flag to up_block
                up_block._parent_is_step1 = True
            
            x = up_block(x, skip, t_emb, encoder_hidden_states)
            
            if is_step1:
                torch.cuda.synchronize()
                ub_time = (time.time() - ub_start) * 1000
                up_block_times.append(ub_time)
                print(f"[DEUS] [U-Net Internal] Up block {i}: {ub_time:.2f}ms")
                # Clear flag
                if hasattr(up_block, '_parent_is_step1'):
                    delattr(up_block, '_parent_is_step1')
        
        if is_step1:
            torch.cuda.synchronize()
            up_blocks_time = (time.time() - up_blocks_start) * 1000
            print(f"[DEUS] [U-Net Internal] Up blocks total: {up_blocks_time:.2f}ms")

        # Output projection
        if is_step1:
            conv_out_start = time.time()
            torch.cuda.synchronize()
        
        x = self.conv_out(x)
        
        if is_step1:
            torch.cuda.synchronize()
            conv_out_time = (time.time() - conv_out_start) * 1000
            total_unet_time = (time.time() - unet_start_time) * 1000
            print(f"[DEUS] [U-Net Internal] Output projection (conv_out): {conv_out_time:.2f}ms")
            print(f"[DEUS] [U-Net Internal] ========== U-NET TOTAL TIME: {total_unet_time:.2f}ms ==========")
            print(f"[DEUS] [U-Net Internal] Breakdown:")
            print(f"  - Time embedding: {time_emb_time:.2f}ms ({time_emb_time/total_unet_time*100:.1f}%)")
            print(f"  - Input projection: {conv_in_time:.2f}ms ({conv_in_time/total_unet_time*100:.1f}%)")
            print(f"  - RoPE 2D: {rope_time:.2f}ms ({rope_time/total_unet_time*100:.1f}%)")
            print(f"  - Down blocks: {down_blocks_time:.2f}ms ({down_blocks_time/total_unet_time*100:.1f}%)")
            print(f"  - Mid block: {mid_time:.2f}ms ({mid_time/total_unet_time*100:.1f}%)")
            print(f"  - Up blocks: {up_blocks_time:.2f}ms ({up_blocks_time/total_unet_time*100:.1f}%)")
            print(f"  - Output projection: {conv_out_time:.2f}ms ({conv_out_time/total_unet_time*100:.1f}%)")
            print()

        # Ensure output matches input spatial size (fix upsampling rounding errors)
        # This follows the Diffusers approach of enforcing exact size match
        if x.shape[2:] != input_size:
            x = torch.nn.functional.interpolate(
                x,
                size=input_size,
                mode='nearest'
            )

        return x

    @staticmethod
    def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if embedding_dim % 2 == 1:  # Zero pad if odd
            emb = torch.nn.functional.pad(emb, (0, 1))

        return emb


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    for variant in ["small", "medium", "large"]:
        config = UNetConfig.from_variant(variant)
        model = DeusUNet(config)
        print(f"\n{variant.upper()} model: {count_parameters(model) / 1e9:.2f}B parameters")
