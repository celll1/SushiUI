"""
DEUS U-Net Architecture (Dual-Embeddings U-Net Structure)

Based on SDXL U-Net with modifications:
- RoPE 2D positional encoding (from Z-Image)
- Sparse skip connections (every N blocks instead of every block)
- Improved Conv blocks
- 16-channel latent input (FLUX VAE)
- Multi-modal conditioning (SigLIP-2 text + optional images)

Architecture:
- Input: 16-channel latents
- Conditioning: SigLIP-2 embeddings (1152-dim, variable length)
- Down blocks: 3 stages with sparse skip connections
- Mid block: Attention + FFN
- Up blocks: 3 stages with sparse skip connections from down blocks
- Output: 16-channel latents

Model sizes:
- Small: ~1.5B params
- Medium: ~2.8B params (default)
- Large: ~4.0B params
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class UNetConfig:
    """U-Net configuration."""

    # Latent specs
    in_channels: int = 16  # FLUX VAE latent channels
    out_channels: int = 16

    # Model size
    model_channels: int = 320  # Base channel count
    channel_mult: Tuple[int, ...] = (1, 2, 4)  # Channel multipliers per stage
    num_res_blocks: int = 2  # Residual blocks per stage

    # Attention
    num_attention_heads: int = 20  # Attention heads
    transformer_depth: int = 2  # Transformer blocks per attention layer
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
                # Target: ~1.5B params
                "model_channels": 320,
                "channel_mult": (1, 2, 4, 4),  # 320, 640, 1280, 1280
                "num_res_blocks": 2,
                "num_attention_heads": 16,
                "transformer_depth": 6,
            },
            "medium": {
                # Target: ~2.8B params (SDXL-like)
                # Matches SDXL Base structure
                "model_channels": 384,
                "channel_mult": (1, 2, 4, 4),  # 384, 768, 1536, 1536
                "num_res_blocks": 2,
                "num_attention_heads": 24,  # 384 / 24 = 16 (head_dim)
                "transformer_depth": 10,  # SDXL has 10 transformer layers
            },
            "large": {
                # Target: ~4.0B params
                # Wider than SDXL
                "model_channels": 448,
                "channel_mult": (1, 2, 4, 4),  # 448, 896, 1792, 1792
                "num_res_blocks": 3,
                "num_attention_heads": 28,  # 448 / 28 = 16 (head_dim)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply resolution-adaptive 2D RoPE to input tensor.

        Args:
            x: Input [batch, channels, height, width]

        Returns:
            x with RoPE applied [batch, channels, height, width]
        """
        B, C, H, W = x.shape

        # Resolution scaling factors (normalize by training resolution)
        scale_h = H / self.train_resolution
        scale_w = W / self.train_resolution

        # Normalized position indices (relative to training resolution)
        # At 2x training resolution: [0, 0.5, 1.0, 1.5, ..., 127.5]
        # At 1x training resolution: [0, 1, 2, ..., 127]
        # At 0.5x training resolution: [0, 2, 4, ..., 126]
        pos_h = torch.arange(H, device=x.device, dtype=torch.float32) / scale_h
        pos_w = torch.arange(W, device=x.device, dtype=torch.float32) / scale_w

        # Compute sinusoidal embeddings
        # For height dimension
        freqs_h = torch.einsum("i,j->ij", pos_h, self.inv_freq)  # [H, dim//2]
        emb_h = torch.cat([freqs_h.sin(), freqs_h.cos()], dim=-1)  # [H, dim]

        # For width dimension
        freqs_w = torch.einsum("i,j->ij", pos_w, self.inv_freq)  # [W, dim//2]
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

        # Reshape to [1, C, H, W] and add to input
        emb_2d = emb_2d.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        return x + emb_2d


class ImprovedConvBlock(nn.Module):
    """
    Improved Convolutional Block.

    Enhancements:
    - GroupNorm instead of BatchNorm
    - SiLU activation
    - Residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.act(x)

        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention block for multi-modal conditioning.

    Attends to SigLIP-2 text/image embeddings.
    """

    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()

        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads

        # Layer norm
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention (to conditioning)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            dropout=dropout,
            batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )

        self.norm3 = nn.LayerNorm(channels)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Latent features [batch, channels, height, width]
            context: Conditioning [batch, seq_len, context_dim]

        Returns:
            Updated features [batch, channels, height, width]
        """
        B, C, H, W = x.shape

        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)

        # Self-attention
        x_norm = self.norm1(x_flat)
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + x_attn

        # Cross-attention
        x_norm = self.norm2(x_flat)
        x_cross, _ = self.cross_attn(x_norm, context, context)
        x_flat = x_flat + x_cross

        # FFN
        x_norm = self.norm3(x_flat)
        x_ffn = self.ffn(x_norm)
        x_flat = x_flat + x_ffn

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        x = x_flat.permute(0, 2, 1).view(B, C, H, W)

        return x


class ResnetBlock(nn.Module):
    """
    Residual block with time embedding and improved conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()

        self.conv_block = ImprovedConvBlock(in_channels, out_channels, dropout=dropout)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [batch, in_channels, height, width]
            time_emb: Time embedding [batch, time_embed_dim]

        Returns:
            Output [batch, out_channels, height, width]
        """
        # Apply conv block
        x = self.conv_block(x)

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

        # Resnet blocks
        self.resnets = nn.ModuleList([
            ResnetBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim,
                dropout
            )
            for i in range(num_res_blocks)
        ])

        # Attention blocks
        self.attentions = nn.ModuleList([
            CrossAttentionBlock(
                out_channels,
                context_dim,
                num_attention_heads,
                dropout
            )
            for _ in range(transformer_depth)
        ])

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
        # Resnet blocks
        for resnet in self.resnets:
            x = resnet(x, time_emb)

        # Attention blocks
        for attn in self.attentions:
            x = attn(x, context)

        # Save skip connection before downsampling
        skip = x

        # Downsample
        if self.downsample_conv is not None:
            x = self.downsample_conv(x)

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

        # Attention blocks
        self.attentions = nn.ModuleList([
            CrossAttentionBlock(
                out_channels,
                context_dim,
                num_attention_heads,
                dropout
            )
            for _ in range(transformer_depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor],
        time_emb: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        # Upsample
        if self.upsample_conv is not None:
            x = self.upsample_conv(x)

        # Concatenate with skip connection (if provided)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        # Resnet blocks
        for resnet in self.resnets:
            x = resnet(x, time_emb)

        # Attention blocks
        for attn in self.attentions:
            x = attn(x, context)

        return x


class DeusUNet(nn.Module):
    """
    DEUS U-Net architecture with multi-modal conditioning.
    (Dual-Embeddings U-Net Structure)

    Features:
    - 16-channel latent input/output (FLUX VAE)
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

            self.down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    time_embed_dim,
                    config.context_dim,
                    config.num_res_blocks,
                    config.num_attention_heads,
                    config.transformer_depth,
                    config.dropout,
                    downsample
                )
            )
            in_ch = out_ch

        # Mid block
        mid_ch = config.model_channels * config.channel_mult[-1]
        self.mid_block = nn.ModuleList([
            ResnetBlock(mid_ch, mid_ch, time_embed_dim, config.dropout),
            CrossAttentionBlock(mid_ch, config.context_dim, config.num_attention_heads, config.dropout),
            ResnetBlock(mid_ch, mid_ch, time_embed_dim, config.dropout)
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

            self.up_blocks.append(
                UpBlock(
                    in_ch,
                    out_ch,
                    skip_ch,
                    time_embed_dim,
                    config.context_dim,
                    config.num_res_blocks,
                    config.num_attention_heads,
                    config.transformer_depth,
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
        print(f"  Attention heads: {config.num_attention_heads}")
        print(f"  Transformer depth: {config.transformer_depth}")
        print(f"  Latent channels: {config.in_channels} -> {config.out_channels}")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sample: Noisy latents [batch, 16, height, width]
            timestep: Timesteps [batch] or [1]
            encoder_hidden_states: SigLIP-2 embeddings [batch, seq_len, 1152]

        Returns:
            Predicted noise [batch, 16, height, width]
        """
        # Time embedding
        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)

        t_emb = self.get_timestep_embedding(timestep, self.config.model_channels)
        t_emb = self.time_embed(t_emb)

        # Input projection
        x = self.conv_in(sample)

        # Apply RoPE
        x = self.rope_2d(x)

        # Down blocks (with sparse skip connections)
        skip_connections = []
        for i, down_block in enumerate(self.down_blocks):
            x, skip = down_block(x, t_emb, encoder_hidden_states)

            # Save skip only if interval matches
            if i % self.config.skip_connection_interval == 0:
                skip_connections.append(skip)
            else:
                skip_connections.append(None)

        # Mid block
        for layer in self.mid_block:
            if isinstance(layer, CrossAttentionBlock):
                x = layer(x, encoder_hidden_states)
            else:
                x = layer(x, t_emb)

        # Up blocks (with sparse skip connections)
        skip_connections = list(reversed(skip_connections))
        for up_block, skip in zip(self.up_blocks, skip_connections):
            x = up_block(x, skip, t_emb, encoder_hidden_states)

        # Output projection
        x = self.conv_out(x)

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
