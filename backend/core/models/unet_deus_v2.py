"""
DEUS U-Net Architecture v2 (SDXL-Compatible Rewrite)

Complete rewrite based on diffusers UNet2DConditionModel to achieve SDXL parameter parity.

Key changes from v1:
- Added layers_per_block parameter (SDXL standard: 2)
- Each down/up block creates multiple (ResNet + Transformer) pairs
- Proper skip connection handling (tuple-based, matching diffusers)
- Expected parameters: ~2.6B (matching SDXL Base)

Architecture:
- Input: 4-channel latents (SDXL VAE)
- Conditioning: SigLIP-2 embeddings (1152-dim, variable length)
- Down blocks: 3 logical blocks with layers_per_block=2 each
  - Block 0: 320ch, no attention, 2 ResNet layers
  - Block 1: 640ch, 2 transformer layers, 2 (ResNet + Transformer) pairs
  - Block 2: 1280ch, 10 transformer layers, 2 (ResNet + Transformer) pairs
- Mid block: 1280ch, ResNet + Transformer(10 layers) + ResNet
- Up blocks: 3 logical blocks (mirror of down blocks)
  - Block 0: 1280ch, 10 transformer layers, 3 (ResNet + Transformer) pairs
  - Block 1: 640ch, 2 transformer layers, 3 (ResNet + Transformer) pairs
  - Block 2: 320ch, no attention, 3 ResNet layers
- Output: 4-channel latents

Differences from SDXL:
- RoPE 2D positional encoding (from Z-Image)
- SigLIP-2 conditioning (1152-dim) instead of dual CLIP encoders (2048-dim)
- No added_cond_kwargs (time_ids, text_embeds) - using RoPE for positional info
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.resnet import ResnetBlock2D, Downsample2D, Upsample2D


@dataclass
class UNetConfig:
    """U-Net configuration (SDXL-compatible)."""

    # Latent specs
    in_channels: int = 4  # SDXL VAE latent channels
    out_channels: int = 4

    # Model size (SDXL standard)
    block_out_channels: Tuple[int, ...] = (320, 640, 1280)  # Channel progression
    layers_per_block: int = 2  # SDXL standard: 2 ResNet layers per block
    layers_per_up_block: int = 3  # SDXL up blocks have 3 ResNet layers

    # Skip connections (DEUS original: sparse skip for memory efficiency)
    # Explicitly specify which down blocks output skip connections (by index)
    skip_connection_blocks: Tuple[int, ...] = (0, 1, 2)  # All down blocks output 1 skip each
    # Number of skip connections each up block receives (must match total skips from down blocks)
    skip_connections_per_up_block: Tuple[int, ...] = (1, 1, 1)  # Each up block receives 1 skip

    # Attention (SDXL style: block-specific settings)
    attention_head_dim: int = 64  # SDXL fixed at 64-dim per head
    num_attention_heads: Tuple[int, ...] = (5, 10, 20)  # Attention heads per block
    transformer_layers_per_block: Tuple[int, ...] = (1, 2, 10)  # Transformer layers: Block0=1, Block1=2, Block2=10
    transformer_layers_per_mid_block: int = 10  # SDXL: Mid block has 10 layers
    context_dim: int = 1152  # SigLIP-2 hidden size

    # Time embedding
    time_embed_dim: int = 1280  # Time embedding dimension

    # Dropout and other settings
    dropout: float = 0.0
    resnet_eps: float = 1e-6
    resnet_act_fn: str = "silu"
    resnet_groups: int = 32

    # Model variant
    variant: str = "medium"

    @property
    def latent_channels(self) -> int:
        """Latent channels (for compatibility with checkpoint_utils)."""
        return self.in_channels

    @classmethod
    def from_variant(cls, variant: str = "medium"):
        """Create config from variant name.

        SDXL-compatible variants:
        - medium: ~2.6B params (SDXL Base equivalent)
        """
        if variant == "medium":
            # SDXL Base equivalent
            return cls(
                block_out_channels=(320, 640, 1280),
                layers_per_block=2,
                layers_per_up_block=3,
                attention_head_dim=64,
                num_attention_heads=(5, 10, 20),
                transformer_layers_per_block=(1, 2, 10),
                transformer_layers_per_mid_block=10,
                variant=variant
            )
        else:
            raise ValueError(f"Unknown variant: {variant}. Only 'medium' is supported.")


class RoPE2D(nn.Module):
    """
    Resolution-Adaptive 2D Rotary Position Embedding.
    Applies resolution-adaptive rotary embeddings to spatial dimensions (H, W) of latents.
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
        self.train_resolution = train_resolution
        self.max_resolution = max_resolution
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Output tensor [B, C, H, W] with RoPE applied
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # Resolution ratio for adaptive scaling
        resolution_ratio = max(H, W) / self.train_resolution
        adjusted_base = self.base * (resolution_ratio ** (self.dim / (self.dim - 2)))

        # Create frequency bands
        half_dim = C // 2
        freqs = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32, device=device) *
            (-math.log(adjusted_base) / half_dim)
        )

        # Position indices (normalized by resolution ratio)
        h_pos = torch.arange(H, dtype=torch.float32, device=device) / resolution_ratio
        w_pos = torch.arange(W, dtype=torch.float32, device=device) / resolution_ratio

        # Compute rotary embeddings
        h_freqs = torch.outer(h_pos, freqs)  # [H, half_dim]
        w_freqs = torch.outer(w_pos, freqs)  # [W, half_dim]

        # Combine height and width frequencies
        h_emb = torch.cat([torch.sin(h_freqs), torch.cos(h_freqs)], dim=-1)  # [H, C]
        w_emb = torch.cat([torch.sin(w_freqs), torch.cos(w_freqs)], dim=-1)  # [W, C]

        # Broadcast to [B, C, H, W]
        h_emb = h_emb.t()[None, :, :, None]  # [H, C] -> [C, H] -> [1, C, H, 1]
        w_emb = w_emb.t()[None, :, None, :]  # [W, C] -> [C, W] -> [1, C, 1, W]

        # Apply rotary embedding (element-wise multiplication)
        x = x.to(torch.float32)  # Convert to float32 for sin/cos operations
        h_emb = h_emb.to(torch.float32)
        w_emb = w_emb.to(torch.float32)

        x = x * torch.cos(h_emb) * torch.cos(w_emb) + \
            x * torch.sin(h_emb) * torch.sin(w_emb)

        return x.to(dtype)


class CrossAttnDownBlock2D(nn.Module):
    """
    Down block with cross-attention (SDXL-compatible).
    Creates multiple (ResNet + Transformer) pairs based on num_layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: int = 1,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 1152,
        add_downsample: bool = True,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        # Expand transformer_layers_per_block to list if needed
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels_layer = in_channels if i == 0 else out_channels

            # ResNet block
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels_layer,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=1.0,
                    pre_norm=True,
                )
            )

            # Transformer block
            attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=True,
                    only_cross_attention=False,
                    upcast_attention=False,
                    attention_type="default",
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        # Downsample at the end of the block
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=1, name="op")
            ])
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Returns:
            (output, output_states) - output_states is a tuple of all intermediate outputs
        """
        output_states = ()

        # Process each (ResNet + Transformer) pair
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing for ResNet
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet,
                    hidden_states,
                    temb,
                    use_reentrant=False
                )
            else:
                hidden_states = resnet(hidden_states, temb)

            # Transformer (has its own gradient checkpointing)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

            output_states = output_states + (hidden_states,)

        # Downsample
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    """
    Down block without cross-attention (SDXL-compatible).
    Used for first block (320ch, no attention).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        add_downsample: bool = True,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
    ):
        super().__init__()

        resnets = []

        for i in range(num_layers):
            in_channels_layer = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels_layer,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=1.0,
                    pre_norm=True,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        # Downsample at the end of the block
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=1, name="op")
            ])
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Returns:
            (output, output_states) - output_states is a tuple of all intermediate outputs
        """
        output_states = ()

        # Process each ResNet
        for resnet in self.resnets:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet,
                    hidden_states,
                    temb,
                    use_reentrant=False
                )
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states = output_states + (hidden_states,)

        # Downsample
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    """
    Up block with cross-attention (SDXL-compatible).
    Creates multiple (ResNet + Transformer) pairs based on num_layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 3,
        num_skip_connections: int = 3,  # Number of skip connections this block receives
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: int = 1,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 1152,
        add_upsample: bool = True,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.num_skip_connections = num_skip_connections

        # Expand transformer_layers_per_block to list if needed
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        resnets = []
        attentions = []

        for i in range(num_layers):
            # Determine if this layer receives skip connection
            # Skips are assigned from the END (last layer gets first skip)
            # For num_skip_connections=2, num_layers=3:
            #   i=0: no skip (layer 0 from end = 2 >= 2)
            #   i=1: skip (layer 1 from end = 1 < 2)
            #   i=2: skip (layer 2 from end = 0 < 2)
            layer_from_end = num_layers - 1 - i
            has_skip = layer_from_end < num_skip_connections

            # Base ResNet in_channels (before concatenating skip)
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # Skip connection channels
            if has_skip:
                # Skip comes from down block → same as this up block's output channels
                res_skip_channels = out_channels
            else:
                res_skip_channels = 0

            # ResNet block
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=1.0,
                    pre_norm=True,
                )
            )

            # Transformer block
            attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=True,
                    only_cross_attention=False,
                    upcast_attention=False,
                    attention_type="default",
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        # Upsample at the end of the block
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
            ])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Current hidden states
            res_hidden_states_tuple: Skip connections from down blocks
            temb: Time embedding
            encoder_hidden_states: Cross-attention conditioning
        """
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            # Check if this layer receives skip connection (last num_skip_connections layers get skips)
            layer_from_end = len(self.resnets) - 1 - i
            has_skip = layer_from_end < self.num_skip_connections

            if has_skip and len(res_hidden_states_tuple) > 0:
                # Pop skip connection
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # Concatenate skip connection
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # ResNet
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet,
                    hidden_states,
                    temb,
                    use_reentrant=False
                )
            else:
                hidden_states = resnet(hidden_states, temb)

            # Transformer (has its own gradient checkpointing)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

        # Upsample at the END (for next up block)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):
    """
    Up block without cross-attention (SDXL-compatible).
    Used for last block (320ch, no attention).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 3,
        num_skip_connections: int = 0,  # Number of skip connections this block receives
        add_upsample: bool = True,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
    ):
        super().__init__()

        self.num_skip_connections = num_skip_connections

        resnets = []

        for i in range(num_layers):
            # Determine if this layer receives skip connection
            # Skips are assigned from the END (last layer gets first skip)
            layer_from_end = num_layers - 1 - i
            has_skip = layer_from_end < num_skip_connections

            # Base ResNet in_channels (before concatenating skip)
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # Skip connection channels
            if has_skip:
                # Skip comes from down block → same as this up block's output channels
                res_skip_channels = out_channels
            else:
                res_skip_channels = 0

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm="default",
                    non_linearity=resnet_act_fn,
                    output_scale_factor=1.0,
                    pre_norm=True,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        # Upsample at the end of the block
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
            ])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Current hidden states
            res_hidden_states_tuple: Skip connections from down blocks
            temb: Time embedding
        """
        for i, resnet in enumerate(self.resnets):
            # Check if this layer receives skip connection (last num_skip_connections layers get skips)
            layer_from_end = len(self.resnets) - 1 - i
            has_skip = layer_from_end < self.num_skip_connections

            if has_skip and len(res_hidden_states_tuple) > 0:
                # Pop skip connection
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # Concatenate skip connection
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # ResNet
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet,
                    hidden_states,
                    temb,
                    use_reentrant=False
                )
            else:
                hidden_states = resnet(hidden_states, temb)

        # Upsample at the END (for next up block)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class DeusUNet(nn.Module):
    """
    DEUS U-Net v2: SDXL-compatible architecture with RoPE 2D and SigLIP-2 conditioning.

    Expected parameters: ~2.6B (matching SDXL Base)
    """

    def __init__(self, config: Optional[UNetConfig] = None):
        super().__init__()

        if config is None:
            config = UNetConfig.from_variant("medium")

        self.config = config

        # Time embedding
        time_embed_dim = config.time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(config.block_out_channels[0], time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # RoPE 2D
        self.rope_2d = RoPE2D(
            dim=config.block_out_channels[0],
            train_resolution=128,
            max_resolution=512,
            base=10000.0
        )

        # Input projection
        self.conv_in = nn.Conv2d(config.in_channels, config.block_out_channels[0], kernel_size=3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList()

        # Down block 0: 320ch, no attention
        self.down_blocks.append(
            DownBlock2D(
                in_channels=config.block_out_channels[0],
                out_channels=config.block_out_channels[0],
                temb_channels=time_embed_dim,
                dropout=config.dropout,
                num_layers=config.layers_per_block,
                add_downsample=True,
                resnet_eps=config.resnet_eps,
                resnet_act_fn=config.resnet_act_fn,
                resnet_groups=config.resnet_groups,
            )
        )

        # Down blocks 1-2: 640ch, 1280ch with attention
        for i in range(1, len(config.block_out_channels)):
            self.down_blocks.append(
                CrossAttnDownBlock2D(
                    in_channels=config.block_out_channels[i - 1],
                    out_channels=config.block_out_channels[i],
                    temb_channels=time_embed_dim,
                    dropout=config.dropout,
                    num_layers=config.layers_per_block,
                    transformer_layers_per_block=config.transformer_layers_per_block[i],
                    num_attention_heads=config.num_attention_heads[i],
                    attention_head_dim=config.attention_head_dim,
                    cross_attention_dim=config.context_dim,
                    add_downsample=(i < len(config.block_out_channels) - 1),
                    resnet_eps=config.resnet_eps,
                    resnet_act_fn=config.resnet_act_fn,
                    resnet_groups=config.resnet_groups,
                )
            )

        # Mid block
        mid_channels = config.block_out_channels[-1]
        self.mid_block = nn.ModuleList([
            ResnetBlock2D(
                in_channels=mid_channels,
                out_channels=mid_channels,
                temb_channels=time_embed_dim,
                eps=config.resnet_eps,
                groups=config.resnet_groups,
                dropout=config.dropout,
                time_embedding_norm="default",
                non_linearity=config.resnet_act_fn,
                output_scale_factor=1.0,
                pre_norm=True,
            ),
            Transformer2DModel(
                num_attention_heads=config.num_attention_heads[-1],
                attention_head_dim=config.attention_head_dim,
                in_channels=mid_channels,
                num_layers=config.transformer_layers_per_mid_block,
                cross_attention_dim=config.context_dim,
                norm_num_groups=config.resnet_groups,
                use_linear_projection=True,
                only_cross_attention=False,
                upcast_attention=False,
                attention_type="default",
            ),
            ResnetBlock2D(
                in_channels=mid_channels,
                out_channels=mid_channels,
                temb_channels=time_embed_dim,
                eps=config.resnet_eps,
                groups=config.resnet_groups,
                dropout=config.dropout,
                time_embedding_norm="default",
                non_linearity=config.resnet_act_fn,
                output_scale_factor=1.0,
                pre_norm=True,
            ),
        ])

        # Up blocks
        self.up_blocks = nn.ModuleList()
        reversed_block_out_channels = list(reversed(config.block_out_channels))
        reversed_num_attention_heads = list(reversed(config.num_attention_heads))
        reversed_transformer_layers = list(reversed(config.transformer_layers_per_block))

        # Up blocks 0-1: 1280ch, 640ch with attention
        # IMPORTANT: DEUS maintains sparse skip connections
        for i in range(len(config.block_out_channels) - 1):
            in_channels = reversed_block_out_channels[i]
            out_channels = in_channels  # Maintain channel size within up block

            # prev_output_channel: output from previous block
            # up_blocks[0]: from mid_block (1280ch)
            # up_blocks[1]: from up_blocks[0] (1280ch)
            if i == 0:
                prev_output_channel = reversed_block_out_channels[0]  # mid_block output (1280ch)
            else:
                prev_output_channel = reversed_block_out_channels[i - 1]  # Previous up block output

            self.up_blocks.append(
                CrossAttnUpBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    dropout=config.dropout,
                    num_layers=config.layers_per_up_block,
                    num_skip_connections=config.skip_connections_per_up_block[i],
                    transformer_layers_per_block=reversed_transformer_layers[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    attention_head_dim=config.attention_head_dim,
                    cross_attention_dim=config.context_dim,
                    add_upsample=True,
                    resnet_eps=config.resnet_eps,
                    resnet_act_fn=config.resnet_act_fn,
                    resnet_groups=config.resnet_groups,
                )
            )

        # Up block 2: 320ch, no attention
        # prev_output_channel: from up_blocks[1] (640ch)
        self.up_blocks.append(
            UpBlock2D(
                in_channels=reversed_block_out_channels[-2],  # 640ch
                out_channels=reversed_block_out_channels[-1],  # 320ch
                prev_output_channel=reversed_block_out_channels[-2],  # Previous up block output (640ch)
                temb_channels=time_embed_dim,
                dropout=config.dropout,
                num_layers=config.layers_per_up_block,
                num_skip_connections=config.skip_connections_per_up_block[-1],
                add_upsample=False,
                resnet_eps=config.resnet_eps,
                resnet_act_fn=config.resnet_act_fn,
                resnet_groups=config.resnet_groups,
            )
        )

        # Output projection
        self.conv_norm_out = nn.GroupNorm(32, config.block_out_channels[0])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(config.block_out_channels[0], config.out_channels, kernel_size=3, padding=1)

        # Gradient checkpointing flag
        self._gradient_checkpointing = False

        print(f"[UNet] DEUS U-Net v2 (SDXL-compatible) initialized:")
        print(f"  Block out channels: {config.block_out_channels}")
        print(f"  Layers per block: {config.layers_per_block}")
        print(f"  Layers per up block: {config.layers_per_up_block}")
        print(f"  Transformer layers: {config.transformer_layers_per_block}")
        print(f"  Attention head dim: {config.attention_head_dim}")
        print(f"  Attention heads: {config.num_attention_heads}")

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient training."""
        self._gradient_checkpointing = True

        # Enable gradient checkpointing for all blocks
        for down_block in self.down_blocks:
            down_block.gradient_checkpointing = True
            # Transformer2DModel needs _set_gradient_checkpointing
            if hasattr(down_block, 'attentions'):
                for attn in down_block.attentions:
                    if hasattr(attn, '_set_gradient_checkpointing'):
                        attn._set_gradient_checkpointing(value=True)
                    else:
                        attn.gradient_checkpointing = True

        # Mid block transformer
        mid_attn = self.mid_block[1]
        if hasattr(mid_attn, '_set_gradient_checkpointing'):
            mid_attn._set_gradient_checkpointing(value=True)
        else:
            mid_attn.gradient_checkpointing = True

        for up_block in self.up_blocks:
            up_block.gradient_checkpointing = True
            if hasattr(up_block, 'attentions'):
                for attn in up_block.attentions:
                    if hasattr(attn, '_set_gradient_checkpointing'):
                        attn._set_gradient_checkpointing(value=True)
                    else:
                        attn.gradient_checkpointing = True

        print(f"[UNet] DEUS U-Net v2: Gradient checkpointing enabled")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor
    ):
        """
        Args:
            sample: Noisy latents [batch, 4, height, width]
            timestep: Timesteps [batch] or scalar
            encoder_hidden_states: SigLIP-2 embeddings [batch, seq_len, 1152]

        Returns:
            Predicted noise [batch, 4, height, width]
        """
        # Time embedding
        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)

        t_emb = self.get_timestep_embedding(timestep, self.config.block_out_channels[0])
        t_emb = t_emb.to(dtype=sample.dtype)
        t_emb = self.time_embed(t_emb)

        # Input projection
        x = self.conv_in(sample)

        # Apply RoPE
        x = self.rope_2d(x)

        # Down blocks (sparse skip: only specified blocks output skip connections)
        down_block_res_samples = ()

        for i, down_block in enumerate(self.down_blocks):
            if hasattr(down_block, 'has_cross_attention') and down_block.has_cross_attention:
                x, res_samples = down_block(x, t_emb, encoder_hidden_states)
            else:
                x, res_samples = down_block(x, t_emb)

            # Only collect skip connections from specified blocks
            # Each block outputs only the LAST ResNet output (before downsample)
            # For layers_per_block=2: res_samples = [resnet0, resnet1, downsample_out]
            # We want resnet1 (last ResNet before downsample)
            if i in self.config.skip_connection_blocks:
                # Take second-to-last (last ResNet output, before downsample)
                # If no downsample, take last
                if len(res_samples) > 2:
                    down_block_res_samples += (res_samples[-2],)  # Last ResNet (before downsample)
                else:
                    down_block_res_samples += (res_samples[-1],)  # Last output (no downsample)

        # Mid block
        x = self.mid_block[0](x, t_emb)
        x = self.mid_block[1](x, encoder_hidden_states, return_dict=False)[0]
        x = self.mid_block[2](x, t_emb)

        # Up blocks (sparse skip: each block specifies how many skips it needs)
        for up_block in self.up_blocks:
            # Get skip connections for this up block
            num_skips = up_block.num_skip_connections
            res_samples = down_block_res_samples[-num_skips:] if num_skips > 0 else ()
            down_block_res_samples = down_block_res_samples[:-num_skips] if num_skips > 0 else down_block_res_samples

            if hasattr(up_block, 'has_cross_attention') and up_block.has_cross_attention:
                x = up_block(x, res_samples, t_emb, encoder_hidden_states)
            else:
                x = up_block(x, res_samples, t_emb)

        # Output projection
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        # Return as namespace with .sample attribute for diffusers compatibility
        class UNetOutput:
            def __init__(self, sample):
                self.sample = sample

        return UNetOutput(x)

    @staticmethod
    def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        return emb


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    config = UNetConfig.from_variant("medium")
    model = DeusUNet(config)
    params = count_parameters(model)
    print(f"\nDEUS U-Net v2 (medium): {params / 1e9:.2f}B parameters")
    print(f"Expected: ~2.6B parameters (SDXL Base equivalent)")
