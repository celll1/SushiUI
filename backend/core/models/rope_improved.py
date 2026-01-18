"""
Improved RoPE 2D Implementations

Provides resolution-adaptive and frequency-adaptive RoPE variants
for better extrapolation to different resolutions.

References:
- RoFormer: https://arxiv.org/abs/2104.09864
- NTK-Aware Scaled RoPE: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RoPE2D(nn.Module):
    """
    Standard 2D Rotary Position Embedding (baseline).

    Applies sinusoidal positional encoding to spatial dimensions (H, W) of latents.
    """

    def __init__(self, dim: int, max_resolution: int = 256, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_resolution = max_resolution
        self.base = base

        # Precompute frequency bands: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D RoPE to input tensor.

        Args:
            x: Input [batch, channels, height, width]

        Returns:
            x with RoPE applied [batch, channels, height, width]
        """
        B, C, H, W = x.shape

        # Generate position indices
        pos_h = torch.arange(H, device=x.device, dtype=torch.float32)
        pos_w = torch.arange(W, device=x.device, dtype=torch.float32)

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

        # Combine (simple addition)
        emb_2d = emb_h + emb_w  # [H, W, dim]

        # Match channel count (repeat or slice)
        if self.dim < C:
            emb_2d = emb_2d.repeat(1, 1, (C // self.dim) + 1)[:, :, :C]
        else:
            emb_2d = emb_2d[:, :, :C]

        # Reshape to [1, C, H, W] and add to input
        emb_2d = emb_2d.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        return x + emb_2d


class ResolutionAdaptiveRoPE2D(nn.Module):
    """
    Resolution-Adaptive RoPE with position normalization.

    Key idea: Normalize position indices by resolution ratio to maintain
    consistent positional information across different resolutions.

    This allows the model to:
    - Generate at resolutions different from training
    - Maintain consistent behavior at integer multiples of training resolution
    - Smoothly interpolate for arbitrary resolutions
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

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply resolution-adaptive RoPE.

        Args:
            x: Input [batch, channels, height, width]

        Returns:
            x with RoPE applied [batch, channels, height, width]
        """
        B, C, H, W = x.shape

        # Compute resolution scaling factors
        scale_h = H / self.train_resolution
        scale_w = W / self.train_resolution

        # Normalized position indices (relative to training resolution)
        # At 2x training resolution: [0, 0.5, 1.0, 1.5, ..., 127.5]
        # At 1x training resolution: [0, 1, 2, ..., 127]
        # At 0.5x training resolution: [0, 2, 4, ..., 126]
        pos_h = torch.arange(H, device=x.device, dtype=torch.float32) / scale_h
        pos_w = torch.arange(W, device=x.device, dtype=torch.float32) / scale_w

        # Compute sinusoidal embeddings
        freqs_h = torch.einsum("i,j->ij", pos_h, self.inv_freq)  # [H, dim//2]
        emb_h = torch.cat([freqs_h.sin(), freqs_h.cos()], dim=-1)  # [H, dim]

        freqs_w = torch.einsum("i,j->ij", pos_w, self.inv_freq)  # [W, dim//2]
        emb_w = torch.cat([freqs_w.sin(), freqs_w.cos()], dim=-1)  # [W, dim]

        # Expand to 2D grid
        emb_h = emb_h.unsqueeze(1).expand(-1, W, -1)  # [H, W, dim]
        emb_w = emb_w.unsqueeze(0).expand(H, -1, -1)  # [H, W, dim]

        # Combine
        emb_2d = emb_h + emb_w  # [H, W, dim]

        # Match channel count
        if self.dim < C:
            emb_2d = emb_2d.repeat(1, 1, (C // self.dim) + 1)[:, :, :C]
        else:
            emb_2d = emb_2d[:, :, :C]

        # Reshape and add
        emb_2d = emb_2d.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        return x + emb_2d


class NTKScaledRoPE2D(nn.Module):
    """
    NTK-Aware Scaled RoPE for extreme resolution extrapolation.

    Based on Neural Tangent Kernel (NTK) theory, this variant scales
    the base frequency to preserve high-frequency components while
    extending low-frequency coverage.

    References:
    - Reddit: NTK-Aware Scaled RoPE
    - Allows extrapolation to 8x or more of training resolution
    """

    def __init__(
        self,
        dim: int,
        train_resolution: int = 128,
        max_resolution: int = 512,
        base: float = 10000.0,
        alpha: Optional[float] = None
    ):
        super().__init__()
        self.dim = dim
        self.train_resolution = train_resolution
        self.max_resolution = max_resolution
        self.base = base

        # Auto-compute alpha if not provided
        # alpha = (max_res / train_res) ^ (dim / (dim - 2))
        if alpha is None:
            scale_factor = max_resolution / train_resolution
            alpha = scale_factor ** (dim / (dim - 2))
        self.alpha = alpha

        # NTK-scaled base frequency
        scaled_base = base * alpha
        print(f"[NTKScaledRoPE2D] base={base:.0f}, alpha={alpha:.2f}, scaled_base={scaled_base:.0f}")

        # Precompute frequency bands with scaled base
        inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply NTK-scaled RoPE.

        Args:
            x: Input [batch, channels, height, width]

        Returns:
            x with RoPE applied [batch, channels, height, width]
        """
        B, C, H, W = x.shape

        # Position indices (no normalization needed, scaling is in frequencies)
        pos_h = torch.arange(H, device=x.device, dtype=torch.float32)
        pos_w = torch.arange(W, device=x.device, dtype=torch.float32)

        # Compute sinusoidal embeddings
        freqs_h = torch.einsum("i,j->ij", pos_h, self.inv_freq)
        emb_h = torch.cat([freqs_h.sin(), freqs_h.cos()], dim=-1)

        freqs_w = torch.einsum("i,j->ij", pos_w, self.inv_freq)
        emb_w = torch.cat([freqs_w.sin(), freqs_w.cos()], dim=-1)

        # Expand to 2D grid
        emb_h = emb_h.unsqueeze(1).expand(-1, W, -1)
        emb_w = emb_w.unsqueeze(0).expand(H, -1, -1)

        # Combine
        emb_2d = emb_h + emb_w

        # Match channel count
        if self.dim < C:
            emb_2d = emb_2d.repeat(1, 1, (C // self.dim) + 1)[:, :, :C]
        else:
            emb_2d = emb_2d[:, :, :C]

        # Reshape and add
        emb_2d = emb_2d.permute(2, 0, 1).unsqueeze(0)

        return x + emb_2d


class DynamicNTKRoPE2D(nn.Module):
    """
    Dynamic NTK-Scaled RoPE with runtime resolution adaptation.

    Automatically adjusts alpha based on current resolution at runtime,
    allowing seamless extrapolation without retraining.
    """

    def __init__(
        self,
        dim: int,
        train_resolution: int = 128,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.train_resolution = train_resolution
        self.base = base

        # Base frequency bands (will be scaled dynamically)
        base_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("base_inv_freq", base_inv_freq)

    def compute_dynamic_alpha(self, current_res: int) -> float:
        """
        Compute alpha based on current resolution.

        Args:
            current_res: Current resolution (max of H, W)

        Returns:
            alpha: Scaling factor for base frequency
        """
        if current_res <= self.train_resolution:
            return 1.0  # No scaling for smaller resolutions

        scale_factor = current_res / self.train_resolution
        alpha = scale_factor ** (self.dim / (self.dim - 2))
        return alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic NTK-scaled RoPE.

        Args:
            x: Input [batch, channels, height, width]

        Returns:
            x with RoPE applied [batch, channels, height, width]
        """
        B, C, H, W = x.shape

        # Compute dynamic alpha based on current resolution
        current_res = max(H, W)
        alpha = self.compute_dynamic_alpha(current_res)

        # Scale frequency bands
        if alpha != 1.0:
            scaled_base = self.base * alpha
            inv_freq = 1.0 / (scaled_base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
        else:
            inv_freq = self.base_inv_freq

        # Position indices
        pos_h = torch.arange(H, device=x.device, dtype=torch.float32)
        pos_w = torch.arange(W, device=x.device, dtype=torch.float32)

        # Compute sinusoidal embeddings
        freqs_h = torch.einsum("i,j->ij", pos_h, inv_freq)
        emb_h = torch.cat([freqs_h.sin(), freqs_h.cos()], dim=-1)

        freqs_w = torch.einsum("i,j->ij", pos_w, inv_freq)
        emb_w = torch.cat([freqs_w.sin(), freqs_w.cos()], dim=-1)

        # Expand to 2D grid
        emb_h = emb_h.unsqueeze(1).expand(-1, W, -1)
        emb_w = emb_w.unsqueeze(0).expand(H, -1, -1)

        # Combine
        emb_2d = emb_h + emb_w

        # Match channel count
        if self.dim < C:
            emb_2d = emb_2d.repeat(1, 1, (C // self.dim) + 1)[:, :, :C]
        else:
            emb_2d = emb_2d[:, :, :C]

        # Reshape and add
        emb_2d = emb_2d.permute(2, 0, 1).unsqueeze(0)

        return x + emb_2d


def create_rope_2d(
    variant: str = "standard",
    dim: int = 320,
    train_resolution: int = 128,
    max_resolution: int = 512,
    base: float = 10000.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create RoPE 2D module.

    Args:
        variant: "standard", "adaptive", "ntk", or "dynamic_ntk"
        dim: Embedding dimension
        train_resolution: Training resolution (in latent space)
        max_resolution: Maximum expected resolution
        base: Base for frequency calculation

    Returns:
        RoPE2D module
    """
    if variant == "standard":
        return RoPE2D(dim=dim, max_resolution=max_resolution, base=base)
    elif variant == "adaptive":
        return ResolutionAdaptiveRoPE2D(
            dim=dim,
            train_resolution=train_resolution,
            max_resolution=max_resolution,
            base=base
        )
    elif variant == "ntk":
        return NTKScaledRoPE2D(
            dim=dim,
            train_resolution=train_resolution,
            max_resolution=max_resolution,
            base=base,
            alpha=kwargs.get("alpha")
        )
    elif variant == "dynamic_ntk":
        return DynamicNTKRoPE2D(
            dim=dim,
            train_resolution=train_resolution,
            base=base
        )
    else:
        raise ValueError(f"Unknown RoPE variant: {variant}")
