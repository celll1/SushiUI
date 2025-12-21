"""
Aesthetic Model - Lightweight CNN for Latent Quality Scoring

Ultra-lightweight convolutional network that predicts quality scores from predicted latents.
Designed for minimal VRAM overhead (<10MB) when integrated into training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file, load_file
from typing import Dict, Any


class LatentCNN(nn.Module):
    """
    Ultra-lightweight CNN for latent quality scoring.

    Architecture:
        Input: [B, 16, H, W] (Z-Image latent, arbitrary H/W)
        Conv2d(16→32, stride=2) → ReLU
        Conv2d(32→64, stride=2) → ReLU
        Conv2d(64→128, stride=2) → ReLU
        AdaptiveAvgPool2d(1, 1)  # Global pooling for arbitrary input size
        Linear(128→1) → Sigmoid
        Output: [B, 1] (score: 0=best quality, 1=worst quality)

    Parameters: ~50K (~200KB)
    VRAM: <5MB (forward pass with batch=4, 1152x832 latent)
    """

    def __init__(self, in_channels: int = 16):
        """
        Initialize LatentCNN.

        Args:
            in_channels: Number of latent channels (16 for Z-Image)
        """
        super().__init__()

        self.in_channels = in_channels

        # Convolutional layers with stride=2 for spatial downsampling
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Global average pooling (handles arbitrary input sizes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected head
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, 16, H, W] Predicted latent

        Returns:
            score: [B, 1] Quality score (0=best, 1=worst)
        """
        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))  # [B, 32, H/2, W/2]
        x = F.relu(self.conv2(x))  # [B, 64, H/4, W/4]
        x = F.relu(self.conv3(x))  # [B, 128, H/8, W/8]

        # Global pooling
        x = self.pool(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]

        # Fully connected + Sigmoid
        score = torch.sigmoid(self.fc(x))  # [B, 1]

        return score

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def save_safetensors(self, path: Path):
        """
        Save model to safetensors format.

        Args:
            path: Output .safetensors file path
        """
        state_dict = self.state_dict()
        save_file(state_dict, path)
        print(f"[LatentCNN] Saved to {path}")

    def load_safetensors(self, path: Path):
        """
        Load model from safetensors format.

        Args:
            path: Input .safetensors file path
        """
        state_dict = load_file(path)
        self.load_state_dict(state_dict)
        print(f"[LatentCNN] Loaded from {path}")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "architecture": "LatentCNN",
            "in_channels": self.in_channels,
            "num_parameters": self.count_parameters(),
        }


class LatentTransformer(nn.Module):
    """
    Lightweight Vision Transformer for latent scoring.

    Alternative to LatentCNN for higher accuracy (at cost of more parameters).

    Architecture:
        Input: [B, 16, H, W]
        Patch Embedding (patch_size=8)
        Transformer Encoder (depth=2, heads=4, dim=128)
        MLP Head
        Output: [B, 1]

    Parameters: ~500K (~2MB)
    VRAM: ~20-30MB
    """

    def __init__(
        self,
        in_channels: int = 16,
        patch_size: int = 8,
        dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 4,
    ):
        """
        Initialize LatentTransformer.

        Args:
            in_channels: Number of latent channels (16 for Z-Image)
            patch_size: Patch size for embedding
            dim: Transformer dimension
            depth: Number of transformer blocks
            heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
        """
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Positional embedding (learnable)
        # Max sequence length: assume max 256x256 latent → 32x32 patches
        max_seq_len = (256 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, 16, H, W] Predicted latent

        Returns:
            score: [B, 1] Quality score (0=best, 1=worst)
        """
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # [B, dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]

        # Add positional embedding
        num_patches = x.size(1)
        x = x + self.pos_embed[:, :num_patches, :]

        # Transformer
        x = self.transformer(x)  # [B, num_patches, dim]

        # Global average pooling
        x = x.mean(dim=1)  # [B, dim]

        # MLP head
        score = self.head(x)  # [B, 1]

        return score

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def save_safetensors(self, path: Path):
        """Save model to safetensors format."""
        state_dict = self.state_dict()
        save_file(state_dict, path)
        print(f"[LatentTransformer] Saved to {path}")

    def load_safetensors(self, path: Path):
        """Load model from safetensors format."""
        state_dict = load_file(path)
        self.load_state_dict(state_dict)
        print(f"[LatentTransformer] Loaded from {path}")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "architecture": "LatentTransformer",
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "dim": self.dim,
            "num_parameters": self.count_parameters(),
        }


def create_aesthetic_model(architecture: str = "LatentCNN", **kwargs) -> nn.Module:
    """
    Factory function to create aesthetic model.

    Args:
        architecture: "LatentCNN" or "LatentTransformer"
        **kwargs: Model-specific arguments

    Returns:
        Aesthetic model instance
    """
    if architecture == "LatentCNN":
        model = LatentCNN(**kwargs)
    elif architecture == "LatentTransformer":
        model = LatentTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    print(f"[AestheticModel] Created {architecture}")
    print(f"[AestheticModel] Parameters: {model.count_parameters():,}")

    return model


if __name__ == "__main__":
    # Test LatentCNN
    print("=" * 60)
    print("Testing LatentCNN")
    print("=" * 60)

    model = LatentCNN(in_channels=16)
    print(f"Parameters: {model.count_parameters():,}")

    # Test with various input sizes
    for H, W in [(144, 104), (128, 128), (192, 192)]:
        x = torch.randn(4, 16, H, W)
        score = model(x)
        print(f"Input: [4, 16, {H}, {W}] → Output: {score.shape}")

    # Test LatentTransformer
    print("\n" + "=" * 60)
    print("Testing LatentTransformer")
    print("=" * 60)

    model_transformer = LatentTransformer(in_channels=16)
    print(f"Parameters: {model_transformer.count_parameters():,}")

    for H, W in [(144, 104), (128, 128), (192, 192)]:
        x = torch.randn(4, 16, H, W)
        score = model_transformer(x)
        print(f"Input: [4, 16, {H}, {W}] → Output: {score.shape}")
