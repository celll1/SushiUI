"""
Aesthetic Loss Module for SushiUI Training Integration

Provides aesthetic quality loss as a regularization term to prevent overbaked images.
Trained aesthetic model is frozen and used only for loss calculation.
"""

import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file
from typing import Optional
import sys

# Add aesthetic scorer to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "subapps" / "aesthetic_scorer" / "backend"))

from core.aesthetic_model import LatentCNN, LatentTransformer


class AestheticLoss:
    """
    Aesthetic loss module for training integration.

    Usage:
        aesthetic_loss_module = AestheticLoss(
            model_path="subapps/aesthetic_scorer/models/aesthetic_best.safetensors",
            architecture="LatentCNN"
        )
        aesthetic_loss = aesthetic_loss_module(predicted_latent)
        total_loss = mse_loss + aesthetic_weight * aesthetic_loss
    """

    def __init__(
        self,
        model_path: str,
        architecture: str = "LatentCNN",
        device: str = "cuda",
        in_channels: int = 16,
    ):
        """
        Initialize aesthetic loss module.

        Args:
            model_path: Path to trained aesthetic model (.safetensors)
            architecture: Model architecture ("LatentCNN" or "LatentTransformer")
            device: Device to use (cuda/cpu)
            in_channels: Number of latent channels (16 for Z-Image, 4 for SD/SDXL)
        """
        self.device = device
        self.architecture = architecture

        # Create model
        if architecture == "LatentCNN":
            self.model = LatentCNN(in_channels=in_channels).to(device)
        elif architecture == "LatentTransformer":
            self.model = LatentTransformer(in_channels=in_channels).to(device)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Load trained weights
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Aesthetic model not found: {model_path}")

        state_dict = load_file(model_path)
        self.model.load_state_dict(state_dict)

        # Freeze model (no gradient computation)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        num_params = sum(p.numel() for p in self.model.parameters())

        print(f"[AestheticLoss] Loaded {architecture} from {model_path}")
        print(f"[AestheticLoss] Parameters: {num_params:,} (~{num_params * 4 / 1024:.1f} KB)")
        print(f"[AestheticLoss] Model frozen (no gradient)")

    def __call__(self, predicted_latent: torch.Tensor) -> torch.Tensor:
        """
        Compute aesthetic loss from predicted latent.

        The model outputs a score where:
        - 0.0 = best quality (good predicted latent)
        - 1.0 = worst quality (overbaked/poor predicted latent)

        We use this score directly as a loss to minimize.

        Args:
            predicted_latent: [B, C, H, W] Predicted latent tensor

        Returns:
            aesthetic_loss: Scalar tensor (mean score across batch)
        """
        with torch.no_grad():
            # Model outputs quality score [B, 1]
            # 0.0 = best quality, 1.0 = worst quality
            scores = self.model(predicted_latent)  # [B, 1]

        # Mean score across batch
        aesthetic_loss = scores.mean()

        return aesthetic_loss

    def get_batch_scores(self, predicted_latent: torch.Tensor) -> torch.Tensor:
        """
        Get individual scores for each sample in batch (for logging).

        Args:
            predicted_latent: [B, C, H, W]

        Returns:
            scores: [B, 1] Individual scores
        """
        with torch.no_grad():
            scores = self.model(predicted_latent)

        return scores


def load_aesthetic_loss(
    model_path: Optional[str] = None,
    architecture: str = "LatentCNN",
    device: str = "cuda",
    in_channels: int = 16,
) -> Optional[AestheticLoss]:
    """
    Convenience function to load aesthetic loss module.

    Args:
        model_path: Path to trained model (None to disable)
        architecture: Model architecture
        device: Device to use
        in_channels: Number of latent channels

    Returns:
        AestheticLoss instance or None
    """
    if model_path is None or model_path == "":
        return None

    if not Path(model_path).exists():
        print(f"[AestheticLoss] WARNING: Model not found at {model_path}, skipping")
        return None

    try:
        aesthetic_loss = AestheticLoss(
            model_path=model_path,
            architecture=architecture,
            device=device,
            in_channels=in_channels,
        )
        return aesthetic_loss
    except Exception as e:
        print(f"[AestheticLoss] ERROR: Failed to load model: {e}")
        return None


# Example integration into BaseTrainer
"""
# In backend/core/training/base_trainer.py:

from core.training.aesthetic_loss import load_aesthetic_loss

class BaseTrainer:
    def __init__(
        self,
        ...,
        aesthetic_loss_weight: float = 0.0,
        aesthetic_model_path: Optional[str] = None,
        aesthetic_architecture: str = "LatentCNN",
    ):
        # ... existing code ...

        # Aesthetic loss (optional)
        self.aesthetic_loss_weight = aesthetic_loss_weight
        self.aesthetic_loss_module = None

        if aesthetic_loss_weight > 0 and aesthetic_model_path:
            # Latent channels from the loaded VAE config (handles SDXL VAE migration to
            # a high-spec VAE, e.g. FLUX.1 16ch); falls back to the Z-Image/SD-SDXL guess.
            in_channels = getattr(self, "vae_latent_channels", None)
            if not in_channels:
                try:
                    in_channels = int(self.vae.config.latent_channels)
                except Exception:
                    in_channels = 16 if self.is_zimage else 4
            self.aesthetic_loss_module = load_aesthetic_loss(
                model_path=aesthetic_model_path,
                architecture=aesthetic_architecture,
                device=self.device,
                in_channels=in_channels,
            )

            if self.aesthetic_loss_module:
                print(f"{self.log_prefix} Aesthetic loss enabled (weight={aesthetic_loss_weight})")

    def train_step_zimage(self, ...):
        # ... existing code for forward pass ...

        # Calculate predicted latent
        t_expanded = timesteps.view(-1, 1, 1, 1)
        predicted_latent = noisy_latents - t_expanded * model_pred

        # MSE loss (existing)
        loss = F.mse_loss(model_pred, target, reduction="none")
        # ... SNR weighting, etc. ...

        # Aesthetic loss (new, optional)
        if self.aesthetic_loss_module is not None:
            aesthetic_loss = self.aesthetic_loss_module(predicted_latent)
            total_loss = loss + self.aesthetic_loss_weight * aesthetic_loss

            if step_callback:
                step_callback({
                    "mse_loss": loss.item(),
                    "aesthetic_loss": aesthetic_loss.item(),
                    "total_loss": total_loss.item(),
                })

            loss = total_loss

        return loss, recon_loss_value
"""
