"""
Energy-based Regularization Loss for preventing overbaking in diffusion model training.

This module implements a regularization technique based on L2 energy (norm) preservation.
Overbaking (excessive denoising) tends to reduce the energy of predicted latents compared
to ground truth, as the model removes both noise and fine details.

Based on empirical analysis showing:
- Timestep 0.0-0.2: Mean energy ratio 0.923 (7.7% energy loss)
- Timestep 0.8-1.0: Mean energy ratio 1.002 (normal)
- Correlation (Timestep vs Energy ratio): -0.445 (moderate negative)

References:
    Analysis: test_debug_latent_batch.py
    Report: test_output/debug_latent_summary.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyRegularizationLoss(nn.Module):
    """
    Energy-based regularization to prevent overbaking in diffusion model training.

    Penalizes predicted latents whose L2 energy deviates from ground truth energy.
    Overbaking typically manifests as energy loss (ratio < 1.0), as excessive
    denoising removes fine details along with noise.

    The regularization is timestep-adaptive: lower timesteps (t→0) receive
    stronger penalties, as overbaking is most severe in this region.

    Args:
        weight (float): Global weight for the regularization loss. Default: 0.05
        timestep_adaptive (bool): If True, apply stronger penalty at lower timesteps.
                                  Default: True
        penalty_mode (str): How to penalize energy deviation:
                           - "abs": Penalize any deviation from 1.0 (two-sided)
                           - "under": Only penalize energy loss (ratio < 1.0)
                           Default: "abs"
        normalize_by_pixels (bool): If True, normalize energy by number of pixels.
                                    This makes the metric resolution-independent.
                                    Default: True

    Example:
        >>> energy_loss = EnergyRegularizationLoss(weight=0.05, penalty_mode="abs")
        >>> # In training loop
        >>> mse_loss = F.mse_loss(model_pred, target)
        >>> energy_reg = energy_loss(model_pred, latents, timesteps / 1000.0)
        >>> total_loss = mse_loss + energy_reg
    """

    def __init__(self, weight: float = 0.05, timestep_adaptive: bool = True,
                 penalty_mode: str = "abs", normalize_by_pixels: bool = True):
        super().__init__()
        self.weight = weight
        self.timestep_adaptive = timestep_adaptive
        self.penalty_mode = penalty_mode
        self.normalize_by_pixels = normalize_by_pixels

        assert penalty_mode in ["abs", "under"], f"Invalid penalty_mode: {penalty_mode}"

    def compute_energy(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 energy (norm) of latent tensors.

        Energy is defined as:
            E = sqrt(sum(x^2)) / sqrt(N)  (if normalize_by_pixels=True)
            E = sqrt(sum(x^2))            (if normalize_by_pixels=False)

        Where N = C * H * W (number of elements per sample).

        Args:
            latent: Latent tensor of shape [B, C, H, W]

        Returns:
            energy: Energy per sample, shape [B]

        Mathematical details:
            For each sample in batch:
            1. Flatten to [C*H*W]
            2. Compute L2 norm: E = sqrt(sum(x_i^2))
            3. Optionally normalize: E_norm = E / sqrt(N)
        """
        B, C, H, W = latent.shape
        latent_flat = latent.view(B, -1)  # [B, C*H*W]

        # L2 norm (Frobenius norm for matrices)
        energy = latent_flat.pow(2).sum(dim=1).sqrt()  # [B]

        # Normalize by number of pixels (makes metric resolution-independent)
        if self.normalize_by_pixels:
            num_elements = C * H * W
            energy = energy / math.sqrt(num_elements)

        return energy

    def forward(self, predicted_latent: torch.Tensor, true_latent: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
        """
        Compute energy regularization loss.

        Args:
            predicted_latent: Model prediction, shape [B, C, H, W]
            true_latent: Ground truth latent (x_0), shape [B, C, H, W]
            timestep: Normalized timestep in [0, 1], shape [B] or scalar
                     (0 = clean/x_0, 1 = noise)

        Returns:
            loss: Scalar regularization loss

        Loss computation:
            1. E_pred = compute_energy(predicted_latent)
            2. E_true = compute_energy(true_latent)
            3. energy_ratio = E_pred / E_true
            4. If penalty_mode="abs": penalty = |ratio - 1.0|
               If penalty_mode="under": penalty = relu(1.0 - ratio)
            5. If timestep_adaptive: penalty *= (1 - timestep)
            6. loss = mean(penalty) * weight
        """
        # Compute energy for both latents
        energy_pred = self.compute_energy(predicted_latent)  # [B]
        energy_true = self.compute_energy(true_latent)       # [B]

        # Energy ratio (predicted / true)
        # ratio < 1.0: predicted has less energy (overbaking, lost details)
        # ratio > 1.0: predicted has more energy (added noise/artifacts)
        energy_ratio = energy_pred / (energy_true + 1e-8)  # [B]

        # Apply penalty based on mode
        if self.penalty_mode == "abs":
            # Penalize any deviation from 1.0 (two-sided)
            energy_penalty = (energy_ratio - 1.0).abs()  # [B]
        else:  # "under"
            # Only penalize energy loss (one-sided)
            energy_penalty = torch.relu(1.0 - energy_ratio)  # [B]

        # Timestep-adaptive weighting
        if self.timestep_adaptive:
            # Ensure timestep is a tensor
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor(timestep, device=predicted_latent.device)

            # Broadcast timestep to batch size if scalar
            if timestep.dim() == 0:
                timestep = timestep.expand(predicted_latent.size(0))

            # Lower timestep (t→0) -> higher weight
            # t=0.0 -> weight=1.0, t=1.0 -> weight=0.0
            timestep_weight = 1.0 - timestep  # [B]
            energy_penalty = energy_penalty * timestep_weight

        # Mean loss with global weight
        loss = energy_penalty.mean() * self.weight

        return loss

    def get_metrics(self, predicted_latent: torch.Tensor, true_latent: torch.Tensor,
                    timestep: torch.Tensor) -> dict:
        """
        Get detailed metrics for logging/debugging.

        Args:
            predicted_latent: Model prediction, shape [B, C, H, W]
            true_latent: Ground truth latent, shape [B, C, H, W]
            timestep: Normalized timestep in [0, 1], shape [B] or scalar

        Returns:
            metrics: Dictionary containing:
                - energy_pred_mean: Mean energy of predicted latents
                - energy_true_mean: Mean energy of true latents
                - energy_ratio_mean: Mean energy ratio (pred/true)
                - energy_ratio_min: Min energy ratio
                - energy_ratio_max: Max energy ratio
                - energy_loss_ratio: Fraction of samples with ratio < 1.0
                - loss: The regularization loss value
        """
        with torch.no_grad():
            energy_pred = self.compute_energy(predicted_latent)
            energy_true = self.compute_energy(true_latent)
            energy_ratio = energy_pred / (energy_true + 1e-8)

        # Compute loss (with gradient for actual training)
        loss = self.forward(predicted_latent, true_latent, timestep)

        with torch.no_grad():
            metrics = {
                "energy_pred_mean": energy_pred.mean().item(),
                "energy_true_mean": energy_true.mean().item(),
                "energy_ratio_mean": energy_ratio.mean().item(),
                "energy_ratio_min": energy_ratio.min().item(),
                "energy_ratio_max": energy_ratio.max().item(),
                "energy_loss_ratio": (energy_ratio < 1.0).float().mean().item(),
                "loss": loss.item(),
            }

        return metrics


def create_energy_regularization_loss(config: dict) -> EnergyRegularizationLoss:
    """
    Factory function to create EnergyRegularizationLoss from config dict.

    Args:
        config: Dictionary containing:
            - energy_regularization_weight: float, default 0.05
            - energy_timestep_adaptive: bool, default True
            - energy_penalty_mode: str, default "abs"
            - energy_normalize_by_pixels: bool, default True

    Returns:
        EnergyRegularizationLoss instance

    Example:
        >>> config = {"energy_regularization_weight": 0.1, "energy_penalty_mode": "abs"}
        >>> energy_loss = create_energy_regularization_loss(config)
    """
    weight = config.get("energy_regularization_weight", 0.05)
    timestep_adaptive = config.get("energy_timestep_adaptive", True)
    penalty_mode = config.get("energy_penalty_mode", "abs")
    normalize_by_pixels = config.get("energy_normalize_by_pixels", True)

    return EnergyRegularizationLoss(
        weight=weight,
        timestep_adaptive=timestep_adaptive,
        penalty_mode=penalty_mode,
        normalize_by_pixels=normalize_by_pixels
    )


# Import math for sqrt in compute_energy
import math
