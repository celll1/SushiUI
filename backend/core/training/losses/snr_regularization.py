"""
SNR-based Regularization Loss for preventing overbaking in diffusion model training.

This module implements a regularization technique to address the systematic bias
where predicted latents become overly denoised (high SNR) compared to ground truth,
particularly at low timesteps (t→0).

Based on empirical analysis showing:
- Timestep 0.0-0.2: Mean SNR diff +4.53 dB (severe overbaking)
- Timestep 0.8-1.0: Mean SNR diff +0.04 dB (normal)
- Correlation (Timestep vs SNR diff): -0.633 (strong negative)

References:
    Analysis: test_debug_latent_batch.py
    Report: test_output/debug_latent_summary.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNRRegularizationLoss(nn.Module):
    """
    SNR-based regularization to prevent overbaking in diffusion model training.

    Penalizes predicted latents that have higher SNR (signal-to-noise ratio)
    than the ground truth latents, which indicates over-denoising.

    The regularization is timestep-adaptive: lower timesteps (t→0) receive
    stronger penalties, as overbaking is most severe in this region.

    Args:
        weight (float): Global weight for the regularization loss. Default: 0.1
        timestep_adaptive (bool): If True, apply stronger penalty at lower timesteps.
                                  Default: True
        penalty_mode (str): How to penalize SNR difference:
                           - "relu": Only penalize when SNR_pred > SNR_true (one-sided)
                           - "abs": Penalize any deviation (two-sided)
                           Default: "relu"

    Example:
        >>> snr_loss = SNRRegularizationLoss(weight=0.1, timestep_adaptive=True)
        >>> # In training loop
        >>> mse_loss = F.mse_loss(model_pred, target)
        >>> snr_reg = snr_loss(model_pred, latents, timesteps / 1000.0)
        >>> total_loss = mse_loss + snr_reg
    """

    def __init__(self, weight: float = 0.1, timestep_adaptive: bool = True, penalty_mode: str = "relu"):
        super().__init__()
        self.weight = weight
        self.timestep_adaptive = timestep_adaptive
        self.penalty_mode = penalty_mode

        assert penalty_mode in ["relu", "abs"], f"Invalid penalty_mode: {penalty_mode}"

    def compute_snr(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute SNR (Signal-to-Noise Ratio) in dB for latent tensors.

        Corrected SNR definition:
            SNR(dB) = 10 * log10(total_power / noise_power)

        Where:
            - total_power = mean(x^2) (total signal power)
            - noise_power = variance (noise power)

        This is the correct definition: SNR = mean(x^2) / var(x) = (mean^2 + var) / var

        Args:
            latent: Latent tensor of shape [B, C, H, W]

        Returns:
            snr_db: SNR in dB, shape [B]

        Mathematical details:
            For each sample in batch:
            1. Flatten spatial dimensions: [C, H, W] -> [C, H*W]
            2. Total power: P_total = mean(x_c^2)
            3. Noise power: P_noise = var(x_c)
            4. SNR(dB) = 10 * log10(P_total / P_noise)
        """
        B, C, H, W = latent.shape
        latent_flat = latent.view(B, C, -1)  # [B, C, H*W]

        # Total power: mean(x^2)
        total_power = latent_flat.pow(2).mean(dim=-1).mean(dim=1)  # [B]

        # Noise power: variance
        variance = latent_flat.var(dim=-1).mean(dim=1)  # [B]

        # SNR in dB
        snr = total_power / (variance + 1e-8)
        snr_db = 10 * torch.log10(snr + 1e-8)

        return snr_db

    def forward(self, predicted_latent: torch.Tensor, true_latent: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
        """
        Compute SNR regularization loss.

        Args:
            predicted_latent: Model prediction, shape [B, C, H, W]
            true_latent: Ground truth latent (x_0), shape [B, C, H, W]
            timestep: Normalized timestep in [0, 1], shape [B] or scalar
                     (0 = clean/x_0, 1 = noise)

        Returns:
            loss: Scalar regularization loss

        Loss computation:
            1. SNR_pred = compute_snr(predicted_latent)
            2. SNR_true = compute_snr(true_latent)
            3. SNR_diff = SNR_pred - SNR_true
            4. If penalty_mode="relu": penalty = relu(SNR_diff)  # only penalize over-denoising
            5. If timestep_adaptive: penalty *= (1 - timestep)  # stronger at low timestep
            6. loss = mean(penalty) * weight
        """
        # Compute SNR for both latents
        snr_pred = self.compute_snr(predicted_latent)  # [B]
        snr_true = self.compute_snr(true_latent)       # [B]

        # SNR difference (positive = predicted is "cleaner" than true -> overbaking)
        snr_diff = snr_pred - snr_true  # [B]

        # Apply penalty based on mode
        if self.penalty_mode == "relu":
            # Only penalize when predicted SNR > true SNR (overbaking)
            snr_penalty = torch.relu(snr_diff)  # [B]
        else:  # "abs"
            # Penalize any deviation (two-sided)
            snr_penalty = torch.abs(snr_diff)  # [B]

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
            snr_penalty = snr_penalty * timestep_weight

        # Mean loss with global weight
        loss = snr_penalty.mean() * self.weight

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
                - snr_pred_mean: Mean SNR of predicted latents (dB)
                - snr_true_mean: Mean SNR of true latents (dB)
                - snr_diff_mean: Mean SNR difference (dB)
                - snr_diff_max: Max SNR difference (dB)
                - overbaking_ratio: Fraction of samples with SNR_diff > 0
                - loss: The regularization loss value
        """
        with torch.no_grad():
            snr_pred = self.compute_snr(predicted_latent)
            snr_true = self.compute_snr(true_latent)
            snr_diff = snr_pred - snr_true

            # Compute loss (with gradient for actual training)
        loss = self.forward(predicted_latent, true_latent, timestep)

        with torch.no_grad():
            metrics = {
                "snr_pred_mean": snr_pred.mean().item(),
                "snr_true_mean": snr_true.mean().item(),
                "snr_diff_mean": snr_diff.mean().item(),
                "snr_diff_max": snr_diff.max().item(),
                "overbaking_ratio": (snr_diff > 0).float().mean().item(),
                "loss": loss.item(),
            }

        return metrics


def create_snr_regularization_loss(config: dict) -> SNRRegularizationLoss:
    """
    Factory function to create SNRRegularizationLoss from config dict.

    Args:
        config: Dictionary containing:
            - snr_regularization_weight: float, default 0.1
            - snr_timestep_adaptive: bool, default True
            - snr_penalty_mode: str, default "relu"

    Returns:
        SNRRegularizationLoss instance

    Example:
        >>> config = {"snr_regularization_weight": 0.15, "snr_timestep_adaptive": True}
        >>> snr_loss = create_snr_regularization_loss(config)
    """
    weight = config.get("snr_regularization_weight", 0.1)
    timestep_adaptive = config.get("snr_timestep_adaptive", True)
    penalty_mode = config.get("snr_penalty_mode", "relu")

    return SNRRegularizationLoss(
        weight=weight,
        timestep_adaptive=timestep_adaptive,
        penalty_mode=penalty_mode
    )
