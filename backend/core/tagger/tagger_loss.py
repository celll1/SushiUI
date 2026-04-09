"""
Asymmetric Loss for multi-label tag classification.

Ported from D:/celll1/tagutl/lora.py (AsymmetricLossOptimized).
Handles:
  - Asymmetric focusing (different gamma for positive/negative)
  - Asymmetric clipping (prevents collapse on false negatives)
  - Per-element loss masking (for rating/quality tag exclusion)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AsymmetricLossOptimized(nn.Module):
    """Optimized asymmetric loss for multi-label classification.

    Minimizes memory allocation and GPU uploading; favors inplace operations.

    Parameters
    ----------
    gamma_neg : float
        Focusing parameter for negative examples (hard negatives get higher weight).
    gamma_pos : float
        Focusing parameter for positive examples.
    clip : float
        Clipping value for negative probabilities (prevents false-negative collapse).
    eps : float
        Numerical stability epsilon for log.
    disable_torch_grad_focal_loss : bool
        If True, compute asymmetric weights without gradient tracking (saves memory).
    reduction : str
        'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-6,
        disable_torch_grad_focal_loss: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.reduction = reduction

        # Pre-allocated buffers (filled each forward call)
        self.targets = self.anti_targets = None
        self.xs_pos = self.xs_neg = None
        self.asymmetric_w = self.loss = None

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute asymmetric loss.

        Parameters
        ----------
        x : Tensor [B, num_tags]
            Raw logits (before sigmoid).
        y : Tensor [B, num_tags]
            Binary multi-label targets (0.0 or 1.0).
        loss_mask : Tensor [B, num_tags] or [num_tags], optional
            Per-element mask. 0 = ignore this tag for this sample.
        """
        self.targets = y
        self.anti_targets = 1.0 - y

        # Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric clipping on negatives
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1.0)

        # Basic cross-entropy
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric focusing weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    xs_pos_f = self.xs_pos * self.targets
                    xs_neg_f = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(
                        1.0 - xs_pos_f - xs_neg_f,
                        self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
                    )
            else:
                xs_pos_f = self.xs_pos * self.targets
                xs_neg_f = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(
                    1.0 - xs_pos_f - xs_neg_f,
                    self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
                )
            self.loss *= self.asymmetric_w

        # Apply loss mask
        if loss_mask is not None:
            self.loss = self.loss * loss_mask

        if self.reduction == "mean":
            return -self.loss.mean()
        elif self.reduction == "sum":
            return -self.loss.sum()
        else:
            return -self.loss
