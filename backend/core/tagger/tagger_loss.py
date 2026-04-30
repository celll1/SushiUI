"""
Loss functions for multi-label tag classification.

Classes
-------
AsymmetricLossOptimized  — original ASL (default)
CSASL                    — Continuous Symmetric ASL  (CS-ASL)
HCSASL                   — Hierarchical CS-ASL       (H-CS-ASL)
LASASL                   — Logit-Adjusted Sym. ASL   (LA-S-ASL)
FWBBCE                   — Fisher-Weighted Balanced BCE (FW-BBCE)

All classes share the same forward signature:
    forward(x, y, loss_mask=None) -> scalar Tensor

References
----------
- Ben-Baruch et al., Asymmetric Loss For Multi-Label Classification, ICCV 2021
- Menon et al., Long-tail Learning via Logit Adjustment, ICLR 2021
- Cui et al., Class-Balanced Loss (Effective Number), CVPR 2019
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Original ASL
# ---------------------------------------------------------------------------

class AsymmetricLossOptimized(nn.Module):
    """Optimized asymmetric loss for multi-label classification.

    Parameters
    ----------
    gamma_neg : float
        Focusing parameter for negative examples.
    gamma_pos : float
        Focusing parameter for positive examples.
    clip : float
        Clipping value for negative probabilities (prevents false-negative collapse).
    eps : float
        Numerical stability epsilon for log.
    disable_torch_grad_focal_loss : bool
        Compute focal weights without gradient tracking (saves memory).
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

        self.targets = self.anti_targets = None
        self.xs_pos = self.xs_neg = None
        self.asymmetric_w = self.loss = None

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.targets = y
        self.anti_targets = 1.0 - y

        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1.0)

        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

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

        if loss_mask is not None:
            self.loss = self.loss * loss_mask

        if self.reduction == "mean":
            return -self.loss.mean()
        elif self.reduction == "sum":
            return -self.loss.sum()
        else:
            return -self.loss


# ---------------------------------------------------------------------------
# Shared helper: CS-ASL per-element core
# ---------------------------------------------------------------------------

def _cs_asl_core(
    x: torch.Tensor,        # [B, N] logits
    y: torch.Tensor,        # [B, N] targets {0,1}
    pi: torch.Tensor,       # [N]   positive rate (buffer, detached)
    gamma0: float,
    m0: float,
    rho: float,
    beta: float,
    eps: float,
    disable_grad_focal: bool,
) -> torch.Tensor:
    """Compute CS-ASL loss elements [B, N] (no reduction, no masking)."""
    p = torch.sigmoid(x)

    # Class weights: α+ = (1-π)^ρ,  α- = π^ρ
    a_pos = (1.0 - pi).pow(rho)   # [N]
    a_neg = pi.pow(rho)            # [N]

    # Asymmetry indicators
    phi = (2.0 * pi - 1.0).clamp(min=0.0)   # > 0 when π > 0.5
    psi = (1.0 - 2.0 * pi).clamp(min=0.0)   # > 0 when π < 0.5

    gamma_pos = gamma0 * phi.pow(beta)   # [N]
    gamma_neg = gamma0 * psi.pow(beta)   # [N]
    m_pos = m0 * phi                     # [N]
    m_neg = m0 * psi                     # [N]

    # Shifted probabilities (both sides clamped to prevent log(0) → -inf under AMP/fp16)
    p_pos = (p + m_pos).clamp(min=eps, max=1.0 - eps)   # [B, N]
    p_neg = (p - m_neg).clamp(min=eps, max=1.0 - eps)   # [B, N]

    # Focal weights (optionally stop gradient)
    if disable_grad_focal:
        with torch.no_grad():
            w_pos = (1.0 - p).pow(gamma_pos)   # [B, N]
            w_neg = p_neg.pow(gamma_neg)         # [B, N]
    else:
        w_pos = (1.0 - p).pow(gamma_pos)
        w_neg = p_neg.pow(gamma_neg)

    loss_pos = a_pos * w_pos * p_pos.log()                  # [B, N]
    loss_neg = a_neg * w_neg * (1.0 - p_neg).clamp(min=eps).log()  # [B, N]

    return -(y * loss_pos + (1.0 - y) * loss_neg)           # [B, N]


# ---------------------------------------------------------------------------
# CS-ASL
# ---------------------------------------------------------------------------

class CSASL(nn.Module):
    """Continuous Symmetric ASL.

    Extends ASL by making class weights and focal parameters continuous
    functions of the per-label positive rate π_n, symmetric around π=0.5.

    Parameters
    ----------
    pi : Tensor [num_tags]
        Per-label positive rate computed from the training set (stop-gradient).
    gamma0 : float
        Base focal strength.
    m0 : float
        Base margin strength.
    rho : float
        Class-weight exponent (0 = uniform, 1 = inverse frequency).
    beta : float
        Sharpness of the π=0.5 transition (≥ 2 recommended).
    eps : float
        Numerical stability clip.
    disable_torch_grad_focal_loss : bool
        Stop gradient through focal weights.
    reduction : str
        'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        pi: torch.Tensor,
        gamma0: float = 4.0,
        m0: float = 0.2,
        rho: float = 0.5,
        beta: float = 2.0,
        eps: float = 1e-4,
        disable_torch_grad_focal_loss: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("pi", pi.detach().clamp(eps, 1.0 - eps))
        self.gamma0 = gamma0
        self.m0 = m0
        self.rho = rho
        self.beta = beta
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = _cs_asl_core(
            x, y, self.pi,
            self.gamma0, self.m0, self.rho, self.beta, self.eps,
            self.disable_torch_grad_focal_loss,
        )   # [B, N]

        if loss_mask is not None:
            loss = loss * loss_mask

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# H-CS-ASL
# ---------------------------------------------------------------------------

def _compute_label_weights(
    pi: torch.Tensor,
    N_pos: torch.Tensor,
    N_neg: torch.Tensor,
    method: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute normalized inter-label weights u_n ∈ [0,1], sum=1."""
    if method == "fisher":
        u = (N_pos * N_neg) / (N_pos + N_neg + eps)
    elif method == "entropy_fisher":
        pi_c = pi.clamp(eps, 1.0 - eps)
        H = -(pi_c * pi_c.log() + (1.0 - pi_c) * (1.0 - pi_c).log())
        fisher = (N_pos * N_neg) / (N_pos + N_neg + eps)
        u = H * fisher
    elif method == "effective":
        xi = 0.999
        N_tot = (N_pos + N_neg).clamp(min=1.0)
        u = (1.0 - xi) / (1.0 - xi ** N_tot)
    else:
        raise ValueError(f"Unknown label_weight method: {method!r}")
    total = u.sum()
    if total > 0:
        u = u / total
    return u.detach()


class HCSASL(nn.Module):
    """Hierarchical CS-ASL.

    Extends CS-ASL with inter-label weights u_n derived from label
    frequency statistics (Fisher information, entropy×Fisher, or
    effective number of samples).

    Parameters
    ----------
    pi : Tensor [num_tags]
        Per-label positive rate.
    N_pos, N_neg : Tensor [num_tags]
        Per-label positive/negative sample counts.
    gamma0, m0, rho, beta, eps : float
        CS-ASL hyperparameters.
    label_weight : str
        'fisher' | 'entropy_fisher' | 'effective'
    disable_torch_grad_focal_loss : bool
    reduction : str
    """

    def __init__(
        self,
        pi: torch.Tensor,
        N_pos: torch.Tensor,
        N_neg: torch.Tensor,
        gamma0: float = 4.0,
        m0: float = 0.2,
        rho: float = 0.5,
        beta: float = 2.0,
        label_weight: str = "fisher",
        eps: float = 1e-4,
        disable_torch_grad_focal_loss: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        _pi = pi.detach().clamp(eps, 1.0 - eps)
        self.register_buffer("pi", _pi)
        self.register_buffer(
            "u",
            _compute_label_weights(_pi, N_pos.detach(), N_neg.detach(), label_weight, eps),
        )
        self.gamma0 = gamma0
        self.m0 = m0
        self.rho = rho
        self.beta = beta
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = _cs_asl_core(
            x, y, self.pi,
            self.gamma0, self.m0, self.rho, self.beta, self.eps,
            self.disable_torch_grad_focal_loss,
        )   # [B, N]

        if loss_mask is not None:
            loss = loss * loss_mask

        # Apply inter-label weights: weighted sum over labels, mean over batch
        # loss [B, N] → (u * loss).sum(dim=1).mean()
        weighted = (self.u * loss).sum(dim=1)   # [B]

        if self.reduction == "mean":
            return weighted.mean()
        elif self.reduction == "sum":
            return weighted.sum()
        return loss   # 'none': return unweighted per-element for compatibility


# ---------------------------------------------------------------------------
# LA-S-ASL
# ---------------------------------------------------------------------------

class LASASL(nn.Module):
    """Logit-Adjusted Symmetric ASL.

    Adjusts logits by the per-label prior τ_n = log(π_n / (1-π_n)) before
    computing CS-ASL focal loss, absorbing class imbalance into the logit
    space instead of explicit class weights (α_± is omitted).

    Parameters
    ----------
    pi : Tensor [num_tags]
        Per-label positive rate.
    gamma0, m0, beta, eps : float
        CS-ASL hyperparameters (rho is not used — no α_± weights).
    disable_torch_grad_focal_loss : bool
    reduction : str
    """

    def __init__(
        self,
        pi: torch.Tensor,
        gamma0: float = 4.0,
        m0: float = 0.2,
        beta: float = 2.0,
        eps: float = 1e-4,
        disable_torch_grad_focal_loss: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        _pi = pi.detach().clamp(eps, 1.0 - eps)
        tau = (_pi / (1.0 - _pi)).log()   # logit prior [N]
        self.register_buffer("pi", _pi)
        self.register_buffer("tau", tau)
        self.gamma0 = gamma0
        self.m0 = m0
        self.beta = beta
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Logit adjustment: shift logits by prior before sigmoid
        x_adj = x - self.tau   # [B, N]

        # rho=0 → α_± = 1 (no class weighting; adjustment absorbed into logit)
        loss = _cs_asl_core(
            x_adj, y, self.pi,
            self.gamma0, self.m0, rho=0.0, beta=self.beta, eps=self.eps,
            disable_grad_focal=self.disable_torch_grad_focal_loss,
        )   # [B, N]

        if loss_mask is not None:
            loss = loss * loss_mask

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# FW-BBCE
# ---------------------------------------------------------------------------

class FWBBCE(nn.Module):
    """Fisher-Weighted Balanced Binary Cross-Entropy.

    Minimal design: importance-sampling class weights α_± with Fisher
    inter-label weights u_n.  No focal/margin mechanisms.

    Parameters
    ----------
    pi : Tensor [num_tags]
        Per-label positive rate.
    N_pos, N_neg : Tensor [num_tags]
        Per-label counts.
    alpha_clip : (float, float)
        Clamp range for α_± weights (prevents divergence when π→0 or π→1).
    eps : float
    reduction : str
    """

    def __init__(
        self,
        pi: torch.Tensor,
        N_pos: torch.Tensor,
        N_neg: torch.Tensor,
        alpha_clip: Tuple[float, float] = (0.1, 10.0),
        eps: float = 1e-4,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        _pi = pi.detach().clamp(eps, 1.0 - eps)
        a_pos = (0.5 / _pi).clamp(*alpha_clip)
        a_neg = (0.5 / (1.0 - _pi)).clamp(*alpha_clip)
        u = _compute_label_weights(_pi, N_pos.detach(), N_neg.detach(), "fisher", eps)
        self.register_buffer("a_pos", a_pos)
        self.register_buffer("a_neg", a_neg)
        self.register_buffer("u", u)
        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Numerically stable log-sigmoid
        log_p    = F.logsigmoid(x)       # log σ(x)
        log_1mp  = F.logsigmoid(-x)      # log(1 - σ(x))

        per_elem = -(self.a_pos * y * log_p + self.a_neg * (1.0 - y) * log_1mp)   # [B, N]

        if loss_mask is not None:
            per_elem = per_elem * loss_mask

        # Weighted sum over labels
        weighted = (self.u * per_elem).sum(dim=1)   # [B]

        if self.reduction == "mean":
            return weighted.mean()
        elif self.reduction == "sum":
            return weighted.sum()
        return per_elem   # 'none': return unweighted per-element
