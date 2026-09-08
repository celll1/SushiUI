"""Convergence diagnostics statistical computation module (Phase 1).

Measures symptoms A (global drift/graying) and B (blocky 8px grid noise)
without affecting training dynamics.

All functions operate on PyTorch tensors in eval/no-grad mode and return
finite scalar metrics suitable for log_extra_metric.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def channel_mean_std_gap(
    pred_latent: torch.Tensor,
    ref_latent: torch.Tensor,
) -> Tuple[float, float]:
    """Compute mean and std error per channel between predicted and reference latents.

    Args:
        pred_latent: Predicted latent tensor [B, C, H, W] or [B, C, ...].
        ref_latent: Reference (GT) latent tensor of matching channel count.

    Returns:
        (mean_err, std_err): Mean absolute error across channels for mean and std.
    """
    pred = pred_latent.detach().float()
    ref = ref_latent.detach().float()

    if pred.dim() < 2 or ref.dim() < 2:
        return 0.0, 0.0

    # Assume channel is dim 1
    c_dim = 1 if pred.dim() >= 2 else 0
    reduce_dims_pred = [d for d in range(pred.dim()) if d != c_dim]
    reduce_dims_ref = [d for d in range(ref.dim()) if d != c_dim]

    pred_mean = pred.mean(dim=reduce_dims_pred)
    ref_mean = ref.mean(dim=reduce_dims_ref)

    pred_std = pred.std(dim=reduce_dims_pred) if pred.numel() > pred.shape[c_dim] else torch.zeros_like(pred_mean)
    ref_std = ref.std(dim=reduce_dims_ref) if ref.numel() > ref.shape[c_dim] else torch.zeros_like(ref_mean)

    min_c = min(pred_mean.shape[0], ref_mean.shape[0])
    mean_err = (pred_mean[:min_c] - ref_mean[:min_c]).abs().mean().item()
    std_err = (pred_std[:min_c] - ref_std[:min_c]).abs().mean().item()

    return float(mean_err), float(std_err)


def low_frequency_power_ratio(
    x: torch.Tensor,
    radius_fraction: float = 0.125,
) -> float:
    """Compute ratio of low-frequency power to total spectral power via 2D FFT.

    Args:
        x: Input tensor [..., H, W] (e.g. image or 2D latent map).
        radius_fraction: Normalized cutoff radius (0.0 to 1.0) defining low frequency.

    Returns:
        Fraction of spectral power contained within radius_fraction of origin.
    """
    if x.dim() < 2:
        return 1.0

    xf = x.detach().float()
    h, w = xf.shape[-2], xf.shape[-1]
    if h < 2 or w < 2:
        return 1.0

    # 2D real FFT over spatial dims
    fft = torch.fft.rfft2(xf)
    power = fft.abs() ** 2

    # Frequency grids: ky in [-0.5, 0.5), kx in [0, 0.5]
    ky = torch.fft.fftfreq(h, device=xf.device).unsqueeze(-1)  # [H, 1]
    kx = torch.fft.rfftfreq(w, device=xf.device).unsqueeze(0)   # [1, W//2 + 1]

    # Normalized radius where Nyquist (0.5) maps to 1.0
    norm_r = torch.sqrt((ky / 0.5) ** 2 + (kx / 0.5) ** 2)
    mask = norm_r <= radius_fraction

    low_power = power[..., mask].sum()
    total_power = power.sum().clamp_min(1e-8)

    return float((low_power / total_power).item())


def pixel_stats_gap(
    pred_img: torch.Tensor,
    ref_img: torch.Tensor,
) -> Dict[str, float]:
    """Compute luminance, saturation and variance metrics between images.

    Args:
        pred_img: Predicted image tensor [B, 3, H, W] or [3, H, W], in [0, 1] or [-1, 1].
        ref_img: Reference image tensor in same scale and channel layout.

    Returns:
        Dict with 'lum_err', 'sat_err', 'var_ratio'.
    """
    p = pred_img.detach().float()
    r = ref_img.detach().float()

    if p.dim() == 3:
        p = p.unsqueeze(0)
    if r.dim() == 3:
        r = r.unsqueeze(0)

    # Standard RGB to luminance Y = 0.299 R + 0.587 G + 0.114 B
    p_y = 0.299 * p[:, 0:1] + 0.587 * p[:, 1:2] + 0.114 * p[:, 2:3]
    r_y = 0.299 * r[:, 0:1] + 0.587 * r[:, 1:2] + 0.114 * r[:, 2:3]

    lum_err = (p_y.mean() - r_y.mean()).abs().item()

    p_sat = torch.sqrt(
        (p[:, 0:1] - p_y) ** 2 + (p[:, 1:2] - p_y) ** 2 + (p[:, 2:3] - p_y) ** 2
    ).mean().item()
    r_sat = torch.sqrt(
        (r[:, 0:1] - r_y) ** 2 + (r[:, 1:2] - r_y) ** 2 + (r[:, 2:3] - r_y) ** 2
    ).mean().item()
    sat_err = abs(p_sat - r_sat)

    p_var = p_y.var().item()
    r_var = r_y.var().item()
    var_ratio = p_var / max(r_var, 1e-6)

    return {
        "lum_err": float(lum_err),
        "sat_err": float(sat_err),
        "var_ratio": float(var_ratio),
    }


def cell_periodicity_power(
    x: torch.Tensor,
    period: int = 8,
) -> float:
    """Measure structural periodicity at step `period` (e.g. 8px grid artifact).

    Uses high-pass spatial autocorrelation at lag=period in horizontal and
    vertical directions. High values indicate repetitive grid artifacts.

    Args:
        x: Tensor [..., H, W] (e.g. image channel or latent map).
        period: Spatial lag in pixels/cells to test (default: 8).

    Returns:
        Autocorrelation coefficient in [-1, 1] at lag=period.
    """
    xf = x.detach().float()
    if xf.dim() < 2:
        return 0.0

    h, w = xf.shape[-2], xf.shape[-1]
    if h <= period or w <= period:
        return 0.0

    xf = xf.view(-1, h, w)

    # Spatial differences (high-pass filter to remove smooth low-frequency trends)
    diff_h = xf[:, 1:, :] - xf[:, :-1, :]  # [B, H-1, W]
    diff_w = xf[:, :, 1:] - xf[:, :, :-1]  # [B, H, W-1]

    # Vertical lag autocorrelation
    if diff_h.shape[1] > period:
        y1 = diff_h[:, :-period, :]
        y2 = diff_h[:, period:, :]
        std1 = y1.std().clamp_min(1e-6)
        std2 = y2.std().clamp_min(1e-6)
        corr_v = ((y1 - y1.mean()) * (y2 - y2.mean())).mean() / (std1 * std2)
    else:
        corr_v = torch.tensor(0.0)

    # Horizontal lag autocorrelation
    if diff_w.shape[2] > period:
        x1 = diff_w[:, :, :-period]
        x2 = diff_w[:, :, period:]
        std1 = x1.std().clamp_min(1e-6)
        std2 = x2.std().clamp_min(1e-6)
        corr_h = ((x1 - x1.mean()) * (x2 - x2.mean())).mean() / (std1 * std2)
    else:
        corr_h = torch.tensor(0.0)

    avg_corr = (corr_v.item() + corr_h.item()) / 2.0
    return float(max(-1.0, min(1.0, avg_corr)))


def trajectory_gap(
    single_lum_err: float,
    rollout_lum_err: float,
) -> float:
    """Discrepancy between single-step x0 prediction error and full rollout error."""
    return float(abs(single_lum_err - rollout_lum_err))

