"""Shared NAG (Normalized Attention Guidance) core for DiT / MM-DiT backends.

NAG extrapolates a model's attention output away from a negative text context in
attention-OUTPUT space (unlike CFG, which extrapolates the final noise). This module
holds the model-agnostic guidance math so each per-architecture hook (FLUX.2, Z-Image,
Anima, Lens, Ideogram4, MiniT2I) only has to:
  1. produce the attention output for the image(-token) queries against the POSITIVE
     text context (z_pos) and against the NAG-negative text context (z_neg), and
  2. call nag_guidance(z_pos, z_neg, ...) to get the guided output.

The formula is identical to the SDXL cross-attention NAG processor:
    g   = z_pos * scale - z_neg * (scale - 1)          # extrapolate (scale = nag_scale)
    s   = ||g||_1 / ||z_pos||_1     (over the feature dim)
    g   = g * min(s, tau) / s                          # L1-norm cap at tau
    out = g * alpha + z_pos * (1 - alpha)              # blend

NAG and Spectrum are orthogonal and compose: when both are on, the NAG-modified output
is what Spectrum records on anchor steps and forecasts on skip steps, so NAG's extra
attention cost is only paid on anchor steps.
"""

import torch


def nag_guidance(z_pos: torch.Tensor, z_neg: torch.Tensor,
                 scale: float = 5.0, tau: float = 3.5, alpha: float = 0.25,
                 feature_dim: int = -1) -> torch.Tensor:
    """Apply NAG output-space guidance. z_pos/z_neg are attention outputs of the same
    shape; the L1 norm is taken over ``feature_dim`` (the channel/head_dim axis).

    scale = nag_scale (phi), tau = nag_tau (L1 cap), alpha = nag_alpha (blend).
    Returns a tensor with the same shape/dtype as z_pos.
    """
    if scale == 1.0:
        # phi=1 -> guidance == z_pos, blend is a no-op; return z_pos unchanged.
        return z_pos
    dtype = z_pos.dtype
    zp = z_pos.float()
    zn = z_neg.float()
    guidance = zp * scale - zn * (scale - 1.0)
    norm_pos = torch.norm(zp, p=1, dim=feature_dim, keepdim=True)
    norm_g = torch.norm(guidance, p=1, dim=feature_dim, keepdim=True)
    s = norm_g / (norm_pos + 1e-8)
    cap = torch.minimum(s, torch.full_like(s, float(tau)))
    guidance = guidance * (cap / (s + 1e-8))
    out = guidance * alpha + zp * (1.0 - alpha)
    return out.to(dtype)


def nag_active(nag_enable: bool, nag_scale: float, nag_negative_embeds) -> bool:
    """Whether NAG should run: enabled, scale != 1, and a negative context is available."""
    return bool(nag_enable) and abs(float(nag_scale) - 1.0) > 1e-5 and nag_negative_embeds is not None
