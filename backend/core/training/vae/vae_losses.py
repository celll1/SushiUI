"""Reconstruction loss bank + validation metrics for VAE decoder fine-tuning.

Loss set and default weights follow design.md §5.1 **as revised by §9.2** (the
Phase-0 measurement outcomes):

  ON by default
    mse       1.0   -- stabilityai/sd-vae-ft-mse's base term. ft-MSE differs from
                       ft-EMA in exactly two ways: L1 -> MSE and LPIPS 1.0 -> 0.1.
    lpips     0.1   -- ft-MSE's weight. NOT 1.0: LPIPS is the term that *creates*
                       plausible high frequency, so a larger weight works against
                       the artifact this fine-tune is meant to suppress.
    ycbcr_dc  0.1   -- PiD's own colour-drift term (Charbonnier on YCbCr, luma
                       downweighted) PLUS an explicit Charbonnier on the
                       spatial-mean (DC) difference, under the same weight.
                       Phase 0 measured 39-51/255 of red DC drift over 8
                       encode/decode roundtrips on the SDXL VAEs -- a
                       spatial-mean defect, which a purely per-pixel penalty
                       barely constrains. This is the term that stops a
                       "successful" fine-tune from regressing under iterative
                       img2img.

  available, default 0
    l1              -- the LDM/ft-EMA reconstruction term; usable instead of, or
                       alongside, MSE.
    pattern         -- latent-cell grid-phase penalty. Phase 0 (M2) measured the
                       8 px grid artifact at ratio ~1.0 on all four VAEs under
                       three independent metric definitions, i.e. the defect this
                       term targets is ABSENT at measurement level. Opt-in only.

No GAN, no crop-consistency term and no invented-HF term in v1 (design.md §5.2,
§9.2: the crop residual after the free inference-time fix is 0.03-0.16/255, and
a short-run GAN is the single most likely way to make a fine-tune worse).

Under a frozen encoder the KL term is a constant w.r.t. the trainable
parameters, so it is not constructed at all.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# JPEG full-range RGB->YCbCr, on 0..1 inputs (same coefficients as the Phase-0
# harness, scratchpad/vae_training/harness/vae_probe.py:234).
_YCBCR_M = (
    (0.299, 0.587, 0.114),
    (-0.168736, -0.331264, 0.5),
    (0.5, -0.418688, -0.081312),
)
_YCBCR_B = (0.0, 0.5, 0.5)


def rgb01_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] in 0..1 -> YCbCr in 0..1 (Cb/Cr centred at 0.5)."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    out = []
    for (cr, cg, cb), off in zip(_YCBCR_M, _YCBCR_B):
        out.append(cr * r + cg * g + cb * b + off)
    return torch.stack(out, dim=1)


def charbonnier(diff: torch.Tensor, eps: float) -> torch.Tensor:
    """sqrt(diff^2 + eps^2) - eps  (PiD's aux-RGB-head formulation)."""
    return torch.sqrt(diff * diff + eps * eps) - eps


class PatternLoss(torch.nn.Module):
    """Latent-cell grid-phase bias penalty (ostris PatternLoss, in concept).

    For a pattern size p, the residual (x_hat - x) is grouped by its (row % p,
    col % p) phase; the loss is the variance of the per-phase mean residual.
    A decoder that biases specific positions within each latent cell -- the
    classic "8 px block" signature -- has a non-uniform per-phase mean; a decoder
    whose error is phase-independent scores 0. Being a statistic-matching term it
    cannot diverge the way a raw high-frequency penalty can.

    NOTE: this is a re-implementation of the concept described in design.md
    §5.1, not a byte-copy of ai-toolkit's class (which is not vendored here).
    Default weight is 0 -- see the module docstring for why.
    """

    def __init__(self, pattern_size: int = 8):
        super().__init__()
        self.pattern_size = int(pattern_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.pattern_size
        b, c, h, w = pred.shape
        h_c, w_c = (h // p) * p, (w // p) * p
        if h_c < p or w_c < p:
            return pred.new_zeros(())
        diff = (pred[..., :h_c, :w_c] - target[..., :h_c, :w_c])
        # [B, C, h/p, p, w/p, p] -> mean over the cell index axes -> [B, C, p, p]
        phase = diff.reshape(b, c, h_c // p, p, w_c // p, p).mean(dim=(2, 4))
        return phase.var(dim=(-2, -1), unbiased=False).mean()


class VaeLossBank(torch.nn.Module):
    """Weighted sum of the enabled reconstruction terms.

    Inputs are the training tensors in **[-1, 1]** (the VAE's own pixel range),
    exactly as produced by ``vae_dataset`` and returned by ``vae.decode``.
    """

    def __init__(self, cfg: Dict, device: torch.device):
        super().__init__()
        self.mse_weight = float(cfg["mse_weight"])
        self.l1_weight = float(cfg["l1_weight"])
        self.lpips_weight = float(cfg["lpips_weight"])
        self.ycbcr_dc_weight = float(cfg["ycbcr_dc_weight"])
        self.pattern_weight = float(cfg["pattern_weight"])
        self.dc_y_weight = float(cfg["ycbcr_dc_y_weight"])
        self.dc_chroma_weight = float(cfg["ycbcr_dc_chroma_weight"])
        self.dc_eps = float(cfg["ycbcr_dc_eps"])

        self.lpips_model = None
        if self.lpips_weight > 0:
            # Import here, not at module import time: the availability check with
            # an explicit message already ran in vae_config._validate, so by the
            # time we get here the package is known to exist.
            import lpips as _lpips
            self.lpips_model = _lpips.LPIPS(net=str(cfg["lpips_net"]), verbose=False)
            self.lpips_model.to(device).eval()
            for p in self.lpips_model.parameters():
                p.requires_grad_(False)

        self.pattern_loss = None
        if self.pattern_weight > 0:
            self.pattern_loss = PatternLoss(int(cfg["pattern_size"]))

        self.register_buffer(
            "_dc_channel_weights",
            torch.tensor([self.dc_y_weight, self.dc_chroma_weight, self.dc_chroma_weight])
            .view(1, 3, 1, 1),
            persistent=False,
        )
        self.to(device)

    def forward(
        self, recon: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns ``(total_loss, {component_name: float})``."""
        parts: Dict[str, float] = {}
        total = recon.new_zeros(())

        # Everything is computed in fp32: the loss magnitudes here (LPIPS ~0.1,
        # Charbonnier ~1e-3) are well inside bf16's 3-decimal-digit mantissa and
        # would otherwise quantise visibly in the logged metrics.
        recon32 = recon.float()
        target32 = target.float()

        if self.mse_weight > 0:
            mse = F.mse_loss(recon32, target32)
            parts["mse"] = float(mse.detach())
            total = total + self.mse_weight * mse
        if self.l1_weight > 0:
            l1 = F.l1_loss(recon32, target32)
            parts["l1"] = float(l1.detach())
            total = total + self.l1_weight * l1
        if self.lpips_model is not None:
            # lpips.LPIPS expects [-1,1] inputs, which is our native range.
            lp = self.lpips_model(recon32, target32).mean()
            parts["lpips"] = float(lp.detach())
            total = total + self.lpips_weight * lp
        if self.ycbcr_dc_weight > 0:
            dc = self._ycbcr_dc(recon32, target32)
            parts["ycbcr_dc"] = float(dc.detach())
            total = total + self.ycbcr_dc_weight * dc
        if self.pattern_loss is not None:
            pat = self.pattern_loss(recon32, target32)
            parts["pattern"] = float(pat.detach())
            total = total + self.pattern_weight * pat

        return total, parts

    def _ycbcr_dc(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-pixel Charbonnier on YCbCr PLUS a Charbonnier on the spatial-mean
        (true DC) difference, both channel-weighted, under one weight.

        The defect this term exists for is a *spatial-mean* drift: Phase 0
        measured 39-51/255 of red DC accumulating over 8 encode/decode
        roundtrips. A per-pixel penalty counts that drift as one contribution
        among ~250k residuals, so on its own it barely constrains it; the
        explicit per-image, per-channel mean term does.

        No clamp on the recon side: clamping would zero the gradient exactly on
        the overshooting pixels that are the most likely to be colour-drifting.
        (The target is data and is already in range.)
        """
        a = rgb01_to_ycbcr((recon + 1.0) * 0.5)
        b = rgb01_to_ycbcr((target + 1.0) * 0.5)
        w = self._dc_channel_weights.to(a.dtype)
        pixel = (charbonnier(a - b, self.dc_eps) * w).mean()
        # [B,3] per-image per-channel mean difference -> the DC term proper.
        dc = (charbonnier(a.mean(dim=(-2, -1)) - b.mean(dim=(-2, -1)), self.dc_eps)
              * w.view(1, 3)).mean()
        return pixel + dc

    def describe(self) -> str:
        bits = [f"mse={self.mse_weight}", f"l1={self.l1_weight}",
                f"lpips={self.lpips_weight}", f"ycbcr_dc={self.ycbcr_dc_weight}",
                f"pattern={self.pattern_weight}"]
        return ", ".join(bits)


# ---------------------------------------------------------------------------
# Validation metrics (no gradients) -- the user's only signal that a fine-tune
# is going wrong, so they are charted every validation interval.
# ---------------------------------------------------------------------------

def psnr(recon: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB over the 0..1 range, from [-1,1] inputs."""
    a = (recon.float().clamp(-1, 1) + 1.0) * 0.5
    b = (target.float().clamp(-1, 1) + 1.0) * 0.5
    mse = F.mse_loss(a, b)
    if float(mse) <= 0:
        return 99.0
    return float(10.0 * torch.log10(1.0 / mse))


def blockiness(recon: torch.Tensor, target: torch.Tensor, period: int = 8) -> float:
    """M2 ``block_step_ratio``: mean |d residual| ACROSS latent-cell boundaries
    divided by the same WITHIN cells (h and v pooled), on the residual
    ``recon - target`` in 8-bit levels.

    1.0 means "no cell-aligned discontinuity structure"; that is what Phase 0
    measured on all four production VAEs, so this metric is here as a
    *regression guard* -- if a fine-tune starts manufacturing grid structure it
    will move above 1.0. Definition copied from
    scratchpad/vae_training/harness/m2_blockiness.py:90.
    """
    r = ((recon.float().clamp(-1, 1) - target.float().clamp(-1, 1)) * 0.5) * 255.0
    # 48 px interior inset: the Phase-0 harness excludes the zero-padding border
    # band (measured at 64-128 px), which otherwise dominates the statistic.
    inset = 48 if min(r.shape[-2:]) > 4 * 48 else 0
    if inset:
        r = r[..., inset:-inset, inset:-inset]
    if min(r.shape[-2:]) < 2 * period:
        return float("nan")

    boundary, inner = [], []
    for axis in (3, 2):
        d = (r.narrow(axis, 1, r.shape[axis] - 1)
             - r.narrow(axis, 0, r.shape[axis] - 1)).abs().mean(1, keepdim=True)
        n = d.shape[axis]
        idx = torch.arange(n, device=d.device)
        on = ((idx + 1) % period == 0)
        shape = [1, 1, 1, 1]
        shape[axis] = n
        on = on.view(shape).expand_as(d)
        boundary.append(float(d[on].mean()))
        inner.append(float(d[~on].mean()))
    b = 0.5 * (boundary[0] + boundary[1])
    i = 0.5 * (inner[0] + inner[1])
    return b / i if i > 0 else float("nan")
