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
    l_invented      -- flat-region invented-HF penalty (InventedHfLoss). Added
                       after results_flat_region_noise.md measured that the
                       three ON-by-default terms above, being all
                       agreement-with-source objectives, cut the flat-region
                       ERROR 21% while leaving the total high-frequency energy
                       the decoder emits there unchanged (+0.4%, ns). This is
                       the only term in the bank that is not an
                       agreement-with-source objective. Opt-in only; see the
                       class docstring.

No GAN and no crop-consistency term (design.md §5.2, §9.2: the crop residual
after the free inference-time fix is 0.03-0.16/255, and a short-run GAN is the
single most likely way to make a fine-tune worse).

  constructed only when the ENCODER is trainable
    kl        1e-6  -- posterior KL, weighted against a PER-ELEMENT-normalised
                       KL so that this number means what it means in LDM (whose
                       reconstruction term is summed, not averaged, over C*H*W).
                       Under a frozen encoder the term is a constant w.r.t.
                       every trainable parameter, so it is not constructed at
                       all (``kl_enabled=False``) and the weight is ignored.
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


class InventedHfLoss(torch.nn.Module):
    """Flat-region invented-high-frequency penalty (``L_invented``).

    Every other term in this bank is an agreement-with-source objective — the
    same family the SDXL VAE was already trained on. Measured outcome of a full
    fine-tune under that bank (scratchpad/vae_training/results_flat_region_noise.md):
    the *error* fell 21% while the total high-frequency energy the decoder emits
    inside flat regions moved +0.4% (not significant), i.e. the decoder kept
    fabricating the same amount of detail and merely aimed it better. In dark
    flat windows 66% of the fine high frequency is fabricated, and amplified to
    the exposure a user actually inspects it at, the invented luma is 3.3/255
    against a visibility bar of 1/255.

    This term is a **conditional non-generation** objective instead: inside
    windows where the source is flat or a smooth gradient, it penalises the part
    of the decode's high frequency that a least-squares projection onto the
    *source's own* high frequency cannot explain.

    Per window ``w``, per channel, per scale, with ``s = h(target)``,
    ``d = h(recon)``, both window-mean-removed::

        alpha = clamp( <d,s> / (<s,s> + eps) , 0, 2 )        # DETACHED
        r     = d - alpha * s
        L_win = mean(r^2)

    **alpha carries no gradient.** This is the load-bearing detail. At the
    least-squares optimum d(L)/d(alpha) = 0, so the through-alpha gradient is a
    second-order correction rather than signal; and an attached alpha would hand
    the decoder a path to reduce the loss by *raising its correlation with the
    source* instead of emitting less — which is precisely the behaviour
    ``mse+lpips`` was measured to reward. With alpha fixed, each step is a plain
    weighted MSE toward the fixed target ``alpha*s``, and the only way down is
    to stop emitting unexplained energy.

    ``eps`` is set at the MEASURED 8-bit quantisation-floor energy
    (``N * 0.2797^2``; the source floor is 0.2797/255 rms,
    results_flat_region_noise.md §1.4), so in a genuinely flat window ``<s,s>``
    is dominated by eps, alpha -> 0 smoothly and the term degrades to "where the
    source has nothing, emit nothing" — with no division blow-up and no NaN
    path.

    **What this term actually charges — stated against the natural misreading.**
    With ``eps > 0`` the coefficient is ``alpha = g * sigma^2/(sigma^2 + eps)``
    for a decode ``d = g*s`` whose source HF has per-pixel rms ``sigma``, so the
    per-scale charge is ``sigma^2 * (g - alpha)^2``. Three consequences, all of
    which contradict the way an epsilon-free projection would read:

    * **Nothing is exempt.** The charge rises smoothly and monotonically with
      ``g`` from ``g = 0`` upward — exactly as ``g^2`` while alpha tracks, since
      ``alpha = g*k``. The ``ALPHA_MAX`` clamp caps how much exemption a
      strongly-correlated emission can buy; it does not make gains below it
      free. (Measured at sigma ~ 1.5, two scales, Weber-weighted: g=0 -> 0
      exactly, g=0.5 -> 0.00223, g=1 -> 0.00892, g=2 -> 0.03567,
      g=3 -> 0.62209.)
    * **Exact reproduction is charged**, at ``sigma^2*eps^2/(sigma^2+eps)^2``
      per scale — maximal at ``sigma = sqrt(eps)``, where it is ``eps/4``
      (0.0196 levels^2, i.e. 0.140 levels rms per scale, ~0.198 combined).
    * **Blur is charged LESS than exact reproduction**, at every sigma, and
      emitting nothing at all is charged EXACTLY zero. The term's own fixed
      point inside a flat window is ``d = alpha*s`` with ``alpha < 1``: a
      shrink, not the identity, and its global minimum over ``g`` is ``g = 0``.
      Measured on 200 real 512 crops at this geometry, the systematic
      under-emission of *transmitted* HF at the fixed point is 5.1% of the
      in-mask transmitted HF energy amplitude (worst case 0.140 levels per
      scale). This is why the term is opt-in and is not a standalone
      objective: it must be run alongside an agreement-with-source term
      (``mse``/``lpips``), which is what supplies the opposing pull toward
      ``g = 1``. Configuring it as the *only* active weight makes "emit nothing"
      the global optimum of the whole run.

    The clamp refuses to exempt anti-correlated "transmission" (alpha < 0, which
    is invention) and caps the exemption at a 2x-amplified copy of the source's
    texture (alpha > 2, itself a visible defect in a flat region).

    Everything below is in **8-bit levels**, but ``sqrt(the logged value)`` is a
    RELATIVE TREND INDICATOR, NOT an absolute level: the logged number carries
    the Weber photometric weight (0.16 bright .. 0.98 black) and the channel
    weights, so it under-reads true invented luma by 1.1x in dark windows and
    ~2.5x in bright ones (measured by injecting exactly 1.0 level of pure
    uncorrelated invention: sqrt(logged) = 0.94 dark / 0.52 mid / 0.40 bright).
    Absolute levels come from the frozen harness, never from this number.

    Halo, stated because it qualifies "only inside regions that should be
    smooth": ``_highpass`` runs on the full image (h2 has a support radius of 3
    px) while the flat mask is a plane fit on the window's own pixels, so a hard
    edge up to 3 px OUTSIDE a selected window contributes to that window's
    charge (252 of the 576 window pixels are within 3 px of the boundary). The
    direction is anti-blur — it charges a window for edge energy that leaked in
    — so it is not a new softening risk, and the geometry is deliberately left
    alone.

    Fixed internal constants, NOT user parameters: window 24 px / stride 12 /
    inset 32, the two-scale binomial highpass (b3, b7), the alpha epsilon and
    clamp, and the Weber photometric weight mu0 = 48. They are deliberately
    different from the evaluation harness's geometry (32/16/48, a single
    sigma=2.0 Gaussian highpass, a clipped 128/mu exposure gain, an unclamped
    float64 projection) so that agreement between "the loss went down" and "the
    frozen meter went down" is evidence about the picture rather than an
    identity. Exposing them as knobs would invite re-deriving the meter, which
    is why only the five weights/thresholds are user-facing.
    """

    # --- geometry (see the class docstring: deliberately not the meter's) ---
    #
    # LOAD-BEARING VALUES, NOT STYLE. The evidential argument for this whole
    # experiment (design_l_invented.md §3) is that the loss and the frozen
    # evaluation meter are different functionals, so "the loss went down AND the
    # meter went down" is evidence about the picture rather than an identity.
    # The meter's geometry is 32/16/48; these are 24/12/32. "Harmonising" them
    # would silently turn the experiment into a tautology while every behavioural
    # test still passed, which is why test_l_invented_loss.py asserts these exact
    # numbers.
    WINDOW = 24
    STRIDE = 12
    INSET = 32
    # The MEASURED 8-bit quantisation-floor energy per pixel: q_floor = 0.2797
    # levels rms (results_flat_region_noise.md §1.4), so 0.2797^2. NOT the
    # (0.5 level)^2 half-step: the highpass of a uniform requantisation has rms
    # 0.28, not 0.5, and the larger value cost 2x more systematic attenuation of
    # *transmitted* HF (0.354 -> 0.198 levels worst case, 9.5% -> 5.1% of in-mask
    # transmitted HF energy on 200 measured crops) to buy a reduction in
    # spuriously-exempted invented energy of 0.010% -> 0.043% — i.e. it paid in
    # the project's primary failure mode (blur) for a change in a negligible one.
    # See the class docstring and design_l_invented.md §1.3.
    ALPHA_EPS_PER_PIXEL = 0.2797 ** 2
    ALPHA_MAX = 2.0
    # Smooth Weber-style dark emphasis, spanning 48/303 (white) to 48/49
    # (black) — a ~6x dark-over-bright emphasis, and NOT the meter's up-to-256x
    # clipped exposure gain.
    WEBER_MU0 = 48.0

    def __init__(self, y_weight: float, chroma_weight: float,
                 flat_t_y: float, flat_t_c: float):
        super().__init__()
        self.y_weight = float(y_weight)
        self.chroma_weight = float(chroma_weight)
        self.flat_t_y = float(flat_t_y)
        self.flat_t_c = float(flat_t_c)

        n = self.WINDOW
        ones = torch.ones(n)
        # Window-local coordinates, centred so that sum(u) = sum(v) = 0 and
        # sum(u*v) = 0 over the window: that is what makes the plane fit
        # closed-form from four pooled sums.
        ramp = torch.arange(n, dtype=torch.float32) - (n - 1) / 2.0
        self.register_buffer("_ones", ones, persistent=False)
        self.register_buffer("_ramp", ramp, persistent=False)
        # sum over the whole window of u^2 (== of v^2), i.e. n * sum_i u_i^2.
        self._ramp_sq_sum = float(n * (ramp * ramp).sum())
        self._n_pixels = float(n * n)
        self._alpha_eps = self._n_pixels * self.ALPHA_EPS_PER_PIXEL
        self.register_buffer("_b3", torch.tensor([0.25, 0.5, 0.25]),
                             persistent=False)

    # -- primitives ------------------------------------------------------
    def _blur(self, x: torch.Tensor, times: int) -> torch.Tensor:
        """`times` applications of the separable binomial [1,2,1]/4 kernel.

        1 application is b3 (3x3); 3 applications is b7 (sigma_eff ~1.22 px).
        Replicate padding, so the operator is defined everywhere; the border is
        cropped away by INSET before any window is taken anyway.
        """
        c = x.shape[1]
        k = self._b3.to(x.dtype)
        kv = k.view(1, 1, 3, 1).repeat(c, 1, 1, 1)
        kh = k.view(1, 1, 1, 3).repeat(c, 1, 1, 1)
        for _ in range(times):
            x = F.pad(x, (0, 0, 1, 1), mode="replicate")
            x = F.conv2d(x, kv, groups=c)
            x = F.pad(x, (1, 1, 0, 0), mode="replicate")
            x = F.conv2d(x, kh, groups=c)
        return x

    def _highpass(self, x: torch.Tensor):
        """The two scales, covering the measured 0.09-0.53 cyc/px band.

        h1 = x - b3(x)        fine scale
        h2 = b3(x) - b7(x)    mid scale
        """
        b1 = self._blur(x, 1)
        b3 = self._blur(b1, 2)
        return x - b1, b1 - b3

    def _window_sum(self, x: torch.Tensor, vertical: torch.Tensor,
                    horizontal: torch.Tensor) -> torch.Tensor:
        """Pooled sum over every 24x24 window at stride 12, closed-form.

        Separable strided convolution, NOT ``unfold``: the per-window statistics
        this term needs are all sums of x, x*u, x*v and x^2, and materialising
        576 pixels per window per channel would cost two orders of magnitude
        more memory than the pooled form for no extra information.
        """
        c = x.shape[1]
        kv = vertical.to(x.dtype).view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        kh = horizontal.to(x.dtype).view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        x = F.conv2d(x, kv, stride=(self.STRIDE, 1), groups=c)
        return F.conv2d(x, kh, stride=(1, self.STRIDE), groups=c)

    # -- the term --------------------------------------------------------
    def forward(self, recon: torch.Tensor, target: torch.Tensor):
        """Returns ``(loss, coverage)``; both are 0-d tensors.

        ``recon``/``target`` are in [-1, 1]. ``coverage`` is the fraction of
        candidate windows that passed the flat test this step — logged so that a
        run where the term almost never fires is visible on the chart instead of
        silently doing nothing.
        """
        # The VAE trains under autocast; these are 8-bit-level energies summed
        # over 576 pixels, which is not a bf16 quantity. Everything below runs
        # in fp32 regardless of the surrounding autocast state.
        with torch.autocast(device_type=recon.device.type, enabled=False):
            return self._forward_fp32(recon.float(), target.float())

    def _forward_fp32(self, recon: torch.Tensor, target: torch.Tensor):
        n_px = self._n_pixels
        inset, win = self.INSET, self.WINDOW
        h, w = recon.shape[-2], recon.shape[-1]
        if min(h, w) < 2 * inset + win:
            # Too small to hold a single window inside the border inset (the
            # config allows resolutions down to 64 px). The zero is kept
            # CONNECTED to the graph rather than a bare new_zeros: with this as
            # the only active term, an unconnected total would make the
            # trainer's backward() raise instead of taking a null step.
            zero = recon.sum() * 0.0
            return zero, zero.detach()

        def crop(x):
            return x[..., inset:h - inset, inset:w - inset]

        # 8-bit levels: the logged value is then a (levels)^2 energy, readable as
        # a relative trend. It is NOT an absolute level — the Weber and channel
        # weights applied below mean sqrt() under-reads true invented luma (see
        # the class docstring).
        recon_y = rgb01_to_ycbcr((recon + 1.0) * 0.5) * 255.0
        with torch.no_grad():
            target_y = rgb01_to_ycbcr((target + 1.0) * 0.5) * 255.0
            mask, photo = self._flat_mask(crop(target_y))
            n_windows = float(mask.numel())
            selected = mask.sum()
            target_h1, target_h2 = self._highpass(target_y)
            target_h1, target_h2 = crop(target_h1), crop(target_h2)

        recon_h1, recon_h2 = self._highpass(recon_y)
        recon_h1, recon_h2 = crop(recon_h1), crop(recon_h2)

        ones = self._ones
        chan = recon.new_tensor([self.y_weight,
                                 self.chroma_weight * 0.5,
                                 self.chroma_weight * 0.5]).view(1, 3, 1, 1)

        # The two scales are summed with equal weight: the measured spectrum of
        # the defect is broad (0.09-0.53 cyc/px), not peaked.
        numerator = recon.new_zeros(())
        for d_map, s_map in ((recon_h1, target_h1), (recon_h2, target_h2)):
            with torch.no_grad():
                s_sum = self._window_sum(s_map, ones, ones)
                s_sq = self._window_sum(s_map * s_map, ones, ones)
                c_ss = s_sq - s_sum * s_sum / n_px
            d_sum = self._window_sum(d_map, ones, ones)
            d_sq = self._window_sum(d_map * d_map, ones, ones)
            ds = self._window_sum(d_map * s_map, ones, ones)
            c_dd = d_sq - d_sum * d_sum / n_px
            c_ds = ds - d_sum * s_sum / n_px

            # DETACHED. See the class docstring: an attached alpha would let the
            # decoder buy an exemption by correlating with the source instead of
            # emitting less.
            alpha = torch.clamp(c_ds.detach() / (c_ss + self._alpha_eps),
                                0.0, self.ALPHA_MAX)
            # ||d - alpha*s||^2 / N, expanded so nothing per-pixel is
            # materialised. clamp_min guards fp round-off only: the quantity is
            # a squared norm.
            l_win = ((c_dd - 2.0 * alpha * c_ds + alpha * alpha * c_ss)
                     / n_px).clamp_min(0.0)
            numerator = numerator + ((l_win * chan).sum(dim=1) * photo * mask).sum()

        # Reduce by the total WINDOW count across the micro-batch, not by image
        # count: a batch element with no flat window then contributes zero to
        # both numerator and denominator, so it neither drags the mean toward 0
        # nor inflates the other element's windows.
        loss = numerator / torch.clamp(selected, min=1.0)
        coverage = selected / max(n_windows, 1.0)
        return loss, coverage

    def _flat_mask(self, target_ycbcr: torch.Tensor):
        """Plane-fit flat/gradient window selection, on the TARGET only.

        A least-squares plane ``a + b*u + c*v`` is fitted per window per channel
        and the window is FLAT when the residual RMS is within the thresholds.
        A plane fit, not a variance test: a smooth ramp is exactly the case this
        term has to cover, and a variance test would exclude every one of them.

        The target is the source (the encoder is frozen and never trained here),
        so this is source-side selection, computed under ``no_grad`` — it is not
        a gradient path and cannot be gamed by the decoder.
        """
        n_px = self._n_pixels
        ones, ramp = self._ones, self._ramp
        # Remove the per-image per-channel mean before the pooled sums. The
        # plane absorbs any constant, so the residual is unchanged, but it keeps
        # sum(x^2) ~1e4 instead of ~4e7 and so keeps the cancellation in
        # (sum x^2 - (sum x)^2/N) inside fp32's mantissa.
        offset = target_ycbcr.mean(dim=(-2, -1), keepdim=True)
        x = target_ycbcr - offset

        s1 = self._window_sum(x, ones, ones)
        su = self._window_sum(x, ramp, ones)
        sv = self._window_sum(x, ones, ramp)
        s2 = self._window_sum(x * x, ones, ones)
        rss = (s2 - s1 * s1 / n_px
               - su * su / self._ramp_sq_sum
               - sv * sv / self._ramp_sq_sum)
        rms = (rss.clamp_min(0.0) / n_px).sqrt()

        flat = (rms[:, 0] <= self.flat_t_y) & (
            torch.maximum(rms[:, 1], rms[:, 2]) <= self.flat_t_c)
        # Window mean luma, back in absolute 8-bit levels.
        mu = (s1[:, 0] / n_px + offset[:, 0]).clamp_min(0.0)
        photo = self.WEBER_MU0 / (mu + self.WEBER_MU0)
        return flat.to(target_ycbcr.dtype), photo


class VaeLossBank(torch.nn.Module):
    """Weighted sum of the enabled reconstruction terms.

    Inputs are the training tensors in **[-1, 1]** (the VAE's own pixel range),
    exactly as produced by ``vae_dataset`` and returned by ``vae.decode``.
    """

    def __init__(self, cfg: Dict, device: torch.device, *, kl_enabled: bool = False):
        super().__init__()
        # The KL term exists only when the encoder is trainable. With a frozen
        # encoder the posterior does not depend on any trainable parameter, so
        # the term is a constant: adding it would inflate the logged total loss
        # by a fixed amount and contribute exactly zero gradient.
        self.kl_enabled = bool(kl_enabled)
        self.kl_weight = float(cfg["kl_weight"]) if self.kl_enabled else 0.0
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

        # Constructed only when its weight is above 0 (same lazy pattern as
        # pattern_loss), so a default run pays nothing for it.
        self.l_invented_weight = float(cfg["l_invented_weight"])
        self.invented_loss = None
        if self.l_invented_weight > 0:
            self.invented_loss = InventedHfLoss(
                y_weight=float(cfg["l_invented_y_weight"]),
                chroma_weight=float(cfg["l_invented_chroma_weight"]),
                flat_t_y=float(cfg["l_invented_flat_t_y"]),
                flat_t_c=float(cfg["l_invented_flat_t_c"]),
            )

        self.register_buffer(
            "_dc_channel_weights",
            torch.tensor([self.dc_y_weight, self.dc_chroma_weight, self.dc_chroma_weight])
            .view(1, 3, 1, 1),
            persistent=False,
        )
        self.to(device)

    def forward(
        self, recon: torch.Tensor, target: torch.Tensor, posterior=None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns ``(total_loss, {component_name: float})``.

        ``posterior`` is the ``DiagonalGaussianDistribution`` returned by
        ``vae.encode(...).latent_dist``. It is required when ``kl_enabled`` (i.e.
        when the encoder is trainable) and ignored otherwise.
        """
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
        if self.invented_loss is not None:
            inv, cov = self.invented_loss(recon32, target32)
            # Unweighted, in (8-bit levels)^2 — but Weber- and channel-weighted
            # inside the term, so sqrt() of it is a RELATIVE TREND INDICATOR and
            # NOT a level readable against the 1/255 bar (it under-reads true
            # invented luma by 1.1x dark to ~2.5x bright). Absolute levels come
            # from the frozen g1flat harness.
            parts["l_invented"] = float(inv.detach())
            # Fraction of candidate windows that passed the flat test. A run
            # where the term almost never fires must be visible, not silent.
            parts["l_invented_cov"] = float(cov.detach())
            total = total + self.l_invented_weight * inv

        if self.kl_enabled and self.kl_weight > 0:
            if posterior is None:
                raise ValueError(
                    "VaeLossBank was built with kl_enabled=True (the encoder is "
                    "trainable) but no posterior was passed to forward(); the KL "
                    "term cannot be computed."
                )
            # DiagonalGaussianDistribution.kl() sums over the latent dims and
            # returns one value per batch item, matching the LDM formulation
            # (0.5 * sum(mean^2 + var - 1 - logvar)). That is the literature's
            # magnitude, and it is what gets LOGGED.
            kl_raw = posterior.kl().float().mean()
            #
            # ...but it must NOT be weighted as-is. LDM's contperceptual.py pairs
            # kl_weight with a reconstruction term that is SUMMED over C*H*W per
            # image (`nll_loss = torch.sum(nll) / nll.shape[0]`), whereas every
            # reconstruction term above is MEAN-reduced over B*C*H*W. The two
            # conventions differ by exactly C*H*W (~786k at 512px), so applying
            # LDM's 1e-6 to a per-element recon would make the KL ~15x the MSE
            # (measured on this install's SDXL VAE at 512px: 0.519 vs 0.034) —
            # a run that is 90% "pull the posterior to N(0,I)" and only 10%
            # reconstruction, and whose balance would additionally shift 4x
            # between 256 and 512px.
            #
            # Dividing by the per-image element count puts the KL in the same
            # reduction as the reconstruction terms. After it, kl_weight=1e-6 IS
            # the LDM value in the LDM sense, and it is resolution-invariant.
            elements_per_image = int(target32.shape[1] * target32.shape[2]
                                     * target32.shape[3])
            kl = kl_raw / max(elements_per_image, 1)
            contribution = self.kl_weight * kl
            # Logged: the raw KL (comparable with the literature and with other
            # trainers) and the actual weighted contribution to `total`, which is
            # what belongs on the loss chart next to the other components.
            parts["kl"] = float(kl_raw.detach())
            parts["kl_term"] = float(contribution.detach())
            total = total + contribution

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
                f"pattern={self.pattern_weight}",
                f"l_invented={self.l_invented_weight}"]
        bits.append(f"kl={self.kl_weight}" if self.kl_enabled
                    else "kl=not constructed (encoder frozen)")
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
