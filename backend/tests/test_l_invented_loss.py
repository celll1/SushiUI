"""Behavioural guard for the flat-region invented-HF loss (``L_invented``).

Run from the repository root with the repo's virtualenv interpreter
(``venv/Scripts/python.exe`` on Windows, ``venv/bin/python`` on POSIX):

    venv/Scripts/python.exe -m pytest backend/tests/test_l_invented_loss.py -v

or, without pytest:

    venv/Scripts/python.exe -m unittest discover -s backend/tests -p "test_l_invented_loss.py"

CPU-only, no model load, no dataset, real torch. Everything here is a synthetic
tensor built in-process, so the file is hermetic and fast.

What this file is for: ``InventedHfLoss`` is the only term in the VAE loss bank
that is NOT an agreement-with-source objective, and the properties that make it
that are all *silent* — a version with the projection coefficient attached to
the graph, or with the orthogonal remainder replaced by a plain high-frequency
magnitude, trains perfectly happily and produces a loss curve that goes down.
It is simply optimising something else (in the second case, "emit less detail
everywhere", i.e. blur — the failure mode this whole feature exists to avoid).
Each such substitution is pinned by a case below; each case was checked by
mutating the implementation and confirming it fails.

Two further categories, added after an adversarial audit found that the suite
passed while the implementation was mutated:

* ``EpsilonFixedPointTest`` / ``LoggedValueIsNotALevelTest`` pin what the term
  actually charges — including the two facts the docs originally denied: exact
  reproduction is NOT free, and a blur is charged LESS than exact reproduction.
  These are properties, not aspirations, and they are the reason the term is
  opt-in, must not be the only active loss weight, and is gated by R7.
* ``InternalGeometryTest`` pins the window geometry, the highpass basis and the
  projection constants BY VALUE. They are derived-from nowhere, every other case
  adapts to them, and they are the whole evidential separation between this loss
  and the frozen evaluation meter (design_l_invented.md §3): mutating 24/12/32
  to the meter's 32/16/48 previously passed all 38 cases.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

# ── path setup ───────────────────────────────────────────────────────────────
# `backend` itself must be on sys.path: the modules under test import
# `api.param_defaults` / `core.training.*` with backend as the root package dir.
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import torch
import torch.nn.functional as F

from api.param_defaults import VAE_TRAINING_DEFAULTS
from core.training.vae.vae_config import (
    VaeConfigError,
    resolve_vae_training_config,
)
from core.training.vae.vae_losses import (
    InventedHfLoss,
    VaeLossBank,
    rgb01_to_ycbcr,
)

_CPU = torch.device("cpu")

# One window is 24 px at stride 12 inside a 32 px inset, so 88 px is the
# smallest image that holds exactly one window and 112 px holds 3x3 of them.
_ONE_WINDOW = 2 * InventedHfLoss.INSET + InventedHfLoss.WINDOW  # 88


def _gray(values: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] luminance in [-1,1] -> a neutral-grey [B,3,H,W] image."""
    return values.repeat(1, 3, 1, 1)


def _flat_image(size: int, level: float = -0.4, batch: int = 1) -> torch.Tensor:
    return _gray(torch.full((batch, 1, size, size), float(level)))


def _scaled_detail(seed: int, sigma_levels: float, size: int = 112,
                   level: float = -0.4):
    """A CONSTANT base plus a detail field, so that ``base + g*detail`` emits
    exactly ``g`` times the source's own high frequency **at every scale**.

    The highpass is linear and kills the constant, so h(base + g*detail) =
    g*h(base + detail) identically for h1 and h2. Building the gain sweep as
    ``target + (g-1)*(h1+h2)`` instead does NOT have this property — h1 and h2
    are not projections, so the scales leak into each other and the emitted
    per-scale gain is only approximately g (measurably so: it puts a spurious
    minimum near g = 0.25). Every gain-sweep case below uses this construction.
    """
    torch.manual_seed(seed)
    base = torch.full((1, 1, size, size), float(level))
    detail = torch.randn(1, 1, size, size) * (sigma_levels / 255.0 * 2.0)
    return base, detail


def _term(**overrides) -> InventedHfLoss:
    cfg = {
        "y_weight": VAE_TRAINING_DEFAULTS["l_invented_y_weight"],
        "chroma_weight": VAE_TRAINING_DEFAULTS["l_invented_chroma_weight"],
        "flat_t_y": VAE_TRAINING_DEFAULTS["l_invented_flat_t_y"],
        "flat_t_c": VAE_TRAINING_DEFAULTS["l_invented_flat_t_c"],
    }
    cfg.update(overrides)
    return InventedHfLoss(**cfg)


def _bank_cfg(**overrides):
    cfg = dict(VAE_TRAINING_DEFAULTS)
    # LPIPS off everywhere in this file: it pulls a VGG download/checkpoint and
    # is irrelevant to every property under test.
    cfg["lpips_weight"] = 0.0
    cfg.update(overrides)
    return cfg


class FlatWindowSelectionTest(unittest.TestCase):
    """Which windows the term fires on. A plane fit, not a variance test."""

    def test_constant_region_is_flat(self):
        term = _term()
        img = _flat_image(112)
        _, cov = term(img.clone(), img)
        self.assertEqual(float(cov), 1.0)

    def test_smooth_ramp_is_flat(self):
        """The measured defect includes fabricated HF on smooth GRADIENTS, so a
        ramp has to count as flat. A raw-variance window test would reject every
        one of them (this ramp's per-window variance is ~1500x the plane-fit
        residual's), which is why the selection is a least-squares plane fit."""
        term = _term()
        size = 112
        ramp = (torch.linspace(-0.5, 0.5, size).view(1, 1, size, 1)
                .expand(1, 1, size, size).contiguous())
        img = _gray(ramp)
        _, cov = term(img.clone(), img)
        self.assertEqual(float(cov), 1.0)

        # ... and the variance a variance-only rule would look at is large.
        y = rgb01_to_ycbcr((img + 1.0) * 0.5) * 255.0
        window = y[0, 0, 32:56, 32:56]
        self.assertGreater(float(window.var()), 10.0)

    def test_diagonal_ramp_is_flat(self):
        """Both plane coefficients at once, so a fit that only removed the row
        gradient (or only the column one) would fail this."""
        term = _term()
        size = 112
        u = torch.linspace(-0.4, 0.4, size).view(1, 1, size, 1)
        v = torch.linspace(-0.4, 0.4, size).view(1, 1, 1, size)
        img = _gray((u + v) * 0.5)
        _, cov = term(img.clone(), img)
        self.assertEqual(float(cov), 1.0)

    def test_textured_region_is_not_flat(self):
        term = _term()
        torch.manual_seed(11)
        img = _gray(torch.randn(1, 1, 112, 112) * 0.2)
        loss, cov = term(img.clone(), img)
        self.assertEqual(float(cov), 0.0)
        self.assertEqual(float(loss), 0.0)

    def test_threshold_moves_the_selection(self):
        """The two thresholds are the user-facing part of the rule, so they must
        actually gate: the same image is selected under a loose threshold and
        rejected under a tight one."""
        torch.manual_seed(3)
        # ~1.4 levels of luma texture: inside T_Y=2.0, outside T_Y=0.5.
        img = _gray(torch.full((1, 1, 112, 112), -0.4)
                    + torch.randn(1, 1, 112, 112) * (1.4 / 255.0 * 2.0))
        self.assertEqual(float(_term(flat_t_y=2.0)(img.clone(), img)[1]), 1.0)
        self.assertEqual(float(_term(flat_t_y=0.5)(img.clone(), img)[1]), 0.0)

    def test_image_too_small_for_a_window_is_a_real_zero(self):
        term = _term()
        img = _flat_image(_ONE_WINDOW - 8).requires_grad_(True)
        loss, cov = term(img, img.detach())
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(cov), 0.0)
        loss.backward()   # must be a tensor on the graph, not a python float
        self.assertTrue(torch.equal(img.grad, torch.zeros_like(img.grad)))


class CoreInvariantTest(unittest.TestCase):
    """What the term charges for, and what it must not charge for."""

    def test_exact_reproduction_of_a_flat_source_costs_nothing(self):
        term = _term()
        img = _flat_image(112)
        loss, cov = term(img.clone(), img)
        self.assertEqual(float(cov), 1.0)
        self.assertLess(float(loss), 1e-8)

    def test_exact_reproduction_is_charged_far_less_than_a_plain_HF_penalty(self):
        """The discriminating case against a plain HF-magnitude penalty.

        The source here is flat by the plane-fit rule but carries real fine
        detail. A decode that reproduces it EXACTLY has transmitted that detail,
        not invented it. A term that penalised ``||h(recon)||^2`` instead of the
        orthogonal remainder would charge the full emitted magnitude here, and
        the cheapest way to satisfy it would be to blur.

        NOT "costs nothing": eps damps alpha below 1, so exact reproduction IS
        charged, at ``sigma^2*eps^2/(sigma^2+eps)^2`` per scale. See
        ``EpsilonFixedPointTest`` for the exact accounting and for the fact that
        a blur is charged *less* than this. What is pinned here is the ratio
        that makes the projection form worth having at all: two orders of
        magnitude below the plain-HF charge.
        """
        torch.manual_seed(5)
        img = _gray(torch.full((1, 1, 112, 112), -0.4)
                    + torch.randn(1, 1, 112, 112) * (1.5 / 255.0 * 2.0))
        term = _term()
        loss, cov = term(img.clone(), img)
        self.assertEqual(float(cov), 1.0)

        # In 8-bit levels: what is charged, against what a plain-HF penalty
        # would have charged.
        y = rgb01_to_ycbcr((img + 1.0) * 0.5) * 255.0
        h1, h2 = term._highpass(y)
        plain = float((h1[:, 0, 32:-32, 32:-32] ** 2).mean()
                      + (h2[:, 0, 32:-32, 32:-32] ** 2).mean())
        self.assertGreater(math.sqrt(plain), 1.0)
        self.assertGreater(float(loss), 0.0)          # charged, not exempt
        self.assertLess(float(loss), plain / 100.0)   # but ~1% of the plain form

    def test_added_high_frequency_is_penalised_and_scales_with_its_energy(self):
        term = _term()
        torch.manual_seed(7)
        target = _flat_image(112)
        noise = torch.randn(1, 1, 112, 112)
        losses = []
        for amp in (0.002, 0.004, 0.008):
            recon = target + _gray(noise) * amp
            losses.append(float(term(recon, target)[0]))
        self.assertGreater(losses[0], 0.0)
        self.assertGreater(losses[1], losses[0])
        self.assertGreater(losses[2], losses[1])
        # The term is a squared energy, so doubling the amplitude quadruples it.
        for lo, hi in zip(losses, losses[1:]):
            self.assertAlmostEqual(hi / lo, 4.0, delta=0.4)

    def test_invention_that_correlates_with_the_source_is_still_penalised(self):
        """The projection exempts *transmission*, not correlation-by-amplitude.

        The decode here emits nothing but scaled copies of the source's own
        high-frequency content — maximally correlated with it, so a term that
        exempted whatever correlates would charge 0 at every gain. It must not.

        NOTHING IS EXEMPT, not even below the alpha clamp: with eps > 0 the
        coefficient is alpha = g*sigma^2/(sigma^2+eps) < g, so the charge
        sigma^2*(g - alpha)^2 rises smoothly and monotonically from g = 0
        upward. The clamp only caps how much exemption a strongly-correlated
        emission can buy; it does not make gains under 2x free. (The docstring
        and the guide used to imply it did.)
        """
        term = _term()
        base, detail = _scaled_detail(seed=9, sigma_levels=1.5)
        target = _gray(base + detail)
        gains = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 6.0)
        losses = {g: float(term(_gray(base + g * detail), target)[0])
                  for g in gains}

        # Monotone in the gain, with NO exempt interval anywhere below 2x.
        for lo, hi in zip(gains, gains[1:]):
            self.assertGreater(losses[hi], losses[lo],
                               f"gain {hi} must cost more than gain {lo}")
        # Below the clamp the charge is exactly quadratic in the gain, because
        # alpha = g*k tracks it: L = sigma^2 * g^2 * (1-k)^2.
        self.assertAlmostEqual(losses[2.0] / losses[1.0], 4.0, delta=0.02)
        # ... and it keeps growing past the clamp, faster than quadratically.
        self.assertGreater(losses[6.0] / losses[3.0], 9.0)

    def test_emitting_NOTHING_is_free_while_exact_reproduction_is_charged(self):
        """The H2 finding, as two assertions.

        Same construction, two gains. g=0 (the decode emits no high frequency at
        all inside the flat window) is charged EXACTLY zero. g=1 (the decode
        reproduces the source's own detail exactly) is charged a real, positive
        amount. So the term's own preference, taken alone, is for less high
        frequency — the docs said the opposite until this was measured. Every
        operational consequence follows from these two lines; see
        EpsilonFixedPointTest.
        """
        term = _term()
        base, detail = _scaled_detail(seed=9, sigma_levels=1.5)
        target = _gray(base + detail)
        nothing = float(term(_gray(base), target)[0])
        exact = float(term(target.clone(), target)[0])
        self.assertEqual(nothing, 0.0)
        self.assertGreater(exact, 0.0)

    def test_the_alpha_clamp_boundary_is_where_it_says_it_is(self):
        """Pins ALPHA_MAX = 2.0 by behaviour, not just by value.

        Below the clamp the charge is the quadratic sigma^2*(g-alpha(g))^2, i.e.
        it scales as g^2. Above it, alpha stops tracking and the charge turns
        super-quadratic. Lowering ALPHA_MAX (2.0 -> 1.5) moves that knee and
        blows the quadratic ratio up by orders of magnitude, which is what this
        case detects — the previous version of the suite passed with the clamp
        mutated.
        """
        self.assertEqual(InventedHfLoss.ALPHA_MAX, 2.0)
        # Thresholds raised (they are user-facing) so that a source with enough
        # HF of its own to make alpha ~ g is still admitted: at sigma ~ 3 levels
        # per scale, alpha = g*sigma^2/(sigma^2+eps) ~ 0.99*g, so the clamp
        # bites just above g = 2 and the g = 1.9 point is inside it.
        term = _term(flat_t_y=6.0, flat_t_c=6.0)
        base, detail = _scaled_detail(seed=9, sigma_levels=3.0)
        target = _gray(base + detail)

        def at(g):
            return float(term(_gray(base + g * detail), target)[0])

        # Inside the clamp alpha tracks the gain, so the charge is EXACTLY
        # quadratic: 1.9^2 = 3.610. Move the clamp down and this explodes
        # (14.1 at ALPHA_MAX=1.75, 81.9 at 1.5).
        self.assertAlmostEqual(at(1.9) / at(1.0), 3.610, delta=0.05)
        # Past the clamp it leaves the quadratic regime: 511.7 at ALPHA_MAX=2.0.
        # Move the clamp up and it collapses (128.5 at 2.5, 9.0 at 3.0); move it
        # down and it overshoots (799.5 at 1.75, 1151.2 at 1.5).
        beyond = at(3.0) / at(1.0)
        self.assertGreater(beyond, 300.0)
        self.assertLess(beyond, 700.0)

    def test_uncorrelated_invention_costs_more_than_the_same_energy_transmitted(self):
        """Same emitted HF energy, two origins: reproduced from the source, or
        invented. Only the invented one is charged."""
        torch.manual_seed(13)
        term = _term()
        base = torch.full((1, 1, 112, 112), -0.4)
        detail = torch.randn(1, 1, 112, 112) * (1.5 / 255.0 * 2.0)
        target = _gray(base + detail)
        invented = _gray(base + torch.randn(1, 1, 112, 112) * (1.5 / 255.0 * 2.0))
        transmitted = float(term(target.clone(), target)[0])
        fabricated = float(term(invented, target)[0])
        self.assertGreater(fabricated, 10.0 * max(transmitted, 1e-12))


class AlphaDetachmentTest(unittest.TestCase):
    """The projection coefficient must not carry gradient.

    An attached alpha hands the decoder a gradient path that reduces the loss by
    RAISING its correlation with the source instead of emitting less — i.e. the
    "invent things that look a bit more like the input" behaviour the existing
    mse+lpips bank was measured to reward, which is the one behaviour this term
    exists to not reward. A version that trains but has alpha attached is a
    failed implementation, not a variant, so it is pinned here differentially:
    the implementation's gradient must equal the detached-alpha reference and
    must NOT equal the attached-alpha one.
    """

    @staticmethod
    def _reference_grad(term: InventedHfLoss, recon: torch.Tensor,
                        target: torch.Tensor, *, detach_alpha: bool):
        """Recompute the term with alpha attached or detached, reusing the
        implementation's own mask / highpass / pooling helpers so that the ONLY
        difference between the two references is the ``.detach()`` under test."""
        recon = recon.clone().detach().requires_grad_(True)
        inset, win = term.INSET, term.WINDOW
        h, w = recon.shape[-2], recon.shape[-1]

        def crop(x):
            return x[..., inset:h - inset, inset:w - inset]

        n_px = float(win * win)
        ones = term._ones
        recon_y = rgb01_to_ycbcr((recon + 1.0) * 0.5) * 255.0
        with torch.no_grad():
            target_y = rgb01_to_ycbcr((target.float() + 1.0) * 0.5) * 255.0
            mask, photo = term._flat_mask(crop(target_y))
            t_h1, t_h2 = term._highpass(target_y)
            t_h1, t_h2 = crop(t_h1), crop(t_h2)
        r_h1, r_h2 = term._highpass(recon_y)
        r_h1, r_h2 = crop(r_h1), crop(r_h2)

        chan = recon.new_tensor([term.y_weight, term.chroma_weight * 0.5,
                                 term.chroma_weight * 0.5]).view(1, 3, 1, 1)
        numerator = recon.new_zeros(())
        for d_map, s_map in ((r_h1, t_h1), (r_h2, t_h2)):
            s_sum = term._window_sum(s_map, ones, ones)
            s_sq = term._window_sum(s_map * s_map, ones, ones)
            c_ss = s_sq - s_sum * s_sum / n_px
            d_sum = term._window_sum(d_map, ones, ones)
            d_sq = term._window_sum(d_map * d_map, ones, ones)
            ds = term._window_sum(d_map * s_map, ones, ones)
            c_dd = d_sq - d_sum * d_sum / n_px
            c_ds = ds - d_sum * s_sum / n_px
            raw = c_ds.detach() if detach_alpha else c_ds
            alpha = torch.clamp(raw / (c_ss + term._alpha_eps), 0.0, term.ALPHA_MAX)
            l_win = ((c_dd - 2.0 * alpha * c_ds + alpha * alpha * c_ss)
                     / n_px).clamp_min(0.0)
            numerator = numerator + ((l_win * chan).sum(dim=1) * photo * mask).sum()
        loss = numerator / torch.clamp(mask.sum(), min=1.0)
        loss.backward()
        return float(loss.detach()), recon.grad.clone()

    def _scenario(self):
        """A flat window whose source carries enough HF for alpha to be well
        away from 0 and from the clamp — i.e. a case where the through-alpha
        gradient is non-zero and the two references genuinely differ."""
        torch.manual_seed(17)
        base = torch.full((1, 1, 112, 112), -0.35)
        detail = torch.randn(1, 1, 112, 112) * (1.5 / 255.0 * 2.0)
        target = _gray(base + detail)
        recon = target + _gray(torch.randn(1, 1, 112, 112) * (1.0 / 255.0 * 2.0)
                               + detail * 0.4)
        return recon, target

    def test_gradient_matches_the_detached_reference(self):
        term = _term()
        recon, target = self._scenario()
        live = recon.clone().detach().requires_grad_(True)
        loss, _ = term(live, target)
        loss.backward()

        ref_loss, ref_grad = self._reference_grad(term, recon, target,
                                                  detach_alpha=True)
        self.assertAlmostEqual(float(loss.detach()), ref_loss, places=6)
        self.assertTrue(torch.allclose(live.grad, ref_grad, atol=1e-9, rtol=1e-5))

    def test_gradient_differs_from_the_attached_reference(self):
        """Guards the guard: if the two references were indistinguishable in
        this scenario, the test above would pass for an attached implementation
        too."""
        term = _term()
        recon, target = self._scenario()
        _, detached = self._reference_grad(term, recon, target, detach_alpha=True)
        _, attached = self._reference_grad(term, recon, target, detach_alpha=False)
        self.assertFalse(torch.allclose(detached, attached, atol=1e-9, rtol=1e-5))
        rel = float((detached - attached).norm() / detached.norm())
        self.assertGreater(rel, 1e-3)

        # And the live implementation is on the detached side of that gap.
        live = recon.clone().detach().requires_grad_(True)
        term(live, target)[0].backward()
        self.assertLess(float((live.grad - detached).norm()),
                        float((live.grad - attached).norm()))

    def test_the_source_side_carries_no_gradient_at_all(self):
        """The mask, the photometric weight and s = h(target) are all computed
        from the target under no_grad: the selection must not be a gradient path
        the decoder could game."""
        term = _term()
        recon, target = self._scenario()
        target = target.clone().requires_grad_(True)
        loss, _ = term(recon.clone().detach().requires_grad_(True), target)
        loss.backward()
        self.assertIsNone(target.grad)


class EpsilonAndDegenerateAlphaTest(unittest.TestCase):
    """Behaviour where the source has (almost) no high frequency of its own."""

    def test_alpha_goes_to_zero_on_a_perfectly_flat_source(self):
        """With <s,s> = 0 the ratio is 0/eps = 0, not a NaN, and the term
        degrades to "where the source has nothing, emit nothing": the full
        emitted energy is charged."""
        term = _term()
        torch.manual_seed(19)
        target = _flat_image(112)
        recon = target + _gray(torch.randn(1, 1, 112, 112) * 0.004)
        loss, cov = term(recon, target)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertEqual(float(cov), 1.0)

        # alpha == 0 means the charge is exactly the emitted (window-centred)
        # high-frequency energy, which is computable independently here.
        expected = self._alpha_zero_energy(term, recon, target)
        self.assertAlmostEqual(float(loss), expected, delta=expected * 1e-4)

    @staticmethod
    def _alpha_zero_energy(term, recon, target):
        inset, win = term.INSET, term.WINDOW
        h, w = recon.shape[-2], recon.shape[-1]
        n_px = float(win * win)
        with torch.no_grad():
            ry = rgb01_to_ycbcr((recon + 1.0) * 0.5) * 255.0
            ty = rgb01_to_ycbcr((target + 1.0) * 0.5) * 255.0
            mask, photo = term._flat_mask(ty[..., inset:h - inset, inset:w - inset])
            total = 0.0
            for d_map in term._highpass(ry):
                d = d_map[..., inset:h - inset, inset:w - inset]
                d_sum = term._window_sum(d, term._ones, term._ones)
                d_sq = term._window_sum(d * d, term._ones, term._ones)
                l_win = (d_sq - d_sum * d_sum / n_px).clamp_min(0.0) / n_px
                chan = torch.tensor([term.y_weight, term.chroma_weight * 0.5,
                                     term.chroma_weight * 0.5]).view(1, 3, 1, 1)
                total += float(((l_win * chan).sum(dim=1) * photo * mask).sum())
            return total / max(float(mask.sum()), 1.0)

    def test_exactly_constant_source_and_decode_produce_no_nan(self):
        term = _term()
        img = _flat_image(112, level=0.0)
        loss, cov = term(img.clone().requires_grad_(True), img)
        value = float(loss.detach())
        self.assertTrue(math.isfinite(value))
        self.assertEqual(value, 0.0)
        self.assertEqual(float(cov), 1.0)

    def test_a_quantisation_floor_source_does_not_buy_a_full_exemption(self):
        """eps sits at the MEASURED 8-bit quantisation-floor energy (0.2797^2
        per pixel, results_flat_region_noise.md §1.4), so a source whose only
        "detail" is at that floor damps alpha rather than licensing a matching
        amount of invention."""
        term = _term()
        torch.manual_seed(23)
        floor = torch.randn(1, 1, 112, 112) * (0.28 / 255.0 * 2.0)
        target = _gray(torch.full((1, 1, 112, 112), -0.4) + floor)
        # A decode that emits 4x the source's floor detail, perfectly correlated.
        recon = target + _gray(floor * 3.0)
        loss, _ = term(recon, target)
        self.assertGreater(float(loss), 0.0)


class EpsilonFixedPointTest(unittest.TestCase):
    """What eps costs, stated as tests rather than as a comforting sentence.

    The regularised projection is NOT the pure one. With eps > 0 the coefficient
    is ``alpha = g*sigma^2/(sigma^2+eps) < g``, so inside a flat window the
    term's own fixed point is ``d = alpha*s`` — a SHRINK of the source's high
    frequency, not the identity — its global minimum over the emitted gain is
    ``g = 0``, and a blur is charged LESS than exact reproduction. That is the
    honest behaviour; these cases pin it so that (a) it cannot be re-described
    as "exact reproduction is free" in a future docstring, and (b) an eps change
    is a deliberate, visible act. The operational consequences (run the term
    with an agreement-with-source term; gate R7 on in-mask transmitted HF) are
    in docs/guides/VAE_TRAINING.md and design_l_invented.md §4.1.
    """

    def test_eps_is_the_measured_quantisation_floor_energy(self):
        """0.2797^2 (the MEASURED source floor, results_flat_region_noise.md
        §1.4), not 0.5^2 (the quantisation half-step). The half-step value cost
        2x the systematic attenuation of transmitted HF for a change in
        spuriously-exempted invented energy of 0.010% -> 0.043%: see
        design_l_invented.md §1.3."""
        self.assertAlmostEqual(InventedHfLoss.ALPHA_EPS_PER_PIXEL,
                               0.2797 ** 2, places=9)
        term = _term()
        self.assertAlmostEqual(term._alpha_eps,
                               576 * (0.2797 ** 2), places=6)

    @staticmethod
    def _sigma_and_stats(term, target):
        """Per-window per-scale source HF variance, plus mask/photo/chan — the
        inputs the closed form needs, taken from the implementation's own
        pooled sums so the prediction below is about eps, not about pooling."""
        inset = term.INSET
        h, w = target.shape[-2], target.shape[-1]
        crop = lambda z: z[..., inset:h - inset, inset:w - inset]
        with torch.no_grad():
            ty = rgb01_to_ycbcr((target + 1.0) * 0.5) * 255.0
            mask, photo = term._flat_mask(crop(ty))
            out = []
            for hmap in term._highpass(ty):
                s_map = crop(hmap)
                s_sum = term._window_sum(s_map, term._ones, term._ones)
                s_sq = term._window_sum(s_map * s_map, term._ones, term._ones)
                out.append((s_sq - s_sum * s_sum / term._n_pixels).clamp_min(0.0))
        return out, mask, photo

    def test_exact_reproduction_is_charged_the_closed_form_amount(self):
        """L(g=1) = sigma^2 * eps^2/(sigma^2+eps)^2 per scale, per window.

        Predicted from the constant alone and compared with the live term. This
        is the case that makes the size of the bias a number: it is bounded by
        eps/4 = 0.0196 levels^2 per scale (0.140 levels rms), attained at
        sigma = sqrt(eps).
        """
        torch.manual_seed(101)
        term = _term()
        target = _gray(torch.full((1, 1, 112, 112), -0.4)
                       + torch.randn(1, 1, 112, 112) * (1.2 / 255.0 * 2.0))
        c_ss_scales, mask, photo = self._sigma_and_stats(term, target)
        e = term.ALPHA_EPS_PER_PIXEL
        n_px = term._n_pixels
        chan = torch.tensor([term.y_weight, term.chroma_weight * 0.5,
                             term.chroma_weight * 0.5]).view(1, 3, 1, 1)
        predicted = torch.zeros(())
        for c_ss in c_ss_scales:
            sig2 = c_ss / n_px
            l_win = sig2 * (e / (sig2 + e)) ** 2
            predicted = predicted + ((l_win * chan).sum(dim=1) * photo * mask).sum()
        predicted = float(predicted / mask.sum().clamp_min(1.0))

        live = float(term(target.clone(), target)[0])
        self.assertGreater(live, 0.0)
        self.assertAlmostEqual(live, predicted, delta=max(predicted * 1e-4, 1e-12))
        # ... and the per-scale bound holds: eps/4 levels^2.
        self.assertLess(live, 2.0 * (e / 4.0))

    def test_blur_is_charged_LESS_than_exact_reproduction(self):
        """The uncomfortable one, pinned deliberately.

        Cheapest is emitting nothing. This is why the term is opt-in, why it
        must not be the only active loss weight, and why design_l_invented.md
        §4 gained gate R7 (in-mask TRANSMITTED HF vs the start arm): under a
        blur, both the primary success metric (invented luma) and the secondary
        (total emitted flat HF) fall, so blur reads as success on every other
        planned instrument.
        """
        torch.manual_seed(103)
        term = _term()
        for sigma_levels in (0.25, 0.5, 1.0, 2.0):
            with self.subTest(sigma=sigma_levels):
                src = (torch.full((1, 1, 112, 112), -0.4)
                       + torch.randn(1, 1, 112, 112) * (sigma_levels / 255 * 2))
                target = _gray(src)
                exact = float(term(target.clone(), target)[0])
                blurred = _gray(term._blur(src, 6))
                blur = float(term(blurred, target)[0])
                self.assertLess(blur, exact)

    def test_the_charge_is_minimised_by_emitting_nothing(self):
        """Sweeping the emitted gain g over [0, 2]: argmin is g = 0, at exactly
        zero. A term whose minimum were at g = 1 would be an
        agreement-with-source objective, which this one is explicitly not — and
        the gap between argmin g=0 and the ideal g=1 is the whole content of
        gate R7."""
        term = _term()
        base, detail = _scaled_detail(seed=107, sigma_levels=1.0)
        target = _gray(base + detail)
        values = {g: float(term(_gray(base + g * detail), target)[0])
                  for g in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)}
        self.assertEqual(min(values, key=values.get), 0.0)
        self.assertEqual(values[0.0], 0.0)
        self.assertGreater(values[1.0], 0.0)

    def test_the_closed_form_matches_an_independent_brute_force_reference(self):
        """Everything recomputed the naive way: an explicit per-window loop with
        an explicit least-squares plane fit and an explicit residual norm. The
        shipped term computes all of it from pooled separable sums, which is a
        very different arithmetic path; this is the numeric equivalence check
        for it (and it re-runs whenever eps or the geometry changes)."""
        torch.manual_seed(109)
        term = _term()
        size = 112
        target = _gray(torch.full((1, 1, size, size), -0.3)
                       + torch.randn(1, 1, size, size) * (1.2 / 255.0 * 2.0))
        recon = target + _gray(torch.randn(1, 1, size, size) * (0.8 / 255.0 * 2.0))

        win, stride, inset = term.WINDOW, term.STRIDE, term.INSET
        n_px = float(win * win)
        eps = term._alpha_eps
        with torch.no_grad():
            ty = (rgb01_to_ycbcr((target + 1.0) * 0.5) * 255.0).double()
            ry = (rgb01_to_ycbcr((recon + 1.0) * 0.5) * 255.0).double()
            t_h = [z[..., inset:size - inset, inset:size - inset]
                   for z in term._highpass(ty)]
            r_h = [z[..., inset:size - inset, inset:size - inset]
                   for z in term._highpass(ry)]
            ty_c = ty[..., inset:size - inset, inset:size - inset]

        ramp = (torch.arange(win, dtype=torch.float64) - (win - 1) / 2.0)
        u = ramp.view(-1, 1).expand(win, win).reshape(-1)
        v = ramp.view(1, -1).expand(win, win).reshape(-1)
        basis = torch.stack([torch.ones_like(u), u, v], dim=1)

        inner = ty_c.shape[-1]
        starts = list(range(0, inner - win + 1, stride))
        total, n_sel = 0.0, 0
        weights = [term.y_weight, term.chroma_weight * 0.5, term.chroma_weight * 0.5]
        for y0 in starts:
            for x0 in starts:
                patch = ty_c[0, :, y0:y0 + win, x0:x0 + win].reshape(3, -1)
                rms = []
                for c in range(3):
                    sol = torch.linalg.lstsq(basis, patch[c].unsqueeze(1)).solution
                    resid = patch[c] - (basis @ sol).squeeze(1)
                    rms.append(float((resid * resid).mean().sqrt()))
                if not (rms[0] <= term.flat_t_y
                        and max(rms[1], rms[2]) <= term.flat_t_c):
                    continue
                n_sel += 1
                mu = max(float(patch[0].mean()), 0.0)
                photo = term.WEBER_MU0 / (mu + term.WEBER_MU0)
                acc = 0.0
                for s_full, d_full in zip(t_h, r_h):
                    for c in range(3):
                        s = s_full[0, c, y0:y0 + win, x0:x0 + win].reshape(-1)
                        d = d_full[0, c, y0:y0 + win, x0:x0 + win].reshape(-1)
                        s = s - s.mean()
                        d = d - d.mean()
                        a = min(max(float(torch.dot(d, s)
                                          / (torch.dot(s, s) + eps)), 0.0),
                                term.ALPHA_MAX)
                        r = d - a * s
                        acc += weights[c] * float((r * r).mean())
                total += acc * photo
        reference = total / max(n_sel, 1)

        live, cov = term(recon, target)
        self.assertGreater(n_sel, 0)
        self.assertEqual(float(cov), n_sel / len(starts) ** 2)
        self.assertAlmostEqual(float(live), reference, delta=abs(reference) * 1e-5)


class LoggedValueIsNotALevelTest(unittest.TestCase):
    """``sqrt(vae_invented_loss)`` must never be read as a level.

    The failure this prevents: an operator sees `vae_invented_loss ~ 0.9` at
    30k, takes sqrt ~ 0.95, concludes "under the 1/255 bar, ship it", while the
    actual invented luma is 2-3x that. The logged value carries the Weber
    photometric weight and the channel weights, so it is a relative trend
    indicator only; absolute levels come from the frozen g1flat harness.
    """

    @staticmethod
    def _inject_one_level(level: float):
        """A decode that adds exactly 1.0 level rms of pure, uncorrelated luma
        high frequency (measured in the term's own two-scale basis) onto a flat
        source at the given luma."""
        torch.manual_seed(211)
        term = _term()
        target = _gray(torch.full((1, 1, 112, 112), float(level)))
        n = torch.randn(1, 1, 112, 112)
        with torch.no_grad():
            h1, h2 = term._highpass(_gray(n))
            energy = float(((h1[:, 0] ** 2 + h2[:, 0] ** 2)[..., 32:-32, 32:-32]
                            ).mean())
        recon = target + _gray(n) * ((2.0 / 255.0) / math.sqrt(energy))
        with torch.no_grad():
            h1, h2 = term._highpass(rgb01_to_ycbcr((recon + 1.0) * 0.5) * 255.0)
            emitted = math.sqrt(float(((h1[:, 0] ** 2 + h2[:, 0] ** 2)
                                       [..., 32:-32, 32:-32]).mean()))
        return math.sqrt(float(term(recon, target)[0])), emitted

    def test_sqrt_of_the_logged_value_under_reads_true_invented_luma(self):
        for name, level, lo, hi in (("dark", -0.95, 0.85, 0.99),
                                    ("mid", 0.0, 0.45, 0.60),
                                    ("bright", 0.95, 0.33, 0.46)):
            with self.subTest(window=name):
                root, emitted = self._inject_one_level(level)
                self.assertAlmostEqual(emitted, 1.0, delta=0.02)
                # 1.0 level of invention does NOT log as sqrt() == 1.0.
                self.assertGreater(root, lo)
                self.assertLess(root, hi)
        # And the under-read is a factor, not a rounding error: the same
        # invention reads ~2.4x smaller in a bright window than in a dark one.
        dark, _ = self._inject_one_level(-0.95)
        bright, _ = self._inject_one_level(0.95)
        self.assertGreater(dark / bright, 2.0)


class InternalGeometryTest(unittest.TestCase):
    """The loss geometry is load-bearing evidence, so it is pinned by VALUE.

    design_l_invented.md §3's entire argument is that the training loss and the
    frozen evaluation meter are DIFFERENT functionals, so "the loss went down
    and the meter went down" is evidence about the picture rather than an
    identity. The meter is 32 px windows / stride 16 / inset 48 with a single
    sigma=2 Gaussian highpass; the loss is 24/12/32 with a two-scale binomial.
    A future edit "harmonising" the two would pass every behavioural test in
    this file — the derived helpers adapt to any geometry — and would silently
    convert the experiment into a tautology. Hence value assertions.
    """

    # The frozen meter's geometry (scratchpad/vae_training/harness/g1flat/,
    # calib.py W/S and g1_eval.py's inset). Written out so the inequality below
    # is explicit rather than implied.
    _METER_WINDOW, _METER_STRIDE, _METER_INSET = 32, 16, 48

    def test_the_window_geometry_is_exactly_the_designed_one(self):
        self.assertEqual(InventedHfLoss.WINDOW, 24)
        self.assertEqual(InventedHfLoss.STRIDE, 12)
        self.assertEqual(InventedHfLoss.INSET, 32)

    def test_the_geometry_differs_from_the_frozen_meters(self):
        self.assertNotEqual(InventedHfLoss.WINDOW, self._METER_WINDOW)
        self.assertNotEqual(InventedHfLoss.STRIDE, self._METER_STRIDE)
        self.assertNotEqual(InventedHfLoss.INSET, self._METER_INSET)
        # 12 is also not a multiple of 8, which breaks the latent-cell
        # grid-phase alignment of the mask boundary (design §2.2).
        self.assertNotEqual(InventedHfLoss.STRIDE % 8, 0)

    def test_the_photometric_and_projection_constants_are_the_designed_ones(self):
        self.assertEqual(InventedHfLoss.WEBER_MU0, 48.0)
        self.assertEqual(InventedHfLoss.ALPHA_MAX, 2.0)
        self.assertAlmostEqual(InventedHfLoss.ALPHA_EPS_PER_PIXEL,
                               0.2797 ** 2, places=9)

    def test_the_highpass_is_the_two_scale_binomial(self):
        """b3 and b7 = b3 applied three times, not one Gaussian: the meter's
        operator is a single sigma=2.0 Gaussian."""
        term = _term()
        self.assertTrue(torch.equal(term._b3,
                                    torch.tensor([0.25, 0.5, 0.25])))
        x = torch.zeros(1, 1, 9, 9)
        x[0, 0, 4, 4] = 1.0
        h1, h2 = term._highpass(x)
        # h1 = x - b3(x): a 3x3 support around the impulse.
        self.assertAlmostEqual(float(h1[0, 0, 4, 4]), 1.0 - 0.25, places=6)
        self.assertEqual(float(h1[0, 0, 4, 1]), 0.0)
        # h2 = b3(x) - b7(x): support radius 3, which is the halo in the class
        # docstring (an edge up to 3 px outside a window reaches into it).
        self.assertNotEqual(float(h2[0, 0, 4, 1]), 0.0)
        self.assertEqual(float(h2[0, 0, 4, 0]), 0.0)


class BatchReductionTest(unittest.TestCase):
    """Reduction is by total window count across the micro-batch."""

    def _mixed_batch(self):
        torch.manual_seed(29)
        flat_t = _flat_image(112)
        flat_r = flat_t + _gray(torch.randn(1, 1, 112, 112) * 0.004)
        textured = _gray(torch.randn(1, 1, 112, 112) * 0.2)
        target = torch.cat([textured, flat_t], dim=0)
        recon = torch.cat([textured.clone(), flat_r], dim=0)
        return recon, target

    def test_zero_window_element_does_not_distort_the_mean(self):
        term = _term()
        recon, target = self._mixed_batch()
        mixed, mixed_cov = term(recon, target)
        alone, alone_cov = term(recon[1:], target[1:])
        self.assertAlmostEqual(float(mixed), float(alone), places=6)
        # ... and the zero-window element halves the coverage, which is exactly
        # why coverage is logged next to the value.
        self.assertAlmostEqual(float(mixed_cov), float(alone_cov) / 2.0, places=6)

    def test_reduction_is_by_window_count_not_image_count(self):
        """A per-IMAGE reduction of a window-summed numerator would scale with
        the number of windows per image, so the same content at two sizes would
        report two very different values."""
        term = _term()
        torch.manual_seed(31)
        small_t = _flat_image(112)
        small_r = small_t + _gray(torch.randn(1, 1, 112, 112) * 0.004)
        # 4x the window count, statistically the same content.
        big_t = _flat_image(160)
        big_r = big_t + _gray(torch.randn(1, 1, 160, 160) * 0.004)
        small = float(term(small_r, small_t)[0])
        big = float(term(big_r, big_t)[0])
        self.assertGreater(small, 0.0)
        self.assertAlmostEqual(big / small, 1.0, delta=0.25)

    def test_an_all_textured_batch_is_exactly_zero_with_a_live_graph(self):
        term = _term()
        torch.manual_seed(37)
        target = _gray(torch.randn(2, 1, 112, 112) * 0.2)
        recon = (target + 0.01).detach().requires_grad_(True)
        loss, cov = term(recon, target)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(cov), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(recon.grad, torch.zeros_like(recon.grad)))


class OffByDefaultTest(unittest.TestCase):
    """The term must be OFF by default and inert when off."""

    def test_default_weight_is_zero(self):
        self.assertEqual(VAE_TRAINING_DEFAULTS["l_invented_weight"], 0.0)

    def test_bank_does_not_construct_the_term_when_the_weight_is_zero(self):
        bank = VaeLossBank(_bank_cfg(), _CPU)
        self.assertIsNone(bank.invented_loss)

    def test_weight_zero_is_bit_identical_to_the_term_being_absent(self):
        """Not "close": identical. An existing run must not move by one ULP
        because a new term was added to the file."""
        torch.manual_seed(41)
        target = _gray(torch.rand(2, 1, 96, 96) * 2 - 1)
        recon_a = (target + torch.randn_like(target) * 0.01).requires_grad_(True)
        recon_b = recon_a.clone().detach().requires_grad_(True)

        bank = VaeLossBank(_bank_cfg(), _CPU)
        total, parts = bank(recon_a, target)
        total.backward()

        # The pre-existing bank, recomputed from scratch: mse + ycbcr_dc only.
        r32, t32 = recon_b.float(), target.float()
        reference = (bank.mse_weight * F.mse_loss(r32, t32)
                     + bank.ycbcr_dc_weight * bank._ycbcr_dc(r32, t32))
        reference.backward()

        self.assertNotIn("l_invented", parts)
        self.assertNotIn("l_invented_cov", parts)
        self.assertTrue(torch.equal(total.detach(), reference.detach()))
        self.assertTrue(torch.equal(recon_a.grad, recon_b.grad))

    def test_turning_it_on_changes_the_total_and_logs_both_parts(self):
        torch.manual_seed(43)
        target = _flat_image(112, batch=2)
        recon = target + _gray(torch.randn(2, 1, 112, 112) * 0.004)

        off = VaeLossBank(_bank_cfg(), _CPU)
        on = VaeLossBank(_bank_cfg(l_invented_weight=1.0), _CPU)
        total_off, parts_off = off(recon, target)
        total_on, parts_on = on(recon, target)

        self.assertIsNotNone(on.invented_loss)
        self.assertIn("l_invented", parts_on)
        self.assertIn("l_invented_cov", parts_on)
        self.assertEqual(parts_on["l_invented_cov"], 1.0)
        self.assertGreater(parts_on["l_invented"], 0.0)
        self.assertAlmostEqual(float(total_on) - float(total_off),
                               parts_on["l_invented"], places=6)
        self.assertNotIn("l_invented", parts_off)

    def test_the_weight_scales_the_contribution(self):
        torch.manual_seed(47)
        target = _flat_image(112)
        recon = target + _gray(torch.randn(1, 1, 112, 112) * 0.004)
        base = VaeLossBank(_bank_cfg(mse_weight=0.0, ycbcr_dc_weight=0.0,
                                     l_invented_weight=1.0), _CPU)
        doubled = VaeLossBank(_bank_cfg(mse_weight=0.0, ycbcr_dc_weight=0.0,
                                        l_invented_weight=2.0), _CPU)
        self.assertAlmostEqual(float(doubled(recon, target)[0]),
                               2.0 * float(base(recon, target)[0]), places=6)


class ChannelWeightTest(unittest.TestCase):
    """The two channel weights address different halves of the measurement."""

    def test_luma_only_ignores_a_chroma_only_invention(self):
        torch.manual_seed(53)
        target = _flat_image(112)
        chroma_noise = torch.zeros(1, 3, 112, 112)
        # Red and blue perturbed in the ratio that leaves Y = 0.299R + 0.587G +
        # 0.114B exactly unchanged, i.e. a chroma-only invention.
        n = torch.randn(1, 1, 112, 112) * 0.02
        chroma_noise[:, 0] = 0.114 * n[:, 0]
        chroma_noise[:, 2] = -0.299 * n[:, 0]
        recon = target + chroma_noise

        luma_only = float(_term(chroma_weight=0.0)(recon, target)[0])
        chroma_only = float(_term(y_weight=0.0, chroma_weight=1.0)(recon, target)[0])
        self.assertGreater(chroma_only, 1e-3)
        self.assertLess(luma_only, 1e-6 * chroma_only)

    def test_chroma_weight_scales_linearly(self):
        torch.manual_seed(59)
        target = _flat_image(112)
        recon = target + torch.randn(1, 3, 112, 112) * 0.004
        a = float(_term(y_weight=0.0, chroma_weight=1.0)(recon, target)[0])
        b = float(_term(y_weight=0.0, chroma_weight=2.0)(recon, target)[0])
        self.assertGreater(a, 0.0)
        self.assertAlmostEqual(b / a, 2.0, places=5)


class PhotometricWeightTest(unittest.TestCase):
    """Dark windows are weighted up, smoothly."""

    def test_dark_windows_outweigh_bright_ones(self):
        torch.manual_seed(61)
        term = _term()
        noise = _gray(torch.randn(1, 1, 112, 112) * 0.004)
        dark = _flat_image(112, level=-0.95)      # ~6/255
        bright = _flat_image(112, level=0.95)     # ~249/255
        dark_loss = float(term(dark + noise, dark)[0])
        bright_loss = float(term(bright + noise, bright)[0])
        self.assertGreater(dark_loss, bright_loss)
        # Weber 48/(mu+48): ~0.89 at 6/255 against ~0.16 at 249/255.
        self.assertAlmostEqual(dark_loss / bright_loss, 0.89 / 0.16, delta=0.6)


class ProductionDtypeTest(unittest.TestCase):
    """The VAE trains under autocast bf16 with fp32 weights.

    A term verified only in fp32 that cannot execute in the production dtype has
    happened in this project before, so the dtype matrix is part of the guard:
    fp32 input outside autocast, bf16 input, and fp32 input under an active bf16
    autocast (which is what the trainer's forward actually looks like).
    """

    def _run(self, dtype, autocast: bool):
        torch.manual_seed(67)
        target = _flat_image(112, batch=2).to(dtype)
        recon = (target + _gray(torch.randn(2, 1, 112, 112) * 0.004).to(dtype)
                 ).detach().requires_grad_(True)
        bank = VaeLossBank(_bank_cfg(l_invented_weight=1.0), _CPU)
        ctx = (torch.autocast(device_type="cpu", dtype=torch.bfloat16)
               if autocast else torch.autocast(device_type="cpu", enabled=False))
        with ctx:
            total, parts = bank(recon, target)
        total.backward()
        return total, parts, recon.grad

    def test_dtype_matrix_produces_finite_loss_and_gradients(self):
        for dtype, autocast in ((torch.float32, False),
                                (torch.bfloat16, False),
                                (torch.float32, True),
                                (torch.bfloat16, True)):
            with self.subTest(dtype=str(dtype), autocast=autocast):
                total, parts, grad = self._run(dtype, autocast)
                self.assertTrue(math.isfinite(float(total)))
                self.assertTrue(math.isfinite(parts["l_invented"]))
                self.assertGreater(parts["l_invented"], 0.0)
                self.assertEqual(parts["l_invented_cov"], 1.0)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_the_term_is_computed_in_fp32_under_bf16_autocast(self):
        """The pooled sums run over 576 pixels of (8-bit level)^2 values, which
        is not a bf16 quantity; the term disables autocast internally, so the
        value under autocast must match the plain fp32 one closely."""
        plain, _, _ = self._run(torch.float32, False)
        auto, _, _ = self._run(torch.float32, True)
        self.assertAlmostEqual(float(plain), float(auto), places=5)

    def test_three_steps_of_optimisation_stay_finite(self):
        """A ~3-step smoke of the term itself (not a training run): the loss and
        the gradients stay finite while an optimiser actually moves the tensor
        it is charged against."""
        torch.manual_seed(71)
        target = _flat_image(112, batch=2)
        recon = (target + _gray(torch.randn(2, 1, 112, 112) * 0.006)
                 ).detach().requires_grad_(True)
        term = _term()
        opt = torch.optim.AdamW([recon], lr=1e-3)
        history = []
        for _ in range(3):
            opt.zero_grad()
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                loss, cov = term(recon, target)
            loss.backward()
            self.assertTrue(torch.isfinite(recon.grad).all())
            opt.step()
            history.append(float(loss))
        self.assertTrue(all(math.isfinite(v) for v in history))
        # The only way down is to emit less: the value must fall under a step
        # that is free to change the decode.
        self.assertLess(history[-1], history[0])


class ConfigSurfaceTest(unittest.TestCase):
    """The five user-facing keys, and only those five."""

    _BASE_MODEL = os.path.join(_BACKEND, "..", "models", "vae",
                               "placeholder.safetensors")

    def _resolve(self, **vae):
        section = {"lpips_weight": 0.0}
        section.update(vae)
        return resolve_vae_training_config({"vae": section},
                                           base_model_path=self._BASE_MODEL)

    def test_the_five_keys_exist_with_the_designed_defaults(self):
        expected = {
            "l_invented_weight": 0.0,
            "l_invented_y_weight": 1.0,
            "l_invented_chroma_weight": 0.25,
            "l_invented_flat_t_y": 2.0,
            "l_invented_flat_t_c": 1.25,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(VAE_TRAINING_DEFAULTS[key], value)
                self.assertEqual(self._resolve()[key], value)

    def test_the_internal_constants_are_not_config_keys(self):
        """Geometry, the highpass basis, the alpha epsilon/clamp and the Weber
        constant stay internal: they are what keeps the loss distinguishable
        from the frozen evaluation metric, and exposing them would invite
        re-deriving the meter."""
        for absent in ("l_invented_window", "l_invented_stride",
                       "l_invented_inset", "l_invented_alpha_eps",
                       "l_invented_alpha_max", "l_invented_mu0",
                       "l_invented_scales"):
            with self.subTest(key=absent):
                self.assertNotIn(absent, VAE_TRAINING_DEFAULTS)
                with self.assertRaises(VaeConfigError):
                    self._resolve(**{absent: 1})

    def test_it_counts_as_a_training_signal_on_its_own(self):
        """It is in ``_LOSS_WEIGHT_KEYS``, so it satisfies the "at least one
        training signal" check alone — exactly as ``pattern_weight`` does, so
        this is consistent rather than new. It is nonetheless a configuration
        NOBODY SHOULD RUN: the term's own global optimum inside the flat mask
        is "emit no high frequency at all" (see EpsilonFixedPointTest), so a run
        with only this weight has blur as its objective. The refusal is not
        tightened here because pattern_weight would need the same treatment and
        because the check is about "is anything connected to the graph"; the
        warning lives in docs/guides/VAE_TRAINING.md and in the config panel."""
        cfg = self._resolve(mse_weight=0.0, ycbcr_dc_weight=0.0,
                            l_invented_weight=1.0)
        self.assertEqual(cfg["l_invented_weight"], 1.0)

    def test_a_zero_threshold_is_refused_WHEN_THE_TERM_IS_ON(self):
        for key in ("l_invented_flat_t_y", "l_invented_flat_t_c"):
            with self.subTest(key=key):
                with self.assertRaises(VaeConfigError) as ctx:
                    self._resolve(l_invented_weight=1.0, **{key: 0})
                self.assertIn(key, str(ctx.exception))

    def test_both_channel_weights_at_zero_is_refused_WHEN_THE_TERM_IS_ON(self):
        with self.assertRaises(VaeConfigError):
            self._resolve(l_invented_weight=1.0,
                          l_invented_y_weight=0, l_invented_chroma_weight=0)

    def test_the_consistency_refusals_do_not_fire_while_the_term_is_OFF(self):
        """Off by default must be COMPLETELY inert, including in validation.

        Both refusals above reject a combination that is individually legal and
        that nothing reads when ``l_invented_weight`` is 0 — and both messages
        end "or set l_invented_weight=0 to disable the term", which at weight 0
        is advice the user has already taken and which cannot clear the error.
        A config that never mentions this feature must resolve.
        """
        for kwargs in ({"l_invented_y_weight": 0, "l_invented_chroma_weight": 0},
                       {"l_invented_flat_t_y": 0},
                       {"l_invented_flat_t_c": 0}):
            with self.subTest(**kwargs):
                cfg = self._resolve(l_invented_weight=0.0, **kwargs)
                self.assertEqual(cfg["l_invented_weight"], 0.0)

    def test_the_bounds_openapi_declares_are_enforced(self):
        """openapi.yaml is the contract in this repo, and it declares
        ``maximum`` for all five keys. Before this, ``l_invented_weight=1e6``
        and ``l_invented_y_weight=99`` were accepted."""
        for key, over in (("l_invented_weight", 10.5),
                          ("l_invented_weight", 1e6),
                          ("l_invented_y_weight", 99),
                          ("l_invented_chroma_weight", 4.5),
                          ("l_invented_flat_t_y", 8.5),
                          ("l_invented_flat_t_c", 12)):
            with self.subTest(key=key, value=over):
                with self.assertRaises(VaeConfigError) as ctx:
                    self._resolve(**{key: over})
                self.assertIn(key, str(ctx.exception))
        # The declared maxima themselves are accepted (bounds are inclusive).
        cfg = self._resolve(l_invented_weight=10, l_invented_y_weight=4,
                            l_invented_chroma_weight=4, l_invented_flat_t_y=8,
                            l_invented_flat_t_c=8)
        self.assertEqual(cfg["l_invented_weight"], 10.0)
        self.assertEqual(cfg["l_invented_flat_t_c"], 8.0)

    def test_negative_weights_are_refused(self):
        # Negatives are refused unconditionally (the term being off does not
        # make a nonsense value acceptable); only the two CONSISTENCY refusals
        # above are gated on the term being on.
        for key in ("l_invented_weight", "l_invented_y_weight",
                    "l_invented_chroma_weight", "l_invented_flat_t_y",
                    "l_invented_flat_t_c"):
            with self.subTest(key=key):
                with self.assertRaises(VaeConfigError):
                    self._resolve(**{key: -1})

    def test_the_metric_names_are_registered_for_charting(self):
        from core.training.metric_registry import EXTRA_METRIC_DEFS
        from core.training.vae.vae_trainer import M_INVENTED, M_INVENTED_COV
        for name in (M_INVENTED, M_INVENTED_COV):
            with self.subTest(name=name):
                self.assertIn(name, EXTRA_METRIC_DEFS)
        # Coverage is a 0..1 fraction, so it must not pool into the loss Y-range.
        self.assertEqual(EXTRA_METRIC_DEFS[M_INVENTED_COV].get("axis"), "right")


if __name__ == "__main__":
    unittest.main()
