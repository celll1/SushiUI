"""``ActivationDispatcher``'s bucket key must determine the thing it predicts.

The dispatcher predicts a training step's activation footprint from the latent
shape and dispatches CPU activation offload from that prediction. Its key was
``(latent_h, latent_w, batch_size)`` and its regression variable was the latent
AREA ``lh*lw``. For a 4-D image latent ``[B, C, H', W']`` that is the whole
latent extent. For a 5-D video latent ``[B, C, T, H', W']`` it is not: the
temporal extent is missing from both, so two clips that differ only in length
land in the SAME bucket and get the SAME prediction while their measured
activation differs by 3.8x (2.36 GB at 22 frames vs 8.90 GB at 124 frames,
384x640, MiniMax-H3 -- see the scaling measurement; the same shape of latent
reaches the same code path for LTX-2.3).

The fix makes the latent VOLUME ``lh*lw*lt`` the regression variable and
``(lh, lw, lt, bs)`` the key, with ``lt=1`` for 4-D latents.

This file is a CHARACTERIZATION test written before that change:

* ``ImageArchBehaviourUnchangedTest`` pins the full observable behaviour of a
  scripted image-arch (4-D, ``lt=1``) session -- key construction via the
  caches, ``base_act`` / ``predicted_offloadable`` / ``plan_micro_bs``, and
  ``decide()`` swept across the headroom range -- against a table snapshotted
  from the pre-change dispatcher. Every call in the scenario is made with the
  ORIGINAL positional signature, so the same scenario runs on both versions.
* ``FixedClipLengthUnchangedTest`` proves the algebraic invariance that makes
  the change safe for a video run at ONE clip length (which is every LTX-2.3 /
  MiniMax-H3 run whose temporal spec does not mix lengths): scaling every
  sample's regression variable by the same constant ``lt`` leaves the 2-term
  least-squares prediction -- and therefore every decision -- unchanged.
* ``VideoClipLengthSeparationTest`` is the bug itself, and the NEGATIVE
  CONTROL: it fails if the temporal term is ever dropped from the key or from
  the volume again.

Everything here is pure arithmetic on the real class. No GPU, no model.
"""

import sys
import unittest
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

from core.memory_management import ActivationDispatcher  # noqa: E402
from activation_dispatcher_snapshot import IMAGE_SCENARIO_SNAPSHOT  # noqa: E402


# --------------------------------------------------------------------------
# The scripted image-arch session. Uses ONLY the original positional signature
# (lh, lw, bs), so it is runnable against the pre-change dispatcher verbatim.
# --------------------------------------------------------------------------
def _image_scenario():
    """Drive a dispatcher through a cold start -> calibration -> offload
    calibration -> overflow session and return every observable it produced."""
    d = ActivationDispatcher(budget_gb=48.0, margin_gb=1.0)
    out = {}

    def snap(tag, lh, lw, bs):
        out[f"{tag}/base_act"] = d.base_act(lh, lw, bs)
        out[f"{tag}/offloadable"] = d.predicted_offloadable(lh, lw, bs)
        for hr in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
            out[f"{tag}/decide@{hr}"] = d.decide(lh, lw, bs, hr)
            out[f"{tag}/micro@{hr}"] = d.plan_micro_bs(lh, lw, bs, hr)

    # 1. Cold start: no samples at all -> seed_coef slope, residual_frac fallback.
    snap("cold_128x128_bs2", 128, 128, 2)
    snap("cold_192x96_bs4", 192, 96, 4)

    # 2. One measured base step -> exact per-bucket cache for that bucket, and a
    #    single sample (still < 2 -> seed slope) for unseen buckets.
    d.record(128, 128, 2, "base", peak_gb=10.0, resident_gb=8.0)
    snap("after1_128x128_bs2", 128, 128, 2)
    snap("after1_192x96_bs4", 192, 96, 4)

    # 3. A second, differently-sized base step -> the 2-term fit engages.
    d.record(192, 96, 4, "base", peak_gb=17.0, resident_gb=8.0)
    snap("after2_128x128_bs2", 128, 128, 2)
    snap("after2_192x96_bs4", 192, 96, 4)
    snap("after2_unseen_256x256_bs1", 256, 256, 1)
    snap("after2_unseen_256x256_bs3", 256, 256, 3)

    # 4. Two measured OFFLOAD steps -> the offloadable fit supersedes residual_frac.
    d.record(160, 160, 2, "offload", peak_gb=12.0, resident_gb=8.0, offloaded_gb=1.5)
    snap("off1_unseen_256x256_bs2", 256, 256, 2)
    d.record(224, 224, 2, "offload", peak_gb=16.0, resident_gb=8.0, offloaded_gb=3.0)
    snap("off2_unseen_256x256_bs2", 256, 256, 2)
    snap("off2_160x160_bs2", 160, 160, 2)

    # 5. A micro-split step: peak reflects executed_bs, record() scales to bs.
    d.record(320, 320, 4, "offload", peak_gb=14.0, resident_gb=8.0,
             executed_bs=2, offloaded_gb=2.0)
    snap("split_320x320_bs4", 320, 320, 4)

    # 6. An offload measurement taken at a NON-default threshold must recover the
    #    base cost but must NOT calibrate the offloadable predictor.
    d.record(288, 288, 2, "offload", peak_gb=13.0, resident_gb=8.0,
             offloaded_gb=4.0, measured_threshold_bytes=256 * 1024)
    snap("lowthr_288x288_bs2", 288, 288, 2)

    # 7. Overflow flag forces escalate for that bucket forever after.
    d.mark_overflow(128, 128, 2)
    snap("overflow_128x128_bs2", 128, 128, 2)
    snap("overflow_192x96_bs4", 192, 96, 4)

    return out


class ImageArchBehaviourUnchangedTest(unittest.TestCase):
    """Every observable of the scripted 4-D session is bit-identical to the
    pre-change snapshot. This is what "does not regress image architectures"
    means concretely."""

    def test_scenario_matches_pre_change_snapshot(self):
        got = _image_scenario()
        expected = IMAGE_SCENARIO_SNAPSHOT
        self.assertEqual(sorted(got), sorted(expected),
                         "the set of observables changed")
        for k in sorted(expected):
            with self.subTest(observable=k):
                if isinstance(expected[k], str):
                    self.assertEqual(got[k], expected[k])
                else:
                    # Exact: for lt=1 the volume is the area times 1, so the
                    # arithmetic is unchanged, not merely close.
                    self.assertEqual(got[k], expected[k])

    def test_four_d_latent_keys_are_unchanged_in_meaning(self):
        """A 4-D image latent's bucket must still be identified by exactly
        (spatial, spatial, batch): recording one and asking for it back must hit
        the exact per-bucket cache, and a different spatial size must not."""
        d = ActivationDispatcher(budget_gb=48.0)
        d.record(64, 64, 2, "base", peak_gb=9.0, resident_gb=8.0)
        self.assertAlmostEqual(d.base_act(64, 64, 2), 1.0, places=9)
        self.assertNotAlmostEqual(d.base_act(64, 128, 2), 1.0, places=3)


class FixedClipLengthUnchangedTest(unittest.TestCase):
    """A video run at a SINGLE clip length decides exactly as before -- ONCE
    CALIBRATED. The cold-start seed deliberately does not, and that is pinned
    here too rather than glossed over.

    The 2-term model is ``y = a + b*x``. Replacing ``x = area`` with
    ``x = area*lt`` for a constant ``lt`` scales the fitted slope by ``1/lt``
    and leaves the intercept alone, so every FITTED prediction -- and therefore
    every decide() / plan_micro_bs() outcome from it -- is the same number.
    That is the proof that LTX-2.3 / MiniMax-H3 runs which do not mix clip
    lengths are behaviour-preserved after their first two recorded steps.

    Before those samples exist, ``base_act`` uses the fixed ``seed_coef`` slope,
    which is NOT invariant: it now scales with the clip length, by exactly a
    factor ``lt``. That is intended -- the old cold-start prediction for a
    MiniMax-H3 384x640 bucket was 0.09 GB against a measured 2.36-8.90 GB, i.e.
    26-96x low -- and it moves predictions UP, so the only decision it can flip
    is fast -> offload, which is value-exact and measured at <=2.4% step time.
    """

    LT = 13  # a fixed latent temporal extent, e.g. one clip-length bucket

    def _drive(self, lt):
        d = ActivationDispatcher(budget_gb=48.0, margin_gb=1.0)
        pts = [(48, 80, 1, 9.0), (64, 96, 2, 12.5), (48, 80, 2, 11.0)]
        for lh, lw, bs, peak in pts:
            d.record(lh, lw, bs, "base", peak_gb=peak, resident_gb=8.0, lt=lt)
        d.record(64, 96, 1, "offload", peak_gb=10.0, resident_gb=8.0,
                 offloaded_gb=1.25, lt=lt)
        d.record(48, 80, 4, "offload", peak_gb=13.0, resident_gb=8.0,
                 offloaded_gb=2.5, lt=lt)
        out = []
        for lh, lw, bs in ((48, 80, 1), (64, 96, 2), (80, 128, 3), (48, 80, 8)):
            out.append(("base", lh, lw, bs, d.base_act(lh, lw, bs, lt=lt)))
            out.append(("off", lh, lw, bs, d.predicted_offloadable(lh, lw, bs, lt=lt)))
            for hr in (0.0, 0.5, 1.0, 2.0, 5.0, 20.0):
                out.append(("dec", lh, lw, bs, hr, d.decide(lh, lw, bs, hr, lt=lt)))
                out.append(("mic", lh, lw, bs, hr, d.plan_micro_bs(lh, lw, bs, hr, lt=lt)))
        return out

    def test_cold_start_seed_scales_with_clip_length_and_only_upward(self):
        """The one place a fixed-clip-length video run does change: the seed."""
        ref = ActivationDispatcher(budget_gb=48.0)
        new = ActivationDispatcher(budget_gb=48.0)
        base = ref.base_act(48, 80, 1)                       # lt defaults to 1
        with_t = new.base_act(48, 80, 1, lt=self.LT)
        self.assertAlmostEqual(with_t, base * self.LT, places=9)
        self.assertGreater(with_t, base)  # never predicts LESS than before

    def test_constant_lt_gives_the_same_decisions_as_lt_1(self):
        """Calibrated regime: the 2-term fit is invariant to a constant lt."""
        ref = self._drive(1)
        got = self._drive(self.LT)
        self.assertEqual(len(ref), len(got))
        for r, g in zip(ref, got):
            with self.subTest(observable=r[:4]):
                self.assertEqual(r[:-1], g[:-1])
                if isinstance(r[-1], str) or isinstance(r[-1], int):
                    self.assertEqual(r[-1], g[-1])
                else:
                    self.assertAlmostEqual(r[-1], g[-1], places=9)


class VideoClipLengthSeparationTest(unittest.TestCase):
    """The bug, and the negative control for the fix.

    Both tests below PASS only while the temporal extent is part of the bucket
    key and of the regression variable. Remove either and they fail.
    """

    # MiniMax-H3 384x640: latent grid 48x80, T_lat = ceil(T/17)*5-3.
    LH, LW = 48, 80
    LT_SHORT, LT_LONG = 7, 37          # 22-frame and 124-frame clips
    ACT_SHORT, ACT_LONG = 2.36, 8.90   # measured GB, batch 1

    def test_a_measured_short_clip_does_not_answer_for_a_long_one(self):
        """The exact per-bucket cache must not be shared across clip lengths.

        With a single sample the 2-term fit has no slope yet and falls back to
        ``seed_coef``, so the long clip gets the SEED prediction rather than a
        calibrated one -- the point being tested here is only that it does not
        get the short clip's cached MEASUREMENT. (The seed slope is an
        image-tuned constant and under-predicts video by ~2.4x until one video
        step has been recorded; that is a seed-calibration property, not this
        bug, and it is what ``test_prediction_tracks_the_measured_3_8x_ratio``
        covers once samples exist.)
        """
        d = ActivationDispatcher(budget_gb=48.0, margin_gb=1.0)
        d.record(self.LH, self.LW, 1, "base",
                 peak_gb=20.0 + self.ACT_SHORT, resident_gb=20.0, lt=self.LT_SHORT)
        short = d.base_act(self.LH, self.LW, 1, lt=self.LT_SHORT)
        long_ = d.base_act(self.LH, self.LW, 1, lt=self.LT_LONG)
        self.assertAlmostEqual(short, self.ACT_SHORT, places=6)
        # NEGATIVE CONTROL: without the temporal term both are the cached 2.36 GB.
        self.assertNotAlmostEqual(
            long_, short, places=3,
            msg="the long clip is being answered by the short clip's cached "
                "measurement -- the temporal term is gone")
        # And it is the seed prediction for the LONG volume, not for the short one.
        self.assertAlmostEqual(long_, 24.0e-6 * self.LH * self.LW * self.LT_LONG,
                               places=6)

    def test_prediction_tracks_the_measured_3_8x_ratio(self):
        """Calibrated from two real clip lengths, the predictor reproduces the
        measured 3.8x separation instead of collapsing it to 1.0x."""
        d = ActivationDispatcher(budget_gb=48.0, margin_gb=1.0)
        # Two measured points at the same spatial size, different clip lengths.
        d.record(self.LH, self.LW, 1, "base",
                 peak_gb=20.0 + self.ACT_SHORT, resident_gb=20.0, lt=self.LT_SHORT)
        d.record(self.LH, self.LW, 1, "base",
                 peak_gb=20.0 + self.ACT_LONG, resident_gb=20.0, lt=self.LT_LONG)
        # An UNSEEN intermediate clip length must land between them, monotonically.
        mid = d.base_act(self.LH, self.LW, 1, lt=22)
        self.assertGreater(mid, self.ACT_SHORT)
        self.assertLess(mid, self.ACT_LONG)
        # NEGATIVE CONTROL on the FIT (not just on the cache): two unseen clip
        # lengths must get different predictions. Drop lt from the regression
        # variable while keeping it in the key and this is where it shows up --
        # both recorded points collapse onto one x, the fit degenerates to a
        # constant, and every unseen clip length gets the same answer.
        self.assertNotAlmostEqual(mid, d.base_act(self.LH, self.LW, 1, lt=30),
                                  places=3)
        ratio = (d.base_act(self.LH, self.LW, 1, lt=self.LT_LONG)
                 / d.base_act(self.LH, self.LW, 1, lt=self.LT_SHORT))
        self.assertAlmostEqual(ratio, self.ACT_LONG / self.ACT_SHORT, places=6)

    def test_decisions_diverge_where_the_footprints_diverge(self):
        """With 6 GB of headroom the short clip fits and the long one does not.
        Pre-change, both got the short clip's answer."""
        d = ActivationDispatcher(budget_gb=48.0, margin_gb=1.0)
        d.record(self.LH, self.LW, 1, "base",
                 peak_gb=20.0 + self.ACT_SHORT, resident_gb=20.0, lt=self.LT_SHORT)
        d.record(self.LH, self.LW, 1, "base",
                 peak_gb=20.0 + self.ACT_LONG, resident_gb=20.0, lt=self.LT_LONG)
        self.assertEqual(d.decide(self.LH, self.LW, 1, 6.0, lt=self.LT_SHORT), "fast")
        self.assertNotEqual(d.decide(self.LH, self.LW, 1, 6.0, lt=self.LT_LONG), "fast")

    def test_overflow_flag_is_per_clip_length(self):
        d = ActivationDispatcher(budget_gb=48.0, margin_gb=1.0)
        d.mark_overflow(self.LH, self.LW, 1, lt=self.LT_LONG)
        self.assertEqual(d.decide(self.LH, self.LW, 1, 40.0, lt=self.LT_LONG), "escalate")
        self.assertEqual(d.decide(self.LH, self.LW, 1, 40.0, lt=self.LT_SHORT), "fast")

    def test_offloadable_cache_is_per_clip_length(self):
        d = ActivationDispatcher(budget_gb=48.0, margin_gb=1.0)
        d.record(self.LH, self.LW, 1, "offload", peak_gb=21.0, resident_gb=20.0,
                 offloaded_gb=1.36, lt=self.LT_SHORT)
        self.assertAlmostEqual(
            d.predicted_offloadable(self.LH, self.LW, 1, lt=self.LT_SHORT), 1.36, places=6)
        self.assertNotAlmostEqual(
            d.predicted_offloadable(self.LH, self.LW, 1, lt=self.LT_LONG), 1.36, places=3)


class LatentKeyExtractionTest(unittest.TestCase):
    """``BaseTrainer`` must read the temporal extent off the latent tensor, and
    must read 1 for a 4-D image latent (whose ``shape[-3]`` is the CHANNEL
    count -- using it would be a silent, large miskey)."""

    def _key(self, t):
        from core.training.base_trainer import BaseTrainer
        return BaseTrainer._actdispatch_latent_key(t)

    def test_image_latent_is_temporally_inert(self):
        # [B, C, H', W'] -- SDXL-ish. C=4 must NOT become lt.
        self.assertEqual(self._key(torch.empty(2, 4, 128, 128)), (128, 128, 1, 2))

    def test_video_latent_carries_the_temporal_extent(self):
        # [B, C, T, H', W'] -- LTX-2.3 / MiniMax-H3 both use this layout.
        self.assertEqual(self._key(torch.empty(1, 128, 37, 48, 80)), (48, 80, 37, 1))
        self.assertEqual(self._key(torch.empty(1, 128, 7, 48, 80)), (48, 80, 7, 1))

    def test_short_and_long_clips_get_different_keys(self):
        """NEGATIVE CONTROL at the extraction layer."""
        short = self._key(torch.empty(1, 128, 7, 48, 80))
        long_ = self._key(torch.empty(1, 128, 37, 48, 80))
        self.assertNotEqual(short, long_)

    def test_three_d_latent_does_not_crash(self):
        self.assertEqual(self._key(torch.empty(2, 16, 64))[3], 2)


if __name__ == "__main__":
    unittest.main()
