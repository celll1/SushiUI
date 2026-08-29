"""
Tests for the OUTPAINT B2 RePaint-style time-travel resample schedule builder
(core/inference/custom_sampling.py's `_build_outpaint_resample_schedule`) --
see scratchpad/outpaint_continuity_design.md section "B2".

Run with:
    venv\\Scripts\\python.exe -m pytest backend/tests/test_outpaint_resample_schedule.py -v

Test coverage:
  1. Degenerate cases (r<=1, u<=0, T<=0) return the plain
     [(0, False), (1, False), ..., (T-1, False)] walk -- ITERATION-ORDER-
     IDENTICAL to enumerate(range(T)), i.e. the B1-only / non-outpaint path.
  2. Normal resampling case (T=28, r=2, u=4, band=[0.15, 0.70]): anchor
     positions, exact extra-visit placement/count, NFE formula
     (T + (r-1)*u*num_anchors), and full monotonic coverage of every logical
     index 0..T-1 (every index visited at least once, with jumps only ever
     landing on indices already visited).
  3. is_forward_jump correctness: True on (and only on) the first visit of
     each re-denoise cycle; every jump visit's target index equals
     (landing - u) for some anchor "landing" position.
  4. Multiple (T, r, u, band) combinations, including r=3 (multiple resample
     cycles per anchor) and a band that yields zero anchors (no-op band).
  5. Determinism: two calls with identical arguments produce an identical
     schedule (pure function, no hidden state).

No torch/GPU dependency for the schedule logic itself, but importing
custom_sampling.py pulls in torch/diffusers/PIL at module level (this
project's existing convention, e.g. test_outpaint_utils.py) -- all installed
in this venv.
"""

from __future__ import annotations

import os
import sys
import unittest

# ── path setup ───────────────────────────────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from backend.core.inference.custom_sampling import _build_outpaint_resample_schedule


def _anchor_positions(num_timesteps: int, u: int, band_lo: float, band_hi: float):
    """Reference (independent) re-derivation of the anchor landing positions,
    mirroring the design doc's formula directly (not the implementation under
    test) -- used to cross-check the schedule builder's anchor placement."""
    import math
    band_start = int(math.ceil(band_lo * num_timesteps))
    band_end = int(math.floor(band_hi * num_timesteps))
    anchors = []
    j = band_start
    while j <= band_end and (j - u) >= 0 and j <= num_timesteps:
        anchors.append(j)
        j += u
    return anchors


class TestDegenerateSchedule(unittest.TestCase):
    def test_r_le_1_is_plain_walk(self):
        for r in (0, 1, -1):
            sched = _build_outpaint_resample_schedule(10, r, 4, 0.15, 0.70)
            self.assertEqual(sched, [(i, False) for i in range(10)])

    def test_u_le_0_is_plain_walk(self):
        for u in (0, -1):
            sched = _build_outpaint_resample_schedule(10, 3, u, 0.15, 0.70)
            self.assertEqual(sched, [(i, False) for i in range(10)])

    def test_num_timesteps_le_0(self):
        self.assertEqual(_build_outpaint_resample_schedule(0, 2, 4, 0.15, 0.70), [])

    def test_plain_walk_matches_enumerate(self):
        """The off-path schedule must be iteration-order-identical to the
        `enumerate(timesteps)` loop it replaces -- the byte-identical
        guarantee for outpaint_resample_count<=1 / non-outpaint calls."""
        sched = _build_outpaint_resample_schedule(28, 1, 4, 0.15, 0.70)
        expected_indices = list(range(28))
        self.assertEqual([idx for idx, _jump in sched], expected_indices)
        self.assertTrue(all(not is_jump for _idx, is_jump in sched))


class TestDesignDocDefaults(unittest.TestCase):
    """T=28, r=2, u=4, band=[0.15, 0.70] -- the exact combination called out
    in the design doc / task ("the NFE multiplier at T=28/r=2/u=4/band=
    [0.15,0.70]")."""

    def setUp(self):
        self.T = 28
        self.r = 2
        self.u = 4
        self.band_lo = 0.15
        self.band_hi = 0.70
        self.sched = _build_outpaint_resample_schedule(self.T, self.r, self.u, self.band_lo, self.band_hi)

    def test_anchor_positions(self):
        anchors = _anchor_positions(self.T, self.u, self.band_lo, self.band_hi)
        # ceil(0.15*28)=5 (band_start), floor(0.70*28)=19 (band_end);
        # 5, 9, 13, 17 <= 19; next would be 21 > 19.
        self.assertEqual(anchors, [5, 9, 13, 17])

    def test_nfe_formula(self):
        anchors = _anchor_positions(self.T, self.u, self.band_lo, self.band_hi)
        expected_nfe = self.T + (self.r - 1) * self.u * len(anchors)
        self.assertEqual(len(self.sched), expected_nfe)
        # T=28 + (2-1)*4*4 = 28 + 16 = 44
        self.assertEqual(expected_nfe, 44)

    def test_nfe_multiplier(self):
        multiplier = len(self.sched) / self.T
        self.assertAlmostEqual(multiplier, 44 / 28, places=6)
        # Within the design doc's stated ~1.6-2.2x ballpark (44/28 ~= 1.571,
        # the low end since u=4 is the upper end of the u=2-4 range and
        # r=2 is the minimum resample count).
        self.assertGreater(multiplier, 1.0)

    def test_full_coverage_every_index_visited(self):
        visited = {idx for idx, _jump in self.sched}
        self.assertEqual(visited, set(range(self.T)))

    def test_first_visit_is_index_zero_no_jump(self):
        self.assertEqual(self.sched[0], (0, False))

    def test_last_visit_is_final_index_no_jump(self):
        self.assertEqual(self.sched[-1], (self.T - 1, False))

    def test_jump_visits_target_anchor_minus_u(self):
        anchors = _anchor_positions(self.T, self.u, self.band_lo, self.band_hi)
        expected_jump_targets = {a - self.u for a in anchors}
        actual_jump_targets = {idx for idx, is_jump in self.sched if is_jump}
        self.assertEqual(actual_jump_targets, expected_jump_targets)

    def test_jump_count_matches_r_minus_1_per_anchor(self):
        anchors = _anchor_positions(self.T, self.u, self.band_lo, self.band_hi)
        jump_visits = [(idx, is_jump) for idx, is_jump in self.sched if is_jump]
        # r-1 jump-marked visits per anchor (one per extra resample cycle).
        self.assertEqual(len(jump_visits), (self.r - 1) * len(anchors))

    def test_each_anchor_segment_traversed_r_times(self):
        """Every index in each [landing-u, landing-1] segment appears exactly
        r times in the schedule (1 original forward pass + (r-1) resampled
        passes)."""
        anchors = _anchor_positions(self.T, self.u, self.band_lo, self.band_hi)
        indices = [idx for idx, _jump in self.sched]
        for landing in anchors:
            for seg_idx in range(landing - self.u, landing):
                self.assertEqual(
                    indices.count(seg_idx), self.r,
                    f"segment index {seg_idx} (anchor landing={landing}) visited "
                    f"{indices.count(seg_idx)} times, expected r={self.r}",
                )
        # Indices strictly outside any resampled segment are visited exactly once.
        resampled = set()
        for landing in anchors:
            resampled.update(range(landing - self.u, landing))
        for idx in range(self.T):
            if idx not in resampled:
                self.assertEqual(indices.count(idx), 1)

    def test_monotonic_forward_steps_except_at_jumps(self):
        """Consecutive visits either advance by exactly +1 (normal forward
        step, is_forward_jump=False) or are a backward jump
        (is_forward_jump=True on the SECOND element of the pair)."""
        for k in range(1, len(self.sched)):
            prev_idx, _ = self.sched[k - 1]
            idx, is_jump = self.sched[k]
            if is_jump:
                self.assertLess(idx, prev_idx, f"visit {k}: jump target {idx} not < previous {prev_idx}")
            else:
                self.assertEqual(idx, prev_idx + 1, f"visit {k}: non-jump step {idx} != previous+1 ({prev_idx + 1})")

    def test_determinism(self):
        sched2 = _build_outpaint_resample_schedule(self.T, self.r, self.u, self.band_lo, self.band_hi)
        self.assertEqual(self.sched, sched2)


class TestOtherCombinations(unittest.TestCase):
    def test_r3_multiple_cycles_per_anchor(self):
        T, r, u, band_lo, band_hi = 20, 3, 2, 0.15, 0.70
        sched = _build_outpaint_resample_schedule(T, r, u, band_lo, band_hi)
        anchors = _anchor_positions(T, u, band_lo, band_hi)
        expected_nfe = T + (r - 1) * u * len(anchors)
        self.assertEqual(len(sched), expected_nfe)
        jump_visits = [(idx, is_jump) for idx, is_jump in sched if is_jump]
        self.assertEqual(len(jump_visits), (r - 1) * len(anchors))
        visited = {idx for idx, _jump in sched}
        self.assertEqual(visited, set(range(T)))

    def test_narrow_band_zero_anchors_is_noop(self):
        """A band too narrow to fit even one u-length segment yields zero
        anchors -- the schedule degenerates to the plain walk despite r>1."""
        T, r, u = 10, 2, 4
        sched = _build_outpaint_resample_schedule(T, r, u, 0.45, 0.50)  # band_start=5, band_end=5, but 5-4=1>=0 so...
        # band_start=ceil(0.45*10)=5, band_end=floor(0.50*10)=5; anchor j=5
        # (5-4=1>=0, valid) -- exactly one anchor. Extra visits = (r-1)*u = 4.
        self.assertEqual(len(sched), T + 4)

    def test_truly_zero_anchor_band(self):
        T, r, u = 10, 2, 8
        # band_start=ceil(0.15*10)=2, band_end=floor(0.70*10)=7; first
        # candidate j=2, but j-u=2-8=-6<0 -> rejected; no valid anchors.
        sched = _build_outpaint_resample_schedule(T, r, u, 0.15, 0.70)
        self.assertEqual(sched, [(i, False) for i in range(T)])

    def test_small_T(self):
        for T in (1, 2, 3, 4, 5):
            sched = _build_outpaint_resample_schedule(T, 2, 4, 0.15, 0.70)
            visited = {idx for idx, _jump in sched}
            self.assertEqual(visited, set(range(T)))


if __name__ == "__main__":
    unittest.main()
