"""Frontend-only: a global user preference (Settings -> Default Video Frame
Count) for the frame count a video generation panel starts from, in place of
the per-architecture served default.

Companion to `video_chain_segment_length_test.py` (same source-anchored
style, reading the frontend TS/TSX sources as text). Per-FIELD assertions
throughout, never a whole-line/whole-object match.

Precedence pinned by these tests, matching what Txt2ImgPanel.tsx /
Img2ImgPanel.tsx actually implement:
  1. A panel's own persisted `params.num_frames` (from a previous session)
     always wins -- the global preference is only consulted at the
     "nothing to fall back on yet" seed gate (no `localStorage[STORAGE_KEY]`
     for that panel), so it can never make an already-working panel jump.
  2. Otherwise, the global preference (if set) overrides the schema-served
     `generationDefaults` default for `num_frames` specifically, leaving
     every other seeded field (steps, cfg_scale, etc.) untouched.
  3. The seeded value is applied UNSNAPPED; the pre-existing
     `normalizeVideoFrames` re-snap effect (keyed on
     `[archCapabilities, loadedArch]`) is what puts it on the loaded
     architecture's actual grid, once that is known -- the same helper
     `num_frames` itself is always snapped with, not a second copy.
  4. Unset (`null`) is the initial state and changes nothing for a user who
     never opens Settings.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _read(*parts: str) -> str:
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# 1. videoFrameSettings.ts: the localStorage-backed read/write pair. Kept as
#    its own store (mirroring attentionSettings.ts), NOT added to the
#    backend UserSettings row -- see the module docstring inside the file
#    for the justification.
# ---------------------------------------------------------------------------
class VideoFrameSettingsModuleTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "utils", "videoFrameSettings.ts")

    def test_declares_its_own_storage_key(self):
        self.assertIn('const STORAGE_KEY = "default_video_frame_count";', self.source)

    def test_read_returns_null_for_unset_and_number_for_set(self):
        fn_start = self.source.index("export const readGlobalVideoFrameCount = ")
        fn = self.source[fn_start:fn_start + self.source[fn_start:].index("\n};") + 3]
        self.assertIn("if (raw == null) return null;", fn)
        self.assertIn("Number.isFinite(parsed) && parsed > 0 ? parsed : null", fn)

    def test_write_null_removes_the_key_rather_than_storing_a_sentinel(self):
        fn_start = self.source.index("export const writeGlobalVideoFrameCount = ")
        fn = self.source[fn_start:]
        self.assertIn("window.localStorage.removeItem(STORAGE_KEY);", fn)
        self.assertIn("window.localStorage.setItem(STORAGE_KEY, String(value));", fn)

    def test_module_does_not_snap_the_value_itself(self):
        """No `nearestValidVideoFrameCount` / grid arithmetic call in this
        file -- snapping happens once, at the point of use in each panel,
        reusing the same helper `num_frames` is snapped with."""
        self.assertNotIn("nearestValidVideoFrameCount(", self.source)
        self.assertNotIn("normalizeVideoFrames(", self.source)


# ---------------------------------------------------------------------------
# 2. settings/page.tsx: the checkbox+NumberInput control, its localStorage
#    round trip, and the exact user-visible copy.
# ---------------------------------------------------------------------------
class SettingsPageControlTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "app", "settings", "page.tsx")

    def test_imports_the_dedicated_read_write_pair(self):
        self.assertIn(
            'import { readGlobalVideoFrameCount, writeGlobalVideoFrameCount } from "@/utils/videoFrameSettings";',
            self.source,
        )

    def test_state_declared_and_defaults_to_unset(self):
        self.assertIn(
            "const [defaultVideoFrameCount, setDefaultVideoFrameCount] = useState<number | null>(null);",
            self.source,
        )

    def test_mount_effect_only_overrides_state_when_a_value_was_actually_saved(self):
        self.assertIn("const savedDefaultVideoFrameCount = readGlobalVideoFrameCount();", self.source)
        self.assertIn("if (savedDefaultVideoFrameCount != null) {", self.source)

    def test_unchecking_clears_state_and_storage(self):
        self.assertIn("setDefaultVideoFrameCount(null);", self.source)
        self.assertIn("writeGlobalVideoFrameCount(null);", self.source)

    def test_committing_a_value_writes_through_to_storage(self):
        """The `NumberInput` for this field commits to BOTH the local state
        and storage in the same handler (state alone would not survive a
        reload; storage alone would not update the control on this page)."""
        commit_start = self.source.index('label="Default Video Frame Count"')
        commit_body = self.source[commit_start:commit_start + 300]
        self.assertIn("onCommit={(v) => {", commit_body)
        self.assertIn("setDefaultVideoFrameCount(v);", commit_body)
        self.assertIn("writeGlobalVideoFrameCount(v);", commit_body)

    def test_checkbox_label_is_exact(self):
        self.assertIn("Default Video Frame Count", self.source)

    def test_helper_text_is_exact_and_states_the_precedence(self):
        self.assertIn(
            "Frame count a video generation panel starts from for a new (never-persisted) session, "
            "in place of the architecture&apos;s own served default. Snapped to the loaded "
            "architecture&apos;s frame grid when a video model is loaded. Unchecked uses the "
            "architecture default. A panel that already has its own saved frame count (from a "
            "previous session) keeps that value; this setting only applies where there is nothing "
            "to fall back on yet.",
            self.source,
        )


# ---------------------------------------------------------------------------
# 3. Txt2ImgPanel / Img2ImgPanel: the global preference is consulted ONLY at
#    the pre-existing "no localStorage[STORAGE_KEY]" schema-defaults seed
#    gate, and only overrides `num_frames` -- every other field still comes
#    from `generationDefaults`.
# ---------------------------------------------------------------------------
class PanelPrecedenceTest(unittest.TestCase):
    def _read(self, name: str) -> str:
        return _read("frontend", "src", "components", "generation", name)

    def test_both_panels_import_the_reader(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn(
                    'import { readGlobalVideoFrameCount } from "@/utils/videoFrameSettings";',
                    source,
                )

    def test_both_panels_read_the_global_inside_the_no_stored_params_branch(self):
        """The read call must be textually inside the `if (!stored) {`
        branch of the defaults-seed effect -- reading it unconditionally
        (outside that guard) would let it re-fire and override a panel the
        user is actively working in on every `generationDefaults` change."""
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                gate_start = source.index("const stored = localStorage.getItem(STORAGE_KEY);\n    if (!stored) {")
                # The effect body ends at the closing of the `if (!stored)` block,
                # marked by the following `}, [generationDefaults]);`.
                gate_end = source.index("}, [generationDefaults]);", gate_start)
                gate_body = source[gate_start:gate_end]
                self.assertIn("const globalFrames = readGlobalVideoFrameCount();", gate_body)

    def test_both_panels_only_override_num_frames_not_the_whole_seed(self):
        """Spread order matters: `generationDefaults` is spread first
        (supplying every field), and only `num_frames` is conditionally
        overridden afterwards -- steps/cfg_scale/etc. are never touched by
        this preference."""
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn(
                    "...(globalFrames != null ? { num_frames: globalFrames } : {}),",
                    source,
                )

    def test_both_panels_leave_snapping_to_the_pre_existing_normalize_effect(self):
        """No grid-arithmetic call at the seed site itself -- confirms the
        seeded value is applied raw and relies on the existing
        `[archCapabilities, loadedArch]`-keyed effect (already required by
        `video_chain_segment_length_test.py` for `num_frames` generally) to
        put it on the loaded architecture's grid."""
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                seed_start = source.index("const globalFrames = readGlobalVideoFrameCount();")
                seed_end = source.index("}, [generationDefaults]);", seed_start)
                seed_body = source[seed_start:seed_end]
                self.assertNotIn("nearestValidVideoFrameCount(", seed_body)
                self.assertNotIn("normalizeVideoFrames(", seed_body)


if __name__ == "__main__":
    unittest.main()
