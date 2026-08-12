"""Server-held upper bound for the video frame-count SLIDER TRACK
(`UserSettings.video_frame_slider_max`), replacing an earlier, wrongly-scoped
"Default Video Frame Count" localStorage preference (removed along with this
task; see `git log` for `videoFrameSettings.ts` / `global_video_frame_default_test.py`).

Unlike that removed feature, this setting:
  - is a TRACK bound, not a value/default -- the paired number box in
    VideoFrameCountSlider must stay unbounded by it (a user who sets 600 and
    types 900 must still get 900);
  - is held SERVER-SIDE (UserSettings row, GET/POST /settings/generation),
    following the `inpaint_use_dedicated_model` / `lora_dirs` precedent
    instead of a second localStorage store.

Companion to `video_chain_segment_length_test.py` (same source-anchored
style, reading the frontend TS/TSX sources as text). Per-FIELD assertions
throughout, never a whole-line/whole-object match.
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
# 1. The removed feature is actually gone: no leftover references anywhere
#    that could mask a partial revert.
# ---------------------------------------------------------------------------
class RemovedFeatureGoneTest(unittest.TestCase):
    def test_video_frame_settings_module_file_is_deleted(self):
        self.assertFalse(
            os.path.exists(os.path.join(_REPO, "frontend", "src", "utils", "videoFrameSettings.ts"))
        )

    def test_no_source_file_references_the_removed_module_or_symbols(self):
        needles = (
            "videoFrameSettings",
            "readGlobalVideoFrameCount",
            "writeGlobalVideoFrameCount",
            "Default Video Frame Count",
        )
        roots = (
            os.path.join(_REPO, "frontend", "src"),
            os.path.join(_REPO, "backend"),
        )
        # This test file itself legitimately names the removed symbols (as
        # documentation of what must stay gone) -- exclude it from the scan.
        self_path = os.path.abspath(__file__)
        hits = []
        for root in roots:
            for dirpath, _dirnames, filenames in os.walk(root):
                for fname in filenames:
                    if not fname.endswith((".ts", ".tsx", ".py")):
                        continue
                    path = os.path.join(dirpath, fname)
                    if os.path.abspath(path) == self_path:
                        continue
                    with open(path, encoding="utf-8") as handle:
                        text = handle.read()
                    for needle in needles:
                        if needle in text:
                            hits.append(f"{path}: {needle}")
        self.assertEqual(hits, [], f"Leftover references to the removed feature: {hits}")


# ---------------------------------------------------------------------------
# 2. backend/database/models.py: UserSettings declares the column, nullable,
#    and to_dict() surfaces it (raw, no False-style coercion -- None means
#    unset, distinct from 0).
# ---------------------------------------------------------------------------
class UserSettingsModelTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("backend", "database", "models.py")

    def test_column_declared_nullable_integer(self):
        self.assertIn(
            "video_frame_slider_max = Column(Integer, nullable=True)", self.source
        )

    def test_to_dict_surfaces_the_raw_value(self):
        to_dict_start = self.source.index("def to_dict(self):")
        to_dict_end = self.source.index("\n\nclass GeneratedImage", to_dict_start)
        body = self.source[to_dict_start:to_dict_end]
        self.assertIn('"video_frame_slider_max": self.video_frame_slider_max,', body)


# ---------------------------------------------------------------------------
# 3. backend/api/routes.py: GET/POST /settings/generation round-trip the
#    field, including explicit null-clears-the-setting, and reject <= 0.
# ---------------------------------------------------------------------------
class SettingsRouteTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("backend", "api", "routes.py")

    def test_get_returns_the_field(self):
        fn_start = self.source.index('async def get_generation_settings(')
        fn_end = self.source.index("\n@router.post(\"/settings/generation\")", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn(
            '"video_frame_slider_max": settings_record.video_frame_slider_max,', body
        )

    def test_post_accepts_and_persists_the_field(self):
        fn_start = self.source.index('async def save_generation_settings(')
        fn_end = self.source.index("\n@router.post(\"/system/restart-backend\")", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn('if "video_frame_slider_max" in settings_data:', body)
        self.assertIn("settings_record.video_frame_slider_max = None", body)
        self.assertIn("settings_record.video_frame_slider_max = parsed_slider_max", body)

    def test_post_rejects_non_positive_values(self):
        fn_start = self.source.index('async def save_generation_settings(')
        fn_end = self.source.index("\n@router.post(\"/system/restart-backend\")", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn("if parsed_slider_max <= 0:", body)
        self.assertIn("status_code=400", body)

    def test_post_echoes_the_field_back_in_the_response(self):
        fn_start = self.source.index('async def save_generation_settings(')
        fn_end = self.source.index("\n@router.post(\"/system/restart-backend\")", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn('"video_frame_slider_max": settings_record.video_frame_slider_max,', body)


# ---------------------------------------------------------------------------
# 4. backend/api/param_defaults.py: the UI-only checkbox seed lives here, not
#    as a literal in GenerationSettings.tsx (CLAUDE.md's "never hardcode a
#    default outside param_defaults.py").
# ---------------------------------------------------------------------------
class ParamDefaultsSeedTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("backend", "api", "param_defaults.py")

    def test_video_gen_defaults_declares_the_checkbox_seed(self):
        video_gen_start = self.source.index("VIDEO_GEN_DEFAULTS: Dict[str, Any] = {")
        video_gen_end = self.source.index("\n}\n", video_gen_start)
        body = self.source[video_gen_start:video_gen_end]
        self.assertIn('"video_frame_slider_max_seed": 241,', body)


# ---------------------------------------------------------------------------
# 5. openapi.yaml: the field is documented on both GET and POST, nullable,
#    with no duplicate top-level keys introduced under /settings/generation.
# ---------------------------------------------------------------------------
class OpenApiSpecTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("openapi.yaml")

    def test_get_response_declares_the_field_nullable_integer(self):
        get_start = self.source.index("  /settings/generation:")
        post_start = self.source.index("    post:", get_start)
        body = self.source[get_start:post_start]
        self.assertIn("video_frame_slider_max:", body)
        self.assertIn("type: integer", body)
        self.assertIn("nullable: true", body)

    def test_post_request_body_declares_the_field(self):
        post_start = self.source.index("    post:", self.source.index("  /settings/generation:"))
        responses_start = self.source.index("      responses:", post_start)
        body = self.source[post_start:responses_start]
        self.assertIn("video_frame_slider_max:", body)
        self.assertIn("nullable: true", body)

    def test_no_duplicate_keys_in_the_document(self):
        import yaml

        class DupCheckLoader(yaml.SafeLoader):
            pass

        def no_dup_construct(loader, node):
            mapping = {}
            for k_node, v_node in node.value:
                key = loader.construct_object(k_node, deep=True)
                if key in mapping:
                    raise AssertionError(f"Duplicate key: {key!r} at line {k_node.start_mark.line + 1}")
                mapping[key] = loader.construct_object(v_node, deep=True)
            return mapping

        DupCheckLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_dup_construct
        )
        yaml.load(self.source, Loader=DupCheckLoader)


# ---------------------------------------------------------------------------
# 6. frontend/src/utils/api.ts: fetchGenerationSettings + response type.
# ---------------------------------------------------------------------------
class ApiClientTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "utils", "api.ts")

    def test_response_interface_declares_the_field_nullable(self):
        match = re.search(
            r"export interface GenerationSettingsResponse \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("video_frame_slider_max: number | null;", match.group(1))

    def test_fetch_function_hits_the_settings_endpoint(self):
        fn_start = self.source.index("export const fetchGenerationSettings = ")
        fn = self.source[fn_start:fn_start + 200]
        self.assertIn('api.get("/settings/generation")', fn)


# ---------------------------------------------------------------------------
# 7. frontend/src/contexts/StartupContext.tsx: fetched once at startup
#    alongside the other schema/settings payloads and exposed on the context.
# ---------------------------------------------------------------------------
class StartupContextTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "contexts", "StartupContext.tsx")

    def test_context_type_declares_the_field(self):
        match = re.search(
            r"interface StartupContextType \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("videoFrameSliderMax: number | null;", match.group(1))

    def test_default_context_value_is_null(self):
        match = re.search(
            r"const StartupContext = createContext<StartupContextType>\(\{(.*?)\n\}\);",
            self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("videoFrameSliderMax: null,", match.group(1))

    def test_fetch_startup_payloads_fetches_and_stores_it(self):
        fn_start = self.source.index("const fetchStartupPayloads = ")
        fn_end = self.source.index("\n  }, []);", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn("fetchGenerationSettings()", body)
        self.assertIn(
            "setVideoFrameSliderMax(genSettings.video_frame_slider_max ?? null);", body
        )

    def test_provider_value_exposes_the_field(self):
        provider_start = self.source.index("<StartupContext.Provider value={{")
        provider_end = self.source.index("}}>", provider_start)
        body = self.source[provider_start:provider_end]
        self.assertIn("videoFrameSliderMax,", body)


# ---------------------------------------------------------------------------
# 8. VideoFrameCountSlider.tsx: sliderMaxOverride bounds the TRACK only.
# ---------------------------------------------------------------------------
class SliderTrackBoundTest(unittest.TestCase):
    def setUp(self):
        self.source = _read(
            "frontend", "src", "components", "common", "VideoFrameCountSlider.tsx")

    def test_prop_declared_and_defaults_to_null(self):
        self.assertIn("sliderMaxOverride?: number | null;", self.source)
        self.assertIn("sliderMaxOverride = null,", self.source)

    def test_raw_ceiling_uses_the_override_in_both_uncapped_branches(self):
        raw_ceiling_start = self.source.index("const rawCeiling = c.max_frames ?? (")
        raw_ceiling_end = self.source.index(");", raw_ceiling_start) + 2
        body = self.source[raw_ceiling_start:raw_ceiling_end]
        self.assertIn(
            "Math.max(sliderMaxOverride ?? Math.round(c.trained_max_frames * TRAINED_RANGE_SLIDER_HEADROOM), value)",
            body,
        )
        self.assertIn(
            "Math.max(sliderMaxOverride ?? UNCAPPED_FRAME_SLIDER_CEILING, value)", body
        )

    def test_real_hard_cap_still_wins_over_the_override(self):
        """`c.max_frames` (a real architecture wall, e.g. LTX-2.3) is checked
        BEFORE the override is ever consulted -- the override only decides
        how far the track reaches when the architecture imposes no real
        wall, never shrinks or grows a real one."""
        raw_ceiling_line = self.source[self.source.index("const rawCeiling = c.max_frames ?? ("):]
        raw_ceiling_line = raw_ceiling_line[:raw_ceiling_line.index("\n")]
        self.assertTrue(raw_ceiling_line.strip().startswith("const rawCeiling = c.max_frames ?? ("))

    def test_number_box_stays_unbounded_by_the_override(self):
        """The `NumberInput` for `value` has no `max` prop at all (only
        `min`) -- `sliderMaxOverride` must never reach it, or a user typing
        past the track max would be silently clamped, defeating the whole
        point of leaving the number box uncapped."""
        number_input_start = self.source.index("<NumberInput\n          label={videoFrameLabel(caps, arch)}")
        number_input_end = self.source.index("/>", number_input_start)
        body = self.source[number_input_start:number_input_end]
        self.assertNotIn("max=", body)
        self.assertIn("min={min}", body)


# ---------------------------------------------------------------------------
# 9. Txt2ImgPanel / Img2ImgPanel: both call sites of VideoFrameCountSlider
#    (num_frames AND the chain segment length) thread the setting through --
#    this task's deliberate decision to apply it uniformly to both, since
#    both compute their track ceiling through the same shared expression.
# ---------------------------------------------------------------------------
class PanelWiringTest(unittest.TestCase):
    def _read(self, name: str) -> str:
        return _read("frontend", "src", "components", "generation", name)

    def test_both_panels_destructure_video_frame_slider_max_from_startup(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn("videoFrameSliderMax", source)
                self.assertIn("= useStartup();", source)

    def test_both_panels_pass_the_override_to_both_slider_instances(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                occurrences = source.count("sliderMaxOverride={videoFrameSliderMax}")
                self.assertEqual(
                    occurrences, 2,
                    f"{panel}: expected sliderMaxOverride on both the num_frames "
                    f"and chain-segment-length VideoFrameCountSlider instances, "
                    f"found {occurrences}",
                )


# ---------------------------------------------------------------------------
# 10. GenerationSettings.tsx: the Settings-page control, its round trip, and
#     the exact user-visible help text (factual, no invented numbers).
# ---------------------------------------------------------------------------
class GenerationSettingsControlTest(unittest.TestCase):
    def setUp(self):
        self.source = _read(
            "frontend", "src", "components", "settings", "GenerationSettings.tsx")

    def test_loads_the_field_from_the_settings_response(self):
        load_start = self.source.index("const loadSettings = async () => {")
        load_end = self.source.index("\n  };", load_start)
        body = self.source[load_start:load_end]
        self.assertIn("setVideoFrameSliderMax(data.video_frame_slider_max ?? null);", body)

    def test_save_sends_the_field(self):
        save_start = self.source.index("const handleSave = async () => {")
        save_end = self.source.index("\n  };", save_start)
        body = self.source[save_start:save_end]
        self.assertIn("video_frame_slider_max: videoFrameSliderMax,", body)

    def test_checkbox_unchecked_clears_to_null(self):
        self.assertIn(
            "setVideoFrameSliderMax(e.target.checked ? (videoFrameSliderMax ?? videoFrameSliderMaxSeed) : null);",
            self.source,
        )

    def test_seed_comes_from_the_backend_not_a_bare_literal_in_the_ternary(self):
        """The checked-branch of the seed ternary must read the fetched
        `videoFrameSliderMaxSeed` variable, not a numeric literal -- the
        literal (241) is only permitted as ITS OWN pre-fetch fallback,
        checked separately below."""
        ternary_start = self.source.index("setVideoFrameSliderMax(e.target.checked ? (")
        ternary_line = self.source[ternary_start:self.source.index("\n", ternary_start)]
        self.assertIn("videoFrameSliderMaxSeed", ternary_line)
        self.assertNotRegex(ternary_line, r"\?\?\s*\d+\)")

    def test_seed_variable_sourced_from_generation_defaults_txt2vid(self):
        self.assertIn(
            'generationDefaults?.txt2vid?.video_frame_slider_max_seed', self.source
        )

    def test_help_text_states_the_number_box_is_not_bounded(self):
        self.assertIn(
            "The number box next to the slider is not bounded by this setting\n"
            "            and always accepts a value above it.",
            self.source,
        )

    def test_help_text_does_not_reintroduce_default_value_wording(self):
        """This is a TRACK bound, not a starting/default value -- the wording
        must not resemble the removed feature's framing ("starts from")."""
        self.assertNotIn("starts from", self.source)
        self.assertNotIn("in place of the architecture", self.source)


if __name__ == "__main__":
    unittest.main()
