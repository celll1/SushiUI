"""General user-override mechanism for slider/number-input UPPER BOUNDS
(`backend/api/param_defaults.py`'s `PARAM_BOUNDS` registry +
`UserSettings.slider_bounds`, `GET/POST /settings/generation`).

Companion to `video_frame_slider_max_setting_test.py` (same source-anchored
style, reading the backend/frontend sources as text) -- this test file covers
the GENERAL mechanism that `video_frame_slider_max` predates and deliberately
does NOT fold into (see PARAM_BOUNDS's own docstring + the `slider_bounds`
column comment in models.py for why the two coexist).

Per-FIELD assertions throughout, never a whole-line/whole-object match.
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
# 1. backend/api/param_defaults.py: PARAM_BOUNDS registry shape.
# ---------------------------------------------------------------------------
class ParamBoundsRegistryTest(unittest.TestCase):
    def setUp(self):
        from api.param_defaults import PARAM_BOUNDS
        self.bounds = PARAM_BOUNDS

    def test_registry_has_the_seven_documented_rows_across_four_families(self):
        # 6 registry entries + the pre-existing video_frame_slider_max
        # legacy column (not in this registry) = 7 Settings-page rows.
        self.assertEqual(
            set(self.bounds.keys()),
            {
                "image_width_max", "image_height_max",
                "steps_max", "cfg_scale_max",
                "video_frame_rate_max",
                "upscale_tile_size_max",
            },
        )
        families = {spec["family"] for spec in self.bounds.values()}
        self.assertEqual(families, {"canvas", "sampling", "video", "upscale"})

    def test_every_entry_has_the_required_shape(self):
        for name, spec in self.bounds.items():
            with self.subTest(bound=name):
                self.assertIn("builtin", spec)
                self.assertIn("floor", spec)
                self.assertIn("ceiling", spec)
                self.assertIn("family", spec)
                self.assertIn("label", spec)
                self.assertLessEqual(spec["floor"], spec["builtin"])
                self.assertLessEqual(spec["builtin"], spec["ceiling"])

    def test_builtin_values_match_todays_literals(self):
        # These MUST equal the literal each wiring site replaces -- it is
        # what an unchecked row (no override) resolves to.
        self.assertEqual(self.bounds["image_width_max"]["builtin"], 2048)
        self.assertEqual(self.bounds["image_height_max"]["builtin"], 2048)
        self.assertEqual(self.bounds["steps_max"]["builtin"], 150)
        self.assertEqual(self.bounds["cfg_scale_max"]["builtin"], 30)
        self.assertEqual(self.bounds["video_frame_rate_max"]["builtin"], 60)
        self.assertEqual(self.bounds["upscale_tile_size_max"]["builtin"], 4096)


# ---------------------------------------------------------------------------
# 2. backend/database/models.py: UserSettings.slider_bounds column.
# ---------------------------------------------------------------------------
class UserSettingsModelTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("backend", "database", "models.py")

    def test_column_declared_nullable_json(self):
        self.assertIn("slider_bounds = Column(JSON, nullable=True)", self.source)

    def test_column_declared_after_video_frame_slider_max_not_replacing_it(self):
        """The two must coexist -- `video_frame_slider_max` must still be
        present (not folded into slider_bounds)."""
        self.assertIn("video_frame_slider_max = Column(Integer, nullable=True)", self.source)
        vfsm_idx = self.source.index("video_frame_slider_max = Column(Integer, nullable=True)")
        sb_idx = self.source.index("slider_bounds = Column(JSON, nullable=True)")
        self.assertLess(vfsm_idx, sb_idx)

    def test_to_dict_surfaces_slider_bounds_as_a_dict_not_none(self):
        to_dict_start = self.source.index("def to_dict(self):")
        to_dict_end = self.source.index("\n\nclass GeneratedImage", to_dict_start)
        body = self.source[to_dict_start:to_dict_end]
        self.assertIn('"slider_bounds": self.slider_bounds or {},', body)


# ---------------------------------------------------------------------------
# 3. backend/api/routes.py: GET/POST /settings/generation handle
#    slider_bounds, and GET /schema/generation-defaults serves PARAM_BOUNDS.
# ---------------------------------------------------------------------------
class SettingsRouteTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("backend", "api", "routes.py")

    def test_param_bounds_imported(self):
        import_start = self.source.index("from api.param_defaults import (")
        import_end = self.source.index(")", import_start)
        self.assertIn("PARAM_BOUNDS", self.source[import_start:import_end])

    def test_schema_generation_defaults_serves_param_bounds(self):
        fn_start = self.source.index("async def get_generation_defaults():")
        fn_end = self.source.index("\n\n@router.get(\"/schema/prompt-assist-defaults\")", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn('"param_bounds": PARAM_BOUNDS,', body)

    def test_get_returns_slider_bounds(self):
        fn_start = self.source.index('async def get_generation_settings(')
        fn_end = self.source.index("\n@router.post(\"/settings/generation\")", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn('"slider_bounds": settings_record.slider_bounds or {},', body)

    def test_post_rejects_unknown_bound_keys(self):
        fn_start = self.source.index('async def save_generation_settings(')
        fn_end = self.source.index("\n@router.post(\"/system/restart-backend\")", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn("if bound_name not in PARAM_BOUNDS:", body)
        self.assertIn("Unknown slider bound", body)

    def test_post_rejects_out_of_range_values(self):
        fn_start = self.source.index('async def save_generation_settings(')
        fn_end = self.source.index("\n@router.post(\"/system/restart-backend\")", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn('if parsed_value < spec["floor"] or parsed_value > spec["ceiling"]:', body)
        self.assertIn("status_code=400", body)

    def test_post_null_value_resets_one_key_without_clearing_the_whole_map(self):
        fn_start = self.source.index('async def save_generation_settings(')
        fn_end = self.source.index("\n@router.post(\"/system/restart-backend\")", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn("merged.pop(bound_name, None)", body)

    def test_post_echoes_slider_bounds_back(self):
        fn_start = self.source.index('async def save_generation_settings(')
        fn_end = self.source.index("\n@router.post(\"/system/restart-backend\")", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn('"slider_bounds": settings_record.slider_bounds or {},', body)


# ---------------------------------------------------------------------------
# 4. openapi.yaml: no duplicate top-level keys after this task's edits, and
#    both directions of /settings/generation document slider_bounds.
# ---------------------------------------------------------------------------
class OpenApiSpecTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("openapi.yaml")

    def test_get_and_post_both_declare_slider_bounds(self):
        get_start = self.source.index("  /settings/generation:")
        post_start = self.source.index("    post:", get_start)
        responses_start = self.source.index("      responses:", post_start)
        get_body = self.source[get_start:post_start]
        post_body = self.source[post_start:responses_start]
        self.assertIn("slider_bounds:", get_body)
        self.assertIn("slider_bounds:", post_body)

    def test_generation_defaults_declares_param_bounds(self):
        start = self.source.index("  /schema/generation-defaults:")
        end = self.source.index("  /schema/prompt-assist-defaults:", start)
        body = self.source[start:end]
        self.assertIn("param_bounds:", body)

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
# 5. frontend/src/utils/api.ts: response type + save function.
# ---------------------------------------------------------------------------
class ApiClientTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "utils", "api.ts")

    def test_response_interface_declares_slider_bounds(self):
        match = re.search(
            r"export interface GenerationSettingsResponse \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("slider_bounds: Record<string, number>;", match.group(1))

    def test_generation_defaults_response_declares_param_bounds(self):
        match = re.search(
            r"export interface GenerationDefaultsResponse \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("param_bounds?: ParamBoundsRegistry;", match.group(1))

    def test_save_slider_bounds_function_posts_to_settings_generation(self):
        fn_start = self.source.index("export const saveSliderBounds = ")
        fn = self.source[fn_start:fn_start + 300]
        self.assertIn('api.post("/settings/generation", { slider_bounds: overrides })', fn)


# ---------------------------------------------------------------------------
# 6. frontend/src/utils/paramBounds.ts: resolver precedence.
# ---------------------------------------------------------------------------
class ResolveBoundTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "utils", "paramBounds.ts")

    def test_arch_limit_clamps_after_override_and_builtin_are_resolved(self):
        fn_start = self.source.index("export function resolveBound(")
        fn_end = self.source.index("\n}", fn_start)
        body = self.source[fn_start:fn_end]
        override_idx = body.index("const override = sliderBounds?.[boundName];")
        resolved_idx = body.index("let resolved = override ?? builtin;")
        arch_idx = body.index("if (archLimit != null) {")
        # override/builtin must be computed BEFORE the arch clamp is applied,
        # so the arch clamp can override (min) whatever they produced.
        self.assertLess(override_idx, arch_idx)
        self.assertLess(resolved_idx, arch_idx)

    def test_arch_limit_uses_min_not_max(self):
        self.assertIn("resolved = Math.min(resolved, archLimit);", self.source)

    def test_final_return_never_strands_the_current_value_below_the_track(self):
        self.assertIn("return Math.max(resolved, currentValue);", self.source)
        # This must be the LAST statement in the function body (after the
        # arch-limit clamp), not computed before it.
        arch_idx = self.source.index("if (archLimit != null) {")
        return_idx = self.source.index("return Math.max(resolved, currentValue);")
        self.assertLess(arch_idx, return_idx)

    def test_builtin_precedes_override_as_the_fallback_not_the_winner(self):
        # "override ?? builtin": override wins when set, builtin is only the
        # fallback -- not the other way around.
        self.assertIn("let resolved = override ?? builtin;", self.source)


# ---------------------------------------------------------------------------
# 7. frontend/src/contexts/StartupContext.tsx: sliderBounds plumbing.
# ---------------------------------------------------------------------------
class StartupContextTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "contexts", "StartupContext.tsx")

    def test_context_type_declares_slider_bounds_and_setter(self):
        match = re.search(
            r"interface StartupContextType \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("sliderBounds: Record<string, number>;", match.group(1))
        self.assertIn("setSliderBounds: (value: Record<string, number>) => void;", match.group(1))

    def test_fetch_startup_payloads_seeds_slider_bounds_from_settings(self):
        fn_start = self.source.index("const fetchStartupPayloads = ")
        fn_end = self.source.index("\n  }, []);", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn("setSliderBounds(genSettings.slider_bounds ?? {});", body)

    def test_provider_exposes_slider_bounds_and_setter(self):
        provider_start = self.source.index("<StartupContext.Provider value={{")
        provider_end = self.source.index("}}>", provider_start)
        body = self.source[provider_start:provider_end]
        self.assertIn("sliderBounds,", body)
        self.assertIn("setSliderBounds,", body)


# ---------------------------------------------------------------------------
# 8. Panel wiring: each exposed bound is actually consumed via resolveBound
#    at the sites this task specifies (not left as a bare literal).
# ---------------------------------------------------------------------------
class PanelWiringTest(unittest.TestCase):
    def _read(self, name: str) -> str:
        return _read("frontend", "src", "components", "generation", name)

    def test_txt2img_wires_canvas_sampling_and_video_bounds(self):
        source = self._read("Txt2ImgPanel.tsx")
        self.assertIn('resolveBound("image_width_max"', source)
        self.assertIn('resolveBound("image_height_max"', source)
        self.assertIn('resolveBound("steps_max"', source)
        self.assertIn('resolveBound("cfg_scale_max"', source)
        self.assertIn('resolveBound("video_frame_rate_max"', source)
        # No more bare `max={2048}` for width/height (both replaced).
        self.assertNotIn("max={2048}", source)

    def test_img2img_wires_canvas_sampling_and_video_bounds(self):
        source = self._read("Img2ImgPanel.tsx")
        self.assertIn('resolveBound("image_width_max"', source)
        self.assertIn('resolveBound("image_height_max"', source)
        self.assertIn('resolveBound("steps_max"', source)
        self.assertIn('resolveBound("cfg_scale_max"', source)
        self.assertIn('resolveBound("video_frame_rate_max"', source)
        self.assertNotIn("max={2048}", source)

    def test_upscale_wires_tile_size_and_sampling_bounds(self):
        source = self._read("UpscalePanel.tsx")
        self.assertIn('resolveBound("upscale_tile_size_max"', source)
        self.assertIn('resolveBound("steps_max"', source)
        self.assertIn('resolveBound("cfg_scale_max"', source)

    def test_inpaint_panel_untouched_by_this_task(self):
        """InpaintPanel.tsx is owned by another concurrent session for this
        task -- this mechanism must not have touched it; its two canvas
        literals are documented in PARAM_BOUNDS as still needing wiring."""
        source = self._read("InpaintPanel.tsx")
        self.assertNotIn("resolveBound(", source)


# ---------------------------------------------------------------------------
# 9. settings/page.tsx: the "Slider Bounds" card exists, is generic over the
#    registry, and the moved video_frame_slider_max control still satisfies
#    every assertion video_frame_slider_max_setting_test.py's
#    SettingsPageControlTest makes (verified there, not duplicated here).
# ---------------------------------------------------------------------------
class SettingsPageSliderBoundsCardTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "app", "settings", "page.tsx")

    def test_card_exists(self):
        self.assertIn('<Card title="Slider Bounds">', self.source)

    def test_page_imports_save_slider_bounds_and_is_above_builtin(self):
        self.assertIn("saveSliderBounds", self.source)
        self.assertIn('import { isAboveBuiltin } from "@/utils/paramBounds";', self.source)

    def test_page_destructures_slider_bounds_from_startup(self):
        self.assertIn("sliderBounds: liveSliderBounds,", self.source)
        self.assertIn("setSliderBounds: setLiveSliderBounds,", self.source)

    def test_card_renders_generically_over_the_registry_not_one_block_per_bound(self):
        card_start = self.source.index('<Card title="Slider Bounds">')
        card_end = self.source.index("</Card>", card_start)
        body = self.source[card_start:card_end]
        self.assertIn("sliderBoundFamilies.map((family) =>", body)
        self.assertIn("Object.entries(paramBounds)", body)

    def test_card_has_a_reset_all_control(self):
        card_start = self.source.index('<Card title="Slider Bounds">')
        card_end = self.source.index("</Card>", card_start)
        body = self.source[card_start:card_end]
        self.assertIn("resetAllSliderBounds", body)
        self.assertIn("Reset All", body)

    def test_scope_statement_is_factual_not_a_performance_claim(self):
        self.assertIn(
            "Raises the slider/number-input range for the settings below;\n"
            "                does not change model or hardware limits.",
            self.source,
        )

    def test_per_row_commit_is_debounced_not_per_keystroke(self):
        fn_start = self.source.index(
            "const handleSliderBoundNumberCommit = (boundName: string, v: number) => {")
        fn_end = self.source.index("\n  };", fn_start)
        body = self.source[fn_start:fn_end]
        self.assertIn("setTimeout(", body)
        self.assertIn("void commitSliderBound(boundName, v);", body)

    def test_commit_reverts_on_failure_and_reports_it(self):
        fn_start = self.source.index(
            "const commitSliderBound = async (boundName: string, value: number | null) => {")
        fn_end = self.source.index("\n  };", fn_start)
        body = self.source[fn_start:fn_end]
        catch_start = body.index("} catch (error)")
        catch_body = body[catch_start:]
        self.assertIn('type: "error"', catch_body)
        self.assertIn("setSliderBoundEnabled((prev) => ({ ...prev, [boundName]: liveSliderBounds[boundName] != null }));", catch_body)

    def test_flush_on_unmount(self):
        self.assertIn("Flushed on unmount", self.source)
        self.assertIn("void saveSliderBounds({ [boundName]: pending })", self.source)

    def test_video_frame_slider_max_control_moved_into_this_card(self):
        card_start = self.source.index('<Card title="Slider Bounds">')
        card_end = self.source.index("</Card>", card_start)
        body = self.source[card_start:card_end]
        self.assertIn('id="video_frame_slider_max_enabled"', body)
        self.assertIn('id="video_frame_slider_max"', body)

    def test_generation_behavior_card_no_longer_contains_the_video_control(self):
        gb_start = self.source.index('<Card title="Generation Behavior">')
        gb_end = self.source.index("</Card>", gb_start)
        body = self.source[gb_start:gb_end]
        self.assertNotIn('id="video_frame_slider_max_enabled"', body)


if __name__ == "__main__":
    unittest.main()
