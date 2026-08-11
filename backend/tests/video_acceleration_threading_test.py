"""End-to-end FBCache/Spectrum threading through the VIDEO generation path.

Companion to `video_block_swap_threading_test.py` (same source-anchored
style): before this change, Txt2ImgPanel and Img2ImgPanel exposed Block Swap
in video mode but no FBCache/Spectrum control at all, and InpaintPanel had no
Spectrum control. Every video route already accepted these fields
(`Txt2VidRequest`/the four multipart routes); the gap was purely frontend.

Per-field assertions throughout, not one big object-literal comparison: a
future edit that drops exactly one of these fields (e.g. `spectrum_tail`)
should fail on that field's own assertion, not get lost inside a dict-equality
diff.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from api import routes  # noqa: E402
from api.param_defaults import (  # noqa: E402
    IMG2VID_DEFAULTS,
    REF2VID_DEFAULTS,
    TXT2VID_DEFAULTS,
    VIDEO_GEN_DEFAULTS,
)

_ACCEL_FIELDS = (
    "fbcache_enable", "fbcache_threshold", "fbcache_warmup_steps",
    "spectrum_enable", "spectrum_w", "spectrum_w_decay", "spectrum_delta_cap",
    "spectrum_m", "spectrum_lam", "spectrum_warmup_steps",
    "spectrum_window_size", "spectrum_flex_window", "spectrum_tail",
    "spectrum_max_cache",
)


# ---------------------------------------------------------------------------
# 1. param_defaults.py: every video default map declares the full FBCache/
#    Spectrum field set (already true before this change -- guards against a
#    future regression removing a field from the SSOT).
# ---------------------------------------------------------------------------
class ParamDefaultsTest(unittest.TestCase):
    def test_every_video_default_map_has_every_acceleration_field(self):
        for name, defaults in (
            ("VIDEO_GEN_DEFAULTS", VIDEO_GEN_DEFAULTS),
            ("TXT2VID_DEFAULTS", TXT2VID_DEFAULTS),
            ("IMG2VID_DEFAULTS", IMG2VID_DEFAULTS),
            ("REF2VID_DEFAULTS", REF2VID_DEFAULTS),
        ):
            for field in _ACCEL_FIELDS:
                with self.subTest(defaults=name, field=field):
                    self.assertIn(field, defaults, f"{name} is missing '{field}'")

    def test_fbcache_and_spectrum_default_off(self):
        for name, defaults in (
            ("VIDEO_GEN_DEFAULTS", VIDEO_GEN_DEFAULTS),
            ("TXT2VID_DEFAULTS", TXT2VID_DEFAULTS),
            ("IMG2VID_DEFAULTS", IMG2VID_DEFAULTS),
            ("REF2VID_DEFAULTS", REF2VID_DEFAULTS),
        ):
            with self.subTest(defaults=name):
                self.assertFalse(defaults["fbcache_enable"])
                self.assertFalse(defaults["spectrum_enable"])


# ---------------------------------------------------------------------------
# 2. Backend request shapes: Txt2VidRequest (JSON) and the three multipart
#    video routes (img2vid/ref2vid) all accept every acceleration field. This
#    was already true (backend-side plumbing predates this task); guards
#    against a future regression removing one.
# ---------------------------------------------------------------------------
class BackendRouteFieldsTest(unittest.TestCase):
    def test_txt2vid_request_declares_every_acceleration_field(self):
        for field in _ACCEL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, routes.Txt2VidRequest.model_fields)

    def test_multipart_video_routes_accept_every_acceleration_field(self):
        for func_name in ("generate_img2vid", "generate_ref2vid"):
            params = inspect.signature(getattr(routes, func_name)).parameters
            for field in _ACCEL_FIELDS:
                with self.subTest(route=func_name, field=field):
                    self.assertIn(
                        field, params,
                        f"{func_name} has no '{field}' Form parameter -- a client's "
                        f"choice can never reach this route at all",
                    )


# ---------------------------------------------------------------------------
# 3. frontend/src/utils/api.ts: Txt2VidParams declares every field, and every
#    one of the three video senders (generateTxt2Vid JSON body,
#    generateImg2Vid/generateRef2Vid FormData) sends it.
# ---------------------------------------------------------------------------
class FrontendApiTest(unittest.TestCase):
    @staticmethod
    def _function_source(source: str, name: str) -> str:
        start = source.index(f"export const {name} =")
        end = source.find("\nexport const ", start + 1)
        return source[start:end if end >= 0 else None]

    def setUp(self):
        api_path = os.path.join(_REPO, "frontend", "src", "utils", "api.ts")
        with open(api_path, encoding="utf-8") as handle:
            self.source = handle.read()

    def test_txt2vidparams_declares_every_acceleration_field(self):
        match = re.search(
            r"export interface Txt2VidParams \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        for field in _ACCEL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(f"{field}?:", match.group(1))

    def test_txt2vid_json_body_sends_every_acceleration_field(self):
        fn = self._function_source(self.source, "generateTxt2Vid")
        for field in _ACCEL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(f"{field}: params.{field}", fn)

    def test_img2vid_and_ref2vid_senders_append_every_acceleration_field(self):
        for name in ("generateImg2Vid", "generateRef2Vid"):
            fn = self._function_source(self.source, name)
            for field in _ACCEL_FIELDS:
                with self.subTest(sender=name, field=field):
                    self.assertIn(f'formData.append("{field}"', fn)


# ---------------------------------------------------------------------------
# 4. The four panels: literal `videoParams`/`refParams` object-literal sites
#    carry every acceleration field, source-anchored like the block-swap
#    threading test (a revert of any one field fails this test).
# ---------------------------------------------------------------------------
class PanelLiteralSiteTest(unittest.TestCase):
    @staticmethod
    def _block(source: str, start_marker: str) -> str:
        start = source.index(start_marker)
        end = source.index("};", start)
        return source[start:end]

    def _read(self, *parts):
        path = os.path.join(_REPO, "frontend", "src", "components", "generation", *parts)
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_txt2img_panel_videoparams_carries_every_acceleration_field(self):
        source = self._read("Txt2ImgPanel.tsx")
        block = self._block(source, "const videoParams: Txt2VidParams = {")
        for field in _ACCEL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(f"{field}: params.{field}", block)

    def test_img2img_panel_videoparams_carries_every_acceleration_field(self):
        source = self._read("Img2ImgPanel.tsx")
        block = self._block(source, "const videoParams: Img2VidParams = {")
        for field in _ACCEL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(f"{field}: params.{field}", block)

    def test_img2img_panel_refparams_carries_every_acceleration_field(self):
        source = self._read("Img2ImgPanel.tsx")
        block = self._block(source, "const refParams: Ref2VidParams = {")
        for field in _ACCEL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(f"{field}: params.{field}", block)

    def test_inpaint_panel_video_params_carries_every_acceleration_field(self):
        """Regression guard: InpaintVideoParams already carried these before
        this task (only the Spectrum UI control was missing)."""
        source = self._read("InpaintPanel.tsx")
        block = self._block(source, "const videoParams: InpaintVideoParams = {")
        for field in _ACCEL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(f"{field}: params.{field}", block)

    def test_outpaint_panel_video_params_carries_every_acceleration_field(self):
        """Regression guard: OutpaintVideoParams already carried these before
        this task (both controls already existed)."""
        source = self._read("OutpaintPanel.tsx")
        block = self._block(source, "const videoParams: OutpaintVideoParams = {")
        for field in _ACCEL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(f"{field}: params.{field}", block)


# ---------------------------------------------------------------------------
# 5. Every video-capable panel renders the shared VideoAccelerationControls
#    component (idPrefix + values/onChange/support flags), not a hand-rolled
#    copy -- four near-identical copies of this exact block is what produced
#    the original drift (Txt2Img/Img2Img missing FBCache+Spectrum entirely,
#    Inpaint missing Spectrum).
# ---------------------------------------------------------------------------
class PanelControlPresenceTest(unittest.TestCase):
    def _read(self, name):
        path = os.path.join(
            _REPO, "frontend", "src", "components", "generation", name)
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_every_panel_imports_the_shared_component(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx", "InpaintPanel.tsx", "OutpaintPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn(
                    'import VideoAccelerationControls from "../common/VideoAccelerationControls";',
                    source,
                )

    def test_every_panel_renders_the_shared_component_with_its_own_id_prefix(self):
        for panel, id_prefix in (
            ("Txt2ImgPanel.tsx", "txt2vid"),
            ("Img2ImgPanel.tsx", "img2vid"),
            ("InpaintPanel.tsx", "inpaint_vid"),
            ("OutpaintPanel.tsx", "outpaint_vid"),
        ):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn("<VideoAccelerationControls", source)
                self.assertIn(f'idPrefix="{id_prefix}"', source)

    def test_no_panel_hand_rolls_a_second_fbcache_or_spectrum_checkbox_in_video_mode(self):
        """The pre-fix defect: each panel wrote its own checkbox markup, which
        is what let Txt2Img/Img2Img/Inpaint diverge from Outpaint in the first
        place. None of the four should define the video-mode ids directly any
        more -- they must come from inside the shared component."""
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx", "InpaintPanel.tsx", "OutpaintPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertNotIn('id="outpaint_vid_spectrum_enable"', source)
                self.assertNotIn('id="outpaint_vid_fbcache_enable"', source)
                self.assertNotIn('id="inpaint_vid_fbcache_enable"', source)


# ---------------------------------------------------------------------------
# 6. VideoAccelerationControls itself: the mutual-exclusion rules mirror the
#    backend exactly (core/pipeline_backends/ltx2.py's
#    _ltx2_build_fbcache/_ltx2_build_spectrum and
#    core/models/minimax_h3_block_loop_wrapper.py's attach_fbcache, both
#    read-only references -- this task does not edit either).
# ---------------------------------------------------------------------------
class SharedComponentMutualExclusionTest(unittest.TestCase):
    def setUp(self):
        path = os.path.join(
            _REPO, "frontend", "src", "components", "common", "VideoAccelerationControls.tsx")
        with open(path, encoding="utf-8") as handle:
            self.source = handle.read()

    def test_spectrum_is_disabled_while_block_swap_is_on(self):
        self.assertIn("const spectrumDisabled = blockSwapOn;", self.source)
        self.assertIn("disabled={spectrumDisabled}", self.source)

    def test_fbcache_is_disabled_while_block_swap_or_spectrum_is_on(self):
        # Assert the two disjuncts, not the whole line. Pinning the line meant a
        # caller adding a THIRD reason to disable FBCache (a spatial mask
        # timeline, which the backend also refuses to combine with it) broke a
        # test whose subject -- that block swap and Spectrum each disable it --
        # was still true. A line-shaped assertion fails on every addition and
        # teaches whoever hits it to update the string without reading it.
        line = next(
            (ln for ln in self.source.splitlines() if "const fbcacheDisabled" in ln), None)
        self.assertIsNotNone(line, "fbcacheDisabled is no longer computed")
        for disjunct in ("blockSwapOn", "spectrumOn"):
            self.assertIn(disjunct, line, line)
        self.assertIn("disabled={fbcacheDisabled}", self.source)

    def test_turning_block_swap_on_clears_fbcache_and_spectrum(self):
        self.assertIn("fbcache_enable: false,", self.source)
        self.assertIn("spectrum_enable: false,", self.source)

    def test_turning_spectrum_on_clears_fbcache_to_match_backend_precedence(self):
        self.assertIn(
            "onChange({ spectrum_enable: true, fbcache_enable: false });", self.source)


class BackendMutualExclusionReferenceTest(unittest.TestCase):
    """Read-only assertions against the backend modules this component's
    behaviour is documented to mirror -- catches the component's comment (and
    this task's premise) going stale if that backend logic ever changes.
    Neither file is edited by this task."""

    def test_minimax_h3_wrapper_refuses_fbcache_with_block_swap(self):
        path = os.path.join(
            _BACKEND, "core", "models", "minimax_h3_block_loop_wrapper.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("MiniMax-H3 FBCache cannot run with Block Swap.", source)

    def test_ltx2_disables_fbcache_and_spectrum_under_block_swap(self):
        path = os.path.join(_BACKEND, "core", "pipeline_backends", "ltx2.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("if block_swap_on:", source)

    def test_ltx2_spectrum_takes_precedence_over_fbcache(self):
        path = os.path.join(_BACKEND, "core", "pipeline_backends", "ltx2.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn('params.get("spectrum_enable", False)', source)


# ---------------------------------------------------------------------------
# 7. frontend/src/utils/videoChain.ts: a chain continuation segment carries
#    the SAME FBCache/Spectrum settings as the segment-1 request that started
#    the chain, exactly like `blocks_to_swap`. Before this change these fields
#    were deliberately NOT part of ChainContinuationBase (there was no
#    video-mode source to carry); that comment is now stale and must not
#    remain claiming a limitation that no longer exists.
# ---------------------------------------------------------------------------
class VideoChainAccelerationTest(unittest.TestCase):
    def setUp(self):
        path = os.path.join(_REPO, "frontend", "src", "utils", "videoChain.ts")
        with open(path, encoding="utf-8") as handle:
            self.source = handle.read()

    def test_chain_continuation_base_declares_every_acceleration_field(self):
        match = re.search(
            r"export interface ChainContinuationBase \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        for field in _ACCEL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(f"{field}?:", match.group(1))

    def test_build_chain_continuation_params_forwards_every_acceleration_field(self):
        start = self.source.index("export function buildChainContinuationParams")
        end = self.source.index("\n}", self.source.index("return {", start))
        block = self.source[start:end]
        for field in _ACCEL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(f"{field}: base.{field},", block)

    def test_no_longer_claims_segment_1_has_no_acceleration_source(self):
        """The pre-fix comment explaining the omission must not survive the
        fields it was explaining being added."""
        self.assertNotIn(
            "segment 1 of\n// a video chain has no acceleration setting of its own to replay",
            self.source,
        )
        self.assertNotIn("no video-mode source to carry", self.source)


if __name__ == "__main__":
    unittest.main()
