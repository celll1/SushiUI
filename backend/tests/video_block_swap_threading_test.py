"""End-to-end `blocks_to_swap` threading through the VIDEO generation path,
plus the opt-in-default plumbing that backs it.

Companion to `video_lora_threading_test.py` (same source-anchored style): a
future edit that drops a site should fail loudly instead of silently
regressing to block swap being unreachable from a panel, or reverting to a
hardcoded magic number instead of the `param_defaults.py` SSOT value.

Context (see AGENTS task): video block swap must be OPT-IN (request default
`blocks_to_swap: 0`, unlike the image routes' `enable_block_swap` gate) but,
once a user turns it on, default to a value that actually does something.
Before this change, Inpaint/Outpaint hardcoded `10` in the panel and
Txt2Img/Img2Img had no control at all.
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
    INPAINT_VIDEO_DEFAULTS,
    OUTPAINT_VIDEO_DEFAULTS,
    REF2VID_DEFAULTS,
    TXT2VID_DEFAULTS,
    VIDEO_GEN_DEFAULTS,
)


# ---------------------------------------------------------------------------
# 1. param_defaults.py: the SSOT declares both the OFF request default and the
#    ON-enabled-default (a single UI-only value every panel reads instead of
#    repeating a literal).
# ---------------------------------------------------------------------------
class ParamDefaultsTest(unittest.TestCase):
    def test_every_video_default_map_defaults_blocks_to_swap_off(self):
        for name, defaults in (
            ("VIDEO_GEN_DEFAULTS", VIDEO_GEN_DEFAULTS),
            ("TXT2VID_DEFAULTS", TXT2VID_DEFAULTS),
            ("IMG2VID_DEFAULTS", IMG2VID_DEFAULTS),
            ("REF2VID_DEFAULTS", REF2VID_DEFAULTS),
            ("OUTPAINT_VIDEO_DEFAULTS", OUTPAINT_VIDEO_DEFAULTS),
            ("INPAINT_VIDEO_DEFAULTS", INPAINT_VIDEO_DEFAULTS),
        ):
            with self.subTest(defaults=name):
                self.assertIn("blocks_to_swap", defaults, f"{name} is missing 'blocks_to_swap'")
                self.assertEqual(
                    defaults["blocks_to_swap"], 0,
                    f"{name}['blocks_to_swap'] must default to 0 (opt-in) -- video block "
                    f"swap must not be forced on",
                )

    def test_every_video_default_map_carries_the_enabled_default(self):
        for name, defaults in (
            ("VIDEO_GEN_DEFAULTS", VIDEO_GEN_DEFAULTS),
            ("TXT2VID_DEFAULTS", TXT2VID_DEFAULTS),
            ("IMG2VID_DEFAULTS", IMG2VID_DEFAULTS),
            ("REF2VID_DEFAULTS", REF2VID_DEFAULTS),
            ("OUTPAINT_VIDEO_DEFAULTS", OUTPAINT_VIDEO_DEFAULTS),
            ("INPAINT_VIDEO_DEFAULTS", INPAINT_VIDEO_DEFAULTS),
        ):
            with self.subTest(defaults=name):
                self.assertIn(
                    "blocks_to_swap_enabled_default", defaults,
                    f"{name} is missing the UI-only 'blocks_to_swap_enabled_default'",
                )
                # A real, non-zero value: the whole point of this key is to
                # not be 0 (0 would just be "off", which is the other key).
                self.assertGreater(defaults["blocks_to_swap_enabled_default"], 0)
                # Below the max-value FF-chunking path (49 on MiniMax-H3's 50
                # blocks): the enabled-default is deliberately NOT the max.
                self.assertLess(defaults["blocks_to_swap_enabled_default"], 49)

    def test_blocks_to_swap_enabled_default_is_not_a_real_request_field(self):
        """It must never reach a route's Form/Pydantic signature -- it is a
        schema-response-only value the frontend reads to seed its checkbox."""
        for func_name in (
            "generate_img2vid", "generate_ref2vid",
            "generate_outpaint_video", "generate_inpaint_video",
        ):
            with self.subTest(route=func_name):
                params = inspect.signature(getattr(routes, func_name)).parameters
                self.assertNotIn("blocks_to_swap_enabled_default", params)
        self.assertNotIn("blocks_to_swap_enabled_default", routes.Txt2VidRequest.model_fields)


# ---------------------------------------------------------------------------
# 2. Every video route already accepted (and still accepts) `blocks_to_swap`
#    -- this task added no backend field, only frontend plumbing, so this
#    guards against a future regression removing it.
# ---------------------------------------------------------------------------
_VIDEO_ROUTE_FUNCS = {
    "generate_txt2vid": "txt2vid",
    "generate_img2vid": "img2vid",
    "generate_ref2vid": "ref2vid",
    "generate_outpaint_video": "outpaint_vid",
    "generate_inpaint_video": "inpaint_vid",
}


class VideoRouteBlocksToSwapPlumbingTest(unittest.TestCase):
    def test_txt2vid_request_has_a_blocks_to_swap_field(self):
        self.assertIn("blocks_to_swap", routes.Txt2VidRequest.model_fields)

    def test_omitted_blocks_to_swap_defaults_to_zero(self):
        req = routes.Txt2VidRequest(prompt="x")
        self.assertEqual(req.dict()["blocks_to_swap"], 0)

    def test_the_four_multipart_routes_accept_blocks_to_swap(self):
        for func_name in (
            "generate_img2vid", "generate_ref2vid",
            "generate_outpaint_video", "generate_inpaint_video",
        ):
            with self.subTest(route=func_name):
                params = inspect.signature(getattr(routes, func_name)).parameters
                self.assertIn(
                    "blocks_to_swap", params,
                    f"{func_name} has no 'blocks_to_swap' Form parameter -- a client's "
                    f"choice can never reach this route at all",
                )

    def test_openapi_documents_blocks_to_swap_on_every_video_request_schema(self):
        import yaml

        spec_path = os.path.join(_REPO, "openapi.yaml")
        with open(spec_path, encoding="utf-8") as handle:
            spec = yaml.safe_load(handle)
        schemas = spec["components"]["schemas"]

        self.assertIn("blocks_to_swap", schemas["Txt2VidRequest"]["properties"])
        for name in ("Img2VidRequest", "Ref2VidRequest"):
            with self.subTest(schema=name):
                self.assertEqual(
                    schemas[name]["allOf"][0]["$ref"], "#/components/schemas/Txt2VidRequest")
        for name in ("OutpaintVideoRequest", "InpaintVideoRequest"):
            with self.subTest(schema=name):
                self.assertIn("blocks_to_swap", schemas[name]["properties"])

    def test_no_duplicate_top_level_schema_keys(self):
        """YAML last-key-wins masks a duplicate path/schema block silently."""
        import yaml

        spec_path = os.path.join(_REPO, "openapi.yaml")

        class _UniqueKeyLoader(yaml.SafeLoader):
            pass

        def _construct_mapping(loader, node, deep=False):
            mapping = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in mapping:
                    raise AssertionError(
                        f"Duplicate key {key!r} at line {key_node.start_mark.line + 1}")
                mapping[key] = loader.construct_object(value_node, deep=deep)
            return mapping

        _UniqueKeyLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)
        with open(spec_path, encoding="utf-8") as handle:
            yaml.load(handle, Loader=_UniqueKeyLoader)  # raises on a duplicate


# ---------------------------------------------------------------------------
# 3. /schema/generation-defaults actually returns the enabled-default under
#    every video mode key.
# ---------------------------------------------------------------------------
class SchemaEndpointTest(unittest.TestCase):
    def test_get_generation_defaults_source_returns_every_video_mode(self):
        source = inspect.getsource(routes.get_generation_defaults)
        for key in ("txt2vid", "img2vid", "ref2vid", "outpaint_vid", "inpaint_vid"):
            with self.subTest(key=key):
                self.assertIn(f'"{key}":', source)


# ---------------------------------------------------------------------------
# 4. frontend/src/utils/api.ts: type interfaces + senders + the shared max
#    constant.
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

    def test_txt2vidparams_declares_blocks_to_swap(self):
        match = re.search(
            r"export interface Txt2VidParams \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("blocks_to_swap?: number;", match.group(1))

    def test_generationparams_declares_video_blocks_to_swap(self):
        match = re.search(
            r"export interface GenerationParams \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("video_blocks_to_swap?: number;", match.group(1))

    def test_txt2vid_json_body_includes_blocks_to_swap(self):
        fn = self._function_source(self.source, "generateTxt2Vid")
        self.assertIn("blocks_to_swap: params.blocks_to_swap ?? 0,", fn)

    def test_img2vid_and_ref2vid_senders_append_blocks_to_swap(self):
        for name in ("generateImg2Vid", "generateRef2Vid"):
            with self.subTest(sender=name):
                fn = self._function_source(self.source, name)
                self.assertIn(
                    'formData.append("blocks_to_swap", String(params.blocks_to_swap ?? 0));', fn)

    def test_video_block_swap_max_constant_is_exported(self):
        self.assertIn("export const VIDEO_BLOCK_SWAP_MAX", self.source)


# ---------------------------------------------------------------------------
# 5. The four panels: literal `blocks_to_swap: params.video_blocks_to_swap`
#    sites, source-anchored (a revert of any one of these fails this test).
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

    def test_txt2img_panel_videoparams_carries_blocks_to_swap(self):
        """One literal (`videoParams`) covers both txt2vid and ref2vid: see
        video_lora_threading_test's identical reasoning for `loras`."""
        source = self._read("Txt2ImgPanel.tsx")
        block = self._block(source, "const videoParams: Txt2VidParams = {")
        self.assertIn("blocks_to_swap: params.video_blocks_to_swap", block)

    def test_img2img_panel_videoparams_carries_blocks_to_swap(self):
        source = self._read("Img2ImgPanel.tsx")
        block = self._block(source, "const videoParams: Img2VidParams = {")
        self.assertIn("blocks_to_swap: params.video_blocks_to_swap", block)

    def test_img2img_panel_refparams_carries_blocks_to_swap(self):
        source = self._read("Img2ImgPanel.tsx")
        block = self._block(source, "const refParams: Ref2VidParams = {")
        self.assertIn("blocks_to_swap: params.video_blocks_to_swap", block)

    def test_inpaint_panel_video_params_carries_blocks_to_swap(self):
        source = self._read("InpaintPanel.tsx")
        block = self._block(source, "const videoParams: InpaintVideoParams = {")
        self.assertIn("blocks_to_swap: params.video_blocks_to_swap", block)

    def test_outpaint_panel_video_params_carries_blocks_to_swap(self):
        source = self._read("OutpaintPanel.tsx")
        block = self._block(source, "const videoParams: OutpaintVideoParams = {")
        self.assertIn("blocks_to_swap: params.video_blocks_to_swap", block)


# ---------------------------------------------------------------------------
# 6. Every panel exposes the control (checkbox + NumberInput), reads the
#    enabled-default from generationDefaults rather than a hardcoded literal,
#    and bounds the field with the shared VIDEO_BLOCK_SWAP_MAX constant.
# ---------------------------------------------------------------------------
class PanelControlPresenceTest(unittest.TestCase):
    def _read(self, name):
        path = os.path.join(
            _REPO, "frontend", "src", "components", "generation", name)
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_every_panel_has_the_block_swap_checkbox(self):
        """The checkbox markup itself now lives in ONE place
        (common/VideoAccelerationControls.tsx, shared by all four panels --
        see video_acceleration_threading_test.py); each panel is checked here
        for RENDERING that shared component with its own resolved enabled
        default, rather than for a copy of the markup itself."""
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx", "InpaintPanel.tsx", "OutpaintPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn("<VideoAccelerationControls", source)
                self.assertIn("blocksToSwapEnabledDefault={videoBlocksToSwapEnabledDefault}", source)
        shared_path = os.path.join(
            _REPO, "frontend", "src", "components", "common", "VideoAccelerationControls.tsx")
        with open(shared_path, encoding="utf-8") as handle:
            shared_source = handle.read()
        self.assertIn(
            "video_blocks_to_swap: blocksToSwapEnabledDefault,", shared_source)

    def test_no_panel_hardcodes_the_enabled_default_as_a_bare_literal(self):
        """The pre-fix defect on Inpaint/Outpaint: `? 10 : 0`. Must not
        reappear on any of the four panels."""
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx", "InpaintPanel.tsx", "OutpaintPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertNotIn("video_blocks_to_swap: e.target.checked ? 10 : 0", source)

    def test_every_panel_derives_the_enabled_default_from_generation_defaults(self):
        for panel, key in (
            ("Txt2ImgPanel.tsx", "txt2vid"),
            ("Img2ImgPanel.tsx", "img2vid"),
            ("InpaintPanel.tsx", "inpaint_vid"),
            ("OutpaintPanel.tsx", "outpaint_vid"),
        ):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn("videoBlocksToSwapEnabledDefault =", source)
                self.assertIn(f"generationDefaults?.{key}", source)
                self.assertIn("blocks_to_swap_enabled_default", source)

    def test_every_panel_bounds_the_field_with_the_shared_max_constant(self):
        """The NumberInput itself now lives inside the shared component; each
        panel is checked for passing the shared constant into it as a prop."""
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx", "InpaintPanel.tsx", "OutpaintPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn("blockSwapMax={VIDEO_BLOCK_SWAP_MAX}", source)
                self.assertNotIn("max={48}", source)
        shared_path = os.path.join(
            _REPO, "frontend", "src", "components", "common", "VideoAccelerationControls.tsx")
        with open(shared_path, encoding="utf-8") as handle:
            shared_source = handle.read()
        self.assertIn("max={blockSwapMax}", shared_source)

    def test_every_panel_has_default_params_off_by_default(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx", "InpaintPanel.tsx", "OutpaintPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn("video_blocks_to_swap: 0,", source)


# ---------------------------------------------------------------------------
# 7. frontend/src/utils/videoChain.ts: a chain continuation segment (segment
#    2..N of an opt-in video-length chain) carries the SAME `blocks_to_swap`
#    as the segment-1 request that started the chain. Dropping this is a hard
#    failure, not a slowdown: a user who enabled block swap because their card
#    cannot hold the model resident gets an OOM on a later segment of what is,
#    to them, one clip -- not a fresh request they get to reconfigure.
# ---------------------------------------------------------------------------
class VideoChainBlocksToSwapTest(unittest.TestCase):
    def setUp(self):
        path = os.path.join(_REPO, "frontend", "src", "utils", "videoChain.ts")
        with open(path, encoding="utf-8") as handle:
            self.source = handle.read()

    def test_chain_continuation_base_declares_blocks_to_swap(self):
        match = re.search(
            r"export interface ChainContinuationBase \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("blocks_to_swap?: number;", match.group(1))

    def test_build_chain_continuation_params_forwards_blocks_to_swap(self):
        start = self.source.index("export function buildChainContinuationParams")
        end = self.source.index("\n}", self.source.index("return {", start))
        block = self.source[start:end]
        self.assertIn("blocks_to_swap: base.blocks_to_swap,", block)


if __name__ == "__main__":
    unittest.main()
