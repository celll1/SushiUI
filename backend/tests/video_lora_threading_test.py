"""End-to-end `loras` threading through the VIDEO generation path.

Companion to the image-side plumbing tests (`attention_type_validation_test.py`,
`optimizer_option_threading_test.py`): source-anchored so a future edit that
drops a site fails loudly instead of silently regressing to `lora_names=None`
or a dropped `params["loras"]`.

Backend application already existed for MiniMax-H3
(`core.models.minimax_h3.minimax_h3_lora` + `MiniMaxH3Mixin._load_lora_minimax_h3`,
hooked into `_generate_minimax_h3`, which every one of that architecture's five
video entry points routes through). This test proves the KEY actually reaches
that code and the gallery row, on all five video routes:

    POST /generate/txt2vid
    POST /generate/img2vid
    POST /generate/ref2vid
    POST /generate/outpaint/video
    POST /generate/inpaint/video

Each check below is verified by reasoning through a revert: if the site named
in the assertion message were removed, the assertion fails.
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

import inspect  # noqa: E402

import yaml  # noqa: E402

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
# 1. param_defaults.py: the SSOT carries "loras" for every video default map.
# ---------------------------------------------------------------------------
class ParamDefaultsTest(unittest.TestCase):
    def test_every_video_default_map_declares_loras(self):
        for name, defaults in (
            ("VIDEO_GEN_DEFAULTS", VIDEO_GEN_DEFAULTS),
            ("TXT2VID_DEFAULTS", TXT2VID_DEFAULTS),
            ("IMG2VID_DEFAULTS", IMG2VID_DEFAULTS),
            ("REF2VID_DEFAULTS", REF2VID_DEFAULTS),
            ("OUTPAINT_VIDEO_DEFAULTS", OUTPAINT_VIDEO_DEFAULTS),
            ("INPAINT_VIDEO_DEFAULTS", INPAINT_VIDEO_DEFAULTS),
        ):
            with self.subTest(defaults=name):
                self.assertIn("loras", defaults, f"{name} is missing a 'loras' default")
                self.assertEqual(defaults["loras"], [])


# ---------------------------------------------------------------------------
# 2. Pydantic model: Txt2VidRequest (the one JSON video route) declares the
#    field.
# ---------------------------------------------------------------------------
class Txt2VidRequestTest(unittest.TestCase):
    def test_txt2vid_request_has_a_loras_field(self):
        self.assertIn("loras", routes.Txt2VidRequest.model_fields)

    def test_loras_round_trips_through_the_request_model(self):
        req = routes.Txt2VidRequest(
            prompt="x", loras=[{"path": "a.safetensors", "strength": 0.8}])
        self.assertEqual(req.dict()["loras"][0]["path"], "a.safetensors")
        self.assertEqual(req.dict()["loras"][0]["strength"], 0.8)

    def test_omitted_loras_defaults_to_empty_list(self):
        req = routes.Txt2VidRequest(prompt="x")
        self.assertEqual(req.dict()["loras"], [])


# ---------------------------------------------------------------------------
# 3. Every video route: accepts `loras`, puts it into `params`, and does NOT
#    hardcode `lora_names=None` on the gallery row.
# ---------------------------------------------------------------------------
_VIDEO_ROUTE_FUNCS = {
    "generate_txt2vid": "txt2vid",
    "generate_img2vid": "img2vid",
    "generate_ref2vid": "ref2vid",
    "generate_outpaint_video": "outpaint_vid",
    "generate_inpaint_video": "inpaint_vid",
}


class VideoRouteLoraPlumbingTest(unittest.TestCase):
    def test_every_video_route_function_exists(self):
        for func_name in _VIDEO_ROUTE_FUNCS:
            self.assertTrue(hasattr(routes, func_name), f"routes.{func_name} not found")

    def test_every_video_route_accepts_loras(self):
        """Pydantic body (txt2vid) or a `loras` Form parameter (the four
        multipart routes) -- either way the parameter must be reachable from
        outside the function."""
        # txt2vid: JSON body, the field lives on the Pydantic model, not on
        # the endpoint's own signature.
        sig = inspect.signature(routes.generate_txt2vid)
        request_param = sig.parameters["request"]
        self.assertIs(request_param.annotation, routes.Txt2VidRequest)
        self.assertIn("loras", routes.Txt2VidRequest.model_fields)

        # The four multipart routes: `loras` must be a real Form parameter.
        for func_name in (
            "generate_img2vid", "generate_ref2vid",
            "generate_outpaint_video", "generate_inpaint_video",
        ):
            with self.subTest(route=func_name):
                func = getattr(routes, func_name)
                params = inspect.signature(func).parameters
                self.assertIn(
                    "loras", params,
                    f"{func_name} has no 'loras' Form parameter -- a client's LoRA "
                    f"selection can never reach this route at all",
                )

    def test_every_video_route_writes_loras_into_the_params_dict(self):
        """`params["loras"] = ...` (txt2vid: implicit via `request.dict()`,
        which already contains every declared field; the multipart routes:
        an explicit assignment after parsing the JSON string)."""
        # txt2vid: request.dict() includes every Pydantic field automatically,
        # which is what test_txt2vid_request_has_a_loras_field already proves;
        # confirm the route body actually calls `request.dict()` to build params.
        source = inspect.getsource(routes.generate_txt2vid)
        self.assertIn("params = request.dict()", source)

        for func_name in (
            "generate_img2vid", "generate_ref2vid",
            "generate_outpaint_video", "generate_inpaint_video",
        ):
            with self.subTest(route=func_name):
                source = inspect.getsource(getattr(routes, func_name))
                self.assertRegex(
                    source, r'params\["loras"\]\s*=',
                    f"{func_name} accepts a 'loras' Form field but never assigns "
                    f"it into params[\"loras\"] -- the value never reaches "
                    f"pipeline_manager",
                )

    def test_no_video_route_hardcodes_lora_names_none(self):
        """The pre-fix defect: every video gallery row recorded
        `lora_names=None` unconditionally, even when `loras` was non-empty."""
        for func_name, generation_type in _VIDEO_ROUTE_FUNCS.items():
            with self.subTest(route=func_name):
                source = inspect.getsource(getattr(routes, func_name))
                self.assertNotIn(
                    "lora_names=None", source,
                    f"{func_name} still hardcodes lora_names=None on the gallery row",
                )
                self.assertIn(
                    'lora_names=extract_lora_names(params.get("loras") or [])',
                    source,
                    f"{func_name} does not derive lora_names from params['loras']",
                )
                self.assertIn(f'generation_type="{generation_type}"', source)


# ---------------------------------------------------------------------------
# 4. arch_capabilities.py: LTX-2.3 (no video LoRA loader) is warned when
#    `loras` is non-empty; MiniMax-H3 (has the loader) is not gated.
# ---------------------------------------------------------------------------
class ArchCapabilitiesLoraTest(unittest.TestCase):
    def test_lora_feature_is_declared(self):
        from api.arch_capabilities import FEATURE_LABELS, FEATURE_PARAMS

        self.assertEqual(FEATURE_PARAMS.get("lora"), ["loras"])
        self.assertIn("lora", FEATURE_LABELS)

    def test_ltx2_is_unsupported_and_minimax_h3_is_not(self):
        from api.arch_capabilities import ARCH_UNSUPPORTED

        self.assertIn("lora", ARCH_UNSUPPORTED.get("ltx2", {}))
        self.assertNotIn("lora", ARCH_UNSUPPORTED.get("minimax_h3", {}))

    def test_non_empty_loras_warns_only_on_ltx2(self):
        from api.arch_capabilities import check_arch_capabilities

        params = {"loras": [{"path": "x.safetensors", "strength": 1.0}]}
        ltx2_warnings = check_arch_capabilities(params, "ltx2", defaults=TXT2VID_DEFAULTS)
        self.assertTrue(
            any("loras" in w["message"] for w in ltx2_warnings),
            "a non-empty loras list on ltx2 must warn",
        )
        h3_warnings = check_arch_capabilities(params, "minimax_h3", defaults=TXT2VID_DEFAULTS)
        self.assertFalse(
            any("loras" in w["message"] for w in h3_warnings),
            "minimax_h3 has a real LoRA loader and must not warn",
        )

    def test_empty_loras_never_warns(self):
        from api.arch_capabilities import check_arch_capabilities

        params = {"loras": []}
        warnings = check_arch_capabilities(params, "ltx2", defaults=TXT2VID_DEFAULTS)
        self.assertFalse(any("loras" in w["message"] for w in warnings))


# ---------------------------------------------------------------------------
# 5. openapi.yaml: every video request schema documents `loras`.
# ---------------------------------------------------------------------------
class OpenApiVideoLoraTest(unittest.TestCase):
    def test_every_video_request_schema_has_loras(self):
        spec_path = os.path.join(_REPO, "openapi.yaml")
        with open(spec_path, encoding="utf-8") as handle:
            spec = yaml.safe_load(handle)
        schemas = spec["components"]["schemas"]

        # Txt2VidRequest declares it directly; Img2VidRequest/Ref2VidRequest
        # inherit it via `allOf: [$ref Txt2VidRequest, ...]`.
        self.assertIn("loras", schemas["Txt2VidRequest"]["properties"])

        for name in ("Img2VidRequest", "Ref2VidRequest"):
            with self.subTest(schema=name):
                all_of = schemas[name]["allOf"]
                self.assertEqual(all_of[0]["$ref"], "#/components/schemas/Txt2VidRequest")

        # OutpaintVideoRequest / InpaintVideoRequest are standalone objects and
        # must declare the field themselves.
        for name in ("OutpaintVideoRequest", "InpaintVideoRequest"):
            with self.subTest(schema=name):
                self.assertIn("loras", schemas[name]["properties"])

    def test_no_duplicate_top_level_schema_keys(self):
        """YAML last-key-wins masks a duplicate path/schema block silently;
        this is the parity-maintenance gotcha this repo has been bitten by."""
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
# 6. frontend/src/utils/api.ts: every video sender appends/serializes `loras`.
# ---------------------------------------------------------------------------
class FrontendVideoLoraSenderTest(unittest.TestCase):
    @staticmethod
    def _function_source(source: str, name: str) -> str:
        start = source.index(f"export const {name} =")
        end = source.find("\nexport const ", start + 1)
        return source[start:end if end >= 0 else None]

    def setUp(self):
        api_path = os.path.join(_REPO, "frontend", "src", "utils", "api.ts")
        with open(api_path, encoding="utf-8") as handle:
            self.source = handle.read()

    def test_txt2vid_json_body_includes_loras(self):
        fn = self._function_source(self.source, "generateTxt2Vid")
        self.assertRegex(fn, r"loras:\s*params\.loras\s*\|\|\s*\[\]")

    def test_multipart_video_senders_append_loras(self):
        for name in (
            "generateImg2Vid", "generateRef2Vid",
            "generateOutpaintVideo", "generateInpaintVideo",
        ):
            with self.subTest(sender=name):
                fn = self._function_source(self.source, name)
                self.assertIn('formData.append("loras", JSON.stringify(params.loras || []));', fn)

    def test_every_video_param_interface_declares_loras(self):
        for name in ("Txt2VidParams", "OutpaintVideoParams", "InpaintVideoParams"):
            with self.subTest(interface=name):
                match = re.search(
                    rf"export interface {name} \{{(.*?)\n\}}", self.source, re.DOTALL)
                self.assertIsNotNone(match, f"interface {name} not found")
                self.assertIn("loras?: LoRAConfig[];", match.group(1))


# ---------------------------------------------------------------------------
# 7. The six panel literal sites, source-anchored (a revert of any one of
#    these fails this test).
# ---------------------------------------------------------------------------
class PanelLiteralLoraSiteTest(unittest.TestCase):
    @staticmethod
    def _block(source: str, start_marker: str) -> str:
        """The `{ ... };` object literal opened by `start_marker`. Every
        literal touched here is a flat object (no nested `{`), so the first
        `};` after the marker closes it."""
        start = source.index(start_marker)
        end = source.index("};", start)
        return source[start:end]

    def _read(self, *parts):
        path = os.path.join(_REPO, "frontend", "src", "components", "generation", *parts)
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_txt2img_panel_txt2vid_and_ref2vid_carry_loras(self):
        """One literal (`videoParams`) covers both sites: `fullVideoParams`
        (the ref2vid shape) is built as `{ ...videoParams, ... }`, so adding
        `loras` to `videoParams` alone threads it into both."""
        source = self._read("Txt2ImgPanel.tsx")
        block = self._block(source, "const videoParams: Txt2VidParams = {")
        self.assertIn("loras: params.loras", block)
        # Confirm the spread really does carry it forward into fullVideoParams.
        fvp_start = source.index("const fullVideoParams: Txt2VidParams = isRef2VaRequest")
        fvp_block = source[fvp_start:source.index(": videoParams;", fvp_start)]
        self.assertIn("...videoParams", fvp_block)

    def test_img2img_panel_refparams_carries_loras(self):
        source = self._read("Img2ImgPanel.tsx")
        block = self._block(source, "const refParams: Ref2VidParams = {")
        self.assertIn("loras: params.loras", block)

    def test_img2img_panel_videoparams_carries_loras(self):
        source = self._read("Img2ImgPanel.tsx")
        block = self._block(source, "const videoParams: Img2VidParams = {")
        self.assertIn("loras: params.loras", block)

    def test_inpaint_panel_video_params_carries_loras(self):
        source = self._read("InpaintPanel.tsx")
        block = self._block(source, "const videoParams: InpaintVideoParams = {")
        self.assertIn("loras: params.loras", block)

    def test_outpaint_panel_video_params_carries_loras(self):
        source = self._read("OutpaintPanel.tsx")
        block = self._block(source, "const videoParams: OutpaintVideoParams = {")
        self.assertIn("loras: params.loras", block)


# ---------------------------------------------------------------------------
# 8. videoChain.ts: a chain continuation segment carries the same LoRAs as
#    segment 1.
# ---------------------------------------------------------------------------
class VideoChainLoraTest(unittest.TestCase):
    def setUp(self):
        path = os.path.join(_REPO, "frontend", "src", "utils", "videoChain.ts")
        with open(path, encoding="utf-8") as handle:
            self.source = handle.read()

    def test_chain_continuation_base_declares_loras(self):
        match = re.search(
            r"export interface ChainContinuationBase \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("loras?: LoRAConfig[];", match.group(1))

    def test_build_chain_continuation_params_forwards_loras(self):
        start = self.source.index("export function buildChainContinuationParams")
        end = self.source.index("\n}", self.source.index("return {", start))
        block = self.source[start:end]
        self.assertIn("loras: base.loras,", block)


if __name__ == "__main__":
    unittest.main()
