"""End-to-end `loras` threading through the VIDEO generation path.

Companion to the image-side plumbing tests (`attention_type_validation_test.py`,
`optimizer_option_threading_test.py`): source-anchored so a future edit that
drops a site fails loudly instead of silently regressing to `lora_names=None`
or a dropped `params["loras"]`.

Both video architectures apply the key in their own backend: MiniMax-H3 via
`core.models.minimax_h3.minimax_h3_lora` + `MiniMaxH3Mixin._load_lora_minimax_h3`
(hooked into `_generate_minimax_h3`, which all five of that architecture's video
entry points route through), and LTX-2.3 via `core.models.ltx2.ltx2_lora` +
`LTX2Mixin._load_lora_ltx2` (hooked into each of the three video entry points
LTX-2.3 has; ref2vid and temporal inpaint are MiniMax-H3-only mechanisms and
refuse for LTX-2.3 before any LoRA is consulted). This test proves the KEY
actually reaches that code and the gallery row, on all five video routes:

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
# 4. arch_capabilities.py: neither video architecture gates `loras` -- both
#    have a real generation-side loader, so a correct request must not be told
#    its LoRA was ignored.
# ---------------------------------------------------------------------------
class ArchCapabilitiesLoraTest(unittest.TestCase):
    def test_lora_feature_is_declared(self):
        from api.arch_capabilities import FEATURE_LABELS, FEATURE_PARAMS

        self.assertEqual(FEATURE_PARAMS.get("lora"), ["loras"])
        self.assertIn("lora", FEATURE_LABELS)

    def test_neither_video_arch_declares_lora_unsupported(self):
        from api.arch_capabilities import ARCH_UNSUPPORTED

        for arch in ("ltx2", "minimax_h3"):
            with self.subTest(arch=arch):
                self.assertNotIn("lora", ARCH_UNSUPPORTED.get(arch, {}))

    def test_non_empty_loras_warns_on_neither_video_arch(self):
        from api.arch_capabilities import check_arch_capabilities

        params = {"loras": [{"path": "x.safetensors", "strength": 1.0}]}
        for arch in ("ltx2", "minimax_h3"):
            with self.subTest(arch=arch):
                warnings = check_arch_capabilities(params, arch, defaults=TXT2VID_DEFAULTS)
                self.assertFalse(
                    any("loras" in w["message"] for w in warnings),
                    f"{arch} has a real LoRA loader and must not warn",
                )

    def test_empty_loras_never_warns(self):
        from api.arch_capabilities import check_arch_capabilities

        params = {"loras": []}
        warnings = check_arch_capabilities(params, "ltx2", defaults=TXT2VID_DEFAULTS)
        self.assertFalse(any("loras" in w["message"] for w in warnings))


# ---------------------------------------------------------------------------
# 4b. LTX-2.3 backend application: the mirror of the MiniMax-H3 hookup, across
#     the same five routes -- three of which LTX-2.3 serves and two of which it
#     refuses outright.
# ---------------------------------------------------------------------------
class Ltx2BackendLoraApplicationTest(unittest.TestCase):
    # (pipeline dispatcher, LTX-2.3 entry point) for every route LTX-2.3 serves.
    SERVED = (
        ("generate_txt2vid", "_generate_txt2vid_ltx2"),
        ("generate_img2vid", "_generate_img2vid_ltx2"),
        ("generate_vid_outpaint", "_generate_vidoutpaint_ltx2"),
    )
    # Routes with no LTX-2.3 mechanism at all: they must refuse, not silently
    # generate without the LoRA.
    REFUSED = ("generate_ref2vid", "generate_vid_inpaint")

    def setUp(self):
        from core.pipeline_backends.ltx2 import LTX2Mixin
        from core.pipeline import DiffusionPipelineManager

        self.mixin = LTX2Mixin
        self.manager = DiffusionPipelineManager

    def test_the_loader_and_unloader_exist(self):
        for name in ("_load_lora_ltx2", "_unload_lora_ltx2",
                     "_ltx2_sync_block_swap_after_lora"):
            with self.subTest(method=name):
                self.assertTrue(hasattr(self.mixin, name))

    def test_every_served_entry_point_applies_and_restores_the_lora(self):
        for dispatcher, entry_point in self.SERVED:
            with self.subTest(entry_point=entry_point):
                source = inspect.getsource(getattr(self.mixin, entry_point))
                self.assertIn(
                    'self._load_lora_ltx2(params.get("loras"))', source,
                    f"{entry_point} never applies params['loras'] -- the request's "
                    f"LoRA silently does nothing on this route",
                )
                finally_idx = source.rindex("finally:")
                self.assertIn(
                    "self._unload_lora_ltx2()", source[finally_idx:],
                    f"{entry_point} does not restore the wrapped Linears in a "
                    f"finally -- a failed generation would leak the adapters into "
                    f"the next one",
                )

    def test_every_served_route_dispatches_to_the_ltx2_entry_point(self):
        for dispatcher, entry_point in self.SERVED:
            with self.subTest(route=dispatcher):
                source = inspect.getsource(getattr(self.manager, dispatcher))
                self.assertIn("if self.is_ltx2_model:", source)
                self.assertIn(entry_point, source)

    def test_the_two_unserved_routes_refuse_rather_than_ignore(self):
        for dispatcher in self.REFUSED:
            with self.subTest(route=dispatcher):
                source = inspect.getsource(getattr(self.manager, dispatcher))
                self.assertNotIn(
                    "_ltx2(", source,
                    f"{dispatcher} appears to dispatch to an LTX-2.3 entry point; "
                    f"this test's five-route accounting is stale",
                )
                self.assertIn("raise ValidationError", source)

    def test_block_swap_caches_are_invalidated_on_both_wrap_and_unwrap(self):
        """M1: the offloader is persistent across generations and every cache it
        holds (H2D masters AND the standard swap's staging buffers) describes the
        pre-LoRA block tree."""
        from core.pipeline_backends import ltx2 as ltx2_mod

        caches = ("h2d_masters", "h2d_ring", "h2d_slot_futures",
                  "h2d_loaded_block", "staging_buffer_a", "staging_buffer_b",
                  "pinned_buffer", "_dtype_split_paths")

        class FakeOffloader:
            pass

        offloader = FakeOffloader()
        for attr in caches:
            setattr(offloader, attr, ["stale"])
        ltx2_mod._ltx2_invalidate_block_swap_caches(offloader)
        for attr in caches:
            with self.subTest(cache=attr):
                self.assertIsNone(
                    getattr(offloader, attr),
                    f"{attr} survives a LoRA wrap/unwrap and would then describe a "
                    f"block tree that no longer exists",
                )
        for name in ("_ltx2_sync_block_swap_after_lora", "_unload_lora_ltx2"):
            with self.subTest(method=name):
                self.assertIn(
                    "_ltx2_invalidate_block_swap_caches",
                    inspect.getsource(getattr(self.mixin, name)),
                )

    def test_the_offloader_reconciliation_is_not_h2d_only(self):
        """The standard swap needs it too: its staging buffers are sized from the
        first job list and zip-paired with every later one."""
        source = inspect.getsource(self.mixin._ltx2_sync_block_swap_after_lora)
        invalidate = source.index("_ltx2_invalidate_block_swap_caches(offloader)")
        self.assertNotIn(
            'h2d_only", False)', source[:invalidate],
            "the cache drop is gated on h2d_only again; the standard swap path "
            "then keeps staging buffers shaped for the wrapped tree",
        )

    def test_lora_bookkeeping_is_weakref_keyed(self):
        """S1: a reload can allocate the new transformer at the dead one's
        address, so id() is not an identity."""
        source = inspect.getsource(self.mixin._ltx2_lora_state)
        self.assertIn("weakref.ref(transformer)", source)
        self.assertNotIn("id(transformer)",
                         inspect.getsource(self.mixin._load_lora_ltx2))
        self.assertNotIn("id(transformer)",
                         inspect.getsource(self.mixin._unload_lora_ltx2))

    def test_state_is_reset_before_the_empty_config_exit(self):
        source = inspect.getsource(self.mixin._load_lora_ltx2)
        self.assertLess(
            source.index("self._ltx2_lora_state(transformer)"),
            source.index("if not lora_configs:"),
            "an evicted model's bookkeeping survives a request that installs no "
            "LoRA, and the next restore would splice it into the new transformer",
        )

    def test_warnings_carry_a_basename_and_the_shared_warning_codes(self):
        """M2/M3: warnings ride into the PNG metadata chunk and the response's
        warnings[], and their codes are the cross-architecture ones."""
        source = inspect.getsource(self.mixin._load_lora_ltx2)
        self.assertIn("os.path.basename(lora_path)", source)
        self.assertNotIn("{lora_path}", source)
        for code in ("lora_not_found", "lora_incompatible", "lora_partial"):
            with self.subTest(code=code):
                self.assertIn(f'"{code}"', source)
        # LTX-2.3 is on CompositeAdapterLayer: two LoRAs over one module sum, so
        # the refusal this used to require is gone rather than merely unreached.
        self.assertNotIn('"lora_stacking_unsupported"', source)
        for arch_prefixed in ("ltx2_lora_not_found", "ltx2_lora_incompatible",
                              "ltx2_lora_targets_unresolved"):
            with self.subTest(code=arch_prefixed):
                self.assertNotIn(arch_prefixed, source)


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
                    rf"export interface {name}[^{{\n]*\{{(.*?)\n\}}", self.source, re.DOTALL)
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
