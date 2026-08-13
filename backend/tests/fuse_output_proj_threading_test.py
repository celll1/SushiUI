"""End-to-end `fuse_output_proj` threading through the VIDEO generation path.

Companion to `video_acceleration_threading_test.py` / `video_block_swap_threading_test.py`
(same source-anchored style, one field instead of the FBCache/Spectrum set).
`fuse_output_proj` is MiniMax-H3-only, opt-in, and NOT bit-exact (see
`core.models.minimax_h3.adaln_chunking`'s "Head fusion" note) -- unlike
FBCache/Spectrum/Block Swap it has no image-mode namesake to collide with, so
it needs no `video_` prefix at the panel-state level (see `blocks_to_swap`
which does, hence `video_blocks_to_swap`).

Per-field assertions throughout (a single field here, but kept as a loop for
the same reason the acceleration suite is): a future edit that silently drops
this field from one of the five sites should fail at that site's own
assertion, not get lost inside a bigger diff.
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
from api.arch_capabilities import ARCH_UNSUPPORTED, FEATURE_PARAMS  # noqa: E402
from api.param_defaults import (  # noqa: E402
    IMG2VID_DEFAULTS,
    INPAINT_VIDEO_DEFAULTS,
    OUTPAINT_VIDEO_DEFAULTS,
    REF2VID_DEFAULTS,
    TXT2VID_DEFAULTS,
    VIDEO_GEN_DEFAULTS,
)

_FIELD = "fuse_output_proj"


class ParamDefaultsTest(unittest.TestCase):
    def test_every_video_default_map_has_the_field_and_it_defaults_off(self):
        for name, defaults in (
            ("VIDEO_GEN_DEFAULTS", VIDEO_GEN_DEFAULTS),
            ("TXT2VID_DEFAULTS", TXT2VID_DEFAULTS),
            ("IMG2VID_DEFAULTS", IMG2VID_DEFAULTS),
            ("REF2VID_DEFAULTS", REF2VID_DEFAULTS),
            ("OUTPAINT_VIDEO_DEFAULTS", OUTPAINT_VIDEO_DEFAULTS),
            ("INPAINT_VIDEO_DEFAULTS", INPAINT_VIDEO_DEFAULTS),
        ):
            with self.subTest(defaults=name):
                self.assertIn(_FIELD, defaults, f"{name} is missing '{_FIELD}'")
                self.assertFalse(defaults[_FIELD])


class ArchCapabilitiesTest(unittest.TestCase):
    """The feature is MiniMax-H3-only; LTX-2.3 must be declared unsupported so
    the frontend hides the control and a direct API call gets a warning."""

    def test_feature_is_registered(self):
        self.assertIn(_FIELD, FEATURE_PARAMS)
        self.assertEqual(FEATURE_PARAMS[_FIELD], [_FIELD])

    def test_ltx2_is_declared_unsupported(self):
        self.assertIn(_FIELD, ARCH_UNSUPPORTED.get("ltx2", {}))

    def test_minimax_h3_is_not_declared_unsupported(self):
        self.assertNotIn(_FIELD, ARCH_UNSUPPORTED.get("minimax_h3", {}))


class BackendRouteFieldsTest(unittest.TestCase):
    def test_txt2vid_request_declares_the_field(self):
        self.assertIn(_FIELD, routes.Txt2VidRequest.model_fields)

    def test_every_multipart_video_route_accepts_the_field(self):
        for func_name in (
            "generate_img2vid", "generate_ref2vid",
            "generate_outpaint_video", "generate_inpaint_video",
        ):
            with self.subTest(route=func_name):
                params = inspect.signature(getattr(routes, func_name)).parameters
                self.assertIn(
                    _FIELD, params,
                    f"{func_name} has no '{_FIELD}' Form parameter -- a client's "
                    f"choice can never reach this route at all",
                )

    def test_every_multipart_video_route_forwards_the_field_into_params(self):
        source_path = os.path.join(_BACKEND, "api", "routes.py")
        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()
        for func_name in (
            "generate_img2vid", "generate_ref2vid",
            "generate_outpaint_video", "generate_inpaint_video",
        ):
            with self.subTest(route=func_name):
                start = source.index(f"async def {func_name}(")
                end = source.index("\nasync def ", start + 1)
                block = source[start:end if end >= 0 else None]
                self.assertIn(f'"{_FIELD}": {_FIELD},', block)


class BackendPipelineTest(unittest.TestCase):
    """The staging function is the ONE place params["fuse_output_proj"]
    crosses into the transformer instance both call sites (stock forward and
    the block-loop wrapper) read from."""

    def test_ensure_swap_and_offload_sets_the_flag_on_the_transformer(self):
        path = os.path.join(_BACKEND, "core", "pipeline_backends", "minimax_h3.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn(
            f'transformer.{_FIELD} = bool(params.get("{_FIELD}", False))', source)

    def test_vendor_forward_and_wrapper_both_read_the_flag(self):
        vendor_path = os.path.join(
            _BACKEND, "core", "models", "minimax_h3", "vendor", "transformer_minimax_h3.py")
        with open(vendor_path, encoding="utf-8") as handle:
            vendor_source = handle.read()
        self.assertIn(f"if self.{_FIELD}:", vendor_source)

        wrapper_path = os.path.join(
            _BACKEND, "core", "models", "minimax_h3_block_loop_wrapper.py")
        with open(wrapper_path, encoding="utf-8") as handle:
            wrapper_source = handle.read()
        self.assertIn(f'if getattr(t, "{_FIELD}", False):', wrapper_source)

    def test_model_defaults_the_flag_off_in_init(self):
        path = os.path.join(
            _BACKEND, "core", "models", "minimax_h3", "vendor", "transformer_minimax_h3.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn(f"self.{_FIELD} = False", source)


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

    def test_txt2vidparams_declares_the_field(self):
        match = re.search(
            r"export interface Txt2VidParams[^{\n]*\{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn(f"{_FIELD}?:", match.group(1))

    def test_outpaint_and_inpaint_video_params_declare_the_field(self):
        for iface in ("OutpaintVideoParams", "InpaintVideoParams"):
            with self.subTest(interface=iface):
                match = re.search(
                    r"export interface %s[^{\n]*\{(.*?)\n\}" % iface, self.source, re.DOTALL)
                self.assertIsNotNone(match)
                self.assertIn(f"{_FIELD}?:", match.group(1))

    def test_txt2vid_json_body_sends_the_field(self):
        fn = self._function_source(self.source, "generateTxt2Vid")
        self.assertIn(f"{_FIELD}: params.{_FIELD}", fn)

    def test_every_multipart_video_sender_appends_the_field(self):
        for name in ("generateImg2Vid", "generateRef2Vid", "generateOutpaintVideo", "generateInpaintVideo"):
            with self.subTest(sender=name):
                fn = self._function_source(self.source, name)
                self.assertIn(f'formData.append("{_FIELD}"', fn)


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

    def test_txt2img_panel_videoparams_carries_the_field(self):
        source = self._read("Txt2ImgPanel.tsx")
        block = self._block(source, "const videoParams: Txt2VidParams = {")
        self.assertIn(f"{_FIELD}: params.{_FIELD}", block)

    def test_img2img_panel_videoparams_and_refparams_carry_the_field(self):
        source = self._read("Img2ImgPanel.tsx")
        for marker in (
            "const videoParams: Img2VidParams = {",
            "const refParams: Ref2VidParams = {",
        ):
            with self.subTest(marker=marker):
                block = self._block(source, marker)
                self.assertIn(f"{_FIELD}: params.{_FIELD}", block)

    def test_inpaint_panel_video_params_carries_the_field(self):
        source = self._read("InpaintPanel.tsx")
        block = self._block(source, "const videoParams: InpaintVideoParams = {")
        self.assertIn(f"{_FIELD}: params.{_FIELD}", block)

    def test_outpaint_panel_video_params_carries_the_field(self):
        source = self._read("OutpaintPanel.tsx")
        block = self._block(source, "const videoParams: OutpaintVideoParams = {")
        self.assertIn(f"{_FIELD}: params.{_FIELD}", block)

    def test_every_panel_default_params_defaults_the_field_off(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx", "InpaintPanel.tsx", "OutpaintPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn(f"{_FIELD}: false,", source)


class SharedComponentTest(unittest.TestCase):
    """The control lives in VideoAccelerationControls (shared by all four
    panels) rather than four hand-rolled copies -- see the acceleration
    suite's PanelControlPresenceTest for the same rule applied to
    FBCache/Spectrum."""

    def setUp(self):
        path = os.path.join(
            _REPO, "frontend", "src", "components", "common", "VideoAccelerationControls.tsx")
        with open(path, encoding="utf-8") as handle:
            self.source = handle.read()

    def test_values_interface_declares_the_field(self):
        match = re.search(
            r"export interface VideoAccelerationValues \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn(f"{_FIELD}?:", match.group(1))

    def test_component_exposes_a_capability_gated_checkbox(self):
        self.assertIn("supportsFuseOutputProj", self.source)
        self.assertIn(f'onChange({{ {_FIELD}: e.target.checked }})', self.source)

    def test_every_panel_passes_the_support_prop(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx", "InpaintPanel.tsx", "OutpaintPanel.tsx"):
            path = os.path.join(_REPO, "frontend", "src", "components", "generation", panel)
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            with self.subTest(panel=panel):
                self.assertIn(
                    'archSupportsFeature(archCapabilities, %s, "%s")' % (
                        "loadedArchType" if panel in ("InpaintPanel.tsx", "OutpaintPanel.tsx") else "loadedArch",
                        _FIELD,
                    ),
                    source,
                )
                self.assertIn("supportsFuseOutputProj={supportsFuseOutputProj}", source)


# ---------------------------------------------------------------------------
# Video-length chaining: a chain continuation segment must carry the SAME
# setting as the segment-1 request that started the chain, exactly like
# `blocks_to_swap` -- a user who enabled this for a low-VRAM segment 1 needs
# every continuation segment to run with it too.
# ---------------------------------------------------------------------------
class VideoChainTest(unittest.TestCase):
    def setUp(self):
        path = os.path.join(_REPO, "frontend", "src", "utils", "videoChain.ts")
        with open(path, encoding="utf-8") as handle:
            self.source = handle.read()

    def test_chain_continuation_base_declares_the_field(self):
        match = re.search(
            r"export interface ChainContinuationBase \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn(f"{_FIELD}?:", match.group(1))

    def test_build_chain_continuation_params_forwards_the_field(self):
        start = self.source.index("export function buildChainContinuationParams")
        end = self.source.index("\n}", self.source.index("return {", start))
        block = self.source[start:end]
        self.assertIn(f"{_FIELD}: base.{_FIELD},", block)


if __name__ == "__main__":
    unittest.main()
