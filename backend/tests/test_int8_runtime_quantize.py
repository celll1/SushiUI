import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.common import int8_runtime_quantize as runtime_int8


class RuntimeInt8PartialAuditTest(unittest.TestCase):
    def test_failed_replacement_is_not_recorded_as_converted(self):
        model = nn.Sequential(nn.Linear(8, 8, bias=False))
        row = {"name": "0", "chosen": "int8"}
        quantized = torch.zeros((8, 8), dtype=torch.int8)
        scale = torch.ones((8, 1), dtype=torch.float32)

        with patch.object(
            runtime_int8,
            "audit_and_quantize_int8",
            return_value=("int8", quantized, scale, row),
        ), patch.object(
            runtime_int8,
            "_filled_quantized_linear",
            side_effect=RuntimeError("replacement failed"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                runtime_int8.quantize_linears_in_place(
                    model,
                    arch="krea2",
                    min_align=1,
                    skip_below_work_gate=False,
                    work_device=torch.device("cpu"),
                )

        document = ctx.exception._int8_partial_document
        self.assertEqual(document["layers"], [])
        self.assertEqual(document["converted_before_failure"], 0)
        self.assertEqual(document["remaining"], 1)
        self.assertIsInstance(model[0], nn.Linear)


class _FakeManager:
    """Just the attributes the converter reads and latches."""


def _gate_clearing_linear():
    # Above the runtime min-work gate (k >= 2048, n >= 1024) so the ideogram4
    # policy, which filters below-gate layers, still selects it.
    return nn.Linear(2048, 1024, bias=False, dtype=torch.bfloat16)


class RuntimeInt8MultiComponentTest(unittest.TestCase):
    """A two-transformer architecture must convert BOTH, in one request.

    The bookkeeping is per manager: ``_runtime_int8_converted`` latches as soon
    as a conversion completes, so converting the second transformer through a
    second single-module call would hit the "already converted" branch and return
    it untouched -- leaving Ideogram 4's asymmetric CFG running a quantized
    conditional branch against a bf16 unconditional one, with no warning.
    """

    def _convert(self, components, quantization="int8", manager=None):
        from core import vram_optimization as vo

        manager = manager or _FakeManager()
        models, converted = vo.apply_runtime_int8_quantization_multi(
            manager, components, "ideogram4", quantization)
        return manager, models, converted

    @staticmethod
    def _int8_count(model):
        return sum(1 for m in model.modules() if type(m).__name__ == "Int8Linear")

    def test_both_components_are_converted_and_latched_once(self):
        cond = nn.Sequential(_gate_clearing_linear())
        uncond = nn.Sequential(_gate_clearing_linear())
        manager, models, converted = self._convert([
            ("transformer", "cond", cond),
            ("unconditional_transformer", "uncond", uncond),
        ])
        self.assertTrue(converted)
        self.assertEqual([self._int8_count(m) for m in models], [1, 1])
        self.assertTrue(manager._runtime_int8_converted)
        # The merged audit names both components and keeps their rows apart --
        # identical geometry means identical module paths, which would otherwise
        # collapse into one row.
        document = manager._runtime_int8_audit
        self.assertEqual(document["settings"]["components"],
                         ["transformer", "unconditional_transformer"])
        self.assertEqual(sorted(r["name"] for r in document["layers"]),
                         ["transformer.0", "unconditional_transformer.0"])
        # Summed across the components, keeping the per-format keys the
        # single-component document has.
        self.assertEqual(document["converted"]["int8"], 2)
        self.assertEqual(document["format_counts"], {"int8": 2})

    def test_a_second_single_module_call_would_have_skipped_one(self):
        # The failure mode the multi entry point exists to prevent, pinned: once
        # the latch is set, a further request returns the module unconverted.
        from core import vram_optimization as vo

        manager = _FakeManager()
        cond = nn.Sequential(_gate_clearing_linear())
        uncond = nn.Sequential(_gate_clearing_linear())
        _m, _models, converted = self._convert(
            [("transformer", "cond", cond)], manager=manager)
        self.assertTrue(converted)
        _model, converted_again = vo.apply_runtime_int8_quantization(
            manager, uncond, "ideogram4", "int8", label="uncond")
        self.assertFalse(converted_again)
        self.assertEqual(self._int8_count(uncond), 0)

    def test_non_int8_request_leaves_every_component_alone(self):
        cond = nn.Sequential(_gate_clearing_linear())
        uncond = nn.Sequential(_gate_clearing_linear())
        manager, models, converted = self._convert(
            [("transformer", "cond", cond), ("unconditional_transformer", "uncond", uncond)],
            quantization=None)
        self.assertFalse(converted)
        self.assertEqual([self._int8_count(m) for m in models], [0, 0])
        self.assertFalse(getattr(manager, "_runtime_int8_converted", False))

    def test_a_missing_component_is_dropped_not_guessed(self):
        cond = nn.Sequential(_gate_clearing_linear())
        _manager, models, converted = self._convert([
            ("transformer", "cond", cond),
            ("unconditional_transformer", "uncond", None),
        ])
        self.assertTrue(converted)
        self.assertEqual(len(models), 1)

    def test_single_component_audit_rows_are_not_namespaced(self):
        # Krea 2 / Anima / FLUX.2 audits must stay diffable against the committed
        # offline artifacts, whose rows are bare module paths.
        from core import vram_optimization as vo

        manager = _FakeManager()
        model = nn.Sequential(_gate_clearing_linear())
        vo.apply_runtime_int8_quantization(manager, model, "krea2", "int8")
        self.assertEqual([r["name"] for r in manager._runtime_int8_audit["layers"]], ["0"])


class CheckpointQuantizedProvenanceTest(unittest.TestCase):
    """A checkpoint that ARRIVED quantized was not converted in place.

    The distinction is the whole content of the
    ``runtime_quantization_persistent`` warning ("it was quantized in place
    earlier in this session and the conversion is one-way ... load the model
    again"). On an architecture whose published checkpoints are all quantized
    (Ideogram 4: FP8/nf4) a shared latch would make that false sentence the
    NORMAL case: one int8 request, then every later generation warns about a
    conversion that never happened and a reload that would change nothing.

    Keep-hot, which only cares whether the resident transformer is quantized,
    must still key the two the same.
    """

    def _quantized_pair(self):
        pair = []
        for _ in range(2):
            module = nn.Sequential(_gate_clearing_linear())
            runtime_int8.quantize_linears_in_place(
                module, arch="ideogram4", compute_dtype=torch.bfloat16,
                work_device=None)
            pair.append(module)
        return pair

    def _request(self, manager, components, quantization, warnings):
        from core import vram_optimization as vo

        with patch.object(vo, "_add_generation_warning",
                          lambda message, code=None: warnings.append(code)):
            vo.apply_runtime_int8_quantization_multi(
                manager, components, "ideogram4", quantization)

    def test_an_already_quantized_checkpoint_never_claims_a_conversion(self):
        cond, uncond = self._quantized_pair()
        components = [("transformer", "Transformer", cond),
                      ("unconditional_transformer", "Unconditional Transformer", uncond)]
        manager = _FakeManager()

        warnings = []
        self._request(manager, components, "int8", warnings)
        self.assertEqual(warnings, ["quantization_superseded"])
        self.assertTrue(manager._runtime_int8_from_checkpoint)
        self.assertFalse(getattr(manager, "_runtime_int8_converted", False))

        for quantization in (None, "fp8_e4m3fn", None):
            warnings = []
            self._request(manager, components, quantization, warnings)
            self.assertNotIn("runtime_quantization_persistent", warnings)
            self.assertEqual(warnings, [])

    def test_a_real_runtime_conversion_still_warns(self):
        from core import vram_optimization as vo

        manager = _FakeManager()
        model = nn.Sequential(_gate_clearing_linear())
        vo.apply_runtime_int8_quantization(manager, model, "ideogram4", "int8")
        self.assertTrue(manager._runtime_int8_converted)

        warnings = []
        with patch.object(vo, "_add_generation_warning",
                          lambda message, code=None: warnings.append(code)):
            vo.apply_runtime_int8_quantization(manager, model, "ideogram4", None)
        self.assertEqual(warnings, ["runtime_quantization_persistent"])

    def test_keep_hot_keys_both_latches_as_int8(self):
        from core.keep_hot import compute_model_key

        params = {"unet_quantization": None}
        from_checkpoint = _FakeManager()
        from_checkpoint._runtime_int8_from_checkpoint = True
        converted = _FakeManager()
        converted._runtime_int8_converted = True
        untouched = _FakeManager()

        self.assertEqual(compute_model_key(from_checkpoint, params),
                         compute_model_key(converted, params))
        self.assertNotEqual(compute_model_key(untouched, params),
                            compute_model_key(converted, params))

    def test_a_model_reload_clears_the_checkpoint_latch(self):
        # Otherwise the next checkpoint -- possibly an unquantized one -- would
        # still key keep-hot as quantized.
        source = (Path(_BACKEND) / "core" / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("self._runtime_int8_from_checkpoint = False", source)


class QuantizedArchListDocumentationTest(unittest.TestCase):
    """The API spec must name the architectures the code actually enforces.

    The `quantized_gemm_mode` description spelled the set out by hand and drifted
    twice (it named Ideogram 4 as FP8/nf4-only after it gained INT8, and never
    named FLUX.2 at all). The spec now points at
    `GET /schema/arch-capabilities.quantized_linear_archs`, which this pins to
    ``QUANTIZED_LINEAR_ARCHS``, and the description's own arch ids are checked
    against the same tuple.
    """

    @staticmethod
    def _spec():
        import yaml

        path = Path(_BACKEND).parent / "openapi.yaml"
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_the_endpoint_serves_the_tuple(self):
        import asyncio

        from api.arch_capabilities import QUANTIZED_LINEAR_ARCHS
        from api.routes import get_arch_capabilities

        served = asyncio.run(get_arch_capabilities())
        self.assertEqual(served["quantized_linear_archs"], list(QUANTIZED_LINEAR_ARCHS))

    def test_the_schema_declares_the_field(self):
        schema = self._spec()["components"]["schemas"]["ArchCapabilities"]
        self.assertIn("quantized_linear_archs", schema["properties"])
        self.assertIn("quantized_linear_archs", schema["required"])

    def test_the_description_names_exactly_the_quantized_archs(self):
        from core.models.common.int8_runtime_quantize import (
            ARCH_DISPLAY_NAMES, QUANTIZED_LINEAR_ARCHS,
        )

        description = (self._spec()["components"]["schemas"]["GenerationParams"]
                       ["properties"]["quantized_gemm_mode"]["description"])
        for arch in QUANTIZED_LINEAR_ARCHS:
            self.assertIn(f"`{arch}`", description, arch)
        for arch in ARCH_DISPLAY_NAMES:
            if arch not in QUANTIZED_LINEAR_ARCHS:
                self.assertNotIn(f"`{arch}`", description, arch)


if __name__ == "__main__":
    unittest.main()
