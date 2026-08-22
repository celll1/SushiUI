"""`attention_type` validation/provenance, and the quantized-GEMM fallback cause.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/attention_type_validation_test.py -v

THREE MEASURED DEFECTS, all found by a live API call rather than by a test:

F1. `POST /generate/txt2vid` with `attention_type: "banana"` answered HTTP 200,
    ran native, returned an EMPTY `warnings[]`, and stored
    `"attention_type": "banana"` on the gallery row. Three surfaces failed at
    once: no validation (the field was a bare `str` on every image and video
    route), no warning (the conduit's "unknown backend" console line is deduped
    once per process, so a second bad request left no trace at all), and a row
    that recorded a backend which never ran.

F2. A `w8a8` request that resolved to the dequantized matmul was always
    reported as "The W8A8 path is unavailable on this device/build for these
    layers." On MiniMax-H3 that is false: this box runs `torch._scaled_mm` for
    other architectures, and H3's layers are dequant because its LOADER pins
    them (`disable_scaled_mm` over the whole DiT, because the checkpoint marks
    50 tensors `full_precision_matrix_mult` and gives the other 150 an
    `input_scale` this repo does not read). A reader would conclude a newer GPU
    fixes it.

Each group below carries a NEGATIVE CONTROL that reproduces the pre-fix
behaviour and asserts the test would have failed against it, because a
provenance test that passes on the broken code is worth nothing.
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from api import generation_status as gs  # noqa: E402
from api import quantized_gemm as qg  # noqa: E402
from api.generation_utils import record_attention_backend  # noqa: E402
from core.attention import (  # noqa: E402
    is_known_backend,
    known_backends,
    normalize_backend,
    validate_backend,
)
from core.attention import observed as observed_mod  # noqa: E402
from core.attention.registry import BACKENDS  # noqa: E402


# ---------------------------------------------------------------------------
# F1a -- the vocabulary is DERIVED, and unknown values are refused
# ---------------------------------------------------------------------------
class VocabularyTest(unittest.TestCase):
    def test_vocabulary_covers_registry_aliases_and_passthrough(self):
        vocab = set(known_backends())
        self.assertTrue(set(BACKENDS) <= vocab, "registry keys must be accepted")
        for alias in ("normal", "none", "sdpa"):
            self.assertIn(alias, vocab)
        self.assertIn("sla", vocab, "passthrough backend must stay accepted")
        self.assertNotIn(None, vocab, "None is not a string in the vocabulary")

    def test_vocabulary_follows_a_newly_registered_backend(self):
        """A hand-written list is the drift this repo has been bitten by; adding
        a backend to the registry must extend the accepted vocabulary with no
        other edit."""
        self.assertFalse(is_known_backend("phantom"))
        sentinel = BACKENDS["native"]
        BACKENDS["phantom"] = sentinel
        try:
            self.assertIn("phantom", known_backends())
            self.assertEqual(validate_backend("phantom"), "phantom")
        finally:
            BACKENDS.pop("phantom", None)
        self.assertFalse(is_known_backend("phantom"))

    def test_known_values_are_accepted_case_and_space_insensitively(self):
        for value in known_backends():
            with self.subTest(value=value):
                self.assertEqual(validate_backend(f"  {value.upper()} "), value)

    def test_unknown_value_raises_and_names_the_vocabulary(self):
        with self.assertRaises(ValueError) as ctx:
            validate_backend("banana")
        message = str(ctx.exception)
        self.assertIn("banana", message)
        for value in known_backends():
            self.assertIn(value, message)

    def test_missing_and_empty_resolve_to_the_callers_default(self):
        self.assertEqual(validate_backend(None, default="normal"), "normal")
        self.assertEqual(validate_backend("", default="normal"), "normal")
        self.assertEqual(validate_backend("   ", default="normal"), "normal")
        self.assertIsNone(validate_backend(None))

    def test_negative_control_normalize_backend_still_swallows_it(self):
        """The pre-fix behaviour, kept explicit: `normalize_backend` -- what the
        routes used to be gated on -- ANSWERS `native` for junk. A validation
        test written against it would pass while the defect stood."""
        self.assertEqual(normalize_backend("banana"), "native")


class RouteValidationTest(unittest.TestCase):
    """The route helper turns an unknown backend into a 400 (not a 422/500).

    400-vs-accept: an unknown backend is a CLIENT error, and this module already
    answers 400 for every other closed-vocabulary generation parameter
    (`quantized_gemm_mode`, `loop_decode`). A capability-gated but VALID backend
    keeps the accept-and-warn behaviour those parameters also use.
    """

    def test_unknown_backend_is_a_400_before_the_run_opens(self):
        from api.error_handlers import ValidationError
        from api.routes import _validated_attention_type

        with self.assertRaises(ValidationError) as ctx:
            _validated_attention_type("banana", "normal")
        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)

    def test_valid_backend_passes_through_unchanged(self):
        from api.routes import _validated_attention_type

        self.assertEqual(_validated_attention_type("sage", "normal"), "sage")
        self.assertEqual(_validated_attention_type("normal", "normal"), "normal")
        self.assertEqual(_validated_attention_type(None, "normal"), "normal")

    def test_every_route_that_takes_attention_type_validates_it(self):
        """Read off the live router: a route that accepts the parameter and does
        not validate it is the F1 defect, one endpoint at a time."""
        import inspect

        from api.routes import router

        checked = 0
        for route in router.routes:
            endpoint = getattr(route, "endpoint", None)
            if endpoint is None:
                continue
            try:
                sig = inspect.signature(endpoint)
            except (TypeError, ValueError):
                continue
            takes_it = "attention_type" in sig.parameters
            if not takes_it:
                # Pydantic-bodied routes: look for the field on the model.
                for param in sig.parameters.values():
                    fields = getattr(param.annotation, "model_fields", None)
                    if fields and "attention_type" in fields:
                        takes_it = True
                        break
            if not takes_it:
                continue
            try:
                source = inspect.getsource(endpoint)
            except OSError:  # pragma: no cover
                continue
            # The three training-preview endpoints are thin wrappers that hand
            # the whole request to `_run_training_preview`; that is where their
            # validation lives, so follow the delegation rather than demanding
            # the call be inlined in each wrapper.
            if "_run_training_preview" in source:
                from api.routes import _run_training_preview

                source += inspect.getsource(_run_training_preview)
            checked += 1
            with self.subTest(route=getattr(route, "path", endpoint.__name__)):
                self.assertIn(
                    "_validated_attention_type", source,
                    f"{endpoint.__name__} accepts attention_type without validating it",
                )
        self.assertGreaterEqual(checked, 6, "expected every image + video route")


class OpenApiEnumTest(unittest.TestCase):
    def test_spec_enum_matches_the_derived_vocabulary(self):
        spec = yaml.safe_load(open(os.path.join(_REPO, "openapi.yaml"), encoding="utf-8"))
        schemas = spec["components"]["schemas"]
        found = 0
        for name, schema in schemas.items():
            field = (schema.get("properties") or {}).get("attention_type")
            if not isinstance(field, dict) or "enum" not in field:
                continue
            found += 1
            with self.subTest(schema=name):
                self.assertEqual(set(field["enum"]), set(known_backends()))
        self.assertGreaterEqual(found, 2, "image + video request schemas")


class FrontendAttentionSenderTest(unittest.TestCase):
    @staticmethod
    def _function_source(source: str, name: str) -> str:
        start = source.index(f"export const {name} =")
        end = source.find("\nexport const ", start + 1)
        return source[start:end if end >= 0 else None]

    def test_every_generation_sender_resolves_the_global_backend_first(self):
        api_path = os.path.join(_REPO, "frontend", "src", "utils", "api.ts")
        with open(api_path, encoding="utf-8") as handle:
            source = handle.read()

        for name in (
            "generateTxt2Img",
            "generateTxt2ImgTrainingPreview",
            "generateImg2ImgTrainingPreview",
            "generateInpaintTrainingPreview",
            "generateImg2Img",
            "generateUpscale",
            "generateTxt2Vid",
            "generateImg2Vid",
            "generateRef2Vid",
            "generateInpaint",
            "generateOutpaint",
            "generateOutpaintVideo",
            "generateInpaintVideo",
        ):
            function_source = self._function_source(source, name)
            with self.subTest(sender=name):
                self.assertIn(
                    "resolveGlobalAttentionType(params.attention_type)",
                    function_source,
                )
                self.assertNotRegex(function_source, r"params\.attention_type\s*\|\|")

    def test_image_senders_resolve_the_global_flux_implementation_first(self):
        api_path = os.path.join(_REPO, "frontend", "src", "utils", "api.ts")
        with open(api_path, encoding="utf-8") as handle:
            source = handle.read()

        for name in (
            "generateTxt2Img",
            "generateTxt2ImgTrainingPreview",
            "generateImg2ImgTrainingPreview",
            "generateInpaintTrainingPreview",
            "generateImg2Img",
            "generateUpscale",
            "generateInpaint",
            "generateOutpaint",
        ):
            with self.subTest(sender=name):
                self.assertIn(
                    "resolveGlobalAttentionImpl(params.attention_impl)",
                    self._function_source(source, name),
                )

    def test_global_setting_reader_accepts_every_ui_option_including_tq(self):
        helper_path = os.path.join(
            _REPO, "frontend", "src", "utils", "attentionSettings.ts")
        with open(helper_path, encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn('["normal", "sage", "flash", "tq"]', source)
        self.assertIn("readGlobalAttentionType() ?? fallback ?? \"normal\"", source)

    def test_every_generation_panel_uses_the_shared_setting_reader(self):
        for name in (
            "Txt2ImgPanel.tsx",
            "Img2ImgPanel.tsx",
            "InpaintPanel.tsx",
            "OutpaintPanel.tsx",
        ):
            path = os.path.join(
                _REPO, "frontend", "src", "components", "generation", name)
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            with self.subTest(panel=name):
                self.assertIn("readGlobalAttentionType()", source)
                self.assertNotIn("savedAttentionType === 'flash'", source)


class DiffusionUpscaleAttentionTest(unittest.TestCase):
    def test_attention_settings_reach_each_img2img_tile(self):
        from PIL import Image
        from core.upscaler import run_diffusion_upscale

        class PipelineManager:
            current_model_info = object()

            def __init__(self):
                self.calls = []

            def generate_img2img(self, params, image, **_kwargs):
                self.calls.append(dict(params))
                return image, params["seed"], None

        manager = PipelineManager()
        params = {
            "scale_factor": 1.0,
            "pil_resample": "lanczos",
            "tile_size": 0,
            "tile_overlap": 0,
            "diffusion_denoising_strength": 0.3,
            "seed": 7,
            "attention_type": "flash",
            "attention_impl": "conduit",
        }

        run_diffusion_upscale(
            params, Image.new("RGB", (8, 8)), manager)

        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(manager.calls[0]["attention_type"], "flash")
        self.assertEqual(manager.calls[0]["attention_impl"], "conduit")


# ---------------------------------------------------------------------------
# F1b -- the row records what RAN, not what was asked for
# ---------------------------------------------------------------------------
class ObservedBackendTest(unittest.TestCase):
    def setUp(self):
        observed_mod.reset()
        self.addCleanup(observed_mod.reset)
        # Clear this thread's generation identity: the tests that call
        # `begin_generation` directly must not inherit an id another test left
        # in the ContextVar (production always enters through
        # `start_generation`, which sets both).
        token = gs._current_generation.set(0)
        self.addCleanup(lambda: gs._current_generation.reset(token))

    def test_nothing_observed_records_nothing(self):
        observed_mod.begin_generation(7)
        params = {"attention_type": "flash"}
        self.assertIsNone(record_attention_backend(params, 7))
        self.assertNotIn("attention_backend", params)

    def test_records_the_backend_that_ran(self):
        observed_mod.begin_generation(7)
        observed_mod.note_backend("sage")
        params = {"attention_type": "sage"}
        self.assertEqual(record_attention_backend(params, 7), "sage")
        self.assertEqual(params["attention_backend"], "sage")

    def test_a_downgrade_is_recorded_as_what_ran_and_warned(self):
        gid = gs.start_generation("txt2vid")
        observed_mod.begin_generation(gid)
        observed_mod.note_backend("native")          # sage was refused per call
        params = {"attention_type": "sage"}
        record_attention_backend(params, gid)
        self.assertEqual(params["attention_backend"], "native")
        codes = [w["code"] for w in gs.get_warnings(gid)]
        self.assertIn("attention_downgrade", codes)
        gs.complete_generation(generation_id=gid)

    def test_an_alias_request_is_not_reported_as_a_downgrade(self):
        gid = gs.start_generation("txt2img")
        observed_mod.begin_generation(gid)
        observed_mod.note_backend("native")
        params = {"attention_type": "normal"}        # alias OF native
        record_attention_backend(params, gid)
        self.assertEqual(params["attention_backend"], "native")
        self.assertEqual(gs.get_warnings(gid), [])
        gs.complete_generation(generation_id=gid)

    def test_observations_are_not_attributed_to_another_generation(self):
        observed_mod.begin_generation(11)
        observed_mod.note_backend("flash")
        self.assertEqual(observed_mod.observed_backends(11), ("flash",))
        self.assertEqual(observed_mod.observed_backends(12), ())

    def test_overlapping_generations_keep_their_own_observations(self):
        """The queue case, not a race: request B is admitted (and calls
        `start_generation`) while request A is still denoising. A's attention
        calls must stay on A's row -- an earlier draft keyed a single global set
        on "the newest generation", which silently emptied A's record."""
        import contextvars

        ctx_a, ctx_b = contextvars.copy_context(), contextvars.copy_context()
        gid_a = ctx_a.run(lambda: gs.start_generation("txt2img"))
        gid_b = ctx_b.run(lambda: gs.start_generation("txt2img"))   # queued
        ctx_a.run(lambda: observed_mod.note_backend("sage"))        # A is running
        ctx_b.run(lambda: observed_mod.note_backend("native"))
        self.assertEqual(observed_mod.observed_backends(gid_a), ("sage",))
        self.assertEqual(observed_mod.observed_backends(gid_b), ("native",))
        params_a = {"attention_type": "sage"}
        ctx_a.run(lambda: record_attention_backend(params_a, gid_a))
        self.assertEqual(params_a["attention_backend"], "sage")
        gs.complete_generation(generation_id=gid_a)
        gs.complete_generation(generation_id=gid_b)

    def test_start_generation_arms_the_recorder(self):
        gid = gs.start_generation("txt2img")
        observed_mod.note_backend("tq")
        self.assertEqual(observed_mod.observed_backends(gid), ("tq",))
        gs.complete_generation(generation_id=gid)

    def test_the_conduit_records_the_backend_it_dispatched(self):
        """End to end through the real conduit: `native` is the terminal
        backend, so this holds with no optional kernel installed."""
        import torch

        from core.attention import AttentionMode, dispatch_attention

        observed_mod.begin_generation(21)
        q = torch.randn(1, 4, 2, 8)
        dispatch_attention(q, q, q, backend="normal", mode=AttentionMode.INFERENCE)
        self.assertEqual(observed_mod.observed_backends(21), ("native",))

    def test_negative_control_recording_the_request_would_be_wrong(self):
        """Reverting to "record what was requested" reproduces the measured
        defect: the row names a backend that never ran."""
        observed_mod.begin_generation(31)
        observed_mod.note_backend("native")
        params = {"attention_type": "sage"}
        reverted = dict(params, attention_backend=params["attention_type"])
        record_attention_backend(params, 31)
        self.assertEqual(reverted["attention_backend"], "sage")   # the old lie
        self.assertNotEqual(params["attention_backend"], reverted["attention_backend"])


# ---------------------------------------------------------------------------
# F2 -- the quantized-GEMM fallback names the RIGHT cause
# ---------------------------------------------------------------------------
_OLD_GENERIC = "The W8A8 path is unavailable on this device/build for these layers."


class DequantCauseTest(unittest.TestCase):
    """`describe_gemm_path` already separates the causes in its label; the
    message must not flatten them back into one sentence.

    * bare `dequant` + flag ON  -> every layer opted out == the loader's
      architecture policy pin (MiniMax-H3, SenseNova), NOT the hardware.
    * `dequant(scaled_mm unavailable)` -> the per-device probe rejected it: the
      genuine device/build limitation.
    * `dequant(scaled_mm unprobed)`    -> nothing reached the probe.
    """

    def setUp(self):
        self._flags = {"fp8": True, "int8": True}
        self._real = qg._gemm_flags_enabled
        qg._gemm_flags_enabled = lambda: dict(self._flags)
        self.addCleanup(lambda: setattr(qg, "_gemm_flags_enabled", self._real))

    def test_policy_pin_is_not_reported_as_a_hardware_limit(self):
        message = qg._dequant_cause("dequant", "minimax_h3")
        self.assertIn("pinned to the dequantized path", message)
        self.assertIn("disable_scaled_mm", message)
        self.assertIn("minimax_h3", message)
        self.assertIn("NOT a device or build", message)
        self.assertNotIn("unavailable on this device/build", message)

    def test_int8_policy_pin_is_not_reported_as_a_hardware_limit(self):
        """SenseNova's `disable_int8_mm` pin, the INT8-axis sibling of
        `test_policy_pin_is_not_reported_as_a_hardware_limit` above -- a
        DIFFERENT `disabler` name must appear on this stem/format pair."""
        message = qg._dequant_cause("int8_dequant", "sensenova")
        self.assertIn("pinned to the dequantized path", message)
        self.assertIn("disable_int8_mm", message)
        self.assertNotIn("disable_scaled_mm", message)
        self.assertIn("sensenova", message)
        self.assertIn("INT8", message)
        self.assertIn("NOT a device or build", message)
        self.assertNotIn("unavailable on this device/build", message)

    def test_probe_rejection_is_reported_as_a_device_build_limit(self):
        message = qg._dequant_cause("dequant(scaled_mm unavailable)", "krea2")
        self.assertIn("unavailable on this device/build", message)
        self.assertIn("probe", message)
        self.assertNotIn("pinned to the dequantized path", message)

    def test_unprobed_is_reported_as_unprobed(self):
        message = qg._dequant_cause("dequant(scaled_mm unprobed)", "krea2")
        self.assertIn("no quantized Linear forward", message)
        self.assertNotIn("unavailable on this device/build", message)

    def test_a_flag_that_could_not_be_set_is_named_as_such(self):
        self._flags["fp8"] = False
        message = qg._dequant_cause("dequant", "krea2")
        self.assertIn("process flag is off", message)
        self.assertNotIn("pinned to the dequantized path", message)

    def test_int8_stem_is_described_on_its_own_axis(self):
        message = qg._dequant_cause("int8_dequant(int_mm unavailable)", "anima")
        self.assertIn("INT8", message)
        self.assertIn("unavailable on this device/build", message)

    def test_mixed_checkpoint_describes_both_stems(self):
        message = qg._dequant_cause("dequant+int8_dequant(int_mm unavailable)", "krea2")
        self.assertIn("FP8", message)
        self.assertIn("INT8", message)
        self.assertIn("pinned to the dequantized path", message)

    def test_the_warning_carries_the_specific_cause(self):
        gid = gs.start_generation("txt2vid")
        message = qg.report_quantized_gemm_outcome("w8a8", "dequant", "minimax_h3")
        self.assertIsNotNone(message)
        self.assertIn("resolved path: dequant", message)
        self.assertIn("pinned to the dequantized path", message)
        warnings = gs.get_warnings(gid)
        self.assertIn("quantization_fallback", [w["code"] for w in warnings])
        gs.complete_generation(generation_id=gid)

    def test_negative_control_the_old_generic_message_fails_this(self):
        """Revert `_dequant_cause` to the single pre-fix sentence and the
        MiniMax-H3 assertions above stop holding."""
        original = qg._dequant_cause
        qg._dequant_cause = lambda label, arch: _OLD_GENERIC
        self.addCleanup(lambda: setattr(qg, "_dequant_cause", original))
        message = qg.report_quantized_gemm_outcome("w8a8", "dequant", "minimax_h3")
        self.assertIn(_OLD_GENERIC, message)
        self.assertNotIn("pinned to the dequantized path", message)

    def test_w8a8_that_really_ran_reports_nothing(self):
        self.assertIsNone(qg.report_quantized_gemm_outcome(
            "w8a8", "w8a8_scaled_mm(tensorwise)", "krea2"))

    def test_packed_w4a8_is_not_reported_as_missing_quantized_linears(self):
        message = qg.report_quantized_gemm_outcome(
            "w8a8", "w4a8_int8(comfy-kitchen)", "minimax_h3")
        self.assertIn("does not control", message)
        self.assertIn("Comfy-Kitchen", message)
        self.assertNotIn("carries no weight-only quantized", message)


if __name__ == "__main__":
    unittest.main()
