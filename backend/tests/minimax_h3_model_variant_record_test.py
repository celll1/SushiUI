"""`model_variant` must record which MiniMax-H3 partition (fl2va/ref2va)
actually ran, the same way `record_attention_backend` records the attention
backend that actually ran rather than the one requested (see e35f6991).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_model_variant_record_test.py -v

MUTANT (see the last test group): echo whatever the CALLER already put in
`params["model_variant"]` (or a request field) instead of reading the
loader's resolved value off `pipeline_manager.current_model_info`. It passes
whenever nothing stale is lying around, but a request that carries a leftover
or attacker-supplied `model_variant` would have it recorded verbatim -- the
same "assert a kernel that never executed" failure the attention-backend
defect had.
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import unittest  # noqa: E402

from api.generation_utils import record_model_variant  # noqa: E402


class _FakePipelineManager:
    def __init__(self, is_minimax_h3_model, current_model_info):
        self.is_minimax_h3_model = is_minimax_h3_model
        self.current_model_info = current_model_info


class RecordModelVariantTest(unittest.TestCase):
    def test_records_fl2va_when_that_partition_is_loaded(self):
        params = {}
        pm = _FakePipelineManager(True, {"variant": "fl2va"})
        result = record_model_variant(params, pm)
        self.assertEqual(result, "fl2va")
        self.assertEqual(params["model_variant"], "fl2va")

    def test_records_ref2va_when_that_partition_is_loaded(self):
        params = {}
        pm = _FakePipelineManager(True, {"variant": "ref2va"})
        record_model_variant(params, pm)
        self.assertEqual(params["model_variant"], "ref2va")

    def test_no_op_for_a_non_h3_architecture(self):
        params = {}
        pm = _FakePipelineManager(False, {"variant": "fl2va"})
        result = record_model_variant(params, pm)
        self.assertIsNone(result)
        self.assertNotIn("model_variant", params)

    def test_no_op_when_h3_is_loaded_but_variant_is_unknown(self):
        params = {}
        pm = _FakePipelineManager(True, {})
        result = record_model_variant(params, pm)
        self.assertIsNone(result)
        self.assertNotIn("model_variant", params)

    def test_never_raises_on_a_pipeline_manager_missing_attributes(self):
        class _Empty:
            pass

        params = {}
        result = record_model_variant(params, _Empty())
        self.assertIsNone(result)
        self.assertNotIn("model_variant", params)

    def test_reads_the_loader_value_not_a_value_already_in_params(self):
        # The loaded model is ref2va; a stale/attacker-supplied params value
        # must NOT survive -- the row has to say what actually ran.
        params = {"model_variant": "fl2va"}
        pm = _FakePipelineManager(True, {"variant": "ref2va"})
        record_model_variant(params, pm)
        self.assertEqual(params["model_variant"], "ref2va")

    def test_mutant_echo_params_value_would_report_the_wrong_partition(self):
        """MUTANT: record whatever `params` already claims instead of the
        loader's resolved variant -- proves the previous test actually
        distinguishes "reads pipeline_manager" from "trusts the caller".
        """

        def mutant_record_model_variant(params, pipeline_manager):
            if not getattr(pipeline_manager, "is_minimax_h3_model", False):
                return None
            existing = params.get("model_variant")
            if not existing:
                return None
            params["model_variant"] = existing
            return existing

        params = {"model_variant": "fl2va"}
        pm = _FakePipelineManager(True, {"variant": "ref2va"})
        mutant_record_model_variant(params, pm)
        self.assertNotEqual(params["model_variant"], "ref2va")


if __name__ == "__main__":
    unittest.main()
