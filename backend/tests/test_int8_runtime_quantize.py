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


if __name__ == "__main__":
    unittest.main()
