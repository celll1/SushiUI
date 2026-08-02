import json
import sys
import unittest
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.krea2.vendor.single_file import reject_unsupported_quant


def _marker(config):
    return torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


class Krea2QuantRejectionTest(unittest.TestCase):
    def test_scans_all_file_metadata(self):
        with self.assertRaisesRegex(ValueError, "int8_convrot"):
            reject_unsupported_quant(
                "renamed.safetensors",
                {"quantization_format": "int8_convrot"},
            )

    def test_detects_convrot_from_layer_marker(self):
        state_dict = {
            "blocks.0.attn.wq.comfy_quant": _marker({
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            })
        }
        with self.assertRaisesRegex(ValueError, "int8_convrot"):
            reject_unsupported_quant("renamed.safetensors", {}, state_dict)

    def test_detects_nested_convrot_marker(self):
        state_dict = {
            "blocks.0.attn.wq.comfy_quant": _marker({
                "format": "int8_tensorwise",
                "params": {"convrot": True},
            })
        }
        with self.assertRaisesRegex(ValueError, "int8_convrot"):
            reject_unsupported_quant("renamed.safetensors", {}, state_dict)

    def test_plain_int8_marker_is_not_mislabeled_as_convrot(self):
        state_dict = {
            "blocks.0.attn.wq.comfy_quant": _marker({
                "format": "int8_tensorwise",
                "convrot": False,
            })
        }
        reject_unsupported_quant("renamed.safetensors", {}, state_dict)

    def test_detects_other_unsupported_layer_markers(self):
        for token in ("mxfp8", "nvfp4"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, token):
                reject_unsupported_quant(
                    "renamed.safetensors",
                    {},
                    {"blocks.0.attn.wq.comfy_quant": _marker({"format": token})},
                )


if __name__ == "__main__":
    unittest.main()
