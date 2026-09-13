"""Header-only production checkpoint gates; opt in with YUE2_TEST_CHECKPOINT."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from core.models.yue2 import loader


@unittest.skipUnless(os.environ.get("YUE2_TEST_CHECKPOINT"), "Set YUE2_TEST_CHECKPOINT for the staged header gate")
class YuE2Header(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.environ["YUE2_TEST_CHECKPOINT"]
        cls.header = loader.read_header(cls.path)

    def test_complete_census(self):
        result = loader.preflight_yue2(self.path)
        self.assertEqual(result["source_tensor_count"], 1355)
        self.assertEqual(len(result["quantized"]), 397)
        self.assertEqual(set(item.source for item in result["plan"]), set(self.header) - {"__metadata__"})
        self.assertEqual(len(result["plan"]), len({item.target for item in result["plan"]}))
        self.assertFalse(torch.cuda.is_initialized())

    def test_reject_incomplete_or_ambiguous_headers(self):
        weight = "text_encoders.model.layers.0.self_attn.qkv_proj.weight"
        variants = []
        for missing in (loader.TOKENIZER_KEY, "vae.decoder.layers.0.weight_v", weight.replace(".weight", ".comfy_quant")):
            header = copy.deepcopy(self.header)
            del header[missing]
            variants.append(header)
        header = copy.deepcopy(self.header)
        header[weight]["shape"][0] -= 1
        variants.append(header)
        header = copy.deepcopy(self.header)
        header[weight.replace(".weight", ".weight_scale")]["shape"] = [1]
        variants.append(header)
        header = copy.deepcopy(self.header)
        header["text_encoders.model.unknown.weight"] = dict(dtype="F32", shape=[1])
        variants.append(header)
        for header in variants:
            with self.subTest(keys=len(header)), patch.object(loader, "read_header", return_value=header):
                with self.assertRaises(ValueError):
                    loader.preflight_yue2(self.path)

    def test_failed_preflight_preserves_loaded_model(self):
        from core.pipeline import DiffusionPipelineManager
        manager = DiffusionPipelineManager.__new__(DiffusionPipelineManager)
        sentinel = {"type": "existing"}
        manager.current_model_info = sentinel
        manager.current_model = "existing"
        with patch.object(loader, "is_yue2_checkpoint", return_value=True), \
                patch.object(loader, "preflight_yue2", side_effect=ValueError("bad checkpoint")):
            with self.assertRaisesRegex(ValueError, "bad checkpoint"):
                manager._load_model_locked("safetensors", self.path)
        self.assertIs(manager.current_model_info, sentinel)
        self.assertEqual(manager.current_model, "existing")


if __name__ == "__main__":
    unittest.main()
